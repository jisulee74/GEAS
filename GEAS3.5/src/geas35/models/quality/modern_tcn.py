"""ModernTCN-style quality model for feature-level masked reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from geas35.models.quality.base import BaseQualityModel, QualityModelOutput
from geas35.models.quality.deep import (
    TorchUnavailableError,
    build_sliding_windows,
    build_valid_observation_mask,
    confidence_from_scores,
    require_torch,
    score_iqr_scale,
)
from geas35.models.quality.hyperparameters import (
    CandidateTrainingResult,
    EarlyStoppingConfig,
    EarlyStoppingTracker,
    HyperparameterCandidate,
)
from geas35.models.quality.rule_only import RuleOnlyQualityModel


@dataclass(frozen=True)
class ModernTCNConfig:
    """Configuration for the initial ModernTCN quality model implementation."""

    lookback: int = 288
    mask_fraction: float = 0.3
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    channel_width: int = 32
    depth: tuple[int, ...] = (1, 1, 1)
    kernel_size: int = 13
    dropout: float = 0.0
    random_state: int = 0
    expected_frequency: str | pd.Timedelta | None = "5min"

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1.")
        if not 0.0 < self.mask_fraction <= 1.0:
            raise ValueError("mask_fraction must be in the interval (0, 1].")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0.")
        if self.channel_width < 1:
            raise ValueError("channel_width must be >= 1.")
        normalized_depth = tuple(int(value) for value in self.depth)
        if not normalized_depth or any(value < 1 for value in normalized_depth):
            raise ValueError("depth must contain positive stage depths.")
        object.__setattr__(self, "depth", normalized_depth)
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be >= 1.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    @classmethod
    def from_candidate(cls, candidate: HyperparameterCandidate) -> "ModernTCNConfig":
        payload = {**candidate.common, **candidate.model_specific}
        allowed = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in payload.items() if key in allowed}
        return cls(**filtered)

    def to_artifact(self) -> dict[str, object]:
        return dict(asdict(self))


class ModernTCNQualityModel(BaseQualityModel):
    """Observation-only ModernTCN candidate using masked current-state reconstruction."""

    model_name = "modern_tcn"

    def __init__(
        self,
        config: ModernTCNConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.config = config or ModernTCNConfig()
        self.device = device
        self.model_ = None
        self.feature_means_: pd.Series | None = None
        self.feature_stds_: pd.Series | None = None
        self.confidence_scales_: pd.Series | None = None
        self.training_history_: list[dict[str, float | int]] = []
        self.best_epoch_: int | None = None
        self.rule_model_ = RuleOnlyQualityModel()

    def fit(
        self,
        train_df: pd.DataFrame,
        observation_columns: Iterable[str],
        *,
        time_column: str = "reg_date",
        group_columns: Iterable[str] = (),
        valid_mask: pd.DataFrame | pd.Series | None = None,
        validation_df: pd.DataFrame | None = None,
        early_stopping: EarlyStoppingConfig | None = None,
    ) -> "ModernTCNQualityModel":
        columns = self._validate_columns(observation_columns)
        self._require_columns(train_df, columns, label="Training")
        torch = require_torch()
        nn = torch.nn

        self._set_random_seed(torch)
        self.observation_columns_ = columns
        self.rule_model_.fit(train_df, columns)

        train_values = train_df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        train_valid_mask = self._fit_valid_mask(train_df, columns, valid_mask)
        self.feature_means_, self.feature_stds_ = self._fit_scaler(
            train_values,
            train_valid_mask,
        )
        train_windows = build_sliding_windows(
            train_df,
            lookback=self.config.lookback,
            time_column=time_column,
            group_columns=group_columns,
            expected_frequency=self.config.expected_frequency,
        )
        if not train_windows:
            raise ValueError(
                f"{self.__class__.__name__} requires at least one training window."
            )

        model = self._build_network(
            torch,
            nn,
            n_features=len(columns),
        )
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        tracker = EarlyStoppingTracker(
            early_stopping or EarlyStoppingConfig(patience=max(1, self.config.epochs // 4))
        )
        train_array = self._standardized_array(train_df, columns)
        train_valid = train_valid_mask.to_numpy(dtype=bool)
        rng = np.random.default_rng(self.config.random_state)

        best_state = None
        best_epoch = None
        self.training_history_ = []
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(
                torch,
                model,
                optimizer,
                device,
                train_array,
                train_valid,
                train_windows,
                rng,
            )
            metrics: dict[str, float | int] = {
                "epoch": epoch,
                "train_reconstruction_loss": train_loss,
            }
            if validation_df is not None:
                metrics["validation_reconstruction_loss"] = (
                    self._validation_reconstruction_loss(
                        torch,
                        model,
                        device,
                        validation_df,
                        columns,
                        time_column=time_column,
                        group_columns=group_columns,
                    )
                )
            else:
                metrics["validation_reconstruction_loss"] = train_loss
            self.training_history_.append(metrics)

            should_stop = tracker.update(epoch, metrics)
            if tracker.best_epoch == epoch:
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if should_stop:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.best_epoch_ = best_epoch or tracker.best_epoch
        self.model_ = model

        train_prediction = self.reconstruct(
            train_df,
            columns,
            time_column=time_column,
            group_columns=group_columns,
        )
        train_scores = self.anomaly_score(train_df, train_prediction, columns)
        self.confidence_scales_ = score_iqr_scale(train_scores)
        return self

    def reconstruct(
        self,
        df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
        *,
        time_column: str = "reg_date",
        group_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        columns = self._resolved_columns(observation_columns)
        self._require_fitted()
        torch = require_torch()
        values = self._standardized_array(df, columns)
        windows = build_sliding_windows(
            df,
            lookback=self.config.lookback,
            time_column=time_column,
            group_columns=group_columns,
            expected_frequency=self.config.expected_frequency,
        )
        predictions = np.tile(
            self.feature_means_.reindex(list(columns)).to_numpy(dtype=float),
            (len(df), 1),
        )
        if windows:
            device = next(self.model_.parameters()).device
            self.model_.eval()
            with torch.no_grad():
                for start in range(0, len(windows), self.config.batch_size):
                    batch_windows = windows[start : start + self.config.batch_size]
                    batch = np.stack([values[list(window.positions)] for window in batch_windows])
                    tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
                    pred = self.model_(tensor).detach().cpu().numpy()
                    current_positions = [window.current_position for window in batch_windows]
                    predictions[current_positions, :] = self._inverse_standardize(pred, columns)
        return pd.DataFrame(predictions, index=df.index, columns=list(columns))

    def forecast(
        self,
        df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
        *,
        horizon: int = 1,
        time_column: str = "reg_date",
        group_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        return self.reconstruct(
            df,
            observation_columns,
            time_column=time_column,
            group_columns=group_columns,
        )

    def anomaly_score(
        self,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        columns = self._resolved_columns(observation_columns)
        values = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        predictions = prediction_df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        return (values - predictions).abs()

    def predict_outlier(
        self,
        df: pd.DataFrame,
        thresholds: object | None = None,
        observation_columns: Iterable[str] | None = None,
    ) -> QualityModelOutput:
        columns = self._resolved_columns(observation_columns)
        prediction = self.reconstruct(df, columns)
        scores = self.anomaly_score(df, prediction, columns)
        outlier_flags = pd.DataFrame(0, index=df.index, columns=list(columns))
        confidence_scores = None
        if thresholds is not None:
            for col in columns:
                threshold = _resolve_threshold(thresholds, col)
                if threshold is not None:
                    outlier_flags[col] = (
                        pd.to_numeric(scores[col], errors="coerce") >= threshold
                    ).astype(int)
            confidence_scores = confidence_from_scores(
                scores,
                thresholds,
                self._confidence_scales(columns),
            )
        invalid_mask = self.rule_model_.invalid_mask(df, columns) | outlier_flags.astype(bool)
        return QualityModelOutput(
            frame=df.copy(),
            observation_columns=columns,
            anomaly_scores=scores,
            outlier_flags=outlier_flags,
            invalid_mask=invalid_mask,
            confidence_scores=confidence_scores,
        )

    def impute(
        self,
        df: pd.DataFrame,
        invalid_mask: pd.DataFrame,
        prediction_df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        columns = self._resolved_columns(observation_columns)
        result = df.copy()
        for col in columns:
            if col not in result.columns or col not in prediction_df.columns:
                continue
            # Reconstruction values are floats even when the raw sensor column
            # was inferred as an integer dtype. Cast before assignment.
            result[col] = pd.to_numeric(result[col], errors="coerce").astype(float)
            mask = invalid_mask[col].astype(bool) if col in invalid_mask.columns else False
            replacement = pd.to_numeric(prediction_df[col], errors="coerce")
            result.loc[mask, col] = replacement.loc[mask]
        return result

    def to_artifact(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "config": self.config.to_artifact(),
            "observation_columns": []
            if self.observation_columns_ is None
            else list(self.observation_columns_),
            "feature_means": None
            if self.feature_means_ is None
            else self.feature_means_.to_dict(),
            "feature_stds": None
            if self.feature_stds_ is None
            else self.feature_stds_.to_dict(),
            "confidence_scales": None
            if self.confidence_scales_ is None
            else self.confidence_scales_.to_dict(),
            "best_epoch": self.best_epoch_,
            "training_history": self.training_history_,
        }

    def _train_epoch(
        self,
        torch,
        model,
        optimizer,
        device,
        values: np.ndarray,
        valid_mask: np.ndarray,
        windows,
        rng: np.random.Generator,
    ) -> float:
        model.train()
        order = rng.permutation(len(windows))
        losses: list[float] = []
        for start in range(0, len(order), self.config.batch_size):
            selected_windows = [windows[int(idx)] for idx in order[start : start + self.config.batch_size]]
            batch, target, loss_mask = self._masked_batch(values, valid_mask, selected_windows, rng)
            if not bool(loss_mask.any()):
                continue
            batch_tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
            target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(loss_mask, dtype=torch.bool, device=device)
            optimizer.zero_grad()
            prediction = model(batch_tensor)
            loss = ((prediction - target_tensor) ** 2)[mask_tensor].mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    def _validation_reconstruction_loss(
        self,
        torch,
        model,
        device,
        validation_df: pd.DataFrame,
        columns: tuple[str, ...],
        *,
        time_column: str,
        group_columns: Iterable[str],
    ) -> float:
        windows = build_sliding_windows(
            validation_df,
            lookback=self.config.lookback,
            time_column=time_column,
            group_columns=group_columns,
            expected_frequency=self.config.expected_frequency,
        )
        if not windows:
            return float("nan")
        values = self._standardized_array(validation_df, columns)
        valid = build_valid_observation_mask(validation_df, columns).to_numpy(dtype=bool)
        rng = np.random.default_rng(self.config.random_state + 1000)
        losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(windows), self.config.batch_size):
                batch_windows = windows[start : start + self.config.batch_size]
                batch, target, loss_mask = self._masked_batch(values, valid, batch_windows, rng)
                if not bool(loss_mask.any()):
                    continue
                batch_tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
                target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
                mask_tensor = torch.as_tensor(loss_mask, dtype=torch.bool, device=device)
                prediction = model(batch_tensor)
                loss = ((prediction - target_tensor) ** 2)[mask_tensor].mean()
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    def _masked_batch(
        self,
        values: np.ndarray,
        valid_mask: np.ndarray,
        windows,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = np.stack([values[list(window.positions)] for window in windows])
        target = np.stack([values[window.current_position] for window in windows])
        loss_mask = np.zeros_like(target, dtype=bool)
        for row_idx, window in enumerate(windows):
            candidates = np.flatnonzero(valid_mask[window.current_position])
            if len(candidates) == 0:
                continue
            n_mask = max(1, int(round(len(candidates) * self.config.mask_fraction)))
            n_mask = min(n_mask, len(candidates))
            selected = rng.choice(candidates, size=n_mask, replace=False)
            batch[row_idx, -1, selected] = 0.0
            loss_mask[row_idx, selected] = True
        return batch, target, loss_mask

    def _fit_valid_mask(
        self,
        df: pd.DataFrame,
        columns: tuple[str, ...],
        valid_mask: pd.DataFrame | pd.Series | None,
    ) -> pd.DataFrame:
        if valid_mask is None:
            return build_valid_observation_mask(df, columns)
        if isinstance(valid_mask, pd.DataFrame):
            return valid_mask.loc[:, list(columns)].astype(bool)
        row_mask = valid_mask.astype(bool)
        return pd.DataFrame(
            {col: row_mask.to_numpy(dtype=bool) for col in columns},
            index=df.index,
        )

    def _fit_scaler(
        self,
        values: pd.DataFrame,
        valid_mask: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        valid_values = values.where(valid_mask)
        means = valid_values.mean(skipna=True).astype(float)
        medians = values.median(skipna=True).astype(float)
        means = means.fillna(medians).fillna(0.0)
        stds = valid_values.std(skipna=True, ddof=0).astype(float)
        stds = stds.where(np.isfinite(stds) & (stds > 0.0), 1.0)
        return means, stds

    def _standardized_array(
        self,
        df: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> np.ndarray:
        means = self.feature_means_.reindex(list(columns)).astype(float)
        stds = self.feature_stds_.reindex(list(columns)).astype(float)
        values = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        values = values.fillna(means)
        return ((values - means) / stds).to_numpy(dtype=np.float32)

    def _inverse_standardize(
        self,
        values: np.ndarray,
        columns: tuple[str, ...],
    ) -> np.ndarray:
        means = self.feature_means_.reindex(list(columns)).to_numpy(dtype=float)
        stds = self.feature_stds_.reindex(list(columns)).to_numpy(dtype=float)
        return values * stds + means

    def _confidence_scales(self, columns: tuple[str, ...]) -> pd.Series:
        if self.confidence_scales_ is None:
            return pd.Series(1.0, index=list(columns), dtype="float64")
        return self.confidence_scales_.reindex(list(columns)).fillna(1.0).astype(float)

    def _require_fitted(self) -> None:
        if self.model_ is None or self.feature_means_ is None or self.feature_stds_ is None:
            raise ValueError("ModernTCNQualityModel is not fitted.")

    def _require_columns(
        self,
        df: pd.DataFrame,
        columns: tuple[str, ...],
        *,
        label: str,
    ) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview = f"{preview}, ..."
            raise ValueError(f"{label} frame is missing observation columns: {preview}")

    def _set_random_seed(self, torch) -> None:
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    def _build_network(self, torch, nn, *, n_features: int):
        return _build_modern_tcn_network(
            torch,
            nn,
            n_features=n_features,
            channel_width=self.config.channel_width,
            depth=self.config.depth,
            kernel_size=self.config.kernel_size,
            dropout=self.config.dropout,
        )


def train_modern_tcn_candidate(
    candidate: HyperparameterCandidate,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    observation_columns: tuple[str, ...],
    early_stopping: EarlyStoppingConfig,
) -> CandidateTrainingResult:
    """Train one ModernTCN HPO candidate and report validation reconstruction metrics."""

    return _train_deep_quality_candidate(
        candidate,
        train_df,
        validation_df,
        observation_columns,
        early_stopping,
        config_cls=ModernTCNConfig,
        model_cls=ModernTCNQualityModel,
    )


def _train_deep_quality_candidate(
    candidate: HyperparameterCandidate,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    observation_columns: tuple[str, ...],
    early_stopping: EarlyStoppingConfig,
    *,
    config_cls,
    model_cls,
) -> CandidateTrainingResult:
    """Train one deep quality-model HPO candidate with the shared contract."""

    from geas35.models.quality.evaluation import evaluate_synthetic_masking

    config = config_cls.from_candidate(candidate)
    model = model_cls(config)
    model.fit(
        train_df,
        observation_columns,
        validation_df=validation_df,
        early_stopping=early_stopping,
    )
    masking = evaluate_synthetic_masking(
        model,
        validation_df,
        observation_columns,
        mask_fraction=config.mask_fraction,
        random_state=config.random_state,
    )
    return CandidateTrainingResult(
        candidate=candidate,
        validation_metrics={
            "validation_synthetic_masking_rmse": masking.rmse,
            "validation_synthetic_masking_mae": masking.mae,
        },
        training_history=model.training_history_,
        model_artifact=model.to_artifact(),
    )


def _build_modern_tcn_network(
    torch,
    nn,
    *,
    n_features: int,
    channel_width: int,
    depth: tuple[int, ...],
    kernel_size: int,
    dropout: float,
):
    class _TemporalBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=padding),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size, padding=padding),
                nn.GELU(),
            )

        def forward(self, x):
            return x + self.net(x)

    class _ModernTCNNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Conv1d(n_features, channel_width, kernel_size=1)
            total_depth = sum(depth)
            self.blocks = nn.Sequential(
                *[_TemporalBlock(channel_width) for _ in range(total_depth)]
            )
            self.output_projection = nn.Linear(channel_width, n_features)

        def forward(self, x):
            x = x.transpose(1, 2)
            hidden = self.input_projection(x)
            hidden = self.blocks(hidden)
            current = hidden[:, :, -1]
            return self.output_projection(current)

    return _ModernTCNNetwork()


def _resolve_threshold(thresholds: object | None, column: str) -> float | None:
    if thresholds is None:
        return None
    if isinstance(thresholds, dict):
        if column in thresholds:
            value = thresholds[column]
        elif "default" in thresholds:
            value = thresholds["default"]
        elif "threshold" in thresholds:
            value = thresholds["threshold"]
        else:
            return None
    else:
        value = thresholds

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


__all__ = [
    "ModernTCNConfig",
    "ModernTCNQualityModel",
    "TorchUnavailableError",
    "train_modern_tcn_candidate",
]
