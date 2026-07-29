"""TimesNet-style quality model for feature-level masked reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from geas35.models.quality.deep import TorchUnavailableError
from geas35.models.quality.hyperparameters import (
    CandidateTrainingResult,
    EarlyStoppingConfig,
    HyperparameterCandidate,
)
from geas35.models.quality.modern_tcn import (
    ModernTCNQualityModel,
    _train_deep_quality_candidate,
)


@dataclass(frozen=True)
class TimesNetConfig:
    """Configuration for the initial TimesNet quality model implementation."""

    lookback: int = 288
    mask_fraction: float = 0.3
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    temporal_blocks: int = 1
    top_k_periods: int = 2
    period_embedding_dim: int = 32
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
        if self.temporal_blocks < 1:
            raise ValueError("temporal_blocks must be >= 1.")
        if self.top_k_periods < 1:
            raise ValueError("top_k_periods must be >= 1.")
        if self.period_embedding_dim < 1:
            raise ValueError("period_embedding_dim must be >= 1.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    @classmethod
    def from_candidate(cls, candidate: HyperparameterCandidate) -> "TimesNetConfig":
        payload = {**candidate.common, **candidate.model_specific}
        allowed = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in payload.items() if key in allowed}
        return cls(**filtered)

    def to_artifact(self) -> dict[str, object]:
        return {**asdict(self), "d_ff": 2 * self.period_embedding_dim}


class TimesNetQualityModel(ModernTCNQualityModel):
    """Observation-only TimesNet candidate using masked current-state reconstruction."""

    model_name = "timesnet"

    def __init__(
        self,
        config: TimesNetConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        super().__init__(config=config or TimesNetConfig(), device=device)

    def _build_network(self, torch, nn, *, n_features: int):
        return _build_timesnet_network(
            torch,
            nn,
            n_features=n_features,
            lookback=self.config.lookback,
            temporal_blocks=self.config.temporal_blocks,
            top_k_periods=self.config.top_k_periods,
            period_embedding_dim=self.config.period_embedding_dim,
            dropout=self.config.dropout,
        )


def train_timesnet_candidate(
    candidate: HyperparameterCandidate,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    observation_columns: tuple[str, ...],
    early_stopping: EarlyStoppingConfig,
) -> CandidateTrainingResult:
    """Train one TimesNet HPO candidate and report validation reconstruction metrics."""

    return _train_deep_quality_candidate(
        candidate,
        train_df,
        validation_df,
        observation_columns,
        early_stopping,
        config_cls=TimesNetConfig,
        model_cls=TimesNetQualityModel,
    )


def _build_timesnet_network(
    torch,
    nn,
    *,
    n_features: int,
    lookback: int,
    temporal_blocks: int,
    top_k_periods: int,
    period_embedding_dim: int,
    dropout: float,
):
    kernels = _period_kernels(lookback, top_k_periods)
    d_ff = 2 * period_embedding_dim

    class _TimesBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for kernel in kernels
                ]
            )
            self.mix = nn.Sequential(
                nn.Conv1d(channels, d_ff, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(d_ff, channels, kernel_size=1),
            )

        def forward(self, x):
            branch_outputs = torch.stack([branch(x) for branch in self.branches], dim=0)
            period_mixed = branch_outputs.mean(dim=0)
            return x + self.mix(period_mixed)

    class _TimesNetNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Conv1d(
                n_features,
                period_embedding_dim,
                kernel_size=1,
            )
            self.blocks = nn.Sequential(
                *[_TimesBlock(period_embedding_dim) for _ in range(temporal_blocks)]
            )
            self.output_projection = nn.Linear(period_embedding_dim, n_features)

        def forward(self, x):
            x = x.transpose(1, 2)
            hidden = self.input_projection(x)
            hidden = self.blocks(hidden)
            current = hidden[:, :, -1]
            return self.output_projection(current)

    return _TimesNetNetwork()


def _period_kernels(lookback: int, top_k_periods: int) -> tuple[int, ...]:
    max_kernel = max(1, min(lookback, 2 * top_k_periods + 1))
    kernels: list[int] = []
    candidate = 1
    while len(kernels) < top_k_periods and candidate <= max_kernel:
        kernels.append(candidate)
        candidate += 2
    return tuple(kernels or [1])


__all__ = [
    "TimesNetConfig",
    "TimesNetQualityModel",
    "TorchUnavailableError",
    "train_timesnet_candidate",
]
