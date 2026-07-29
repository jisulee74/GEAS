"""PatchTST-style quality model for feature-level masked reconstruction."""

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
class PatchTSTConfig:
    """Configuration for the initial PatchTST quality model implementation."""

    lookback: int = 288
    mask_fraction: float = 0.3
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    patch_length: int = 8
    patch_stride: int | None = None
    transformer_depth: int = 2
    attention_heads: int = 4
    embedding_dim: int = 32
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
        if self.patch_length < 1:
            raise ValueError("patch_length must be >= 1.")
        if self.patch_length > self.lookback:
            raise ValueError("patch_length must be <= lookback.")
        derived_stride = self.patch_length // 2
        object.__setattr__(self, "patch_stride", int(derived_stride))
        if derived_stride < 1:
            raise ValueError("patch_stride must be >= 1.")
        if self.transformer_depth < 1:
            raise ValueError("transformer_depth must be >= 1.")
        if self.attention_heads < 1:
            raise ValueError("attention_heads must be >= 1.")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1.")
        if self.embedding_dim % self.attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    @classmethod
    def from_candidate(cls, candidate: HyperparameterCandidate) -> "PatchTSTConfig":
        payload = {**candidate.common, **candidate.model_specific}
        allowed = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in payload.items() if key in allowed}
        return cls(**filtered)

    def to_artifact(self) -> dict[str, object]:
        return {**asdict(self), "d_ff": 2 * self.embedding_dim}


class PatchTSTQualityModel(ModernTCNQualityModel):
    """Observation-only PatchTST candidate using masked current-state reconstruction."""

    model_name = "patch_tst"

    def __init__(
        self,
        config: PatchTSTConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        super().__init__(config=config or PatchTSTConfig(), device=device)

    def _build_network(self, torch, nn, *, n_features: int):
        return _build_patch_tst_network(
            torch,
            nn,
            n_features=n_features,
            patch_length=self.config.patch_length,
            patch_stride=self.config.patch_stride,
            transformer_depth=self.config.transformer_depth,
            attention_heads=self.config.attention_heads,
            embedding_dim=self.config.embedding_dim,
            dropout=self.config.dropout,
        )


def train_patch_tst_candidate(
    candidate: HyperparameterCandidate,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    observation_columns: tuple[str, ...],
    early_stopping: EarlyStoppingConfig,
) -> CandidateTrainingResult:
    """Train one PatchTST HPO candidate and report validation reconstruction metrics."""

    return _train_deep_quality_candidate(
        candidate,
        train_df,
        validation_df,
        observation_columns,
        early_stopping,
        config_cls=PatchTSTConfig,
        model_cls=PatchTSTQualityModel,
    )


def _build_patch_tst_network(
    torch,
    nn,
    *,
    n_features: int,
    patch_length: int,
    patch_stride: int,
    transformer_depth: int,
    attention_heads: int,
    embedding_dim: int,
    dropout: float,
):
    class _PatchTSTNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_projection = nn.Linear(n_features * patch_length, embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attention_heads,
                dim_feedforward=max(embedding_dim * 2, 4),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_depth,
            )
            self.output_projection = nn.Linear(embedding_dim, n_features)

        def forward(self, x):
            patches = x.unfold(dimension=1, size=patch_length, step=patch_stride)
            patches = patches.permute(0, 1, 3, 2).contiguous()
            batch_size, n_patches, _, _ = patches.shape
            patches = patches.reshape(batch_size, n_patches, -1)
            tokens = self.patch_projection(patches)
            encoded = self.encoder(tokens)
            current_context = encoded[:, -1, :]
            return self.output_projection(current_context)

    return _PatchTSTNetwork()


__all__ = [
    "PatchTSTConfig",
    "PatchTSTQualityModel",
    "TorchUnavailableError",
    "train_patch_tst_candidate",
]
