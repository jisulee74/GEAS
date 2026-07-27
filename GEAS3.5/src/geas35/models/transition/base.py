"""Base interfaces for GEAS transition models.

Transition models predict the next dynamic observation from the current
observation and action. Reward calculation remains outside this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True)
class TransitionModelCapabilities:
    """Static capability metadata for a transition model implementation."""

    supports_uncertainty: bool = False
    supports_sequence_input: bool = False
    supports_native_multi_output: bool = False
    multi_output_strategy: str = "independent"
    optional_dependency: str | None = None


@dataclass(frozen=True)
class TransitionDataset:
    """Tabular transition training or inference dataset."""

    x: pd.DataFrame
    y: pd.DataFrame
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    metadata: pd.DataFrame | None = None
    observation_columns: tuple[str, ...] = ()
    action_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.x.index) != len(self.y.index):
            raise ValueError("TransitionDataset x and y must contain the same rows.")
        if self.metadata is not None and len(self.metadata.index) != len(self.x.index):
            raise ValueError("TransitionDataset metadata must match x row count.")
        if not self.input_columns:
            raise ValueError("TransitionDataset input_columns must not be empty.")
        if not self.target_columns:
            raise ValueError("TransitionDataset target_columns must not be empty.")
        _require_columns(self.x, self.input_columns, label="input")
        _require_columns(self.y, self.target_columns, label="target")


@dataclass(frozen=True)
class TransitionPrediction:
    """Common output container for transition model inference."""

    next_observation: pd.DataFrame
    target_columns: tuple[str, ...]
    uncertainty: pd.DataFrame | None = None
    raw_prediction: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if not self.target_columns:
            raise ValueError("TransitionPrediction target_columns must not be empty.")
        _require_columns(self.next_observation, self.target_columns, label="prediction")
        if self.uncertainty is not None and len(self.uncertainty.index) != len(
            self.next_observation.index
        ):
            raise ValueError("TransitionPrediction uncertainty must match prediction rows.")


class BaseTransitionModel(ABC):
    """Abstract next-state prediction interface for model-based RL."""

    model_name: ClassVar[str] = "base_transition_model"
    capabilities: ClassVar[TransitionModelCapabilities] = TransitionModelCapabilities()

    input_columns_: tuple[str, ...] | None = None
    target_columns_: tuple[str, ...] | None = None

    def _require_fitted(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self.input_columns_ is None or self.target_columns_ is None:
            raise ValueError("Transition model is not fitted.")
        return self.input_columns_, self.target_columns_

    @abstractmethod
    def fit(
        self,
        dataset: TransitionDataset,
        validation_dataset: TransitionDataset | None = None,
    ) -> "BaseTransitionModel":
        """Fit the transition model on current-state/action inputs."""

    @abstractmethod
    def predict(
        self,
        dataset_or_frame: TransitionDataset | pd.DataFrame,
    ) -> TransitionPrediction:
        """Predict the next dynamic observation."""


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview = f"{preview}, ..."
        raise KeyError(f"Missing {label} columns: {preview}")


__all__ = [
    "BaseTransitionModel",
    "TransitionDataset",
    "TransitionModelCapabilities",
    "TransitionPrediction",
]
