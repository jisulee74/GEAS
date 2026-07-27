"""Base interfaces for observation quality models.

Quality models operate only on observation/state columns. Action/control
columns are restored from action logs in preprocessing and must never be passed
to AI, median, or rule-only quality model interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from geas35.preprocessing import ACTION_COLUMNS, STATE_COLUMNS


OBSERVATION_COLUMNS = tuple(STATE_COLUMNS)
ACTION_COLUMN_SET = frozenset(ACTION_COLUMNS)
OBSERVATION_COLUMN_SET = frozenset(OBSERVATION_COLUMNS)


@dataclass(frozen=True)
class QualityModelOutput:
    """Common output container for quality model inference."""

    frame: pd.DataFrame
    observation_columns: tuple[str, ...]
    anomaly_scores: pd.DataFrame
    outlier_flags: pd.DataFrame
    invalid_mask: pd.DataFrame
    confidence_scores: pd.DataFrame | None = None


def validate_observation_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Return validated observation columns and reject action columns."""

    requested = tuple(str(col) for col in columns)
    action_cols = [col for col in requested if col in ACTION_COLUMN_SET]
    if action_cols:
        preview = ", ".join(action_cols[:5])
        if len(action_cols) > 5:
            preview = f"{preview}, ..."
        raise ValueError(
            "Quality models operate on observation/state columns only. "
            f"Action columns are not allowed: {preview}"
        )

    invalid_cols = [col for col in requested if col not in OBSERVATION_COLUMN_SET]
    if invalid_cols:
        preview = ", ".join(invalid_cols[:5])
        if len(invalid_cols) > 5:
            preview = f"{preview}, ..."
        raise ValueError(f"Unknown observation columns: {preview}")

    return requested


class BaseQualityModel(ABC):
    """Abstract observation-only quality model interface."""

    observation_columns_: tuple[str, ...] | None = None

    def _validate_columns(self, columns: Iterable[str]) -> tuple[str, ...]:
        return validate_observation_columns(columns)

    def _resolved_columns(self, columns: Iterable[str] | None) -> tuple[str, ...]:
        if columns is not None:
            return self._validate_columns(columns)
        if self.observation_columns_ is None:
            raise ValueError("Quality model is not fitted and no columns were provided.")
        return self.observation_columns_

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        observation_columns: Iterable[str],
        *,
        time_column: str = "reg_date",
        group_columns: Iterable[str] = (),
        valid_mask: pd.DataFrame | pd.Series | None = None,
    ) -> "BaseQualityModel":
        """Fit the quality model on observation columns only."""

    @abstractmethod
    def reconstruct(
        self,
        df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
        *,
        time_column: str = "reg_date",
        group_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        """Return reconstructed observation values."""

    @abstractmethod
    def forecast(
        self,
        df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
        *,
        horizon: int = 1,
        time_column: str = "reg_date",
        group_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        """Return forecast observation values."""

    @abstractmethod
    def anomaly_score(
        self,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return per-observation anomaly scores."""

    @abstractmethod
    def predict_outlier(
        self,
        df: pd.DataFrame,
        thresholds: object | None,
        observation_columns: Iterable[str] | None = None,
    ) -> QualityModelOutput:
        """Return quality model outlier predictions."""

    @abstractmethod
    def impute(
        self,
        df: pd.DataFrame,
        invalid_mask: pd.DataFrame,
        prediction_df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return a frame with invalid observation values imputed."""


__all__ = [
    "ACTION_COLUMN_SET",
    "BaseQualityModel",
    "OBSERVATION_COLUMNS",
    "OBSERVATION_COLUMN_SET",
    "QualityModelOutput",
    "validate_observation_columns",
]
