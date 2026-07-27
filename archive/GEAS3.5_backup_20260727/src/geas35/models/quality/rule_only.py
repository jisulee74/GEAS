"""Rule-only quality model baseline.

This baseline does not learn, score, or impute values. It exists as the first
ablation point for the quality pipeline: missing flags and rule-based outlier
flags define the invalid observation mask, while feature values are preserved.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from geas35.models.quality.base import BaseQualityModel, QualityModelOutput


def _missing_flag_column(column: str) -> str:
    return f"{column}_missing_flag"


def _rule_flag_column(column: str) -> str:
    return f"{column}_rule_outlier_flag"


class RuleOnlyQualityModel(BaseQualityModel):
    """Observation-only no-op baseline using existing missing/rule flags."""

    model_name = "rule_only"

    def fit(
        self,
        train_df: pd.DataFrame,
        observation_columns: Iterable[str],
        *,
        time_column: str = "reg_date",
        group_columns: Iterable[str] = (),
        valid_mask: pd.DataFrame | pd.Series | None = None,
    ) -> "RuleOnlyQualityModel":
        columns = self._validate_columns(observation_columns)
        missing = [col for col in columns if col not in train_df.columns]
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview = f"{preview}, ..."
            raise ValueError(f"Training frame is missing observation columns: {preview}")
        self.observation_columns_ = columns
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
        return df.loc[:, list(columns)].copy()

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
        return pd.DataFrame(0.0, index=df.index, columns=list(columns))

    def invalid_mask(
        self,
        df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return missing-or-rule invalid mask for observation columns."""

        columns = self._resolved_columns(observation_columns)
        mask = pd.DataFrame(False, index=df.index, columns=list(columns))
        for col in columns:
            missing_flag = _missing_flag_column(col)
            rule_flag = _rule_flag_column(col)
            if missing_flag in df.columns:
                mask[col] |= (
                    pd.to_numeric(df[missing_flag], errors="coerce")
                    .fillna(0)
                    .astype(bool)
                )
            if rule_flag in df.columns:
                mask[col] |= (
                    pd.to_numeric(df[rule_flag], errors="coerce")
                    .fillna(0)
                    .astype(bool)
                )
        return mask

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
        return QualityModelOutput(
            frame=df.copy(),
            observation_columns=columns,
            anomaly_scores=scores,
            outlier_flags=outlier_flags,
            invalid_mask=self.invalid_mask(df, columns),
        )

    def impute(
        self,
        df: pd.DataFrame,
        invalid_mask: pd.DataFrame,
        prediction_df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        self._resolved_columns(observation_columns)
        return df.copy()


__all__ = ["RuleOnlyQualityModel"]
