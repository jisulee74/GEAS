"""Deterministic baseline quality models."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from geas35.models.quality.base import BaseQualityModel, QualityModelOutput
from geas35.models.quality.rule_only import RuleOnlyQualityModel


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


class MedianQualityModel(BaseQualityModel):
    """Observation-only median imputation baseline.

    The model fits per-observation medians on train data. It does not produce AI
    outlier flags; missing/rule invalid masks are inherited from the rule-only
    baseline and invalid observations can be imputed with train medians.
    """

    model_name = "median"

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None
        self.rule_model_ = RuleOnlyQualityModel()

    def fit(
        self,
        train_df: pd.DataFrame,
        observation_columns: Iterable[str],
        *,
        time_column: str = "reg_date",
        group_columns: Iterable[str] = (),
        valid_mask: pd.DataFrame | pd.Series | None = None,
    ) -> "MedianQualityModel":
        columns = self._validate_columns(observation_columns)
        missing = [col for col in columns if col not in train_df.columns]
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview = f"{preview}, ..."
            raise ValueError(f"Training frame is missing observation columns: {preview}")

        train_values = train_df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        if valid_mask is not None:
            if isinstance(valid_mask, pd.DataFrame):
                train_values = train_values.where(
                    valid_mask.loc[:, list(columns)].astype(bool)
                )
            else:
                valid_rows = valid_mask.astype(bool)
                train_values = train_values.loc[valid_rows]

        medians = train_values.median(skipna=True)
        self.medians_ = medians.astype(float)
        self.observation_columns_ = columns
        self.rule_model_.fit(train_df, columns)
        return self

    def _require_medians(self) -> pd.Series:
        if self.medians_ is None:
            raise ValueError("MedianQualityModel is not fitted.")
        return self.medians_

    def reconstruct(
        self,
        df: pd.DataFrame,
        observation_columns: Iterable[str] | None = None,
        *,
        time_column: str = "reg_date",
        group_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        columns = self._resolved_columns(observation_columns)
        medians = self._require_medians().reindex(list(columns))
        return pd.DataFrame(
            {col: medians[col] for col in columns},
            index=df.index,
            dtype="float64",
        )

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
        predictions = prediction_df.loc[:, list(columns)].apply(
            pd.to_numeric,
            errors="coerce",
        )
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
        for col in columns:
            threshold = _resolve_threshold(thresholds, col)
            if threshold is not None:
                outlier_flags[col] = (
                    pd.to_numeric(scores[col], errors="coerce") >= threshold
                ).astype(int)
        invalid_mask = self.rule_model_.invalid_mask(df, columns) | outlier_flags.astype(bool)
        return QualityModelOutput(
            frame=df.copy(),
            observation_columns=columns,
            anomaly_scores=scores,
            outlier_flags=outlier_flags,
            invalid_mask=invalid_mask,
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
            if col not in result.columns:
                continue
            mask = invalid_mask[col].astype(bool) if col in invalid_mask.columns else False
            replacement = pd.to_numeric(prediction_df[col], errors="coerce")
            result.loc[mask, col] = replacement.loc[mask]
        return result


__all__ = ["MedianQualityModel"]
