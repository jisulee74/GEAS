"""Combined missing/outlier handling stage for GEAS quality datasets.

This stage is the Step 7 replacement target for the old
``2_missing_values_handled -> 3_unit_canonicalized -> 4_outliers_flagged``
flow. It expects unit-canonicalized GEAS feature tables and applies:

* feature-level missing flags;
* action restoration from same-timestamp control logs only;
* observation-only rule outlier flags;
* observation-only quality model flags and imputation;
* final invalid/imputed masks for downstream clean dataset creation.

Action/control columns are never passed to the quality model and are never
median- or AI-imputed.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from geas35.preprocessing import (
    ACTION_COLUMNS,
    DEFAULT_DOMAIN_RANGE_PATH,
    DEFAULT_RULE_AGGREGATE_FLAG,
    FEATURE_COLUMNS,
    STATE_COLUMNS,
    DomainRangeRule,
    apply_rule_based_outlier_flags,
    load_domain_range_rules,
    prepare_missing_features,
    raw_value_column,
)

if TYPE_CHECKING:
    from geas35.models.quality.base import BaseQualityModel, QualityModelOutput


AI_ANOMALY_SCORE_SUFFIX = "_ai_anomaly_score"
AI_CONFIDENCE_SUFFIX = "_ai_confidence"
AI_OUTLIER_FLAG_SUFFIX = "_ai_outlier_flag"
INVALID_FLAG_SUFFIX = "_invalid_flag"
QUALITY_IMPUTED_FLAG_SUFFIX = "_imputed_flag"

QUALITY_AI_OUTLIER_FLAG_COLUMN = "ai_outlier_flag"
QUALITY_INVALID_FLAG_COLUMN = "invalid_flag"
QUALITY_IMPUTED_FLAG_COLUMN = "imputed_flag"
DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD = 0.5


def ai_anomaly_score_column(column: str) -> str:
    """Return the per-observation quality-model score column."""

    return f"{column}{AI_ANOMALY_SCORE_SUFFIX}"


def ai_confidence_column(column: str) -> str:
    """Return the per-observation quality-model confidence column."""

    return f"{column}{AI_CONFIDENCE_SUFFIX}"


def quality_ai_outlier_flag_column(column: str) -> str:
    """Return the per-observation quality-model outlier flag column."""

    return f"{column}{AI_OUTLIER_FLAG_SUFFIX}"


def invalid_flag_column(column: str) -> str:
    """Return the per-observation final invalid flag column."""

    return f"{column}{INVALID_FLAG_SUFFIX}"


def quality_imputed_flag_column(column: str) -> str:
    """Return the per-observation quality imputation flag column."""

    return f"{column}{QUALITY_IMPUTED_FLAG_SUFFIX}"


def _observation_columns(columns: Iterable[str] | None) -> tuple[str, ...]:
    requested = tuple(STATE_COLUMNS if columns is None else columns)
    action_cols = [col for col in requested if col in set(ACTION_COLUMNS)]
    if action_cols:
        preview = ", ".join(action_cols[:5])
        if len(action_cols) > 5:
            preview = f"{preview}, ..."
        raise ValueError(
            "Missing/outlier quality handling accepts observation columns only. "
            f"Action columns are not allowed: {preview}"
        )
    return requested


def _ensure_raw_value_columns(
    df_raw: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    df = df_raw.copy()
    for col in columns:
        if col in df.columns:
            df[raw_value_column(col)] = df[col]
    return df


def _numeric_flag_frame(
    frame: pd.DataFrame | None,
    *,
    index: pd.Index,
    columns: Iterable[str],
    fill_value: int | float = 0,
) -> pd.DataFrame:
    result = pd.DataFrame(fill_value, index=index, columns=list(columns))
    if frame is None or frame.empty:
        return result
    for col in result.columns:
        if col in frame.columns:
            result[col] = pd.to_numeric(frame[col], errors="coerce").fillna(fill_value)
    return result


def _bool_mask_frame(
    frame: pd.DataFrame | None,
    *,
    index: pd.Index,
    columns: Iterable[str],
) -> pd.DataFrame:
    numeric = _numeric_flag_frame(
        frame,
        index=index,
        columns=columns,
        fill_value=0,
    )
    return numeric.astype(bool)


def _flag_mask_from_suffixes(
    df: pd.DataFrame,
    *,
    index: pd.Index,
    columns: Iterable[str],
    suffixes: Iterable[str],
) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=index, columns=list(columns))
    for col in mask.columns:
        for suffix in suffixes:
            flag_col = f"{col}{suffix}"
            if flag_col in df.columns:
                mask[col] |= (
                    pd.to_numeric(df[flag_col], errors="coerce")
                    .fillna(0)
                    .astype(bool)
                )
    return mask


def _validate_imputation_confidence_threshold(value: float) -> float:
    threshold = pd.to_numeric(value, errors="coerce")
    if pd.isna(threshold):
        raise ValueError("imputation_confidence_threshold must be numeric.")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("imputation_confidence_threshold must be in [0, 1].")
    return threshold


def _confidence_aware_imputation_mask(
    *,
    invalid_mask: pd.DataFrame,
    ai_flags: pd.DataFrame,
    confidence_scores: pd.DataFrame | None,
    missing_or_rule_mask: pd.DataFrame,
    imputation_confidence_threshold: float,
) -> pd.DataFrame:
    """Return cells eligible for reconstruction/imputation.

    Missing and rule-based invalid cells remain imputation candidates. AI-only
    outliers require confidence above the configured gate. If a model does not
    provide confidence scores, the historical behavior is preserved.
    """

    if confidence_scores is None:
        return invalid_mask

    ai_mask = ai_flags.astype(bool)
    confidence_gate = confidence_scores.ge(imputation_confidence_threshold).fillna(False)
    non_ai_invalid = invalid_mask & ~ai_mask
    return missing_or_rule_mask | non_ai_invalid | (ai_mask & confidence_gate)


def _fit_rule_only_if_needed(
    model: "BaseQualityModel",
    df: pd.DataFrame,
    observation_columns: tuple[str, ...],
) -> "BaseQualityModel":
    from geas35.models.quality.rule_only import RuleOnlyQualityModel

    if isinstance(model, RuleOnlyQualityModel) and model.observation_columns_ is None:
        return model.fit(df, observation_columns)
    return model


def _predict_quality_output(
    model: "BaseQualityModel",
    df: pd.DataFrame,
    thresholds: object | None,
    observation_columns: tuple[str, ...],
) -> tuple["QualityModelOutput", pd.DataFrame]:
    output = model.predict_outlier(df, thresholds, observation_columns)
    prediction = model.reconstruct(df, observation_columns)
    return output, prediction


def _cell_changed(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    both_missing = before.isna() & after.isna()
    equal_values = before.eq(after)
    return ~(both_missing | equal_values)


def prepare_missing_outliers_handled_features(
    df_raw: pd.DataFrame,
    df_control_log: pd.DataFrame | None = None,
    *,
    quality_model: "BaseQualityModel | None" = None,
    thresholds: object | None = None,
    rules: Iterable[DomainRangeRule] | None = None,
    domain_csv_path: str | Path = DEFAULT_DOMAIN_RANGE_PATH,
    observation_columns: Iterable[str] | None = None,
    keep_extra_columns: bool = True,
    add_flags: bool = True,
    imputation_confidence_threshold: float = DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Prepare one unit-canonicalized frame for quality-model clean output.

    Observation/state columns receive missing, rule-outlier, AI-outlier,
    invalid, and imputed flags. Action/control columns receive missing flags and
    optional action-log restoration flags only.
    """

    if df_raw is None or df_raw.empty:
        return pd.DataFrame() if df_raw is None else df_raw.copy()

    imputation_confidence_threshold = _validate_imputation_confidence_threshold(
        imputation_confidence_threshold
    )
    obs_cols = _observation_columns(observation_columns)
    if quality_model is None:
        from geas35.models.quality.rule_only import RuleOnlyQualityModel

        model = RuleOnlyQualityModel()
    else:
        model = quality_model

    df = _ensure_raw_value_columns(df_raw, [*obs_cols, *ACTION_COLUMNS])
    df = prepare_missing_features(
        df,
        df_control_log=df_control_log,
        keep_extra_columns=True,
        add_flags=add_flags,
    )

    rule_set = list(rules) if rules is not None else load_domain_range_rules(domain_csv_path)
    df = apply_rule_based_outlier_flags(
        df,
        rule_set,
        aggregate_flag_col=DEFAULT_RULE_AGGREGATE_FLAG,
        allowed_columns=obs_cols,
    )

    model = _fit_rule_only_if_needed(model, df, obs_cols)
    output, prediction = _predict_quality_output(model, df, thresholds, obs_cols)

    scores = _numeric_flag_frame(
        output.anomaly_scores,
        index=df.index,
        columns=obs_cols,
        fill_value=0.0,
    )
    confidence_scores = None
    if output.confidence_scores is not None:
        confidence_scores = _numeric_flag_frame(
            output.confidence_scores,
            index=df.index,
            columns=obs_cols,
            fill_value=float("nan"),
        )
    ai_flags = _numeric_flag_frame(
        output.outlier_flags,
        index=df.index,
        columns=obs_cols,
        fill_value=0,
    ).astype(int)
    invalid_mask = _bool_mask_frame(
        output.invalid_mask,
        index=df.index,
        columns=obs_cols,
    ) | ai_flags.astype(bool)
    missing_or_rule_mask = _flag_mask_from_suffixes(
        df,
        index=df.index,
        columns=obs_cols,
        suffixes=("_missing_flag", "_rule_outlier_flag"),
    )
    imputation_mask = _confidence_aware_imputation_mask(
        invalid_mask=invalid_mask,
        ai_flags=ai_flags,
        confidence_scores=confidence_scores,
        missing_or_rule_mask=missing_or_rule_mask,
        imputation_confidence_threshold=imputation_confidence_threshold,
    )

    before_impute = df.loc[:, list(obs_cols)].copy()
    df = output.frame.copy()
    for col in obs_cols:
        df[ai_anomaly_score_column(col)] = scores[col]
        if confidence_scores is not None:
            df[ai_confidence_column(col)] = confidence_scores[col]
        df[quality_ai_outlier_flag_column(col)] = ai_flags[col].astype(int)
        df[invalid_flag_column(col)] = invalid_mask[col].astype(int)

    df = model.impute(df, imputation_mask, prediction, obs_cols)
    after_impute = df.loc[:, list(obs_cols)].copy()
    changed = _cell_changed(before_impute, after_impute) & imputation_mask
    for col in obs_cols:
        df[quality_imputed_flag_column(col)] = changed[col].astype(int)

    df[QUALITY_AI_OUTLIER_FLAG_COLUMN] = ai_flags.any(axis=1).astype(int)
    df[QUALITY_INVALID_FLAG_COLUMN] = invalid_mask.any(axis=1).astype(int)
    df[QUALITY_IMPUTED_FLAG_COLUMN] = changed.any(axis=1).astype(int)

    if keep_extra_columns:
        return df

    generated_cols = [
        col
        for col in df.columns
        if col.endswith(
            (
                "_raw_value",
                "_missing_flag",
                "_restored_flag",
                "_rule_outlier_flag",
                AI_ANOMALY_SCORE_SUFFIX,
                AI_CONFIDENCE_SUFFIX,
                AI_OUTLIER_FLAG_SUFFIX,
                INVALID_FLAG_SUFFIX,
                QUALITY_IMPUTED_FLAG_SUFFIX,
            )
        )
        or col
        in {
            DEFAULT_RULE_AGGREGATE_FLAG,
            QUALITY_AI_OUTLIER_FLAG_COLUMN,
            QUALITY_INVALID_FLAG_COLUMN,
            QUALITY_IMPUTED_FLAG_COLUMN,
        }
    ]
    ordered = [col for col in [*FEATURE_COLUMNS, *generated_cols] if col in df.columns]
    return df.loc[:, ordered].copy()


prepare_quality_features = prepare_missing_outliers_handled_features


__all__ = [
    "AI_ANOMALY_SCORE_SUFFIX",
    "AI_CONFIDENCE_SUFFIX",
    "AI_OUTLIER_FLAG_SUFFIX",
    "INVALID_FLAG_SUFFIX",
    "QUALITY_AI_OUTLIER_FLAG_COLUMN",
    "DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD",
    "QUALITY_IMPUTED_FLAG_COLUMN",
    "QUALITY_IMPUTED_FLAG_SUFFIX",
    "QUALITY_INVALID_FLAG_COLUMN",
    "ai_anomaly_score_column",
    "ai_confidence_column",
    "invalid_flag_column",
    "prepare_missing_outliers_handled_features",
    "prepare_quality_features",
    "quality_ai_outlier_flag_column",
    "quality_imputed_flag_column",
]
