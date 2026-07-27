"""Combined missing/outlier handling stage for GEAS quality datasets.

This stage is the Step 7 combined quality-control target. It expects
unit-canonicalized GEAS feature tables and applies:

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
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from geas35.models.quality.base import BaseQualityModel, QualityModelOutput


_input_schema = import_module("geas35.preprocessing.1_input_schema_preparation")
ACTION_COLUMNS = _input_schema.ACTION_COLUMNS
EXTERNAL_STATE_COLUMNS = _input_schema.EXTERNAL_STATE_COLUMNS
FEATURE_COLUMNS = _input_schema.FEATURE_COLUMNS
INTERNAL_STATE_COLUMNS = _input_schema.INTERNAL_STATE_COLUMNS
STATE_COLUMNS = _input_schema.STATE_COLUMNS
SYSTEM_STATE_COLUMNS = _input_schema.SYSTEM_STATE_COLUMNS
TIME_COLUMN = _input_schema.TIME_COLUMN


AI_ANOMALY_SCORE_SUFFIX = "_ai_anomaly_score"
AI_CONFIDENCE_SUFFIX = "_ai_confidence"
AI_OUTLIER_FLAG_SUFFIX = "_ai_outlier_flag"
INVALID_FLAG_SUFFIX = "_invalid_flag"
QUALITY_IMPUTED_FLAG_SUFFIX = "_imputed_flag"
MISSING_FLAG_SUFFIX = "_missing_flag"
ACTION_RESTORED_FLAG_SUFFIX = "_restored_flag"

QUALITY_AI_OUTLIER_FLAG_COLUMN = "ai_outlier_flag"
QUALITY_INVALID_FLAG_COLUMN = "invalid_flag"
QUALITY_IMPUTED_FLAG_COLUMN = "imputed_flag"
DEFAULT_RULE_AGGREGATE_FLAG = "rule_outlier_flag"
DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_DOMAIN_RANGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "offline_dataset_preparation"
    / "configs"
    / "qc"
    / "smartfarm_korea_domain_ranges.csv"
)
OBSERVATION_RULE_COLUMNS = tuple(STATE_COLUMNS)
SEGMENT_COLUMN = "segment_id"
RESAMPLED_ROW_COLUMN = "is_resampled_row"

ACTION_CONTROL_LOG_MAP = {
    "cont_skyl_vol": "pred_ltw",
    "cont_skyr_vol": "pred_rtw",
    "cont_cur_vol": "pred_pc1",
    "cont_kwcur_vol": "pred_pc2",
    "cont_co2_run": "pred_co2",
    "cont_pump1_run": "pred_cp1",
    "cont_pump2_run": "pred_cp2",
    "cont_heater_run": "pred_heater",
    "cont_cooler_run": "pred_cooler",
    "cont_3way1_vol": "pred_tw1",
    "cont_3way2_vol": "pred_tw2",
    "cont_fan_run": "pred_fan",
}

CONTROL_LOG_TIME_CANDIDATES = [
    "reg_date",
    "send_date",
    "request_time",
    "created_at",
    "create_date",
    "timestamp",
]


@dataclass(frozen=True)
class DomainRangeRule:
    """Domain-review bounds and discrete-value rules for one GEAS column."""

    column: str
    lower: float | None
    upper: float | None
    source_code: str
    source_name: str
    unit: str
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    allowed_values: tuple[float, ...] = ()
    integer_only: bool = False


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


def missing_flag_column(feature_col: str) -> str:
    """Return the feature-level missing flag column name."""

    return f"{feature_col}{MISSING_FLAG_SUFFIX}"


def action_restored_flag_column(action_col: str) -> str:
    """Return the action-log restoration flag column name."""

    return f"{action_col}{ACTION_RESTORED_FLAG_SUFFIX}"


def raw_value_column(column: str) -> str:
    """Return the trace column that stores the original feature value."""

    return f"{column}_raw_value"


def rule_outlier_flag_column(column: str) -> str:
    """Return the feature-level rule outlier flag column name."""

    return f"{column}_rule_outlier_flag"


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _require_schema_ready_input(df: pd.DataFrame) -> None:
    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:5])
        if len(missing_columns) > 5:
            preview = f"{preview}, ..."
        raise ValueError(
            "Missing/outlier handling expects GEAS input-schema features. "
            f"Missing required columns: {preview}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df[TIME_COLUMN]):
        raise ValueError(
            "Missing/outlier handling expects reg_date to be parsed before this stage."
        )


def add_missing_flags(
    df_raw: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Attach feature-level missing flags without changing feature values."""

    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()
    for col in columns:
        if col in df.columns:
            df[missing_flag_column(col)] = df[col].isna().astype(int)
    return df


def _ensure_action_restored_flags(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    for col in ACTION_COLUMNS:
        flag_col = action_restored_flag_column(col)
        if flag_col not in df.columns:
            df[flag_col] = 0
        else:
            df[flag_col] = (
                pd.to_numeric(df[flag_col], errors="coerce")
                .fillna(0)
                .astype(int)
            )
    return df


def _control_log_action_wide(df_control_log: pd.DataFrame | None) -> pd.DataFrame:
    """Return same-timestamp action fallback values from control output logs."""

    if df_control_log is None or df_control_log.empty:
        return pd.DataFrame()

    time_col = _first_existing_column(df_control_log, CONTROL_LOG_TIME_CANDIDATES)
    if not time_col:
        return pd.DataFrame()

    available_map = {
        action_col: log_col
        for action_col, log_col in ACTION_CONTROL_LOG_MAP.items()
        if log_col in df_control_log.columns
    }
    if not available_map:
        return pd.DataFrame()

    cols = [time_col, *available_map.values()]
    wide = df_control_log[cols].copy()
    wide[TIME_COLUMN] = pd.to_datetime(wide[time_col], errors="coerce")
    wide = wide.dropna(subset=[TIME_COLUMN]).sort_values(TIME_COLUMN)
    if wide.empty:
        return pd.DataFrame()

    for log_col in available_map.values():
        wide[log_col] = pd.to_numeric(wide[log_col], errors="coerce")

    wide = wide.rename(
        columns={log_col: action_col for action_col, log_col in available_map.items()}
    )
    keep_cols = [TIME_COLUMN, *available_map.keys()]
    return wide[keep_cols].drop_duplicates(subset=[TIME_COLUMN], keep="last")


def restore_action_columns_from_control_log(
    df_raw: pd.DataFrame,
    df_control_log: pd.DataFrame | None = None,
    *,
    add_flags: bool = True,
) -> pd.DataFrame:
    """Restore missing action values only from same-timestamp control logs."""

    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()
    if add_flags:
        df = _ensure_action_restored_flags(df)

    control_wide = _control_log_action_wide(df_control_log)
    if control_wide.empty:
        return df

    left = df[[TIME_COLUMN]].reset_index().dropna(subset=[TIME_COLUMN])
    if left.empty:
        return df

    matched_control = left.merge(control_wide, on=TIME_COLUMN, how="left").set_index(
        "index"
    )
    for action_col in ACTION_COLUMNS:
        if action_col not in matched_control.columns:
            continue
        fallback = matched_control[action_col].reindex(df.index)
        restored = df[action_col].isna() & fallback.notna()
        df[action_col] = df[action_col].where(df[action_col].notna(), fallback)
        if add_flags:
            df.loc[restored, action_restored_flag_column(action_col)] = 1

    return df


def prepare_missing_features(
    df_raw: pd.DataFrame,
    df_control_log: pd.DataFrame | None = None,
    *,
    keep_extra_columns: bool = True,
    add_flags: bool = True,
) -> pd.DataFrame:
    """Record feature-level missingness and restore actions from control logs."""

    if df_raw is None or df_raw.empty:
        return pd.DataFrame() if df_raw is None else df_raw.copy()
    _require_schema_ready_input(df_raw)

    df = df_raw.copy()
    if add_flags:
        df = add_missing_flags(df, [*STATE_COLUMNS, *ACTION_COLUMNS])
    df = restore_action_columns_from_control_log(
        df,
        df_control_log,
        add_flags=add_flags,
    )

    if keep_extra_columns:
        return df
    return df[FEATURE_COLUMNS].copy()


def _parse_bool(value: Any, *, default: bool) -> bool:
    if pd.isna(value):
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y"}


def _parse_allowed_values(value: Any) -> tuple[float, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    parsed: list[float] = []
    for item in text.replace(";", "|").split("|"):
        item = item.strip()
        if not item:
            continue
        numeric = pd.to_numeric(item, errors="coerce")
        if not pd.isna(numeric):
            parsed.append(float(numeric))
    return tuple(parsed)


def _dedupe_rules(rules: Iterable[DomainRangeRule]) -> list[DomainRangeRule]:
    deduped: dict[
        tuple[str, float | None, float | None, bool, bool, tuple[float, ...], bool],
        DomainRangeRule,
    ] = {}
    for rule in rules:
        deduped.setdefault(
            (
                rule.column,
                rule.lower,
                rule.upper,
                rule.lower_inclusive,
                rule.upper_inclusive,
                rule.allowed_values,
                rule.integer_only,
            ),
            rule,
        )
    return list(deduped.values())


def load_domain_range_rules(
    csv_path: str | Path = DEFAULT_DOMAIN_RANGE_PATH,
) -> list[DomainRangeRule]:
    """Load GEAS canonical domain-range rules."""

    table = pd.read_csv(csv_path)
    required = {"column", "lower", "upper"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Domain range file missing required columns: {sorted(missing)}")

    rules: list[DomainRangeRule] = []
    for _, row in table.iterrows():
        column = str(row.get("column", "")).strip()
        lower = pd.to_numeric(row.get("lower"), errors="coerce")
        upper = pd.to_numeric(row.get("upper"), errors="coerce")
        allowed_values = _parse_allowed_values(row.get("allowed_values", ""))
        integer_only = _parse_bool(row.get("integer_only", False), default=False)
        if not column:
            continue
        lower_value = None if pd.isna(lower) else float(lower)
        upper_value = None if pd.isna(upper) else float(upper)
        if lower_value is None and upper_value is None and not allowed_values and not integer_only:
            continue
        rules.append(
            DomainRangeRule(
                column=column,
                lower=lower_value,
                upper=upper_value,
                source_code=str(row.get("source_code", "")).strip(),
                source_name=str(row.get("source_name", "")).strip(),
                unit=str(row.get("unit", "")).strip(),
                lower_inclusive=_parse_bool(
                    row.get("lower_inclusive", True),
                    default=True,
                ),
                upper_inclusive=_parse_bool(
                    row.get("upper_inclusive", True),
                    default=True,
                ),
                allowed_values=allowed_values,
                integer_only=integer_only,
            )
        )
    return _dedupe_rules(rules)


def _as_int_flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _rule_flag(values: pd.Series, rule: DomainRangeRule) -> pd.Series:
    present = values.notna()
    flag = pd.Series(False, index=values.index)
    if rule.lower is not None:
        flag |= present & (
            (values < rule.lower)
            if rule.lower_inclusive
            else (values <= rule.lower)
        )
    if rule.upper is not None:
        flag |= present & (
            (values > rule.upper)
            if rule.upper_inclusive
            else (values >= rule.upper)
        )
    if rule.allowed_values:
        flag |= present & ~values.isin(set(rule.allowed_values))
    if rule.integer_only:
        rounded = values.round()
        flag |= present & ~np.isclose(values, rounded, equal_nan=False)
    return flag


def _assign_columns(df: pd.DataFrame, updates: dict[str, Any]) -> pd.DataFrame:
    if not updates:
        return df.copy()
    update_df = pd.DataFrame(updates, index=df.index)
    base = df.drop(columns=[col for col in updates if col in df.columns])
    return pd.concat([base, update_df], axis=1)


def apply_rule_based_outlier_flags(
    df_raw: pd.DataFrame,
    rules: Iterable[DomainRangeRule],
    *,
    aggregate_flag_col: str = DEFAULT_RULE_AGGREGATE_FLAG,
    allowed_columns: Iterable[str] = OBSERVATION_RULE_COLUMNS,
) -> pd.DataFrame:
    """Attach observation-only inclusive rule-based outlier flags."""

    if df_raw is None or df_raw.empty:
        return pd.DataFrame() if df_raw is None else df_raw.copy()

    df = df_raw.copy()
    aggregate = pd.Series(False, index=df.index)
    updates: dict[str, Any] = {}
    allowed = set(allowed_columns)

    for rule in rules:
        if rule.column not in allowed or rule.column not in df.columns:
            continue
        values = pd.to_numeric(df[rule.column], errors="coerce")
        flag = _rule_flag(values, rule)
        flag_col = rule_outlier_flag_column(rule.column)
        if flag_col in df.columns:
            updates[flag_col] = (
                _as_int_flag(df[flag_col]).astype(bool) | flag
            ).astype(int)
        else:
            updates[flag_col] = flag.astype(int)
        aggregate |= flag

    updates[aggregate_flag_col] = aggregate.astype(int)
    return _assign_columns(df, updates)


def apply_default_rule_based_outlier_flags(
    df_raw: pd.DataFrame,
    csv_path: str | Path = DEFAULT_DOMAIN_RANGE_PATH,
) -> pd.DataFrame:
    """Load canonical domain ranges and attach rule-based outlier flags."""

    return apply_rule_based_outlier_flags(df_raw, load_domain_range_rules(csv_path))


# Compatibility aliases for earlier domain-QC names.
normalize_missing_input = prepare_missing_features
apply_domain_range_flags = apply_rule_based_outlier_flags
apply_default_domain_flags = apply_default_rule_based_outlier_flags
load_smartfarm_korea_domain_rules = load_domain_range_rules
apply_smartfarm_korea_domain_flags = apply_default_rule_based_outlier_flags


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
    "ACTION_COLUMNS",
    "ACTION_CONTROL_LOG_MAP",
    "ACTION_RESTORED_FLAG_SUFFIX",
    "CONTROL_LOG_TIME_CANDIDATES",
    "DEFAULT_DOMAIN_RANGE_PATH",
    "DEFAULT_RULE_AGGREGATE_FLAG",
    "DomainRangeRule",
    "EXTERNAL_STATE_COLUMNS",
    "FEATURE_COLUMNS",
    "INVALID_FLAG_SUFFIX",
    "INTERNAL_STATE_COLUMNS",
    "MISSING_FLAG_SUFFIX",
    "OBSERVATION_RULE_COLUMNS",
    "QUALITY_AI_OUTLIER_FLAG_COLUMN",
    "DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD",
    "QUALITY_IMPUTED_FLAG_COLUMN",
    "QUALITY_IMPUTED_FLAG_SUFFIX",
    "QUALITY_INVALID_FLAG_COLUMN",
    "RESAMPLED_ROW_COLUMN",
    "SEGMENT_COLUMN",
    "STATE_COLUMNS",
    "SYSTEM_STATE_COLUMNS",
    "TIME_COLUMN",
    "action_restored_flag_column",
    "add_missing_flags",
    "ai_anomaly_score_column",
    "ai_confidence_column",
    "apply_default_domain_flags",
    "apply_default_rule_based_outlier_flags",
    "apply_domain_range_flags",
    "apply_rule_based_outlier_flags",
    "apply_smartfarm_korea_domain_flags",
    "invalid_flag_column",
    "load_domain_range_rules",
    "load_smartfarm_korea_domain_rules",
    "missing_flag_column",
    "normalize_missing_input",
    "prepare_missing_outliers_handled_features",
    "prepare_missing_features",
    "prepare_quality_features",
    "quality_ai_outlier_flag_column",
    "quality_imputed_flag_column",
    "raw_value_column",
    "restore_action_columns_from_control_log",
    "rule_outlier_flag_column",
]
