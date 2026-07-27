"""Rule-based and TCN-ready outlier handling for GEAS feature tables.

This stage keeps original feature values immutable and creates explicit
controller input columns. The policy is:

* rule normal + TCN normal: use raw value;
* rule normal + TCN warning: use raw value and set AI warning flag;
* rule normal + TCN outlier: use TCN prediction when available;
* rule outlier: use TCN prediction when available, regardless of TCN state;
* rule outlier + TCN outlier: mark high-confidence outlier.

The actual TCN model is intentionally not implemented here. Future code can
inject ``{feature}_tcn_pred_value`` and ``{feature}_tcn_outlier_score`` columns
or pass a model object with a compatible output, and this policy will operate
without changing the surrounding pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_input_schema = import_module("geas35.preprocessing.1_input_schema_preparation")
ACTION_COLUMNS = _input_schema.ACTION_COLUMNS
STATE_COLUMNS = _input_schema.STATE_COLUMNS


DEFAULT_DOMAIN_RANGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "qc"
    / "smartfarm_korea_domain_ranges.csv"
)

DEFAULT_RULE_FLAG_PREFIX = "rule_outlier"
DEFAULT_RULE_AGGREGATE_FLAG = "rule_outlier_flag"
DEFAULT_TCN_SCORE_COLUMN = "tcn_outlier_score"
DEFAULT_TCN_WARNING_FLAG_COLUMN = "tcn_warning_flag"
DEFAULT_TCN_FLAG_COLUMN = "tcn_outlier_flag"
DEFAULT_TCN_PRED_VALUE_COLUMN = "tcn_pred_value"
DEFAULT_AI_WARNING_FLAG_COLUMN = "ai_warning_flag"
DEFAULT_AI_OUTLIER_FLAG_COLUMN = "ai_outlier_flag"
DEFAULT_IMPUTED_FLAG_COLUMN = "outlier_imputed_flag"
DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN = "high_confidence_outlier_flag"
DEFAULT_AGGREGATE_FLAG = "outlier_flag"
OBSERVATION_RULE_COLUMNS = tuple(STATE_COLUMNS)


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


@dataclass(frozen=True)
class TCNOutlierConfig:
    """Runtime options for pluggable TCN-based outlier outputs.

    ``warning_threshold`` is T1 and ``outlier_threshold`` is T2. ``score_threshold``
    is kept for compatibility with the earlier one-threshold placeholder and,
    when provided, overrides ``outlier_threshold``.
    """

    warning_threshold: float = 0.5
    outlier_threshold: float = 0.8
    score_column: str = DEFAULT_TCN_SCORE_COLUMN
    warning_flag_column: str = DEFAULT_TCN_WARNING_FLAG_COLUMN
    flag_column: str = DEFAULT_TCN_FLAG_COLUMN
    pred_value_column: str = DEFAULT_TCN_PRED_VALUE_COLUMN
    score_threshold: float | None = None

    @property
    def resolved_outlier_threshold(self) -> float:
        return (
            self.outlier_threshold
            if self.score_threshold is None
            else self.score_threshold
        )


TCNOutlierModel = Callable[[pd.DataFrame], Any]


def raw_value_column(column: str) -> str:
    return f"{column}_raw_value"


def controller_value_column(column: str) -> str:
    return f"{column}_controller_value"


def rule_outlier_flag_column(column: str) -> str:
    return f"{column}_rule_outlier_flag"


def tcn_score_column(column: str) -> str:
    return f"{column}_tcn_outlier_score"


def tcn_warning_flag_column(column: str) -> str:
    return f"{column}_tcn_warning_flag"


def tcn_outlier_flag_column(column: str) -> str:
    return f"{column}_tcn_outlier_flag"


def tcn_pred_value_column(column: str) -> str:
    return f"{column}_tcn_pred_value"


def ai_warning_flag_column(column: str) -> str:
    return f"{column}_ai_warning_flag"


def ai_outlier_flag_column(column: str) -> str:
    return f"{column}_ai_outlier_flag"


def imputed_flag_column(column: str) -> str:
    return f"{column}_outlier_imputed_flag"


def high_confidence_outlier_flag_column(column: str) -> str:
    return f"{column}_high_confidence_outlier_flag"


def load_domain_range_rules(
    csv_path: str | Path = DEFAULT_DOMAIN_RANGE_PATH,
) -> list[DomainRangeRule]:
    """Load GEAS canonical domain-range rules.

    The canonical CSV uses ASCII column names:
    ``column, lower, upper, source_code, source_name, unit``. Optional
    ``lower_inclusive``, ``upper_inclusive``, ``allowed_values``, and
    ``integer_only`` columns refine boundary behavior when domain guidance uses
    wording such as "greater than or equal to".
    """

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


def _target_columns(
    df: pd.DataFrame,
    rules: Iterable[DomainRangeRule],
    *,
    allowed_columns: Iterable[str] = OBSERVATION_RULE_COLUMNS,
) -> list[str]:
    allowed = set(allowed_columns)
    columns: list[str] = []
    for rule in rules:
        if (
            rule.column in allowed
            and rule.column in df.columns
            and rule.column not in columns
        ):
            columns.append(rule.column)
    return columns


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
    """Attach observation-only inclusive rule-based outlier flags.

    Non-missing values outside ``[lower, upper]`` get flag 1. Missing values are
    not flagged here because missingness is represented by the previous stage.
    Action/control columns are intentionally excluded even when a rule is
    supplied for them.
    """

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


def _model_output(model: Any, df: pd.DataFrame) -> Any:
    if callable(model):
        return model(df)
    if hasattr(model, "predict_outliers"):
        return model.predict_outliers(df)
    if hasattr(model, "predict_scores"):
        return model.predict_scores(df)
    if hasattr(model, "score_samples"):
        return model.score_samples(df)
    if hasattr(model, "predict"):
        return model.predict(df)
    raise TypeError(
        "TCN outlier model must be callable or expose predict_outliers, "
        "predict_scores, score_samples, or predict."
    )


def _normalize_model_output(raw_output: Any, index: pd.Index) -> pd.DataFrame:
    if raw_output is None:
        return pd.DataFrame(index=index)
    if isinstance(raw_output, pd.DataFrame):
        if len(raw_output) != len(index):
            raise ValueError("TCN output DataFrame must have one row per input row.")
        result = raw_output.copy()
        result.index = index
        return result
    if isinstance(raw_output, dict):
        result = pd.DataFrame(raw_output, index=index)
        if len(result) != len(index):
            raise ValueError("TCN output dict must have one value per input row.")
        return result

    scores = pd.Series(raw_output, index=index)
    if len(scores) != len(index):
        raise ValueError("TCN outlier model must return one score per input row.")
    return pd.DataFrame({DEFAULT_TCN_SCORE_COLUMN: scores}, index=index)


def _inject_tcn_model_outputs(
    df_raw: pd.DataFrame,
    model: TCNOutlierModel | Any | None,
) -> pd.DataFrame:
    if model is None:
        return df_raw.copy()

    df = df_raw.copy()
    outputs = _normalize_model_output(_model_output(model, df), df.index)
    for col in outputs.columns:
        df[col] = outputs[col]
    return df


def apply_tcn_outlier_flags(
    df_raw: pd.DataFrame,
    model: TCNOutlierModel | Any | None = None,
    *,
    config: TCNOutlierConfig = TCNOutlierConfig(),
) -> pd.DataFrame:
    """Attach row-level placeholder TCN score/prediction/flag columns.

    This preserves the earlier placeholder interface. If no TCN model is
    supplied, no replacement can be performed: score/prediction are NA and flags
    are 0. Future TCN code can also inject per-feature columns such as
    ``in_temp_tcn_pred_value`` and ``in_temp_tcn_outlier_score``.
    """

    if df_raw is None or df_raw.empty:
        return pd.DataFrame() if df_raw is None else df_raw.copy()

    df = _inject_tcn_model_outputs(df_raw, model)

    if config.score_column not in df.columns:
        df[config.score_column] = pd.NA
    if config.pred_value_column not in df.columns:
        df[config.pred_value_column] = pd.NA

    scores = pd.to_numeric(df[config.score_column], errors="coerce")
    warning = scores.notna() & (scores >= config.warning_threshold)
    outlier = scores.notna() & (scores >= config.resolved_outlier_threshold)

    df[config.warning_flag_column] = warning.astype(int)
    df[config.flag_column] = outlier.astype(int)
    return df


def _ensure_feature_tcn_columns(
    df_raw: pd.DataFrame,
    columns: Iterable[str],
    *,
    config: TCNOutlierConfig,
) -> pd.DataFrame:
    df = df_raw.copy()
    updates: dict[str, Any] = {}
    for col in columns:
        feature_score_col = tcn_score_column(col)
        feature_pred_col = tcn_pred_value_column(col)
        feature_tcn_warning_col = tcn_warning_flag_column(col)
        feature_tcn_outlier_col = tcn_outlier_flag_column(col)

        if feature_score_col not in df.columns:
            if config.score_column in df.columns:
                updates[feature_score_col] = df[config.score_column]
            else:
                updates[feature_score_col] = pd.Series(pd.NA, index=df.index)
        if feature_pred_col not in df.columns:
            if config.pred_value_column in df.columns:
                updates[feature_pred_col] = df[config.pred_value_column]
            else:
                updates[feature_pred_col] = pd.Series(pd.NA, index=df.index)

        score_values = updates.get(feature_score_col, df.get(feature_score_col))
        scores = pd.to_numeric(score_values, errors="coerce")
        warning = scores.notna() & (scores >= config.warning_threshold)
        outlier = scores.notna() & (scores >= config.resolved_outlier_threshold)
        updates[feature_tcn_warning_col] = warning.astype(int)
        updates[feature_tcn_outlier_col] = outlier.astype(int)
    return _assign_columns(df, updates)


def apply_controller_value_policy(
    df_raw: pd.DataFrame,
    columns: Iterable[str],
    *,
    tcn_config: TCNOutlierConfig = TCNOutlierConfig(),
) -> pd.DataFrame:
    """Create raw/controller values and traceable policy flags per feature."""

    if df_raw is None or df_raw.empty:
        return pd.DataFrame() if df_raw is None else df_raw.copy()

    df = df_raw.copy()
    updates: dict[str, Any] = {}
    aggregate_outlier = pd.Series(False, index=df.index)
    aggregate_ai_warning = pd.Series(False, index=df.index)
    aggregate_ai_outlier = pd.Series(False, index=df.index)
    aggregate_imputed = pd.Series(False, index=df.index)
    aggregate_high_conf = pd.Series(False, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        raw_col = raw_value_column(col)
        controller_col = controller_value_column(col)
        rule_col = rule_outlier_flag_column(col)
        feature_tcn_warning_col = tcn_warning_flag_column(col)
        feature_tcn_outlier_col = tcn_outlier_flag_column(col)
        feature_pred_col = tcn_pred_value_column(col)
        ai_warning_col = ai_warning_flag_column(col)
        ai_outlier_col = ai_outlier_flag_column(col)
        imputed_col = imputed_flag_column(col)
        high_conf_col = high_confidence_outlier_flag_column(col)

        updates[raw_col] = df[col]
        rule_flag = (
            _as_int_flag(df[rule_col]).astype(bool)
            if rule_col in df.columns
            else pd.Series(False, index=df.index)
        )
        tcn_warning_flag = (
            _as_int_flag(df[feature_tcn_warning_col]).astype(bool)
            if feature_tcn_warning_col in df.columns
            else pd.Series(False, index=df.index)
        )
        tcn_outlier_flag = (
            _as_int_flag(df[feature_tcn_outlier_col]).astype(bool)
            if feature_tcn_outlier_col in df.columns
            else pd.Series(False, index=df.index)
        )
        pred = (
            pd.to_numeric(df[feature_pred_col], errors="coerce")
            if feature_pred_col in df.columns
            else pd.Series(pd.NA, index=df.index, dtype="Float64")
        )

        replace_requested = rule_flag | tcn_outlier_flag
        replacement_available = pred.notna()
        replaced = replace_requested & replacement_available

        controller_values = df[col].where(~replaced, pred)
        ai_warning_values = (tcn_warning_flag & ~tcn_outlier_flag).astype(int)
        ai_outlier_values = tcn_outlier_flag.astype(int)
        imputed_values = replaced.astype(int)
        high_conf_values = (rule_flag & tcn_outlier_flag).astype(int)

        updates[controller_col] = controller_values
        updates[ai_warning_col] = ai_warning_values
        updates[ai_outlier_col] = ai_outlier_values
        updates[imputed_col] = imputed_values
        updates[high_conf_col] = high_conf_values

        aggregate_outlier |= rule_flag | tcn_outlier_flag
        aggregate_ai_warning |= ai_warning_values.astype(bool)
        aggregate_ai_outlier |= tcn_outlier_flag
        aggregate_imputed |= replaced
        aggregate_high_conf |= high_conf_values.astype(bool)

    updates[DEFAULT_AGGREGATE_FLAG] = aggregate_outlier.astype(int)
    updates[DEFAULT_AI_WARNING_FLAG_COLUMN] = aggregate_ai_warning.astype(int)
    updates[DEFAULT_AI_OUTLIER_FLAG_COLUMN] = aggregate_ai_outlier.astype(int)
    updates[DEFAULT_IMPUTED_FLAG_COLUMN] = aggregate_imputed.astype(int)
    updates[DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN] = aggregate_high_conf.astype(int)
    return _assign_columns(df, updates)


def apply_outlier_flags(
    df_raw: pd.DataFrame,
    *,
    rules: Iterable[DomainRangeRule] | None = None,
    domain_csv_path: str | Path = DEFAULT_DOMAIN_RANGE_PATH,
    tcn_model: TCNOutlierModel | Any | None = None,
    tcn_config: TCNOutlierConfig = TCNOutlierConfig(),
    aggregate_flag_col: str = DEFAULT_AGGREGATE_FLAG,
) -> pd.DataFrame:
    """Apply rule/TCN detection and create controller-value columns."""

    if df_raw is None or df_raw.empty:
        return pd.DataFrame() if df_raw is None else df_raw.copy()

    rule_set = list(rules) if rules is not None else load_domain_range_rules(domain_csv_path)
    target_cols = _target_columns(df_raw, rule_set)

    df = apply_rule_based_outlier_flags(df_raw, rule_set)
    df = apply_tcn_outlier_flags(df, tcn_model, config=tcn_config)
    df = _ensure_feature_tcn_columns(df, target_cols, config=tcn_config)
    df = apply_controller_value_policy(df, target_cols, tcn_config=tcn_config)

    if aggregate_flag_col != DEFAULT_AGGREGATE_FLAG:
        df[aggregate_flag_col] = df[DEFAULT_AGGREGATE_FLAG]
    return df


# Compatibility aliases for earlier domain-QC names.
apply_domain_range_flags = apply_rule_based_outlier_flags
apply_default_domain_flags = apply_default_rule_based_outlier_flags
load_smartfarm_korea_domain_rules = load_domain_range_rules
apply_smartfarm_korea_domain_flags = apply_default_rule_based_outlier_flags


__all__ = [
    "DEFAULT_AGGREGATE_FLAG",
    "DEFAULT_AI_OUTLIER_FLAG_COLUMN",
    "DEFAULT_AI_WARNING_FLAG_COLUMN",
    "DEFAULT_DOMAIN_RANGE_PATH",
    "DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN",
    "DEFAULT_IMPUTED_FLAG_COLUMN",
    "DEFAULT_RULE_AGGREGATE_FLAG",
    "DEFAULT_RULE_FLAG_PREFIX",
    "DEFAULT_TCN_FLAG_COLUMN",
    "DEFAULT_TCN_PRED_VALUE_COLUMN",
    "DEFAULT_TCN_SCORE_COLUMN",
    "DEFAULT_TCN_WARNING_FLAG_COLUMN",
    "DomainRangeRule",
    "OBSERVATION_RULE_COLUMNS",
    "TCNOutlierConfig",
    "ai_outlier_flag_column",
    "ai_warning_flag_column",
    "apply_controller_value_policy",
    "apply_default_domain_flags",
    "apply_default_rule_based_outlier_flags",
    "apply_domain_range_flags",
    "apply_outlier_flags",
    "apply_rule_based_outlier_flags",
    "apply_smartfarm_korea_domain_flags",
    "apply_tcn_outlier_flags",
    "controller_value_column",
    "high_confidence_outlier_flag_column",
    "imputed_flag_column",
    "load_domain_range_rules",
    "load_smartfarm_korea_domain_rules",
    "raw_value_column",
    "rule_outlier_flag_column",
    "tcn_outlier_flag_column",
    "tcn_pred_value_column",
    "tcn_score_column",
    "tcn_warning_flag_column",
]
