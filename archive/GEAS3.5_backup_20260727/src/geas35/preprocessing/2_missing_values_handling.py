"""Missing-value flagging and action-log restoration for GEAS input features.

This module expects input that has already passed
``1_input_schema_preparation.py``. Step 2 of the preprocessing refactor removes
LOCF entirely:

* state/action feature missingness is recorded with feature-level flags;
* no aggregate row-level ``missing_flag`` is created in this stage;
* missing action variables are restored only from same-timestamp control logs;
* state variables and unresolved action variables remain NaN for later stages.
"""

from __future__ import annotations

from importlib import import_module
from typing import Iterable

import pandas as pd

_input_schema = import_module("geas35.preprocessing.1_input_schema_preparation")
ACTION_COLUMNS = _input_schema.ACTION_COLUMNS
EXTERNAL_STATE_COLUMNS = _input_schema.EXTERNAL_STATE_COLUMNS
FEATURE_COLUMNS = _input_schema.FEATURE_COLUMNS
INTERNAL_STATE_COLUMNS = _input_schema.INTERNAL_STATE_COLUMNS
STATE_COLUMNS = _input_schema.STATE_COLUMNS
SYSTEM_STATE_COLUMNS = _input_schema.SYSTEM_STATE_COLUMNS
TIME_COLUMN = _input_schema.TIME_COLUMN

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

SEGMENT_COLUMN = "segment_id"
RESAMPLED_ROW_COLUMN = "is_resampled_row"
MISSING_FLAG_SUFFIX = "_missing_flag"
ACTION_RESTORED_FLAG_SUFFIX = "_restored_flag"


def missing_flag_column(feature_col: str) -> str:
    """Return the feature-level missing flag column name."""

    return f"{feature_col}{MISSING_FLAG_SUFFIX}"


def action_restored_flag_column(action_col: str) -> str:
    """Return the action-log restoration flag column name."""

    return f"{action_col}{ACTION_RESTORED_FLAG_SUFFIX}"


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
            "Missing-value handling expects output from "
            "1_input_schema_preparation.py. "
            f"Missing required GEAS feature columns: {preview}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df[TIME_COLUMN]):
        raise ValueError(
            "Missing-value handling expects reg_date to be parsed before this "
            "stage. Run 1_input_schema_preparation.py first."
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
    """Restore missing action values only from same-timestamp control logs.

    Existing action values are preserved and receive ``restored_flag=0``.
    Missing actions with matching log values receive ``restored_flag=1``.
    Actions missing from both input and control log stay NaN with
    ``restored_flag=0``.
    """

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
    """Apply Step 2 missing handling to the 39 GEAS input features.

    This stage records feature-level missingness and restores missing actions
    from control logs. It does not perform LOCF, AI imputation, median
    imputation, row-level aggregate missing flags, or outlier handling.
    """

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


# Backward-compatible name for pipeline code.
normalize_missing_input = prepare_missing_features


__all__ = [
    "ACTION_COLUMNS",
    "ACTION_CONTROL_LOG_MAP",
    "ACTION_RESTORED_FLAG_SUFFIX",
    "CONTROL_LOG_TIME_CANDIDATES",
    "EXTERNAL_STATE_COLUMNS",
    "FEATURE_COLUMNS",
    "INTERNAL_STATE_COLUMNS",
    "MISSING_FLAG_SUFFIX",
    "RESAMPLED_ROW_COLUMN",
    "SEGMENT_COLUMN",
    "STATE_COLUMNS",
    "SYSTEM_STATE_COLUMNS",
    "TIME_COLUMN",
    "action_restored_flag_column",
    "add_missing_flags",
    "missing_flag_column",
    "normalize_missing_input",
    "prepare_missing_features",
    "restore_action_columns_from_control_log",
]
