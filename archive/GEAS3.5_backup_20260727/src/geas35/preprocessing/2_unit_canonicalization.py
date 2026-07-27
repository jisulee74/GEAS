"""Unit and scale canonicalization for GEAS input features.

This module is shared by offline dataset preparation and realtime inference.
It assumes ``1_input_schema_preparation`` already parsed timestamps, ensured
the GEAS feature columns, and coerced state/action features to numeric values.
This stage only converts feature units/scales and never clips values. Values
outside domain ranges must remain visible so QC can attach outlier flags.
"""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

_input_schema = import_module("geas35.preprocessing.1_input_schema_preparation")
ACTION_COLUMNS = _input_schema.ACTION_COLUMNS
EXTERNAL_STATE_COLUMNS = _input_schema.EXTERNAL_STATE_COLUMNS
INTERNAL_STATE_COLUMNS = _input_schema.INTERNAL_STATE_COLUMNS
STATE_COLUMNS = _input_schema.STATE_COLUMNS
SYSTEM_STATE_COLUMNS = _input_schema.SYSTEM_STATE_COLUMNS
TIME_COLUMN = _input_schema.TIME_COLUMN


PERCENT_ACTION_COLUMNS = [
    "cont_skyl_vol",
    "cont_skyr_vol",
    "cont_cur_vol",
    "cont_kwcur_vol",
    "cont_3way1_vol",
    "cont_3way2_vol",
]

BINARY_ACTION_COLUMNS = [
    "cont_heater_run",
    "cont_cooler_run",
    "cont_co2_run",
    "cont_pump1_run",
    "cont_pump2_run",
    "cont_fan_run",
]

HUMIDITY_PERCENT_COLUMNS = [
    "in_hum",
    "in_hum2",
    "in_medium_hum1",
    "in_medium_hum2",
    "out_hum",
]

NON_NEGATIVE_COLUMNS = [
    "out_windsp",
    "out_light",
    "out_light_sum",
    "out_rainfall",
]

DERIVED_RENAME_MAP = {
    "in_hum": "in_rh",
    "out_windsp": "wind_speed",
    "cont_skyl_vol": "window_pct",
    "cont_heater_run": "heater_duty",
}


def _require_numeric(series: pd.Series, column: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"{column} must be numeric before unit canonicalization. "
            "Run 1_input_schema_preparation first."
        )
    return series


def _scale_percent_0_1(series: pd.Series, *, column: str) -> pd.Series:
    values = _require_numeric(series, column).copy()
    if values.notna().any() and values.max(skipna=True) > 1.5:
        values = values / 100.0
    return values


def canonicalize_feature_units(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert GEAS feature units without clipping values.

    For percentage-like action columns, values that appear to be in ``[0, 100]``
    scale are converted to ``[0, 1]`` scale. Converted values are not clipped;
    for example, ``120`` becomes ``1.2`` and remains available for QC flagging.
    """

    if df_raw is None:
        return pd.DataFrame()
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()
    for col in PERCENT_ACTION_COLUMNS:
        if col in df.columns:
            df[col] = _scale_percent_0_1(df[col], column=col)

    return df


def canonicalize_for_derived(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Create derived-feature aliases without clipping values.

    This keeps old GEAS derived-feature naming available while preserving
    out-of-range evidence for QC. The aliases are ``in_hum -> in_rh``,
    ``out_windsp -> wind_speed``, ``cont_skyl_vol -> window_pct``, and
    ``cont_heater_run -> heater_duty``.
    """

    if df_raw is None:
        return pd.DataFrame()
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()
    for source_col, target_col in DERIVED_RENAME_MAP.items():
        if source_col not in df.columns:
            continue
        if target_col in df.columns:
            source = _require_numeric(df[source_col], source_col)
            target = _require_numeric(df[target_col], target_col)
            df[target_col] = target.where(target.notna(), source)
        else:
            df[target_col] = df[source_col]

    if "window_pct" in df.columns:
        df["window_pct"] = _scale_percent_0_1(
            df["window_pct"],
            column="window_pct",
        )
    else:
        df["window_pct"] = 0.0

    if "heater_duty" in df.columns:
        df["heater_duty"] = _require_numeric(df["heater_duty"], "heater_duty")
    else:
        df["heater_duty"] = np.nan

    if "in_rh" in df.columns:
        df["in_rh"] = _require_numeric(df["in_rh"], "in_rh")

    if "wind_speed" in df.columns:
        df["wind_speed"] = _require_numeric(df["wind_speed"], "wind_speed")
    else:
        df["wind_speed"] = np.nan

    return df


# Backward-compatible names while terminology is being cleaned up.
normalize_feature_ranges = canonicalize_feature_units
normalize_for_derived = canonicalize_for_derived


__all__ = [
    "BINARY_ACTION_COLUMNS",
    "DERIVED_RENAME_MAP",
    "HUMIDITY_PERCENT_COLUMNS",
    "NON_NEGATIVE_COLUMNS",
    "PERCENT_ACTION_COLUMNS",
    "canonicalize_feature_units",
    "canonicalize_for_derived",
    "normalize_feature_ranges",
    "normalize_for_derived",
]
