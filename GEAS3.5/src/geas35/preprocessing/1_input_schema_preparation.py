"""Input schema preparation for GEAS 3.5 feature tables.

This module is intentionally narrower than scaling/normalization. It only makes
incoming offline or realtime data conform to the GEAS input contract:

* parse and validate ``reg_date``;
* ensure the 39 input feature columns exist;
* coerce state/action features to numeric values.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


TIME_COLUMN = "reg_date"

INTERNAL_STATE_COLUMNS = [
    "in_medium_temp1",
    "in_medium_temp2",
    "in_temp",
    "in_temp2",
    "in_water_hot",
    "in_water_cold",
    "in_hum",
    "in_hum2",
    "in_medium_hum1",
    "in_medium_hum2",
    "in_co2",
    "in_co2_2",
    "in_medium_ec1",
    "in_medium_ec2",
]

SYSTEM_STATE_COLUMNS = [
    "etc_blackout",
    "etc_plc_abnorm",
    "etc_plc_norm",
]

EXTERNAL_STATE_COLUMNS = [
    "out_temp",
    "out_hum",
    "out_winddirec",
    "out_windsp",
    "out_light",
    "out_light_sum",
    "out_rainfall",
    "out_rain",
    "out_airpress",
]

ACTION_COLUMNS = [
    "cont_skyl_vol",
    "cont_skyr_vol",
    "cont_cur_vol",
    "cont_kwcur_vol",
    "cont_co2_run",
    "cont_pump1_run",
    "cont_pump2_run",
    "cont_heater_run",
    "cont_cooler_run",
    "cont_3way1_vol",
    "cont_3way2_vol",
    "cont_fan_run",
]

STATE_COLUMNS = [
    *INTERNAL_STATE_COLUMNS,
    *SYSTEM_STATE_COLUMNS,
    *EXTERNAL_STATE_COLUMNS,
]

NUMERIC_FEATURE_COLUMNS = [
    *STATE_COLUMNS,
    *ACTION_COLUMNS,
]

FEATURE_COLUMNS = [
    TIME_COLUMN,
    *NUMERIC_FEATURE_COLUMNS,
]


def ensure_columns(
    df_raw: pd.DataFrame,
    columns: Iterable[str],
    *,
    default: float = np.nan,
) -> pd.DataFrame:
    """Return a copy with missing columns added using ``default``."""

    df = df_raw.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = default
    return df


def coerce_numeric_columns(
    df_raw: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Return a copy with selected columns coerced to numeric values."""

    df = df_raw.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_reg_date(
    df_raw: pd.DataFrame,
    *,
    drop_invalid: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Parse ``reg_date`` and optionally drop invalid timestamps."""

    if df_raw is None or df_raw.empty:
        return pd.DataFrame() if df_raw is None else df_raw.copy()
    if TIME_COLUMN not in df_raw.columns:
        return df_raw.iloc[0:0].copy()

    df = df_raw.copy()
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    if drop_invalid:
        df = df.dropna(subset=[TIME_COLUMN])
    if sort:
        df = df.sort_values(TIME_COLUMN)
    return df.reset_index(drop=True)


def ensure_geas_feature_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Ensure all GEAS state/action feature columns exist."""

    return ensure_columns(df_raw, NUMERIC_FEATURE_COLUMNS)


def coerce_geas_numeric_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce GEAS state/action feature columns to numeric values."""

    return coerce_numeric_columns(df_raw, NUMERIC_FEATURE_COLUMNS)


def prepare_geas_input_schema(
    df_raw: pd.DataFrame,
    *,
    keep_extra_columns: bool = True,
    drop_invalid_time: bool = True,
    sort_by_time: bool = True,
) -> pd.DataFrame:
    """Prepare the 39-column GEAS input schema without imputing values.

    Invalid or missing ``reg_date`` rows are removed by default. Missing
    state/action feature columns are added as NaN, and feature values that
    cannot be parsed as numbers become NaN. No LOCF, action-log restoration,
    clipping, or outlier flagging is performed here.
    """

    if df_raw is None or df_raw.empty:
        return pd.DataFrame() if df_raw is None else df_raw.copy()

    df = parse_reg_date(
        df_raw,
        drop_invalid=drop_invalid_time,
        sort=sort_by_time,
    )
    if df.empty:
        return df

    df = ensure_geas_feature_columns(df)
    df = coerce_geas_numeric_features(df)

    if keep_extra_columns:
        return df
    return df[FEATURE_COLUMNS].copy()


# Alias kept for readability in pipeline code. This is schema preparation, not
# value normalization/scaling.
standardize_geas_input_schema = prepare_geas_input_schema


__all__ = [
    "ACTION_COLUMNS",
    "EXTERNAL_STATE_COLUMNS",
    "FEATURE_COLUMNS",
    "INTERNAL_STATE_COLUMNS",
    "NUMERIC_FEATURE_COLUMNS",
    "STATE_COLUMNS",
    "SYSTEM_STATE_COLUMNS",
    "TIME_COLUMN",
    "coerce_geas_numeric_features",
    "coerce_numeric_columns",
    "ensure_columns",
    "ensure_geas_feature_columns",
    "parse_reg_date",
    "prepare_geas_input_schema",
    "standardize_geas_input_schema",
]
