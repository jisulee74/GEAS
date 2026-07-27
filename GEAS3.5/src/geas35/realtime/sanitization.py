"""Realtime controller-only value sanitization.

This module creates a clipped controller view after preprocessing and QC flags
are available. It should not be used for offline training data preparation,
because clipping would hide out-of-range evidence from outlier handling.
"""

from __future__ import annotations

import pandas as pd

from geas35.preprocessing import (
    BINARY_ACTION_COLUMNS,
    HUMIDITY_PERCENT_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    PERCENT_ACTION_COLUMNS,
    SYSTEM_STATE_COLUMNS,
    canonicalize_feature_units,
)


def sanitize_for_controller(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Return a clipped realtime controller view after QC flags are preserved."""

    df = canonicalize_feature_units(df_raw)
    if df.empty:
        return df

    for col in HUMIDITY_PERCENT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 100.0)

    for col in SYSTEM_STATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)

    for col in PERCENT_ACTION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)

    for col in BINARY_ACTION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)

    if "out_winddirec" in df.columns:
        df["out_winddirec"] = df["out_winddirec"].clip(0.0, 360.0)

    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    if "out_rain" in df.columns:
        df["out_rain"] = df["out_rain"].clip(0.0, 1.0)

    return df


__all__ = ["sanitize_for_controller"]
