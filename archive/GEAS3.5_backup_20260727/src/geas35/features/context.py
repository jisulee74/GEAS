"""Context feature builders shared by MDP and transition-model inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from geas35.core import classify_solar_period, classify_sunshine_ratio


SOLAR_PERIOD_VALUES = ("day", "early_night", "late_night")
DAYLIGHT_CONDITION_VALUES = ("sunny", "partly_cloudy", "cloudy", "unknown")


def _safe_string_series(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    default: str,
) -> pd.Series:
    result = pd.Series(default, index=df.index, dtype="object")
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].astype("object")
        result = result.where(result.ne(default), values)
    return result.fillna(default).astype(str)


def _fallback_solar_period(timestamps: pd.Series) -> pd.Series:
    hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
    period = pd.Series("late_night", index=timestamps.index, dtype="object")
    period.loc[(hours >= 7.0) & (hours < 18.0)] = "day"
    period.loc[(hours >= 18.0) & (hours < 24.0)] = "early_night"
    return period


def _solar_period_series(
    df: pd.DataFrame,
    *,
    time_col: str,
    lat: float | None,
    lon: float | None,
    tz: str,
    early_night_hours: float,
) -> pd.Series:
    existing = _safe_string_series(
        df,
        ("solar_period", "period", "day_night_period"),
        default="",
    )
    valid_existing = existing.isin(SOLAR_PERIOD_VALUES)
    if bool(valid_existing.all()):
        return existing

    timestamps = pd.to_datetime(df[time_col], errors="coerce")
    fallback = _fallback_solar_period(timestamps)
    if lat is None or lon is None:
        return existing.where(valid_existing, fallback)

    computed = []
    for ts, fallback_value in zip(timestamps, fallback, strict=False):
        if pd.isna(ts):
            computed.append(fallback_value)
            continue
        try:
            computed.append(
                classify_solar_period(
                    now=ts,
                    lat=float(lat),
                    lon=float(lon),
                    tz=tz,
                    early_night_hours=early_night_hours,
                )
            )
        except Exception:
            computed.append(fallback_value)
    computed_series = pd.Series(computed, index=df.index, dtype="object")
    return existing.where(valid_existing, computed_series)


def _daylight_condition_series(df: pd.DataFrame) -> pd.Series:
    existing = _safe_string_series(
        df,
        ("daylight_condition", "weather_condition", "sky_condition"),
        default="",
    )
    valid_existing = existing.isin(DAYLIGHT_CONDITION_VALUES)
    if bool(valid_existing.all()):
        return existing

    ratio = None
    for column in ("sunshine_ratio_pct", "daylight_ratio_pct"):
        if column in df.columns:
            ratio = pd.to_numeric(df[column], errors="coerce")
            break
    if ratio is None:
        for column in ("sunshine_ratio", "daylight_ratio"):
            if column in df.columns:
                ratio = pd.to_numeric(df[column], errors="coerce") * 100.0
                break

    if ratio is None:
        condition = pd.Series("unknown", index=df.index, dtype="object")
    else:
        condition = pd.Series("partly_cloudy", index=df.index, dtype="object")
        condition.loc[ratio >= 70.0] = "sunny"
        condition.loc[ratio <= 30.0] = "cloudy"
        condition.loc[~np.isfinite(ratio.to_numpy(dtype=float))] = "unknown"

    if {
        "actual_sunshine_minutes",
        "possible_sunshine_minutes",
    }.issubset(df.columns):
        actual = pd.to_numeric(df["actual_sunshine_minutes"], errors="coerce")
        possible = pd.to_numeric(df["possible_sunshine_minutes"], errors="coerce")
        condition = pd.Series(
            [
                classify_sunshine_ratio(
                    actual_sunshine_minutes=a,
                    possible_sunshine_minutes_value=p,
                )
                for a, p in zip(actual, possible, strict=False)
            ],
            index=df.index,
            dtype="object",
        )

    return existing.where(valid_existing, condition)


def add_mdp_context_features(
    df_raw: pd.DataFrame,
    *,
    time_col: str = "reg_date",
    lat: float | None = None,
    lon: float | None = None,
    tz: str = "Asia/Seoul",
    early_night_hours: float = 6.0,
    prefix: str = "obs",
) -> pd.DataFrame:
    """Add solar-period and daylight-condition one-hot context features."""

    if df_raw is None:
        return pd.DataFrame()
    if df_raw.empty:
        return df_raw.copy()
    if time_col not in df_raw.columns:
        raise KeyError(f"Missing timestamp column: {time_col}")

    df = df_raw.copy()
    solar_period = _solar_period_series(
        df,
        time_col=time_col,
        lat=lat,
        lon=lon,
        tz=tz,
        early_night_hours=early_night_hours,
    )
    daylight_condition = _daylight_condition_series(df)

    df[f"{prefix}_solar_period_code"] = solar_period
    df[f"{prefix}_daylight_condition_code"] = daylight_condition
    for value in SOLAR_PERIOD_VALUES:
        df[f"{prefix}_solar_period_{value}"] = solar_period.eq(value).astype(float)
    for value in DAYLIGHT_CONDITION_VALUES:
        df[f"{prefix}_daylight_condition_{value}"] = daylight_condition.eq(value).astype(float)
    return df


__all__ = [
    "DAYLIGHT_CONDITION_VALUES",
    "SOLAR_PERIOD_VALUES",
    "add_mdp_context_features",
]
