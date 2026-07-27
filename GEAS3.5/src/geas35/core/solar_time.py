"""Solar day/night period helpers.

This follows the backup controller's sunrise/sunset calculation: use pvlib
when available, and fall back to fixed 07:00/18:00 local times otherwise.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

try:
    import pvlib

    _HAS_PVLIB = True
except ImportError:  # pragma: no cover - depends on optional runtime package
    _HAS_PVLIB = False


SolarPeriod = Literal["day", "early_night", "late_night"]
DaylightCondition = Literal["sunny", "partly_cloudy", "cloudy", "unknown"]


def _local_timestamp(value: object, tz: str) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def get_sun_times(
    lat: float,
    lon: float,
    when: object,
    tz: str = "Asia/Seoul",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return approximate sunrise and sunset for ``when`` in local time."""

    when_ts = _local_timestamp(when, tz)
    day = when_ts.normalize()

    if _HAS_PVLIB:
        times = pd.date_range(start=day, periods=24 * 4, freq="15min", tz=tz)
        solar_position = pvlib.solarposition.get_solarposition(times, lat, lon)
        daylight = solar_position["apparent_elevation"] > 0
        if daylight.any():
            daylight_times = times[daylight.to_numpy()]
            return daylight_times[0], daylight_times[-1]

    return day + pd.Timedelta(hours=7), day + pd.Timedelta(hours=18)


def possible_sunshine_minutes(
    *,
    lat: float,
    lon: float,
    when: object,
    tz: str = "Asia/Seoul",
) -> float:
    """Return theoretical daylight duration in minutes for ``when``."""

    sunrise, sunset = get_sun_times(lat, lon, when, tz=tz)
    return float((sunset - sunrise) / pd.Timedelta(minutes=1))


def classify_sunshine_ratio(
    *,
    actual_sunshine_minutes: float | int | None,
    possible_sunshine_minutes_value: float | int | None,
) -> DaylightCondition:
    """Classify weather by WMO-style sunshine ratio thresholds."""

    if actual_sunshine_minutes is None or possible_sunshine_minutes_value is None:
        return "unknown"
    possible = float(possible_sunshine_minutes_value)
    if possible <= 0:
        return "unknown"
    ratio_pct = float(actual_sunshine_minutes) / possible * 100.0
    if ratio_pct >= 70.0:
        return "sunny"
    if ratio_pct <= 30.0:
        return "cloudy"
    return "partly_cloudy"


def classify_daylight_condition(
    *,
    actual_sunshine_minutes: float | int | None,
    lat: float,
    lon: float,
    when: object,
    tz: str = "Asia/Seoul",
) -> DaylightCondition:
    """Classify clear, partly cloudy, or cloudy using sunshine ratio."""

    possible = possible_sunshine_minutes(lat=lat, lon=lon, when=when, tz=tz)
    return classify_sunshine_ratio(
        actual_sunshine_minutes=actual_sunshine_minutes,
        possible_sunshine_minutes_value=possible,
    )


def classify_solar_period_from_bounds(
    *,
    now: object,
    sunrise: object,
    sunset: object,
    previous_sunset: object | None = None,
    next_sunrise: object | None = None,
    tz: str = "Asia/Seoul",
    early_night_hours: float = 6.0,
) -> SolarPeriod:
    """Classify day, early night, or late night from explicit solar bounds."""

    now_ts = _local_timestamp(now, tz)
    sunrise_ts = _local_timestamp(sunrise, tz)
    sunset_ts = _local_timestamp(sunset, tz)
    early_duration = pd.Timedelta(hours=float(early_night_hours))

    if sunrise_ts <= now_ts < sunset_ts:
        return "day"

    if now_ts < sunrise_ts:
        prev_sunset_ts = (
            _local_timestamp(previous_sunset, tz)
            if previous_sunset is not None
            else sunset_ts - pd.Timedelta(days=1)
        )
        return "early_night" if now_ts < prev_sunset_ts + early_duration else "late_night"

    next_sunrise_ts = (
        _local_timestamp(next_sunrise, tz)
        if next_sunrise is not None
        else sunrise_ts + pd.Timedelta(days=1)
    )
    if now_ts < next_sunrise_ts:
        return "early_night" if now_ts < sunset_ts + early_duration else "late_night"

    return "day"


def classify_solar_period(
    *,
    now: object,
    lat: float,
    lon: float,
    tz: str = "Asia/Seoul",
    early_night_hours: float = 6.0,
) -> SolarPeriod:
    """Classify the current solar period using calculated sunrise/sunset."""

    now_ts = _local_timestamp(now, tz)
    sunrise, sunset = get_sun_times(lat, lon, now_ts, tz=tz)
    previous_sunrise, previous_sunset = get_sun_times(
        lat, lon, now_ts - pd.Timedelta(days=1), tz=tz
    )
    next_sunrise, next_sunset = get_sun_times(
        lat, lon, now_ts + pd.Timedelta(days=1), tz=tz
    )
    del previous_sunrise, next_sunset

    return classify_solar_period_from_bounds(
        now=now_ts,
        sunrise=sunrise,
        sunset=sunset,
        previous_sunset=previous_sunset,
        next_sunrise=next_sunrise,
        tz=tz,
        early_night_hours=early_night_hours,
    )


__all__ = [
    "DaylightCondition",
    "SolarPeriod",
    "classify_daylight_condition",
    "classify_solar_period",
    "classify_solar_period_from_bounds",
    "classify_sunshine_ratio",
    "get_sun_times",
    "possible_sunshine_minutes",
]
