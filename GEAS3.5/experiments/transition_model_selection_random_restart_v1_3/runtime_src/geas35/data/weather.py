"""Database adapter for weather-derived day classification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from geas35.core import DaylightCondition, classify_daylight_condition
from geas35.data.crop_info import DbConfig, connect, fetch_growth_stage_at


@dataclass(frozen=True)
class WeatherStationRecord:
    iot_data_idx: int
    city_code: str | None
    stn_nm: str | None
    stn_code: str


def weather_station_record_from_row(row: dict[str, Any]) -> WeatherStationRecord:
    return WeatherStationRecord(
        iot_data_idx=int(row["idx"]),
        city_code=row.get("city_code"),
        stn_nm=row.get("stn_nm"),
        stn_code=str(row["stn_code"]),
    )


def fetch_weather_station_for_iot(
    connection,
    *,
    iot_data_idx: int,
) -> WeatherStationRecord | None:
    """Fetch the weather station linked to a greenhouse."""

    query = """
        SELECT idx, city_code, stn_nm, stn_code
        FROM iot_data_info
        WHERE idx = %s
          AND stn_code IS NOT NULL
          AND TRIM(stn_code) <> ''
        LIMIT 1
    """
    with connection.cursor() as cur:
        cur.execute(query, (iot_data_idx,))
        row = cur.fetchone()
    return weather_station_record_from_row(row) if row else None


def fetch_daily_sunshine_minutes(
    connection,
    *,
    station_code: str,
    target_day: date,
) -> float | None:
    """Fetch daily accumulated sunshine minutes from ``weather_data.sun_Time``."""

    start = datetime.combine(target_day, time.min)
    end = start + timedelta(days=1)
    query = """
        SELECT
            MAX(
                CASE
                    WHEN sun_Time REGEXP '^-?[0-9]+(\\\\.[0-9]+)?$'
                    THEN CAST(sun_Time AS DECIMAL(10,3))
                END
            ) AS daily_sunshine_minutes
        FROM weather_data
        WHERE stn_Code = %s
          AND date >= %s
          AND date < %s
    """
    with connection.cursor() as cur:
        cur.execute(query, (station_code, start, end))
        row = cur.fetchone()
    if not row or row.get("daily_sunshine_minutes") is None:
        return None
    return float(row["daily_sunshine_minutes"])


def fetch_daylight_condition_at(
    connection,
    *,
    iot_data_idx: int,
    target_ts: datetime,
    lat: float,
    lon: float,
    tz: str = "Asia/Seoul",
) -> DaylightCondition:
    """Classify the target date using WMO-style sunshine ratio."""

    station = fetch_weather_station_for_iot(connection, iot_data_idx=iot_data_idx)
    if station is None:
        return "unknown"
    sunshine_minutes = fetch_daily_sunshine_minutes(
        connection,
        station_code=station.stn_code,
        target_day=target_ts.date(),
    )
    return classify_daylight_condition(
        actual_sunshine_minutes=sunshine_minutes,
        lat=lat,
        lon=lon,
        when=target_ts,
        tz=tz,
    )


def fetch_daylight_condition_at_from_db(
    *,
    iot_data_idx: int,
    target_ts: datetime,
    lat: float,
    lon: float,
    tz: str = "Asia/Seoul",
    config: DbConfig | None = None,
) -> DaylightCondition:
    """Open a DB connection and classify the target date's daylight condition."""

    with connect(config) as connection:
        return fetch_daylight_condition_at(
            connection,
            iot_data_idx=iot_data_idx,
            target_ts=target_ts,
            lat=lat,
            lon=lon,
            tz=tz,
        )


def fetch_growth_stage_controller_config_at(
    connection,
    *,
    iot_data_idx: int,
    target_ts: datetime,
    lat: float,
    lon: float,
    tz: str = "Asia/Seoul",
) -> dict[str, object] | None:
    """Fetch growth stage and weather condition, then build controller config."""

    stage = fetch_growth_stage_at(
        connection,
        iot_data_idx=iot_data_idx,
        target_ts=target_ts,
    )
    if stage is None:
        return None
    daylight_condition = fetch_daylight_condition_at(
        connection,
        iot_data_idx=iot_data_idx,
        target_ts=target_ts,
        lat=lat,
        lon=lon,
        tz=tz,
    )
    return stage.to_controller_stage_config(
        target_ts=target_ts,
        lat=lat,
        lon=lon,
        tz=tz,
        daylight_condition=daylight_condition,
    )


def fetch_growth_stage_controller_config_at_from_db(
    *,
    iot_data_idx: int,
    target_ts: datetime,
    lat: float,
    lon: float,
    tz: str = "Asia/Seoul",
    config: DbConfig | None = None,
) -> dict[str, object] | None:
    """Open a DB connection and build weather-aware controller config."""

    with connect(config) as connection:
        return fetch_growth_stage_controller_config_at(
            connection,
            iot_data_idx=iot_data_idx,
            target_ts=target_ts,
            lat=lat,
            lon=lon,
            tz=tz,
        )


__all__ = [
    "WeatherStationRecord",
    "fetch_daily_sunshine_minutes",
    "fetch_daylight_condition_at",
    "fetch_daylight_condition_at_from_db",
    "fetch_weather_station_for_iot",
    "fetch_growth_stage_controller_config_at",
    "fetch_growth_stage_controller_config_at_from_db",
    "weather_station_record_from_row",
]
