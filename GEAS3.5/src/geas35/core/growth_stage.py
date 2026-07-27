"""Crop growth-stage lookup from transplant date.

The rule table is based on DAT (days after transplanting). DAT is one-based:
the transplant calendar date itself is DAT 1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from geas35.core.solar_time import DaylightCondition, SolarPeriod, classify_solar_period


DEFAULT_GROWTH_STAGE_RULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "offline_dataset_preparation"
    / "configs"
    / "crops"
    / "growth_stage_rules.csv"
)

SUBJ_CD_TO_CROP = {
    "01": "strawberry",
    "04": "cucumber",
    "05": "melon",
}


@dataclass(frozen=True)
class TemperatureRange:
    """Inclusive temperature range in Celsius."""

    min_c: float | None
    max_c: float | None

    @property
    def midpoint_c(self) -> float | None:
        if self.min_c is None and self.max_c is None:
            return None
        if self.min_c is None:
            return self.max_c
        if self.max_c is None:
            return self.min_c
        return (self.min_c + self.max_c) / 2.0

    def as_tuple(self) -> tuple[float | None, float | None]:
        return self.min_c, self.max_c


@dataclass(frozen=True)
class GrowthStageRule:
    crop: str
    subj_cd: str
    stage_order: int
    stage_name: str
    dat_start: int
    dat_end: int | None
    day_temp: TemperatureRange
    day_cloudy_temp: TemperatureRange
    night_temp: TemperatureRange
    night_early_temp: TemperatureRange
    night_late_temp: TemperatureRange
    root_temp: TemperatureRange
    notes: str = ""

    def contains_dat(self, dat: int) -> bool:
        if dat < self.dat_start:
            return False
        return self.dat_end is None or dat <= self.dat_end


@dataclass(frozen=True)
class GrowthStage:
    crop: str
    subj_cd: str
    stage_order: int
    stage_name: str
    dat: int
    dat_start: int
    dat_end: int | None
    day_temp: TemperatureRange
    day_cloudy_temp: TemperatureRange
    night_temp: TemperatureRange
    night_early_temp: TemperatureRange
    night_late_temp: TemperatureRange
    root_temp: TemperatureRange
    notes: str = ""

    @property
    def target_day_temp_c(self) -> float | None:
        return self.day_temp.midpoint_c

    @property
    def target_night_temp_c(self) -> float | None:
        return self.night_temp.midpoint_c

    def day_temp_for_condition(
        self,
        daylight_condition: DaylightCondition | str | None = None,
    ) -> TemperatureRange:
        """Return day range, using cloudy range for cloudy/partly-cloudy days."""

        if (
            daylight_condition in {"partly_cloudy", "cloudy"}
            and self.day_cloudy_temp.midpoint_c is not None
        ):
            return self.day_cloudy_temp
        return self.day_temp

    def temperature_range_for_period(
        self,
        period: SolarPeriod | str,
        *,
        daylight_condition: DaylightCondition | str | None = None,
    ) -> TemperatureRange:
        """Return the applicable temperature range for a solar period."""

        if period == "day":
            return self.day_temp_for_condition(daylight_condition)
        if period == "early_night" and self.night_early_temp.midpoint_c is not None:
            return self.night_early_temp
        if period == "late_night" and self.night_late_temp.midpoint_c is not None:
            return self.night_late_temp
        if period in {"early_night", "late_night", "night"}:
            return self.night_temp
        raise ValueError(f"Unknown solar period: {period!r}")

    def target_temp_c_for_period(
        self,
        period: SolarPeriod | str,
        *,
        daylight_condition: DaylightCondition | str | None = None,
    ) -> float | None:
        """Return the midpoint target for the applicable solar-period range."""

        return self.temperature_range_for_period(
            period,
            daylight_condition=daylight_condition,
        ).midpoint_c

    def effective_temperature_target_for_period(
        self,
        period: SolarPeriod | str,
        daylight_condition: DaylightCondition | str | None = None,
    ) -> dict[str, object]:
        """Return the current period-specific target and allowed range."""

        temp_range = self.temperature_range_for_period(
            period,
            daylight_condition=daylight_condition,
        )
        return {
            "solar_period": period,
            "daylight_condition": daylight_condition,
            "target_temp_c": temp_range.midpoint_c,
            "temp_range_c": temp_range.as_tuple(),
            "day_temp_range_c": self.day_temp.as_tuple(),
            "day_cloudy_temp_range_c": self.day_cloudy_temp.as_tuple(),
            "night_temp_range_c": self.night_temp.as_tuple(),
            "night_early_temp_range_c": self.night_early_temp.as_tuple(),
            "night_late_temp_range_c": self.night_late_temp.as_tuple(),
        }

    def effective_temperature_target(
        self,
        *,
        target_ts: datetime,
        lat: float,
        lon: float,
        tz: str = "Asia/Seoul",
        early_night_hours: float = 6.0,
        daylight_condition: DaylightCondition | str | None = None,
    ) -> dict[str, object]:
        """Return the target and range for the solar period at ``target_ts``."""

        period = classify_solar_period(
            now=target_ts,
            lat=lat,
            lon=lon,
            tz=tz,
            early_night_hours=early_night_hours,
        )
        return self.effective_temperature_target_for_period(
            period,
            daylight_condition=daylight_condition,
        )

    def to_controller_stage_config(
        self,
        *,
        max_sun_light: float = 1200.0,
        target_ts: datetime | None = None,
        lat: float | None = None,
        lon: float | None = None,
        tz: str = "Asia/Seoul",
        early_night_hours: float = 6.0,
        daylight_condition: DaylightCondition | str | None = None,
    ) -> dict[str, object]:
        """Return the legacy controller shape used by ``policy.stage``."""

        if self.target_day_temp_c is None or self.target_night_temp_c is None:
            raise ValueError(
                "This growth stage does not define both day and night numeric targets: "
                f"{self.crop}/{self.stage_name}"
            )
        config: dict[str, object] = {
            "temp": {
                "day": self.target_day_temp_c,
                "night": self.target_night_temp_c,
            },
            "max_sun_light": float(max_sun_light),
            "crop": self.crop,
            "subj_cd": self.subj_cd,
            "stage_name": self.stage_name,
            "stage_order": self.stage_order,
            "dat": self.dat,
            "day_temp_range_c": self.day_temp.as_tuple(),
            "day_cloudy_temp_range_c": self.day_cloudy_temp.as_tuple(),
            "night_temp_range_c": self.night_temp.as_tuple(),
            "night_early_temp_range_c": self.night_early_temp.as_tuple(),
            "night_late_temp_range_c": self.night_late_temp.as_tuple(),
            "root_temp_range_c": self.root_temp.as_tuple(),
        }
        solar_args = (target_ts, lat, lon)
        if any(value is not None for value in solar_args):
            if not all(value is not None for value in solar_args):
                raise ValueError("target_ts, lat, and lon must be provided together.")
            effective = self.effective_temperature_target(
                target_ts=target_ts,
                lat=float(lat),
                lon=float(lon),
                tz=tz,
                early_night_hours=early_night_hours,
                daylight_condition=daylight_condition,
            )
            period_target = effective["target_temp_c"]
            if period_target is None:
                raise ValueError(
                    "This growth stage does not define a numeric target for "
                    f"{effective['solar_period']}: {self.crop}/{self.stage_name}"
                )
            temp_config = dict(config["temp"])
            if effective["solar_period"] == "day":
                temp_config["day"] = period_target
            else:
                temp_config["night"] = period_target
            config["temp"] = temp_config
            config["solar_period"] = effective["solar_period"]
            config["daylight_condition"] = effective["daylight_condition"]
            config["current_target_temp_c"] = period_target
            config["current_temp_range_c"] = effective["temp_range_c"]
        return config


def _optional_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(str(value).strip())


def _optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(str(value).strip())


def normalize_crop(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in SUBJ_CD_TO_CROP:
        return SUBJ_CD_TO_CROP[key]
    return key or None


def load_growth_stage_rules(
    path: Path | str = DEFAULT_GROWTH_STAGE_RULE_PATH,
) -> list[GrowthStageRule]:
    """Load DAT-based growth-stage rules from a CSV file."""

    rules: list[GrowthStageRule] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rules.append(
                GrowthStageRule(
                    crop=str(row["crop"]).strip().lower(),
                    subj_cd=str(row["subj_cd"]).strip(),
                    stage_order=int(row["stage_order"]),
                    stage_name=str(row["stage_name"]).strip(),
                    dat_start=int(row["dat_start"]),
                    dat_end=_optional_int(row.get("dat_end")),
                    day_temp=TemperatureRange(
                        _optional_float(row.get("day_temp_min_c")),
                        _optional_float(row.get("day_temp_max_c")),
                    ),
                    day_cloudy_temp=TemperatureRange(
                        _optional_float(row.get("day_cloudy_temp_min_c")),
                        _optional_float(row.get("day_cloudy_temp_max_c")),
                    ),
                    night_temp=TemperatureRange(
                        _optional_float(row.get("night_temp_min_c")),
                        _optional_float(row.get("night_temp_max_c")),
                    ),
                    night_early_temp=TemperatureRange(
                        _optional_float(row.get("night_early_temp_min_c")),
                        _optional_float(row.get("night_early_temp_max_c")),
                    ),
                    night_late_temp=TemperatureRange(
                        _optional_float(row.get("night_late_temp_min_c")),
                        _optional_float(row.get("night_late_temp_max_c")),
                    ),
                    root_temp=TemperatureRange(
                        _optional_float(row.get("root_temp_min_c")),
                        _optional_float(row.get("root_temp_max_c")),
                    ),
                    notes=str(row.get("notes") or "").strip(),
                )
            )
    return rules


def calculate_dat(target_ts: datetime | date, transplant_date: datetime | date) -> int:
    """Return one-based days after transplanting for the target timestamp."""

    target_day = target_ts.date() if isinstance(target_ts, datetime) else target_ts
    transplant_day = (
        transplant_date.date() if isinstance(transplant_date, datetime) else transplant_date
    )
    dat = (target_day - transplant_day).days + 1
    if dat < 1:
        raise ValueError(
            f"target_ts {target_ts!r} is before transplant_date {transplant_date!r}"
        )
    return dat


def find_growth_stage(
    *,
    target_ts: datetime | date,
    transplant_date: datetime | date,
    crop: str | None = None,
    subj_cd: str | None = None,
    rules: Iterable[GrowthStageRule] | None = None,
) -> GrowthStage:
    """Find the current growth stage for a crop and transplant date."""

    rule_list = list(rules) if rules is not None else load_growth_stage_rules()
    crop_key = normalize_crop(crop) or normalize_crop(subj_cd)
    subj_key = str(subj_cd).strip() if subj_cd is not None else None
    if crop_key is None and subj_key is None:
        raise ValueError("Either crop or subj_cd is required.")

    candidates = [
        rule
        for rule in rule_list
        if (crop_key is not None and rule.crop == crop_key)
        or (subj_key is not None and rule.subj_cd == subj_key)
    ]
    if not candidates:
        raise ValueError(f"No growth-stage rules found for crop={crop!r}, subj_cd={subj_cd!r}")

    dat = calculate_dat(target_ts, transplant_date)
    for rule in sorted(candidates, key=lambda item: item.dat_start):
        if rule.contains_dat(dat):
            return GrowthStage(
                crop=rule.crop,
                subj_cd=rule.subj_cd,
                stage_order=rule.stage_order,
                stage_name=rule.stage_name,
                dat=dat,
                dat_start=rule.dat_start,
                dat_end=rule.dat_end,
                day_temp=rule.day_temp,
                day_cloudy_temp=rule.day_cloudy_temp,
                night_temp=rule.night_temp,
                night_early_temp=rule.night_early_temp,
                night_late_temp=rule.night_late_temp,
                root_temp=rule.root_temp,
                notes=rule.notes,
            )

    last_rule = max(candidates, key=lambda item: item.dat_start)
    if last_rule.dat_end is not None:
        raise ValueError(
            f"No growth-stage rule covers DAT {dat} for crop={crop!r}, subj_cd={subj_cd!r}"
        )
    return GrowthStage(
        crop=last_rule.crop,
        subj_cd=last_rule.subj_cd,
        stage_order=last_rule.stage_order,
        stage_name=last_rule.stage_name,
        dat=dat,
        dat_start=last_rule.dat_start,
        dat_end=last_rule.dat_end,
        day_temp=last_rule.day_temp,
        day_cloudy_temp=last_rule.day_cloudy_temp,
        night_temp=last_rule.night_temp,
        night_early_temp=last_rule.night_early_temp,
        night_late_temp=last_rule.night_late_temp,
        root_temp=last_rule.root_temp,
        notes=last_rule.notes,
    )


__all__ = [
    "DEFAULT_GROWTH_STAGE_RULE_PATH",
    "SUBJ_CD_TO_CROP",
    "GrowthStage",
    "GrowthStageRule",
    "TemperatureRange",
    "calculate_dat",
    "find_growth_stage",
    "load_growth_stage_rules",
    "normalize_crop",
]
