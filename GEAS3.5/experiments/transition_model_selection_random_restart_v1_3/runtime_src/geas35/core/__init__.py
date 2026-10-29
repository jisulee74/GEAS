"""Shared constants, schema definitions, and domain rules."""

from geas35.core.growth_stage import (
    DEFAULT_GROWTH_STAGE_RULE_PATH,
    SUBJ_CD_TO_CROP,
    GrowthStage,
    GrowthStageRule,
    TemperatureRange,
    calculate_dat,
    find_growth_stage,
    load_growth_stage_rules,
    normalize_crop,
)
from geas35.core.solar_time import (
    DaylightCondition,
    SolarPeriod,
    classify_daylight_condition,
    classify_solar_period,
    classify_solar_period_from_bounds,
    classify_sunshine_ratio,
    get_sun_times,
    possible_sunshine_minutes,
)

__all__ = [
    "DEFAULT_GROWTH_STAGE_RULE_PATH",
    "SUBJ_CD_TO_CROP",
    "GrowthStage",
    "GrowthStageRule",
    "DaylightCondition",
    "SolarPeriod",
    "TemperatureRange",
    "calculate_dat",
    "classify_daylight_condition",
    "classify_solar_period",
    "classify_solar_period_from_bounds",
    "classify_sunshine_ratio",
    "find_growth_stage",
    "get_sun_times",
    "load_growth_stage_rules",
    "normalize_crop",
    "possible_sunshine_minutes",
]
