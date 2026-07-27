"""Sensor quality-control compatibility exports."""

from importlib import import_module

_quality_stage = import_module("geas35.preprocessing.3_missing_outliers_handling")

DEFAULT_DOMAIN_RANGE_PATH = _quality_stage.DEFAULT_DOMAIN_RANGE_PATH
DEFAULT_RULE_AGGREGATE_FLAG = _quality_stage.DEFAULT_RULE_AGGREGATE_FLAG
DomainRangeRule = _quality_stage.DomainRangeRule
OBSERVATION_RULE_COLUMNS = _quality_stage.OBSERVATION_RULE_COLUMNS
apply_default_domain_flags = _quality_stage.apply_default_domain_flags
apply_default_rule_based_outlier_flags = (
    _quality_stage.apply_default_rule_based_outlier_flags
)
apply_domain_range_flags = _quality_stage.apply_domain_range_flags
apply_rule_based_outlier_flags = _quality_stage.apply_rule_based_outlier_flags
apply_smartfarm_korea_domain_flags = _quality_stage.apply_smartfarm_korea_domain_flags
load_domain_range_rules = _quality_stage.load_domain_range_rules
load_smartfarm_korea_domain_rules = _quality_stage.load_smartfarm_korea_domain_rules

__all__ = [
    "DEFAULT_DOMAIN_RANGE_PATH",
    "DEFAULT_RULE_AGGREGATE_FLAG",
    "DomainRangeRule",
    "OBSERVATION_RULE_COLUMNS",
    "apply_default_domain_flags",
    "apply_default_rule_based_outlier_flags",
    "apply_domain_range_flags",
    "apply_rule_based_outlier_flags",
    "apply_smartfarm_korea_domain_flags",
    "load_domain_range_rules",
    "load_smartfarm_korea_domain_rules",
]
