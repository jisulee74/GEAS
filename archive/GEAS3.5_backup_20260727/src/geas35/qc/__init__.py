"""Sensor quality-control compatibility exports."""

from importlib import import_module

_outlier = import_module("geas35.preprocessing.4_outlier_handling")

DEFAULT_AGGREGATE_FLAG = _outlier.DEFAULT_AGGREGATE_FLAG
DEFAULT_DOMAIN_RANGE_PATH = _outlier.DEFAULT_DOMAIN_RANGE_PATH
DEFAULT_RULE_AGGREGATE_FLAG = _outlier.DEFAULT_RULE_AGGREGATE_FLAG
DEFAULT_RULE_FLAG_PREFIX = _outlier.DEFAULT_RULE_FLAG_PREFIX
DEFAULT_AI_OUTLIER_FLAG_COLUMN = _outlier.DEFAULT_AI_OUTLIER_FLAG_COLUMN
DEFAULT_AI_WARNING_FLAG_COLUMN = _outlier.DEFAULT_AI_WARNING_FLAG_COLUMN
DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN = _outlier.DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN
DEFAULT_IMPUTED_FLAG_COLUMN = _outlier.DEFAULT_IMPUTED_FLAG_COLUMN
DEFAULT_TCN_FLAG_COLUMN = _outlier.DEFAULT_TCN_FLAG_COLUMN
DEFAULT_TCN_PRED_VALUE_COLUMN = _outlier.DEFAULT_TCN_PRED_VALUE_COLUMN
DEFAULT_TCN_SCORE_COLUMN = _outlier.DEFAULT_TCN_SCORE_COLUMN
DEFAULT_TCN_WARNING_FLAG_COLUMN = _outlier.DEFAULT_TCN_WARNING_FLAG_COLUMN
DomainRangeRule = _outlier.DomainRangeRule
OBSERVATION_RULE_COLUMNS = _outlier.OBSERVATION_RULE_COLUMNS
TCNOutlierConfig = _outlier.TCNOutlierConfig
apply_default_domain_flags = _outlier.apply_default_domain_flags
apply_default_rule_based_outlier_flags = (
    _outlier.apply_default_rule_based_outlier_flags
)
apply_domain_range_flags = _outlier.apply_domain_range_flags
apply_outlier_flags = _outlier.apply_outlier_flags
apply_rule_based_outlier_flags = _outlier.apply_rule_based_outlier_flags
apply_smartfarm_korea_domain_flags = _outlier.apply_smartfarm_korea_domain_flags
apply_tcn_outlier_flags = _outlier.apply_tcn_outlier_flags
load_domain_range_rules = _outlier.load_domain_range_rules
load_smartfarm_korea_domain_rules = _outlier.load_smartfarm_korea_domain_rules

__all__ = [
    "DEFAULT_AGGREGATE_FLAG",
    "DEFAULT_AI_OUTLIER_FLAG_COLUMN",
    "DEFAULT_AI_WARNING_FLAG_COLUMN",
    "DEFAULT_DOMAIN_RANGE_PATH",
    "DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN",
    "DEFAULT_IMPUTED_FLAG_COLUMN",
    "DEFAULT_RULE_AGGREGATE_FLAG",
    "DEFAULT_RULE_FLAG_PREFIX",
    "DEFAULT_TCN_FLAG_COLUMN",
    "DEFAULT_TCN_PRED_VALUE_COLUMN",
    "DEFAULT_TCN_SCORE_COLUMN",
    "DEFAULT_TCN_WARNING_FLAG_COLUMN",
    "DomainRangeRule",
    "OBSERVATION_RULE_COLUMNS",
    "TCNOutlierConfig",
    "apply_default_domain_flags",
    "apply_default_rule_based_outlier_flags",
    "apply_domain_range_flags",
    "apply_outlier_flags",
    "apply_rule_based_outlier_flags",
    "apply_smartfarm_korea_domain_flags",
    "apply_tcn_outlier_flags",
    "load_domain_range_rules",
    "load_smartfarm_korea_domain_rules",
]
