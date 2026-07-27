"""Compatibility wrapper for domain-range outlier flags.

The implementation moved to ``geas35.preprocessing.4_outlier_handling`` so the
offline preprocessing pipeline can keep numbered stages together.
"""

from __future__ import annotations

from importlib import import_module

_outliers = import_module("geas35.preprocessing.4_outlier_handling")

DEFAULT_DOMAIN_RANGE_PATH = _outliers.DEFAULT_DOMAIN_RANGE_PATH
DomainRangeRule = _outliers.DomainRangeRule
OBSERVATION_RULE_COLUMNS = _outliers.OBSERVATION_RULE_COLUMNS
apply_default_domain_flags = _outliers.apply_default_domain_flags
apply_domain_range_flags = _outliers.apply_domain_range_flags
apply_smartfarm_korea_domain_flags = _outliers.apply_smartfarm_korea_domain_flags
load_domain_range_rules = _outliers.load_domain_range_rules
load_smartfarm_korea_domain_rules = _outliers.load_smartfarm_korea_domain_rules

__all__ = [
    "DEFAULT_DOMAIN_RANGE_PATH",
    "DomainRangeRule",
    "OBSERVATION_RULE_COLUMNS",
    "apply_default_domain_flags",
    "apply_domain_range_flags",
    "apply_smartfarm_korea_domain_flags",
    "load_domain_range_rules",
    "load_smartfarm_korea_domain_rules",
]
