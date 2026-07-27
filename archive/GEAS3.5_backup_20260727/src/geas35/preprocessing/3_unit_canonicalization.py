"""Backward-compatible wrapper for unit canonicalization.

The implementation moved to ``2_unit_canonicalization.py`` when the offline
preprocessing pipeline changed to:

``input_schema_prepared -> unit_canonicalized -> missing_outliers_handled``.
"""

from __future__ import annotations

from importlib import import_module

_unit = import_module("geas35.preprocessing.2_unit_canonicalization")

BINARY_ACTION_COLUMNS = _unit.BINARY_ACTION_COLUMNS
DERIVED_RENAME_MAP = _unit.DERIVED_RENAME_MAP
HUMIDITY_PERCENT_COLUMNS = _unit.HUMIDITY_PERCENT_COLUMNS
NON_NEGATIVE_COLUMNS = _unit.NON_NEGATIVE_COLUMNS
PERCENT_ACTION_COLUMNS = _unit.PERCENT_ACTION_COLUMNS
canonicalize_feature_units = _unit.canonicalize_feature_units
canonicalize_for_derived = _unit.canonicalize_for_derived
normalize_feature_ranges = _unit.normalize_feature_ranges
normalize_for_derived = _unit.normalize_for_derived

__all__ = [
    "BINARY_ACTION_COLUMNS",
    "DERIVED_RENAME_MAP",
    "HUMIDITY_PERCENT_COLUMNS",
    "NON_NEGATIVE_COLUMNS",
    "PERCENT_ACTION_COLUMNS",
    "canonicalize_feature_units",
    "canonicalize_for_derived",
    "normalize_feature_ranges",
    "normalize_for_derived",
]
