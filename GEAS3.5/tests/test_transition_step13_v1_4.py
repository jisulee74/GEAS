from __future__ import annotations

import math

import pytest

from geas35.experiments.transition.reproducibility_review_v1_4 import (
    Step13V14ReviewError,
    _assert_test_locked,
    _compare_numeric_tree,
)


def test_numeric_tree_comparison_accepts_equal_and_nan_values() -> None:
    _compare_numeric_tree({"a": 1.0, "b": math.nan}, {"a": 1.0, "b": math.nan}, "metric", 1e-9, 1e-11)


def test_numeric_tree_comparison_rejects_metric_change() -> None:
    with pytest.raises(Step13V14ReviewError, match="metric mismatch"):
        _compare_numeric_tree({"a": 1.0}, {"a": 1.1}, "metric", 1e-9, 1e-11)


def test_test_lock_rejects_any_test_access_marker() -> None:
    _assert_test_locked({"test_opened": False, "nested": {"test_used_for_hpo": False}}, __file__)
    with pytest.raises(Step13V14ReviewError, match="Test-lock violation"):
        _assert_test_locked({"test_opened": True}, __file__)
