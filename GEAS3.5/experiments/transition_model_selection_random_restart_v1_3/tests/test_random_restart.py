from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geas35.experiments.transition.random_restart import (
    _paired_bootstrap,
    aggregate_restart_results,
    finite_preflight,
    load_random_restart_config,
)


def test_config_contract(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("x: 1\n", encoding="utf-8")
    config = tmp_path / "restart.yaml"
    config.write_text(
        "base_config: base.yaml\noutput_root: out\nseeds: [42, 43, 44]\n"
        "restart_models: [mlp]\nselection_rule: median_validation_score\n",
        encoding="utf-8",
    )
    loaded = load_random_restart_config(config)
    assert loaded.seeds == (42, 43, 44)
    assert loaded.output_root == tmp_path / "out"


def test_finite_preflight_reports_column_and_position() -> None:
    clean = pd.DataFrame({"a": [1.0, 2.0]})
    bad = pd.DataFrame({"b": [1.0, np.inf]})
    report = finite_preflight(clean, clean, bad, clean)
    assert not report["passed"]
    assert report["problems"] == [{
        "split_component": "validation_x", "column": "b",
        "count": 1, "row_positions": [1],
    }]


def test_median_seed_selection_and_statistics() -> None:
    rows = [
        {"seed": 42, "validation_rollout_weighted_score": 0.1, "physical_violation_rate": 0.0, "cvar90": 2.0, "deployment_eligible": True},
        {"seed": 43, "validation_rollout_weighted_score": 0.3, "physical_violation_rate": 0.0, "cvar90": 2.0, "deployment_eligible": True},
        {"seed": 44, "validation_rollout_weighted_score": 0.2, "physical_violation_rate": 0.0, "cvar90": 2.0, "deployment_eligible": True},
    ]
    report = aggregate_restart_results(rows)
    assert report["selected"]["seed"] == 44
    assert report["score_statistics"]["mean"] == pytest.approx(0.2)


def test_paired_bootstrap_effect_direction() -> None:
    improved = _paired_bootstrap([-0.2] * 20, seed=42, replicates=100)
    degraded = _paired_bootstrap([0.2] * 20, seed=42, replicates=100)
    tied = _paired_bootstrap([-0.2, 0.2] * 20, seed=42, replicates=500)
    assert improved["effect"] == "improved"
    assert degraded["effect"] == "degraded"
    assert tied["effect"] == "no_significant_difference"
