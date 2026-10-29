from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geas35.experiments.rl.step19_v1_4 import STEP19_VERSION


ARTIFACT_ROOT = PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step19_v1_4"
CROPS = ("strawberry", "melon", "cucumber")


def _load(name: str) -> dict:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def test_step19_v1_4_integrity_hash_chain_and_scope() -> None:
    integrity = _load("step19_integrity_manifest.json")
    assert integrity["schema_version"] == STEP19_VERSION
    assert integrity["status"] == "passed"
    assert integrity["checks"]["test_accessed"] is False
    assert integrity["checks"]["hpo_started"] is False
    assert integrity["checks"]["step19_5_started"] is False
    assert integrity["checks"]["step20_implemented"] is False
    records = [
        integrity["step18_integrity"], integrity["pilot_config"],
        integrity["implementation"], *integrity["artifacts"].values(),
    ]
    for record in records:
        path = PROJECT_ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_all_crop_trainability_gates_and_resume_pass() -> None:
    diagnostics = _load("pilot_learning_diagnostics.json")
    decision = _load("trainability_decision.json")
    assert decision["overall_decision"] == "continue_to_step19_5_calibration"
    assert decision["scope"] == "trainability_only_no_baseline_performance_judgment"
    assert decision["test_accessed"] is False
    assert decision["hpo_started"] is False
    for crop in CROPS:
        record = diagnostics["crops"][crop]
        crop_decision = decision["crops"][crop]
        assert crop_decision["trainable"] is True
        assert crop_decision["failed_gates"] == []
        assert crop_decision["baseline_comparison_performed"] is False
        assert crop_decision["reward_performance_gate_applied"] is False
        assert record["training_steps"] == 24
        assert record["updates"] == 2
        assert record["invalid_action_count"] == 0
        assert record["checkpoint_resume_passed"] is True
        assert record["seed_reproducible"] is True
        assert all(record["gates"].values())
        assert record["signal"]["reward_std"] > 0.0
        assert record["signal"]["raw_advantage_std"] > 0.0
        assert record["state_response"]["distribution_parameter_max_range"] > 0.0
        assert record["state_response"]["deterministic_action_max_range"] > 0.0
        assert record["state_response"]["value_range"] > 0.0
        for update in record["update_metrics"]:
            numeric = [value for value in update.values() if not isinstance(value, bool)]
            assert np.isfinite(np.asarray(numeric, dtype=float)).all()
            assert update["gradient_norm"] > 0.0
            assert update["parameter_update_l2"] > 0.0
        checkpoint = PROJECT_ROOT / record["checkpoint"]["path"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record["checkpoint"]["sha256"]


def test_support_layer_is_active_without_blocking_pilot() -> None:
    diagnostics = _load("pilot_learning_diagnostics.json")
    curves = _load("ood_intervention_curves.json")
    assert curves["status"] == "passed"
    for crop in CROPS:
        ood = diagnostics["crops"][crop]["ood_summary"]
        assert ood["support_restriction_rate"] >= 0.01
        assert ood["projection_rate"] < 1.0
        assert ood["fallback_rate"] < 1.0
        assert ood["severe_ood_rate"] <= 0.5
        curve = curves["crops"][crop]
        assert curve["steps"] == [16, 24]
        assert all(np.isfinite(curve[key]).all() for key in (
            "reward_mean", "raw_advantage_std", "approx_kl", "entropy", "gradient_norm"
        ))
