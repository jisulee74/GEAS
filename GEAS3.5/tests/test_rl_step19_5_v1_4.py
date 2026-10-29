from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geas35.experiments.rl.step19_5_calibration_v1_4 import (
    STEP19_5_VERSION,
    analyze_crop_learning_curves,
)
from geas35.experiments.rl.step19_5_runner_v1_4 import _job_payloads
from geas35.experiments.rl.step20_runner_v1_4 import _artifact_hashes


ARTIFACT_ROOT = PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step19_5_v1_4"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v1_4_preflight_hash_chain_and_test_lock() -> None:
    readiness = _load(ARTIFACT_ROOT / "step19_5_readiness.json")
    reps = _load(ARTIFACT_ROOT / "representative_configurations.json")
    gates = _load(ARTIFACT_ROOT / "train_derived_safety_gates.json")
    assert readiness["schema_version"] == STEP19_5_VERSION
    assert readiness["status"] == "ready_to_run"
    assert readiness["actual_calibration_started"] is False
    assert readiness["test_accessed"] is False
    assert len(reps["configurations"]) == 5
    assert reps["selection"] == "default_plus_deterministic_maximin"
    assert reps["outcome_blind"] is True
    assert gates["fit_split"] == "train_only"
    for crop, gate in gates["crops"].items():
        assert crop in {"strawberry", "melon", "cucumber"}
        assert gate["hard_constraint_violations_max"] == 0
        assert 0.0 <= gate["severe_ood_rate_max"] <= 1.0
        assert 0.0 <= gate["fallback_rate_max"] <= 1.0
        assert gate["thresholds_hpo_tunable"] is False
    for name, expected in readiness["dependencies"].items():
        if name == "step19_integrity_sha256":
            path = PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step19_v1_4/step19_integrity_manifest.json"
        elif name == "step17_coverage_sha256":
            path = PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step17_v1_4/support_coverage_report.json"
        elif name == "pilot_diagnostics_sha256":
            path = PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step19_v1_4/pilot_learning_diagnostics.json"
        elif name == "config_sha256":
            path = PROJECT_ROOT / "experiments/rl_policy_training/configs/step19_5_calibration_v2_1_v1_4.json"
        elif name == "search_space_sha256":
            path = PROJECT_ROOT / "experiments/rl_policy_training/configs/step20_hpo_v1.json"
        elif name == "safety_gates_sha256":
            path = ARTIFACT_ROOT / "train_derived_safety_gates.json"
        else:
            raise AssertionError(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_job_grid_has_five_configs_times_three_seeds_and_55_observations() -> None:
    reps = _load(ARTIFACT_ROOT / "representative_configurations.json")["configurations"]
    payloads = _job_payloads(reps, [42, 43, 44], 55, 4096)
    assert len(payloads) == 15
    identities = {(row["config_id"], row["seed"]) for row in payloads.values()}
    assert len(identities) == 15
    assert all(row["config"]["ppo"]["observation_dim"] == 55 for row in payloads.values())
    assert all(row["budget_steps"] == 4096 for row in payloads.values())


def test_checkpoint_dependency_contract_uses_v1_4_chain() -> None:
    for crop in ("strawberry", "melon", "cucumber"):
        hashes = _artifact_hashes(PROJECT_ROOT, crop)
        assert set(hashes) == {
            "observation_scaler", "environment", "transition_model", "support",
            "step15_protocol", "step17_integrity",
        }
        assert all(len(value) == 64 for value in hashes.values())


def test_learning_curve_analysis_rejects_test_rows() -> None:
    config = _load(PROJECT_ROOT / "experiments/rl_policy_training/configs/step19_5_calibration_v2_1_v1_4.json")
    row = {
        "crop": "strawberry", "config_id": "representative_0", "seed": 42,
        "steps": 4096, "split": "test", "test_accessed": True,
        "safety_gate_passed": True, "mean_reward_per_scheduled_valid_step": 0.0,
    }
    import pytest
    with pytest.raises(ValueError, match="Validation-only"):
        analyze_crop_learning_curves([row], config)
