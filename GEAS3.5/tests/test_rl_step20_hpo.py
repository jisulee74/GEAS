from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from geas35.experiments.rl.step20_hpo import (DeterministicSearchSpace, EpisodeEvaluation, HpoDatabase, STEP20_VERSION, aggregate_validation_metrics, apply_safety_gate, normalized_objective_audit, trial_rank_key)

def _episode(*, scheduled=10, executed=10, reward=-10.0, hard=0, severe=0, fallback=0, warning=0, instability=0.1):
    return EpisodeEvaluation("e", scheduled, executed, reward, reward, hard, severe, fallback, warning, instability)

def test_length_normalized_objective_uses_scheduled_not_executed_steps():
    complete = aggregate_validation_metrics([_episode()])
    early = aggregate_validation_metrics([_episode(executed=2)])
    assert complete["mean_reward_per_scheduled_valid_step"] == early["mean_reward_per_scheduled_valid_step"] == -1.0
    assert early["mean_reward_per_executed_step"] == -5.0
    assert normalized_objective_audit()["same_primary_for_same_reward_and_schedule"] is True

def test_safety_is_gate_before_reward_ranking():
    unsafe_metrics = aggregate_validation_metrics([_episode(hard=1, reward=100.0)])
    feasible, reasons = apply_safety_gate(unsafe_metrics, severe_ood_rate_max=0.1, fallback_rate_max=0.1)
    assert feasible is False and reasons == ["hard_constraint_violation"]
    unsafe = {"trial_number": 0, "feasible": False, "metrics": unsafe_metrics}
    safe_metrics = aggregate_validation_metrics([_episode(reward=-100.0)])
    safe = {"trial_number": 1, "feasible": True, "metrics": safe_metrics}
    assert trial_rank_key(safe) > trial_rank_key(unsafe)

def test_sampler_and_sqlite_are_deterministic_and_resumable(tmp_path):
    config = json.loads((PROJECT_ROOT / "experiments/rl_policy_training/configs/step20_hpo_v1.json").read_text())
    sampler = DeterministicSearchSpace(config["search_space"], seed=42)
    first = sampler.sample_trial(3, 59)
    assert first == sampler.sample_trial(3, 59)
    assert first["rollout_length"] in config["search_space"]["rollout_length"]
    db = HpoDatabase(tmp_path / "study.sqlite3", "study")
    metrics = aggregate_validation_metrics([_episode()])
    db.upsert(crop="melon", trial_number=3, rung=0, status="complete", config=first, seed=45, budget_steps=100, metrics=metrics, feasible=True)
    assert HpoDatabase(tmp_path / "study.sqlite3", "study").records("melon")[0]["config"] == first

def test_step20_protocol_artifacts_are_fail_closed_and_hash_chained():
    root = PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step20"
    manifest = json.loads((root / "step20_integrity_manifest.json").read_text())
    status = json.loads((root / "hpo_execution_status.json").read_text())
    gates = json.loads((root / "train_derived_safety_gates.json").read_text())
    assert manifest["schema_version"] == STEP20_VERSION
    assert manifest["status"] == "protocol_ready_official_trials_pending"
    assert manifest["checks"]["test_accessed"] is False and manifest["checks"]["step21_started"] is False
    assert manifest["checks"]["official_hpo_complete"] is False
    assert status["completed_trials"] == {"strawberry": 0, "melon": 0, "cucumber": 0}
    assert status["smoke_results_eligible_for_selection"] is False
    assert gates["fit_split"] == "train_only"
    assert all(not x["thresholds_hpo_tunable"] for x in gates["crops"].values())
    for record in [manifest["config"], manifest["step15"], manifest["step19"],
                   *manifest["implementations"].values(), *manifest["artifacts"].values()]:
        path = PROJECT_ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
