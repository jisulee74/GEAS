from __future__ import annotations
import json
from pathlib import Path
import sys
import pytest
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from geas35.experiments.rl.step20_hpo import HpoDatabase, STEP20_VERSION
from geas35.experiments.rl.step20_runner import finalize_official_hpo, select_promotions

METRICS = {"mean_reward_per_scheduled_valid_step": -1.0, "mean_reward_per_executed_step": -1.0,
           "mean_episodic_return": -10.0, "mean_discounted_return": -9.0,
           "scheduled_valid_steps": 10, "executed_steps": 10,
           "early_termination_missing_steps": 0, "hard_constraint_violations": 0,
           "severe_ood_rate": 0.0, "fallback_rate": 0.0,
           "warning_ood_rate": 0.0, "action_instability": 0.1}
CONFIG = {"ppo": {"seed": 42}, "rollout_length": 8}

def test_promotions_are_feasible_first_and_objective_ranked():
    records = []
    for trial, objective, feasible in ((0, 100.0, False), (1, -2.0, True), (2, -1.0, True)):
        metrics = {**METRICS, "mean_reward_per_scheduled_valid_step": objective}
        records.append({"trial_number": trial, "rung": 0, "status": "complete",
                        "feasible": feasible, "metrics": metrics})
    assert select_promotions(records, rung=0, fraction=1 / 3) == [2]

def _root(tmp_path: Path) -> tuple[Path, HpoDatabase]:
    config = json.loads((PROJECT_ROOT / "experiments/rl_policy_training/configs/step20_hpo_v1.json").read_text())
    config_path = tmp_path / "experiments/rl_policy_training/configs/step20_hpo_v1.json"
    config_path.parent.mkdir(parents=True); config_path.write_text(json.dumps(config))
    output = tmp_path / "experiments/rl_policy_training/artifacts/step20"
    output.mkdir(parents=True)
    (output / "step20_integrity_manifest.json").write_text(json.dumps({
        "schema_version": STEP20_VERSION, "status": "pending", "artifacts": {"execution_status": {}}
    }))
    return tmp_path, HpoDatabase(output / "hpo_study.sqlite3", STEP20_VERSION)

def test_finalize_refuses_incomplete_official_campaign(tmp_path):
    root, _ = _root(tmp_path)
    with pytest.raises(RuntimeError, match="requires 30 completed"):
        finalize_official_hpo(project_root=root)

def test_finalize_requires_and_aggregates_multi_seed_results(tmp_path):
    root, db = _root(tmp_path)
    for crop in ("strawberry", "melon", "cucumber"):
        for trial in range(30):
            db.upsert(crop=crop, trial_number=trial, rung=0, status="complete",
                      config=CONFIG, seed=42 + trial, budget_steps=1_000_000,
                      metrics=METRICS, feasible=True)
        for seed_index, seed in enumerate((42, 43, 44)):
            metrics = {**METRICS, "mean_reward_per_scheduled_valid_step": -1.0 + seed_index * 0.1}
            db.upsert(crop=crop, trial_number=0, rung=10 + seed_index, status="complete",
                      config={"ppo": {"seed": seed}, "rollout_length": 8}, seed=seed,
                      budget_steps=2_000_000, metrics=metrics, feasible=True)
    manifest_path = finalize_official_hpo(project_root=root)
    manifest = json.loads(manifest_path.read_text())
    best = json.loads((manifest_path.parent / "best_configs.json").read_text())
    assert manifest["status"] == "passed"
    assert manifest["checks"]["official_hpo_complete"] is True
    assert manifest["checks"]["test_accessed"] is False
    for crop in ("strawberry", "melon", "cucumber"):
        assert best["crops"][crop]["replication_seeds"] == [42, 43, 44]
        assert best["crops"][crop]["metrics"]["mean_reward_per_scheduled_valid_step"] == pytest.approx(-0.9)
