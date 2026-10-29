from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.rl.step17_v1_4 import STEP17_VERSION


ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "experiments/rl_policy_training/artifacts/step17_v1_4"
)
CROPS = ("strawberry", "melon", "cucumber")


def _load(name: str) -> dict:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def test_step17_v1_4_integrity_and_hash_chain() -> None:
    integrity = _load("step17_integrity_manifest.json")
    assert integrity["schema_version"] == STEP17_VERSION
    assert integrity["status"] == "passed"
    assert integrity["checks"]["test_parquet_opened"] is False
    assert integrity["checks"]["step18_implemented"] is False
    for record in [*integrity["reports"].values(), *integrity["implementations"].values()]:
        path = PROJECT_ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_step17_v1_4_rollout_reward_and_ood_contracts() -> None:
    rollout = _load("rollout_parity_report.json")
    validation = _load("environment_validation_report.json")
    matrix = _load("constraint_ood_test_matrix.json")
    assert rollout["status"] == validation["status"] == matrix["status"] == "passed"
    for crop in CROPS:
        parity = rollout["crops"][crop]
        assert parity["source_split"] == "validation"
        assert parity["horizon_steps"] == [3, 6, 12]
        assert parity["max_absolute_metric_difference"] <= 1e-10

        reward = validation["crops"][crop]["reward_parity"]
        assert reward["status"] == "passed"
        assert len(reward["penalty_terms"]) == 6
        assert reward["reward"] == reward["base_reward"]

        termination = validation["crops"][crop]["termination_truncation"]
        assert termination["data_boundary"]["termination_reason"] == "episode_boundary"
        assert termination["time_limit"]["truncation_reason"] == "training_time_limit"

        ood = matrix["crops"][crop]["ood_response_sequence"]
        assert ood["warning_action"]["projection"]["changes"]
        assert ood["warning_action"]["penalty"] > 0.0
        assert ood["severe_action"]["fallback"] is True
        assert ood["severe_state"]["remaining_scheduled_steps_charged"] > 0


def test_step17_v1_4_support_and_split_isolation() -> None:
    coverage = _load("support_coverage_report.json")
    validation = _load("environment_validation_report.json")
    assert coverage["fit_split"] == "train_only"
    for crop in CROPS:
        record = coverage["crops"][crop]
        assert record["overall"]["state_severe_coverage"] >= 0.95
        assert record["overall"]["joint_severe_coverage"] >= 0.95
        assert record["subgroup_diagnostic"]["hard_gate"] is False

        registry = validation["crops"][crop]["split_registry"]
        assert registry["pairwise_disjoint"] is True
        assert registry["test_parquet_opened"] is False
        assert set(registry["episode_counts"]) == {"train", "validation", "test"}


def test_step17_did_not_open_rl_test() -> None:
    assert not (
        PROJECT_ROOT
        / "experiments/rl_policy_training/artifacts/step15_v1_4/rl_test_access_log.json"
    ).exists()
