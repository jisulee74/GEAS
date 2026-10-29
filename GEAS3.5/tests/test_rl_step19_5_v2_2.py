from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geas35.experiments.rl.step19_5_calibration_v2_2 import (
    DIAGNOSTICS, analyze_v2_2, kendall_tau_b, practical_equivalence_groups,
)
from geas35.experiments.rl.step19_5_runner_v2_2 import prepare_v2_2


def _config() -> dict:
    return json.loads((PROJECT_ROOT / "experiments/rl_policy_training/configs/step19_5_calibration_v2_2.json").read_text())


def _rows(*, last_delta: float = 0.0, unsafe: set[str] | None = None) -> list[dict]:
    unsafe = unsafe or set()
    output = []
    names = ("a", "b", "c")
    steps = (262144, 524288, 1048576)
    base = {"a": -0.3, "b": -0.3002, "c": -0.5}
    for step_index, step in enumerate(steps):
        for name in names:
            for seed in (42, 43, 44):
                reward = base[name] + (seed - 43) * 0.00005
                if step_index:
                    reward += last_delta * step_index
                row = {"config_id": name, "seed": seed, "steps": step,
                       "split": "validation", "test_accessed": False,
                       "safety_gate_passed": name not in unsafe,
                       "mean_reward_per_scheduled_valid_step": reward}
                row.update({metric: 0.1 for metric in DIAGNOSTICS})
                output.append(row)
    return output


def test_practical_ties_and_tau_b_are_tie_aware() -> None:
    rewards = {
        "a": {42: 1.0, 43: 1.0001, 44: 0.9999},
        "b": {42: 1.0002, 43: 1.0, 44: 1.0001},
        "c": {42: 0.8, 43: 0.8, 44: 0.8},
    }
    groups, comparisons = practical_equivalence_groups(rewards, _config()["practical_equivalence"])
    assert set(groups[0]) == {"a", "b"}
    assert groups[1] == ["c"]
    assert any(item["equivalent"] for item in comparisons)
    assert kendall_tau_b(groups, groups) == pytest.approx(1.0)


def test_v2_2_passes_two_stable_intervals_and_emits_three_rungs() -> None:
    result = analyze_v2_2(_rows(), ["a", "b", "c"], _config())
    assert result["status"] == "calibrated"
    assert result["rung_budgets"] == [262144, 524288, 1048576]
    assert result["prospective_extension_confirmed"] is True
    assert result["test_accessed"] is False


def test_v2_2_blocks_missing_safe_configurations_and_large_delta() -> None:
    no_safe = analyze_v2_2(_rows(), [], _config())
    assert no_safe["status"] == "no_safe_configuration"
    one_safe = analyze_v2_2(_rows(), ["a"], _config())
    assert one_safe["status"] == "insufficient_eligible_configurations"
    unstable = analyze_v2_2(_rows(last_delta=0.01), ["a", "b", "c"], _config())
    assert unstable["status"] == "calibration_inconclusive_v2_2"
    assert unstable["rung_budgets"] == []


def test_v2_2_rejects_test_rows() -> None:
    rows = _rows()
    rows[0]["split"] = "test"; rows[0]["test_accessed"] = True
    with pytest.raises(ValueError, match="Validation-only"):
        analyze_v2_2(rows, ["a", "b", "c"], _config())


def test_prepare_freezes_existing_source_without_copying_checkpoints() -> None:
    readiness = prepare_v2_2(PROJECT_ROOT)
    payload = json.loads(readiness.read_text())
    assert payload["status"] == "ready_for_safety_audit"
    assert payload["test_accessed"] is False
    output = PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step19_5_v2_2"
    freeze = json.loads((output / "source_v1_4_freeze_manifest.json").read_text())
    assert freeze["source_mutation_allowed"] is False
    assert freeze["status"] == "inconclusive_under_strict_stability_contract"
    assert not list(output.rglob("*.pt"))
