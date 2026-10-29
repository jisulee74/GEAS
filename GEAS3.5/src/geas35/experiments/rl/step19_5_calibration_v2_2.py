"""Prospective Step 19.5 v2.2 stability contract.

The v1.4 calibration remains immutable.  This module consumes its two final
checkpoints plus one prospectively extended checkpoint and never reads Test.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

STEP19_5_V2_2_VERSION = "geas35.rl.step19_5.v2_2"
STEP19_5_V2_2_BUDGET_SCHEMA = "geas35.rl.step19_5.budget.v2_2"
OFFICIAL_CROPS = ("strawberry", "melon", "cucumber")
DIAGNOSTICS = (
    "policy_loss", "value_loss", "approx_kl", "entropy", "explained_variance",
    "gradient_norm", "in_support_rate", "warning_ood_rate", "severe_ood_rate",
    "projection_rate", "fallback_rate", "termination_rate",
)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(target)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bootstrap_median_difference_ci(
    left: Sequence[float], right: Sequence[float], *, samples: int,
    confidence: float, seed: int,
) -> tuple[float, float]:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not len(a):
        raise ValueError("Paired bootstrap inputs must be non-empty and aligned")
    rng = np.random.default_rng(seed)
    deltas = a - b
    values = [float(np.median(deltas[rng.integers(0, len(deltas), len(deltas))]))
              for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def practical_equivalence_groups(
    seed_rewards: Mapping[str, Mapping[int, float]], contract: Mapping[str, Any],
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Return ordered connected tie groups and auditable pairwise decisions."""
    names = sorted(seed_rewards)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    comparisons: list[dict[str, Any]] = []
    samples = int(contract["bootstrap_samples"])
    confidence = float(contract["bootstrap_confidence"])
    epsilon = float(contract["absolute_median_reward_difference_max"])
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            seeds = sorted(set(seed_rewards[left]) & set(seed_rewards[right]))
            a = [seed_rewards[left][seed] for seed in seeds]
            b = [seed_rewards[right][seed] for seed in seeds]
            difference = float(np.median(np.asarray(a) - np.asarray(b)))
            ci = _bootstrap_median_difference_ci(
                a, b, samples=samples, confidence=confidence,
                seed=42 + index * len(names) + names.index(right),
            )
            equivalent = abs(difference) <= epsilon and ci[0] <= 0.0 <= ci[1]
            if equivalent:
                union(left, right)
            comparisons.append({
                "left": left, "right": right, "paired_seeds": seeds,
                "median_difference": difference, "bootstrap_ci": list(ci),
                "equivalent": equivalent,
            })
    components: dict[str, list[str]] = {}
    for name in names:
        components.setdefault(find(name), []).append(name)
    medians = {name: float(np.median(list(seed_rewards[name].values()))) for name in names}
    groups = sorted(components.values(), key=lambda group: (
        -max(medians[name] for name in group), tuple(sorted(group))))
    return [sorted(group) for group in groups], comparisons


def kendall_tau_b(left_groups: Sequence[Sequence[str]], right_groups: Sequence[Sequence[str]]) -> float:
    left_rank = {name: rank for rank, group in enumerate(left_groups) for name in group}
    right_rank = {name: rank for rank, group in enumerate(right_groups) for name in group}
    names = sorted(set(left_rank) & set(right_rank))
    concordant = discordant = left_ties = right_ties = 0
    for index, a in enumerate(names):
        for b in names[index + 1:]:
            dl = np.sign(left_rank[a] - left_rank[b])
            dr = np.sign(right_rank[a] - right_rank[b])
            if dl == 0 and dr == 0:
                continue
            if dl == 0:
                left_ties += 1
            elif dr == 0:
                right_ties += 1
            elif dl == dr:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt((concordant + discordant + left_ties) *
                          (concordant + discordant + right_ties))
    return 1.0 if denominator == 0.0 else float((concordant - discordant) / denominator)


def _bootstrap_slope_ci(
    rewards: Mapping[int, Mapping[str, Mapping[int, float]]], steps: Sequence[int],
    *, samples: int, confidence: float,
) -> tuple[float, float]:
    rng = np.random.default_rng(20220911)
    x = np.arange(len(steps), dtype=float)
    slopes = []
    for _ in range(samples):
        values = []
        for step in steps:
            resampled = []
            for seed_values in rewards[step].values():
                candidates = list(seed_values.values())
                resampled.append(float(rng.choice(candidates)))
            values.append(float(np.median(resampled)))
        slopes.append(float(np.polyfit(x, np.asarray(values), 1)[0]))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(slopes, alpha)), float(np.quantile(slopes, 1.0 - alpha))


def analyze_v2_2(
    rows: Sequence[Mapping[str, Any]], eligible_config_ids: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if any(row.get("split") != "validation" or row.get("test_accessed") is not False
           for row in rows):
        raise ValueError("Step 19.5 v2.2 accepts Validation-only rows with Test locked")
    eligible = sorted(set(map(str, eligible_config_ids)))
    minimum = int(config["configuration_eligibility"]["minimum_eligible_configurations"])
    if not eligible:
        return {"status": "no_safe_configuration", "rung_budgets": [],
                "eligible_configurations": [], "test_accessed": False}
    if len(eligible) < minimum:
        return {"status": "insufficient_eligible_configurations", "rung_budgets": [],
                "eligible_configurations": eligible, "test_accessed": False}
    required_steps = [*map(int, config["source_steps"]), int(config["extension_budget_steps"])]
    seeds = sorted(map(int, config["seeds"]))
    lookup = {(int(row["steps"]), str(row["config_id"]), int(row["seed"])): row for row in rows}
    expected = {(step, name, seed) for step in required_steps for name in eligible for seed in seeds}
    missing = sorted(expected - set(lookup))
    if missing:
        raise ValueError(f"Incomplete v2.2 step/config/seed grid: {missing[:5]}")
    rewards = {step: {name: {seed: float(lookup[(step, name, seed)]["mean_reward_per_scheduled_valid_step"])
                                  for seed in seeds} for name in eligible} for step in required_steps}
    finite = all(np.isfinite(float(lookup[key][metric])) for key in expected for metric in DIAGNOSTICS)
    groups: dict[int, list[list[str]]] = {}
    pairwise: dict[int, list[dict[str, Any]]] = {}
    for step in required_steps:
        groups[step], pairwise[step] = practical_equivalence_groups(
            rewards[step], config["practical_equivalence"])
    ranking = []
    for previous, current in zip(required_steps, required_steps[1:]):
        tau = kendall_tau_b(groups[previous], groups[current])
        top_stable = set(groups[previous][0]) == set(groups[current][0])
        stable = tau >= float(config["ranking_stability"]["kendall_tau_b_min"]) and top_stable
        ranking.append({"previous_steps": previous, "steps": current,
                        "kendall_tau_b": tau, "previous_top_group": groups[previous][0],
                        "top_group": groups[current][0], "top_group_stable": top_stable,
                        "stable": stable})
    learning = []
    lc = config["learning_stability"]
    for previous, current in zip(required_steps, required_steps[1:]):
        deltas = [abs(rewards[current][name][seed] - rewards[previous][name][seed])
                  for name in eligible for seed in seeds]
        median, q90 = float(np.median(deltas)), float(np.quantile(deltas, 0.9))
        stable = median <= float(lc["absolute_reward_delta_median_max"]) and q90 <= float(lc["absolute_reward_delta_q90_max"])
        learning.append({"previous_steps": previous, "steps": current,
                         "absolute_delta_median": median, "absolute_delta_q90": q90,
                         "stable": stable})
    slope_ci = _bootstrap_slope_ci(
        rewards, required_steps, samples=int(lc["bootstrap_samples"]),
        confidence=float(lc["bootstrap_confidence"]),
    )
    slope_stable = slope_ci[0] <= 0.0 <= slope_ci[1]
    consecutive = int(config["ranking_stability"]["consecutive_intervals"])
    rank_stable = len(ranking) >= consecutive and all(item["stable"] for item in ranking[-consecutive:])
    learning_stable = (len(learning) >= int(lc["consecutive_intervals"])
                       and all(item["stable"] for item in learning[-int(lc["consecutive_intervals"]):])
                       and slope_stable and finite)
    passed = rank_stable and learning_stable
    rungs = required_steps if passed else []
    return {
        "status": "calibrated" if passed else "calibration_inconclusive_v2_2",
        "rung_budgets": rungs, "eligible_configurations": eligible,
        "steps": required_steps, "reward_medians": {
            str(step): {name: float(np.median(list(values.values())))
                        for name, values in rewards[step].items()} for step in required_steps},
        "equivalence_groups": {str(step): groups[step] for step in required_steps},
        "pairwise_equivalence": {str(step): pairwise[step] for step in required_steps},
        "ranking_stability": ranking, "learning_stability": learning,
        "bootstrap_slope_ci": list(slope_ci), "slope_stable": slope_stable,
        "finite_diagnostics": finite, "prospective_extension_confirmed": passed,
        "test_accessed": False,
    }


def make_budget_manifest(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    passed = all(results[crop]["status"] == "calibrated" for crop in OFFICIAL_CROPS)
    return {
        "schema_version": STEP19_5_V2_2_BUDGET_SCHEMA,
        "protocol_version": STEP19_5_V2_2_VERSION,
        "status": "passed" if passed else "calibration_inconclusive_v2_2",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "budget_decision_source": "prospective_extension_with_practical_equivalence",
        "numeric_budget_precommitted": False,
        "crops": {crop: {"status": results[crop]["status"],
                          "rung_budgets": results[crop]["rung_budgets"]}
                  for crop in OFFICIAL_CROPS},
        "test_accessed": False,
    }


__all__ = [
    "DIAGNOSTICS", "OFFICIAL_CROPS", "STEP19_5_V2_2_BUDGET_SCHEMA",
    "STEP19_5_V2_2_VERSION", "analyze_v2_2", "kendall_tau_b",
    "make_budget_manifest", "practical_equivalence_groups", "read_json",
    "sha256", "write_json",
]
