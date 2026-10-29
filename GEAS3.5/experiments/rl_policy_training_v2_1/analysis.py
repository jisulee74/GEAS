from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from geas35.experiments.rl.step19_5_calibration import kendall_tau

DIAGNOSTICS = (
    "policy_loss", "value_loss", "approx_kl", "entropy", "explained_variance",
    "gradient_norm", "in_support_rate", "warning_ood_rate", "severe_ood_rate",
    "projection_rate", "fallback_rate", "termination_rate",
)


def _bootstrap_slope_ci(rows, steps, samples, confidence, seed):
    rng = np.random.default_rng(seed)
    points = [[float(r["mean_reward_per_scheduled_valid_step"]) for r in rows
               if int(r["steps"]) == step] for step in steps]
    x = np.arange(len(steps), dtype=float)
    slopes = []
    for _ in range(samples):
        y = np.asarray([np.median(rng.choice(values, size=len(values), replace=True))
                        for values in points])
        slopes.append(float(np.polyfit(x, y, 1)[0]))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(slopes, alpha)), float(np.quantile(slopes, 1.0 - alpha))


def analyze(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Learning curve rows are empty")
    if any(r.get("split") != "validation" or r.get("test_accessed") is not False for r in rows):
        raise ValueError("v2.1 accepts Validation-only metrics with Test locked")
    steps = sorted({int(r["steps"]) for r in rows})
    configs = sorted({str(r["config_id"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    expected = {(step, cfg, seed) for step in steps for cfg in configs for seed in seeds}
    observed = {(int(r["steps"]), str(r["config_id"]), int(r["seed"])) for r in rows}
    if expected != observed:
        raise ValueError("Learning curves must form a complete step/config/seed grid")
    medians = {
        step: {
            cfg: float(np.median([float(r["mean_reward_per_scheduled_valid_step"])
                                  for r in rows if int(r["steps"]) == step
                                  and str(r["config_id"]) == cfg]))
            for cfg in configs
        }
        for step in steps
    }
    rc = config["ranking_stability"]
    comparisons = []
    top_count = int(rc["top_count"])
    for previous, current in zip(steps, steps[1:]):
        tau = kendall_tau(medians[previous], medians[current])
        left = set(sorted(medians[previous], key=lambda k: (-medians[previous][k], k))[:top_count])
        right = set(sorted(medians[current], key=lambda k: (-medians[current][k], k))[:top_count])
        shared = len(left & right)
        comparisons.append({
            "previous_steps": previous, "steps": current, "kendall_tau": tau,
            "shared_top_configurations": shared,
            "stable": tau >= float(rc["kendall_tau_min"])
            and shared >= int(rc["minimum_shared_top_configurations"]),
        })
    consecutive = int(rc["consecutive_intervals"])
    first = None
    for index in range(consecutive - 1, len(comparisons)):
        if all(item["stable"] for item in comparisons[index - consecutive + 1:index + 1]):
            first = comparisons[index]["steps"]
            break

    lc = config["learning_stability"]
    plateau = []
    window = int(lc["window_intervals"])
    for index in range(window, len(steps)):
        current = steps[index]
        recent = steps[index - window:index + 1]
        current_values = np.asarray(list(medians[current].values()))
        iqr = float(np.quantile(current_values, 0.75) - np.quantile(current_values, 0.25))
        reward_median = float(np.median(np.abs(current_values)))
        scale = max(iqr, float(lc["iqr_floor_fraction_of_absolute_median_reward"]) * reward_median,
                    np.finfo(float).eps)
        deltas = [abs(medians[b][cfg] - medians[a][cfg])
                  for a, b in zip(recent, recent[1:]) for cfg in configs]
        median_delta = float(np.median(deltas))
        ci = _bootstrap_slope_ci(rows, recent, int(lc["bootstrap_samples"]),
                                 float(lc["bootstrap_confidence"]), 42 + index)
        current_rows = [r for r in rows if int(r["steps"]) == current]
        finite = all(np.isfinite(float(r[key])) for r in current_rows for key in DIAGNOSTICS)
        safe = all(bool(r.get("safety_gate_passed", False)) for r in current_rows)
        stable = (
            median_delta <= float(lc["median_reward_delta_fraction_of_robust_scale_max"]) * scale
            and ci[0] <= 0 <= ci[1] and finite and safe
        )
        plateau.append({
            "steps": current, "reward_iqr": iqr, "robust_scale": scale,
            "median_recent_delta": median_delta, "bootstrap_slope_ci": list(ci),
            "finite_diagnostics": finite, "safety_gate_passed": safe, "stable": stable,
        })

    stable_steps = {item["steps"] for item in comparisons if item["stable"]}
    final = None
    if first is not None:
        for item in plateau:
            if item["steps"] > first and item["stable"] and item["steps"] in stable_steps:
                final = item["steps"]
                break
    decision = "observed_stability"
    max_budget = int(config["checkpoint_grid"]["maximum_environment_steps"])
    if final is None and steps[-1] >= max_budget:
        if len(steps) < 2:
            raise ValueError("Maximum-budget fallback requires two observed budgets")
        rungs = steps[-2:]
        decision = "maximum_budget_fallback"
        status = "passed"
    elif final is not None:
        rungs = [first, final]
        status = "passed"
    else:
        rungs = []
        status = "calibration_inconclusive"
    return {
        "schema_version": "geas35.rl.step19_5.analysis.v2_1", "status": status,
        "decision": decision, "rung_budgets": rungs, "steps": steps,
        "medians": medians, "ranking_comparisons": comparisons,
        "learning_stability": plateau, "test_accessed": False,
    }
