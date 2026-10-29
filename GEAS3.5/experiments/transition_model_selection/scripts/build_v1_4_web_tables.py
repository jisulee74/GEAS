#!/usr/bin/env python3
"""Build five website-ready CSV tables from the official v1.4 validation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = {
    "deduplicated_with_quality_flags": "with_quality_flags",
    "deduplicated_without_quality_flags": "without_quality_flags",
}
CROPS = ("strawberry", "melon", "cucumber")
TARGETS = {
    "indoor_temp_c": "indoor_temperature",
    "indoor_humidity_pct": "indoor_humidity",
    "indoor_co2_ppm": "indoor_co2",
}
HORIZONS = ("15min", "30min", "60min")
HORIZON_WEIGHTS = {"15min": 0.10, "30min": 0.25, "60min": 0.45}


def _bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _candidate_dir(crop_root: Path, crop: str, model: str) -> Path:
    return crop_root / crop / model


def _read_rollout(crop_root: Path, crop: str, model: str) -> dict:
    path = _candidate_dir(crop_root, crop, model) / "rollout_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_difference(
    with_values: np.ndarray,
    without_values: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    difference = with_values - without_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(samples, difference.size))
    boot_means = difference[indices].mean(axis=1)
    return (
        float(difference.mean()),
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )


def _window_scores(rollout: dict) -> dict[str, float]:
    by_horizon: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        metrics = rollout[horizon]
        ids = [str(value) for value in metrics["window_ids"]]
        values = np.asarray(metrics["window_trajectory_nrmse_samples"], dtype=float)
        by_horizon[horizon] = dict(zip(ids, values, strict=True))
    shared = set.intersection(*(set(values) for values in by_horizon.values()))
    weight_sum = sum(HORIZON_WEIGHTS.values())
    return {
        window_id: sum(
            HORIZON_WEIGHTS[horizon] * by_horizon[horizon][window_id]
            for horizon in HORIZONS
        )
        / weight_sum
        for window_id in shared
    }


def build_tables(root: Path, output: Path) -> dict[str, int]:
    table1: list[dict] = []
    table2: list[dict] = []
    table3: list[dict] = []
    table4: list[dict] = []

    for variant_dir, variant_label in VARIANTS.items():
        for crop in CROPS:
            crop_root = root / variant_dir / crop
            candidates = pd.read_csv(crop_root / "candidate_metrics.csv")
            rollouts = pd.read_csv(crop_root / "candidate_rollout_metrics.csv")
            actions = pd.read_csv(crop_root / "candidate_action_response.csv")
            resources = pd.read_csv(crop_root / "candidate_resource_metrics.csv")

            for _, candidate in candidates.iterrows():
                model = str(candidate["candidate_name"])
                rollout = rollouts.loc[rollouts["candidate_name"] == model]
                resource = resources.loc[resources["candidate_name"] == model]
                action = actions.loc[actions["candidate_name"] == model]
                if rollout.empty:
                    continue
                rollout = rollout.iloc[0]

                table1.append({
                    "crop": crop,
                    "feature_variant": variant_label,
                    "model": model,
                    "model_family": candidate["model_family"],
                    "baseline_only": _bool(candidate["baseline_only"]),
                    "deployment_eligible": _bool(candidate["deployment_eligible"]),
                    "candidate_status": candidate["candidate_status"],
                    "validation_rollout_weighted_nrmse": candidate["validation_rollout_weighted_score"],
                    "one_step_nrmse": candidate["validation_one_step_nrmse"],
                    "rollout_15min_nrmse": rollout["15min_trajectory_nrmse"],
                    "rollout_30min_nrmse": rollout["30min_trajectory_nrmse"],
                    "rollout_60min_nrmse": rollout["60min_trajectory_nrmse"],
                    "skill_vs_persistence_15min": rollout["15min_persistence_skill"],
                    "skill_vs_persistence_30min": rollout["30min_persistence_skill"],
                    "skill_vs_persistence_60min": rollout["60min_persistence_skill"],
                    "physical_violation_rate_max": max(
                        rollout[f"{h}_physical_violation_rate"] for h in HORIZONS
                    ),
                    "nan_inf_count_total": sum(rollout[f"{h}_nan_inf_count"] for h in HORIZONS),
                })

                for target_key, target_label in TARGETS.items():
                    prefix = f"validation_{target_key}"
                    table2.append({
                        "crop": crop,
                        "feature_variant": variant_label,
                        "model": model,
                        "target": target_label,
                        "deployment_eligible": _bool(candidate["deployment_eligible"]),
                        "mae": candidate[f"{prefix}_mae"],
                        "rmse": candidate[f"{prefix}_rmse"],
                        "nrmse": candidate[f"{prefix}_nrmse"],
                        "r2": candidate[f"{prefix}_r2"],
                        "absolute_error_q90": candidate[f"{prefix}_q90"],
                        "absolute_error_cvar90": candidate[f"{prefix}_cvar90"],
                    })

                supported = action.loc[action["status"] == "ok"]
                sufficient = action.loc[action["support_count"] >= 100]
                sensitivities = sufficient["counterfactual_mean_absolute_sensitivity"].dropna()
                table3.append({
                    "crop": crop,
                    "feature_variant": variant_label,
                    "model": model,
                    "deployment_eligible": _bool(candidate["deployment_eligible"]),
                    "action_response_eligible": _bool(candidate["action_response_eligible"]),
                    "fully_action_insensitive": _bool(candidate["fully_action_insensitive"]),
                    "actions_total": int(len(action)),
                    "actions_with_sufficient_support": int(len(sufficient)),
                    "actions_insufficient_support": int((action["support_count"] < 100).sum()),
                    "actions_status_ok": int(len(supported)),
                    "actions_marked_insensitive": int(action["action_insensitive"].map(_bool).sum()),
                    "minimum_counterfactual_sensitivity": sensitivities.min() if len(sensitivities) else np.nan,
                    "mean_counterfactual_sensitivity": sensitivities.mean() if len(sensitivities) else np.nan,
                    "physical_violation_rate_15min": rollout["15min_physical_violation_rate"],
                    "physical_violation_rate_30min": rollout["30min_physical_violation_rate"],
                    "physical_violation_rate_60min": rollout["60min_physical_violation_rate"],
                    "nan_inf_count_total": sum(rollout[f"{h}_nan_inf_count"] for h in HORIZONS),
                    "drift_slope_nmae": rollout["drift_slope_nmae"],
                })

                if not resource.empty:
                    resource = resource.iloc[0]
                    table4.append({
                        "crop": crop,
                        "feature_variant": variant_label,
                        "model": model,
                        "deployment_eligible": _bool(candidate["deployment_eligible"]),
                        "benchmark_status": resource["benchmark_status"],
                        "training_time_seconds": resource["training_time_seconds"],
                        "hpo_total_time_seconds": resource["hpo_total_time_seconds"],
                        "inference_latency_median_ms": resource["inference_latency_median_ms"],
                        "inference_latency_p95_ms": resource["inference_latency_p95_ms"],
                        "peak_cpu_memory_mb": resource["peak_cpu_rss_bytes"] / (1024 ** 2),
                        "peak_gpu_memory_mb": resource["peak_gpu_allocated_bytes"] / (1024 ** 2),
                        "serialized_model_size_mb": resource["serialized_model_size_bytes"] / (1024 ** 2),
                        "artifact_directory_size_mb": resource["artifact_dir_size_bytes"] / (1024 ** 2),
                        "device": resource["device"],
                    })

    table5: list[dict] = []
    with_dir = root / "deduplicated_with_quality_flags"
    without_dir = root / "deduplicated_without_quality_flags"
    for crop in CROPS:
        with_metrics = pd.read_csv(with_dir / crop / "candidate_metrics.csv")
        without_metrics = pd.read_csv(without_dir / crop / "candidate_metrics.csv")
        common_models = sorted(set(with_metrics["candidate_name"]) & set(without_metrics["candidate_name"]))
        for model in common_models:
            with_rollout = _read_rollout(with_dir / crop, crop, model)
            without_rollout = _read_rollout(without_dir / crop, crop, model)
            with_scores = _window_scores(with_rollout)
            without_scores = _window_scores(without_rollout)
            shared_ids = sorted(set(with_scores) & set(without_scores))
            with_values = np.asarray([with_scores[value] for value in shared_ids])
            without_values = np.asarray([without_scores[value] for value in shared_ids])
            difference, lower, upper = _bootstrap_difference(with_values, without_values)
            if upper < 0:
                decision = "with_flags_improved"
            elif lower > 0:
                decision = "with_flags_worsened"
            else:
                decision = "no_significant_difference"
            with_row = with_metrics.loc[with_metrics["candidate_name"] == model].iloc[0]
            without_row = without_metrics.loc[without_metrics["candidate_name"] == model].iloc[0]
            table5.append({
                "crop": crop,
                "model": model,
                "baseline_only": _bool(with_row["baseline_only"]),
                "with_flags_deployment_eligible": _bool(with_row["deployment_eligible"]),
                "without_flags_deployment_eligible": _bool(without_row["deployment_eligible"]),
                "paired_validation_window_count": len(shared_ids),
                "with_flags_paired_rollout_nrmse": with_values.mean(),
                "without_flags_paired_rollout_nrmse": without_values.mean(),
                "difference_with_minus_without": difference,
                "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper,
                "quality_flag_effect": decision,
                "bootstrap_samples": 10_000,
                "bootstrap_seed": 42,
            })

    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_1_candidate_screening_and_rollout.csv": pd.DataFrame(table1),
        "table_2_one_step_target_metrics.csv": pd.DataFrame(table2),
        "table_3_safety_and_action_response.csv": pd.DataFrame(table3),
        "table_4_resource_metrics.csv": pd.DataFrame(table4),
        "table_5_quality_flag_ablation.csv": pd.DataFrame(table5),
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False, float_format="%.10g")
    manifest = {
        "source": str(root),
        "scope": "validation_only",
        "test_data_used": False,
        "table_rows": {name: len(frame) for name, frame in tables.items()},
        "table_5_difference_definition": "with_quality_flags minus without_quality_flags",
        "table_5_negative_difference_meaning": "quality flags improve weighted rollout NRMSE",
        "table_5_horizon_weights": HORIZON_WEIGHTS,
        "table_5_weights_normalized": True,
    }
    (output / "table_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest["table_rows"]


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1] / "artifacts" / "step12_v1_4_root_cause" / "official_rerun"
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=default_root / "selection_tables")
    args = parser.parse_args()
    print(json.dumps(build_tables(args.root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
