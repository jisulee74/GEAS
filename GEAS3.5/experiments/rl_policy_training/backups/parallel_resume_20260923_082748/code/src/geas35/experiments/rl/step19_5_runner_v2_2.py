"""Audit, resume, and finalize the Step 19.5 v2.2 calibration campaign."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from geas35.experiments.rl.checkpoint_v2 import load_ppo_checkpoint_v2
from geas35.experiments.rl.step19_5_calibration_v2_2 import (
    OFFICIAL_CROPS, STEP19_5_V2_2_VERSION, analyze_v2_2,
    make_budget_manifest, read_json, sha256, write_json,
)
from geas35.experiments.rl.step19_5_runner_v1_4 import _payload, _train_to_budget_v2
from geas35.experiments.rl.step20_hpo import apply_safety_gate
from geas35.experiments.rl.step20_runner_v1_4 import (
    _artifact_hashes, _limit_transition_threads,
)
from geas35.rl.hybrid_ppo import SupportAwareActionAdapter
from geas35.rl.model_driven_env import load_model_driven_env_from_step15

SOURCE_REL = Path("experiments/rl_policy_training/artifacts/step19_5_v1_4")
OUTPUT_REL = Path("experiments/rl_policy_training/artifacts/step19_5_v2_2")
CONFIG_REL = Path("experiments/rl_policy_training/configs/step19_5_calibration_v2_2.json")


def _paths(root: Path) -> tuple[Path, Path, Path]:
    return root / SOURCE_REL, root / OUTPUT_REL, root / CONFIG_REL


def _record(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path),
            "size_bytes": path.stat().st_size}


def prepare_v2_2(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    source, output, config_path = _paths(root)
    config = read_json(config_path)
    if config.get("protocol_version") != STEP19_5_V2_2_VERSION:
        raise ValueError("Unsupported Step 19.5 v2.2 config")
    required = [source / "representative_configurations.json",
                source / "train_derived_safety_gates.json"]
    required += [source / crop / "learning_curves.json" for crop in OFFICIAL_CROPS]
    for crop in OFFICIAL_CROPS:
        for config_id in range(int(config["representative_configurations"])):
            for seed in config["seeds"]:
                required.append(source / crop / f"representative_{config_id}" / f"seed_{seed}.pt")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v1.4 calibration dependencies: {missing[:5]}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("safety_audit", "retrospective_diagnostics", "extended_training", "calibration_report"):
        (output / name).mkdir(parents=True, exist_ok=True)
    snapshot = {
        "schema_version": STEP19_5_V2_2_VERSION,
        "status": "inconclusive_under_strict_stability_contract",
        "source_root": source.relative_to(root).as_posix(),
        "source_files": [_record(root, path) for path in sorted(required)],
        "source_mutation_allowed": False, "test_accessed": False,
    }
    write_json(output / "source_v1_4_freeze_manifest.json", snapshot)
    dependencies = {
        "config": _record(root, config_path),
        "step15_protocol": _record(root, root / "experiments/rl_policy_training/artifacts/step15_v1_4/rl_protocol_manifest.json"),
        "step16_environment": _record(root, root / "experiments/rl_policy_training/artifacts/step16_v1_4/step16_environment_manifest.json"),
        "step17_integrity": _record(root, root / "experiments/rl_policy_training/artifacts/step17_v1_4/step17_integrity_manifest.json"),
        "safety_gates": _record(root, source / "train_derived_safety_gates.json"),
        "representatives": _record(root, source / "representative_configurations.json"),
    }
    path = output / "step19_5_v2_2_readiness.json"
    write_json(path, {
        "schema_version": STEP19_5_V2_2_VERSION, "status": "ready_for_safety_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dependencies": dependencies, "source_freeze_sha256": sha256(output / "source_v1_4_freeze_manifest.json"),
        "test_accessed": False, "step20_started": False,
    })
    return path


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    numeric = [value for value in metrics.values() if isinstance(value, (int, float))]
    return bool(numeric) and bool(np.isfinite(np.asarray(numeric, dtype=float)).all())


def _evaluate_audit(
    *, root: Path, crop: str, trainer: Any, gates: Mapping[str, Any],
    transition_n_jobs: int,
) -> tuple[dict[str, Any], bool, list[str], dict[str, Any]]:
    """Validation-only evaluator retaining intervention and termination details."""
    env = load_model_driven_env_from_step15(
        project_root=root, crop=crop, split="validation",
        protocol_manifest_path=root / "experiments/rl_policy_training/artifacts/step15_v1_4/rl_protocol_manifest.json",
    )
    _limit_transition_threads(env, transition_n_jobs)
    totals = {"scheduled": 0, "executed": 0, "reward": 0.0, "hard": 0,
              "severe": 0, "fallback": 0, "warning": 0, "projection": 0,
              "abnormal_termination": 0}
    termination_reasons: dict[str, int] = {}
    episodes = []
    for ordinal, episode_index in enumerate(range(len(env.episodes))):
        observation, reset_info = env.reset(seed=42 + ordinal, options={"episode_index": episode_index})
        scheduled = int(reset_info["scheduled_valid_steps"])
        executed = hard = severe = fallback = warning = projection = 0
        reward_sum = 0.0
        terminated = truncated = False
        final_reason = None
        adapter = SupportAwareActionAdapter(env)
        while not (terminated or truncated):
            spec = adapter.feasible_spec()
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=trainer.device).unsqueeze(0)
            with torch.no_grad():
                output = trainer.policy.act(tensor, spec.batched(1, device=trainer.device), deterministic=True)
            action = output["action"][0].cpu().numpy().astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            executed += 1; reward_sum += float(reward)
            severe += int(info.get("joint_ood_level") == "severe" or info.get("state_ood_level") == "severe")
            warning += int(info.get("joint_ood_level") == "warning")
            fallback += int(info.get("safety_fallback_applied", False))
            projection += int(bool(info.get("support_projection", {}).get("changes", {})))
            final_reason = info.get("termination_reason") if terminated or truncated else final_reason
            executed_action = info.get("executed_action")
            if executed_action is not None:
                values = np.asarray(list(executed_action.values()), dtype=float)
                weather = info.get("weather_constraint", {})
                binary_valid = all(float(executed_action[name]) in (0.0, 1.0)
                                   for name in ("heat_run", "cool_run", "fan_run"))
                vent_valid = float(executed_action["vent_pct"]) <= float(weather.get("vent_open_cap", 1.0)) + 1e-7
                hard += int(not np.isfinite(values).all() or np.any(values < 0.0)
                            or np.any(values > 1.0) or not binary_valid or not vent_valid)
        abnormal = int(final_reason not in (None, "episode_boundary"))
        if final_reason is not None:
            termination_reasons[str(final_reason)] = termination_reasons.get(str(final_reason), 0) + 1
        totals["scheduled"] += scheduled; totals["executed"] += executed
        totals["reward"] += reward_sum; totals["hard"] += hard
        totals["severe"] += severe; totals["fallback"] += fallback
        totals["warning"] += warning; totals["projection"] += projection
        totals["abnormal_termination"] += abnormal
        episodes.append({"episode_id": str(reset_info["episode_id"]), "scheduled_steps": scheduled,
                         "executed_steps": executed, "hard_constraint_violations": hard,
                         "termination_reason": final_reason})
    denominator = float(max(totals["executed"], 1))
    metrics = {
        "mean_reward_per_scheduled_valid_step": totals["reward"] / float(totals["scheduled"]),
        "scheduled_valid_steps": totals["scheduled"], "executed_steps": totals["executed"],
        "hard_constraint_violations": totals["hard"],
        "severe_ood_rate": totals["severe"] / denominator,
        "fallback_rate": totals["fallback"] / denominator,
        "warning_ood_rate": totals["warning"] / denominator,
        "support_projection_rate": totals["projection"] / denominator,
        "abnormal_termination_count": totals["abnormal_termination"],
    }
    feasible, reasons = apply_safety_gate(
        metrics, severe_ood_rate_max=float(gates["severe_ood_rate_max"]),
        fallback_rate_max=float(gates["fallback_rate_max"]),
    )
    finite = _finite_metrics(metrics)
    if not finite:
        feasible = False; reasons = [*reasons, "non_finite_validation_metric"]
    details = {"termination_reasons": termination_reasons, "episodes": episodes,
               "finite_metrics": finite, "test_accessed": False}
    return metrics, feasible, reasons, details


def audit_crop(
    project_root: str | Path, crop: str, *, device: str = "cpu",
    transition_n_jobs: int = 1,
) -> Path:
    root = Path(project_root).resolve(); source, output, config_path = _paths(root)
    if crop not in OFFICIAL_CROPS:
        raise ValueError(f"Unsupported crop: {crop}")
    if not (output / "step19_5_v2_2_readiness.json").exists():
        raise RuntimeError("Run --prepare before the safety audit")
    config = read_json(config_path)
    reps = read_json(source / "representative_configurations.json")["configurations"]
    gates = read_json(source / "train_derived_safety_gates.json")["crops"][crop]
    rows = []
    hashes = _artifact_hashes(root, crop)
    for representative in reps:
        config_id = str(representative["config_id"])
        for seed in map(int, config["seeds"]):
            checkpoint = source / crop / config_id / f"seed_{seed}.pt"
            trainer, payload = load_ppo_checkpoint_v2(
                checkpoint, expected_artifact_hashes=hashes, device=device, restore_rng=False)
            if int(payload.get("extra_state", {}).get("training_steps", 0)) != 524288:
                raise ValueError(f"Unexpected source checkpoint step: {checkpoint}")
            metrics, feasible, reasons, details = _evaluate_audit(
                root=root, crop=crop, trainer=trainer, gates=gates,
                transition_n_jobs=transition_n_jobs,
            )
            rows.append({"crop": crop, "config_id": config_id, "seed": seed,
                         "checkpoint_steps": 524288, "checkpoint_sha256": sha256(checkpoint),
                         "metrics": metrics, "safety_gate_passed": feasible,
                         "failure_reasons": reasons, "details": details,
                         "split": "validation", "test_accessed": False})
    by_config = {}
    for representative in reps:
        config_id = str(representative["config_id"])
        selected = [row for row in rows if row["config_id"] == config_id]
        eligible = all(row["safety_gate_passed"] for row in selected)
        by_config[config_id] = {"eligible": eligible,
                                "seed_results": {str(row["seed"]): row["safety_gate_passed"] for row in selected},
                                "failure_reasons": sorted({reason for row in selected for reason in row["failure_reasons"]})}
    eligible = sorted(name for name, value in by_config.items() if value["eligible"])
    status = ("no_safe_configuration" if not eligible else
              "insufficient_eligible_configurations" if len(eligible) < int(config["configuration_eligibility"]["minimum_eligible_configurations"])
              else "eligible_for_extension")
    path = output / "safety_audit" / f"{crop}_safety_audit.json"
    write_json(path, {"schema_version": STEP19_5_V2_2_VERSION, "crop": crop,
                      "status": status, "gates": gates, "configurations": by_config,
                      "eligible_configurations": eligible, "runs": rows,
                      "test_accessed": False})
    return path


def finalize_audit(project_root: str | Path) -> Path:
    root = Path(project_root).resolve(); _, output, _ = _paths(root)
    crops = {}
    for crop in OFFICIAL_CROPS:
        path = output / "safety_audit" / f"{crop}_safety_audit.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing safety audit: {path}")
        audit = read_json(path)
        crops[crop] = {"status": audit["status"],
                       "eligible_configurations": audit["eligible_configurations"],
                       "audit_sha256": sha256(path)}
    combined_audit = output / "safety_audit" / "safety_audit.json"
    write_json(combined_audit, {"schema_version": STEP19_5_V2_2_VERSION,
                                "crops": crops, "test_accessed": False})
    path = output / "eligible_configuration_manifest.json"
    write_json(path, {"schema_version": STEP19_5_V2_2_VERSION, "crops": crops,
                      "require_all_seeds_safe": True,
                      "safety_audit_sha256": sha256(combined_audit),
                      "test_accessed": False})
    return path


def _copy_resume_state(source_checkpoint: Path, target_checkpoint: Path) -> None:
    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_history = source_checkpoint.with_suffix(".training_history.json")
    target_history = target_checkpoint.with_suffix(".training_history.json")
    if not target_checkpoint.exists():
        shutil.copy2(source_checkpoint, target_checkpoint)
    if source_history.exists() and not target_history.exists():
        shutil.copy2(source_history, target_history)


def extend_crop(
    project_root: str | Path, crop: str, *, device: str = "cpu",
    transition_n_jobs: int = 1,
) -> Path:
    root = Path(project_root).resolve(); source, output, config_path = _paths(root)
    config = read_json(config_path)
    audit = read_json(output / "safety_audit" / f"{crop}_safety_audit.json")
    eligible = audit["eligible_configurations"]
    minimum = int(config["configuration_eligibility"]["minimum_eligible_configurations"])
    status_path = output / "extended_training" / crop / "extension_status.json"
    if len(eligible) < minimum:
        status = "no_safe_configuration" if not eligible else "insufficient_eligible_configurations"
        write_json(status_path, {"schema_version": STEP19_5_V2_2_VERSION, "crop": crop,
                                 "status": status, "eligible_configurations": eligible,
                                 "test_accessed": False})
        return status_path
    reps = {item["config_id"]: item for item in read_json(source / "representative_configurations.json")["configurations"]}
    probe = load_model_driven_env_from_step15(
        project_root=root, crop=crop, split="train",
        protocol_manifest_path=root / "experiments/rl_policy_training/artifacts/step15_v1_4/rl_protocol_manifest.json")
    gates = read_json(source / "train_derived_safety_gates.json")["crops"][crop]
    budget = int(config["extension_budget_steps"])
    rows = []
    for config_id in eligible:
        for seed in map(int, config["seeds"]):
            source_checkpoint = source / crop / config_id / f"seed_{seed}.pt"
            checkpoint = output / "extended_training" / crop / config_id / f"seed_{seed}.pt"
            _copy_resume_state(source_checkpoint, checkpoint)
            payload = _payload(reps[config_id]["parameters"], probe.observation_shape[0], seed)
            trainer, history = _train_to_budget_v2(
                root=root, crop=crop, config_payload=payload, seed=seed,
                budget_steps=budget, checkpoint=checkpoint, device=device,
                resume=True, transition_n_jobs=transition_n_jobs)
            metrics, feasible, reasons, details = _evaluate_audit(
                root=root, crop=crop, trainer=trainer, gates=gates,
                transition_n_jobs=transition_n_jobs)
            latest = history[-1]
            row = {"crop": crop, "config_id": config_id, "seed": seed,
                   "steps": budget, "split": "validation", "test_accessed": False,
                   "safety_gate_passed": feasible,
                   "mean_reward_per_scheduled_valid_step": metrics["mean_reward_per_scheduled_valid_step"]}
            for name in ("policy_loss", "value_loss", "approx_kl", "entropy",
                         "explained_variance", "gradient_norm", "in_support_rate",
                         "warning_ood_rate", "severe_ood_rate", "projection_rate",
                         "fallback_rate"):
                row[name] = float(latest[name])
            row["termination_rate"] = float(latest["severe_state_termination_rate"])
            rows.append(row)
            write_json(checkpoint.with_suffix(".v2_2_metrics.json"), {
                "schema_version": STEP19_5_V2_2_VERSION, "validation": metrics,
                "safety_gate_passed": feasible, "failure_reasons": reasons,
                "details": details, "source_checkpoint_sha256": sha256(source_checkpoint),
                "extended_checkpoint_sha256": sha256(checkpoint), "test_accessed": False})
            write_json(checkpoint.with_suffix(".completed.json"), {
                "schema_version": STEP19_5_V2_2_VERSION, "crop": crop,
                "config_id": config_id, "seed": seed, "training_steps": budget,
                "source_checkpoint_sha256": sha256(source_checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "ppo_config": asdict(trainer.config),
                "status": "complete", "test_accessed": False,
            })
    write_json(output / "extended_training" / crop / "learning_curves_extension.json",
               {"schema_version": STEP19_5_V2_2_VERSION, "crop": crop, "rows": rows,
                "test_accessed": False})
    write_json(status_path, {"schema_version": STEP19_5_V2_2_VERSION, "crop": crop,
                             "status": "extension_complete", "budget_steps": budget,
                             "completed_runs": len(rows), "test_accessed": False})
    return status_path


def finalize_v2_2(project_root: str | Path) -> Path:
    root = Path(project_root).resolve(); source, output, config_path = _paths(root)
    config = read_json(config_path)
    results = {}
    for crop in OFFICIAL_CROPS:
        audit = read_json(output / "safety_audit" / f"{crop}_safety_audit.json")
        eligible = audit["eligible_configurations"]
        source_rows = read_json(source / crop / "learning_curves.json")["rows"]
        rows = [row for row in source_rows if int(row["steps"]) in set(config["source_steps"])]
        extension_path = output / "extended_training" / crop / "learning_curves_extension.json"
        if extension_path.exists():
            rows.extend(read_json(extension_path)["rows"])
        result = analyze_v2_2(rows, eligible, config)
        results[crop] = result
        write_json(output / "calibration_report" / f"{crop}_calibration_report.json", result)
        csv_path = output / "calibration_report" / f"{crop}_learning_curves.csv"
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    budget = output / "calibrated_budget_manifest.json"
    write_json(budget, make_budget_manifest(results))
    report = output / "calibration_report" / "combined_calibration_report.json"
    write_json(report, {"schema_version": STEP19_5_V2_2_VERSION,
                        "status": read_json(budget)["status"], "crops": results,
                        "old_contract": {"consecutive_rank_intervals": 3,
                                         "reward_delta_rule": "maximum <= 5% of current configuration IQR"},
                        "v2_2_contract": config, "test_accessed": False})
    artifacts = {
        "budget": _record(root, budget), "combined_report": _record(root, report),
        "eligible_configurations": _record(root, output / "eligible_configuration_manifest.json"),
    }
    for crop in OFFICIAL_CROPS:
        artifacts[f"{crop}_safety_audit"] = _record(root, output / "safety_audit" / f"{crop}_safety_audit.json")
    integrity = output / "step19_5_integrity_manifest.json"
    write_json(integrity, {"schema_version": STEP19_5_V2_2_VERSION,
                           "status": read_json(budget)["status"], "artifacts": artifacts,
                           "budget": {"path": budget.relative_to(root).as_posix(), "sha256": sha256(budget)},
                           "test_accessed": False, "step20_started": False})
    return integrity


__all__ = ["audit_crop", "extend_crop", "finalize_audit", "finalize_v2_2", "prepare_v2_2"]
