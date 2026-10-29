"""Executable GPU/CPU HPO campaign for Step 20 (Step 21 is out of scope)."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from geas35.experiments.rl.step20_hpo import (
    DeterministicSearchSpace, HpoDatabase, OFFICIAL_CROPS, STEP20_VERSION,
    aggregate_validation_metrics, apply_safety_gate,
    evaluate_paired_validation_episodes, trial_rank_key,
)
from geas35.experiments.rl.step20_protocol import prepare_step20_protocol
from geas35.rl.hybrid_ppo import (
    HybridActorCritic, HybridPPOConfig, HybridPPOTrainer, collect_rollout,
    load_ppo_checkpoint, save_ppo_checkpoint, seed_everything,
)
from geas35.rl.model_driven_env import ModelDrivenEnvConfig, load_model_driven_env_from_step15


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_hashes(root: Path, crop: str) -> dict[str, str]:
    protocol = _read(root / "experiments/rl_policy_training/artifacts/step15/rl_protocol_manifest.json")
    record = protocol["crops"][crop]
    return {
        "observation_scaler": record["observation_scaler"]["sha256"],
        "environment": _sha(root / "src/geas35/rl/model_driven_env.py"),
        "transition_model": record["candidate_model"]["sha256"],
        "support": record["state_action_support"]["sha256"],
    }


def _limit_transition_threads(env: Any, n_jobs: int) -> None:
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("transition_n_jobs must be -1 or a positive integer")
    estimators = getattr(env.model, "estimators_", {})
    values = estimators.values() if isinstance(estimators, Mapping) else estimators
    for estimator in values:
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = n_jobs


def _ppo_config(payload: Mapping[str, Any]) -> HybridPPOConfig:
    values = dict(payload["ppo"])
    values["hidden_sizes"] = tuple(values["hidden_sizes"])
    return HybridPPOConfig(**values)


def _ensure_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if resolved.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if resolved.type == "cuda" and resolved.index is not None:
        if resolved.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {resolved.index} is unavailable")
        torch.cuda.set_device(resolved)
    return resolved


def _train_to_budget(
    *, root: Path, crop: str, config_payload: Mapping[str, Any], seed: int,
    budget_steps: int, checkpoint: Path, device: str, resume: bool,
    transition_n_jobs: int,
) -> tuple[HybridPPOTrainer, list[dict[str, Any]]]:
    hashes = _artifact_hashes(root, crop)
    env = load_model_driven_env_from_step15(project_root=root, crop=crop, split="train")
    _limit_transition_threads(env, transition_n_jobs)
    if resume and checkpoint.exists():
        trainer, saved = load_ppo_checkpoint(
            checkpoint, expected_artifact_hashes=hashes, device=device, restore_rng=True,
        )
        if asdict(trainer.config) != asdict(_ppo_config(config_payload)):
            raise ValueError("Resume checkpoint PPO config differs from sampled trial config")
        completed = int(saved.get("extra_state", {}).get("training_steps", 0))
    else:
        seed_everything(seed)
        ppo = _ppo_config(config_payload)
        trainer = HybridPPOTrainer(HybridActorCritic(ppo), ppo, device=device)
        completed = 0
    if completed > budget_steps:
        raise ValueError("Checkpoint training_steps exceed the requested rung budget")
    rollout_length = int(config_payload["rollout_length"])
    history_path = checkpoint.with_suffix(".training_history.json")
    if resume and history_path.exists():
        stored_history = _read(history_path)
        history = list(stored_history.get("history", []))
        history = [item for item in history if int(item["training_steps"]) <= completed]
    else:
        history: list[dict[str, Any]] = []
    while completed < budget_steps:
        count = min(rollout_length, budget_steps - completed)
        buffer = collect_rollout(env, trainer, steps=count, seed=seed + trainer.update_count)
        metrics = trainer.update(buffer)
        completed += count
        history.append({"training_steps": completed, **metrics})
        _write(history_path, {"schema_version": STEP20_VERSION, "crop": crop,
                              "seed": seed, "history": history, "test_accessed": False})
        save_ppo_checkpoint(
            checkpoint, trainer, artifact_hashes=hashes,
            extra_state={"crop": crop, "training_steps": completed, "hpo_step": 20,
                         "trial_seed": seed, "test_accessed": False},
        )
    return trainer, history


def _evaluate(
    *, root: Path, crop: str, trainer: HybridPPOTrainer,
    validation_episode_indices: Sequence[int], evaluation_seed: int,
    gates: Mapping[str, Any], max_episode_steps: int | None = None,
    transition_n_jobs: int = 1,
) -> tuple[dict[str, Any], bool, list[str], dict[str, Any]]:
    env = load_model_driven_env_from_step15(
        project_root=root, crop=crop, split="validation",
        env_config=None if max_episode_steps is None else ModelDrivenEnvConfig(max_episode_steps=max_episode_steps),
    )
    _limit_transition_threads(env, transition_n_jobs)
    indices = list(validation_episode_indices) if validation_episode_indices else list(range(len(env.episodes)))
    episodes, pairing = evaluate_paired_validation_episodes(
        env, trainer, episode_indices=indices, seed=evaluation_seed,
    )
    metrics = aggregate_validation_metrics(episodes)
    feasible, reasons = apply_safety_gate(
        metrics, severe_ood_rate_max=float(gates["severe_ood_rate_max"]),
        fallback_rate_max=float(gates["fallback_rate_max"]),
    )
    return metrics, feasible, reasons, pairing


def _completed_by_rung(records: Sequence[Mapping[str, Any]], rung: int) -> dict[int, Mapping[str, Any]]:
    return {int(record["trial_number"]): record for record in records
            if int(record["rung"]) == rung and record["status"] == "complete"}


def select_promotions(records: Sequence[Mapping[str, Any]], *, rung: int,
                      fraction: float, minimum: int = 1) -> list[int]:
    completed = list(_completed_by_rung(records, rung).values())
    feasible = [record for record in completed if record["feasible"]]
    if not feasible:
        return []
    count = max(minimum, int(math.ceil(len(completed) * fraction)))
    return [int(record["trial_number"]) for record in
            sorted(feasible, key=trial_rank_key, reverse=True)[:count]]


def run_crop_hpo(
    *, project_root: str | Path, crop: str, device: str = "cpu", resume: bool = True,
    official: bool = True, trials_override: int | None = None,
    rung_budgets_override: Sequence[int] | None = None,
    validation_episode_indices: Sequence[int] = (), transition_n_jobs: int = 1,
) -> Path:
    root = Path(project_root).resolve()
    if crop not in OFFICIAL_CROPS:
        raise ValueError(f"Unsupported crop: {crop}")
    resolved_device = _ensure_device(device)
    config = _read(root / "experiments/rl_policy_training/configs/step20_hpo_v1.json")
    if official and (trials_override is not None or rung_budgets_override is not None):
        raise ValueError("Official mode forbids reduced trial or budget overrides")
    if official and validation_episode_indices:
        raise ValueError("Official mode evaluates the complete frozen Validation episode registry")
    if not official and (trials_override is None or rung_budgets_override is None):
        raise ValueError("Smoke mode requires explicit trials and rung budgets")
    output = root / "experiments/rl_policy_training/artifacts" / ("step20" if official else "step20_smoke")
    if official and not (output / "step20_integrity_manifest.json").exists():
        prepare_step20_protocol(project_root=root)
    output.mkdir(parents=True, exist_ok=True)
    trials = int(config["trials_per_crop"] if official else trials_override)
    budgets = list(config["successive_halving"]["rungs_environment_steps"] if official else rung_budgets_override)
    if len(budgets) != 2 or budgets[0] <= 0 or budgets[1] <= budgets[0]:
        raise ValueError("Step 20 requires two increasing successive-halving budgets")
    study_id = STEP20_VERSION if official else STEP20_VERSION + ".smoke"
    database = HpoDatabase(output / "hpo_study.sqlite3", study_id)
    sampler = DeterministicSearchSpace(config["search_space"], config["sampler_seed"])
    gates = _read(root / "experiments/rl_policy_training/artifacts/step20/train_derived_safety_gates.json")["crops"][crop]
    crop_dir = output / crop
    pairing_path = crop_dir / "validation_pairing_manifest.json"
    for trial_number in range(trials):
        existing = _completed_by_rung(database.records(crop), 0)
        if trial_number in existing:
            continue
        env_probe = load_model_driven_env_from_step15(project_root=root, crop=crop, split="train")
        _limit_transition_threads(env_probe, transition_n_jobs)
        sampled = sampler.sample_trial(trial_number, env_probe.observation_shape[0])
        seed = int(config["sampler_seed"]) + trial_number
        checkpoint = crop_dir / f"trial_{trial_number:03d}" / "checkpoint.pt"
        database.upsert(crop=crop, trial_number=trial_number, rung=0, status="running",
                        config=sampled, seed=seed, budget_steps=budgets[0])
        try:
            trainer, history = _train_to_budget(
                root=root, crop=crop, config_payload=sampled, seed=seed,
                budget_steps=budgets[0], checkpoint=checkpoint,
                device=str(resolved_device), resume=resume,
                transition_n_jobs=transition_n_jobs,
            )
            metrics, feasible, reasons, pairing = _evaluate(
                root=root, crop=crop, trainer=trainer,
                validation_episode_indices=validation_episode_indices,
                evaluation_seed=int(config["sampler_seed"]), gates=gates,
                max_episode_steps=None if official else 2,
                transition_n_jobs=transition_n_jobs,
            )
            _write(pairing_path, pairing)
            _write(checkpoint.with_suffix(".rung0_metrics.json"),
                   {"history": history, "validation": metrics, "test_accessed": False})
            database.upsert(crop=crop, trial_number=trial_number, rung=0, status="complete",
                            config=sampled, seed=seed, budget_steps=budgets[0], metrics=metrics,
                            feasible=feasible, failure_reasons=reasons)
        except Exception as exc:
            database.upsert(crop=crop, trial_number=trial_number, rung=0, status="failed",
                            config=sampled, seed=seed, budget_steps=budgets[0],
                            feasible=False, failure_reasons=[type(exc).__name__, str(exc)])
            raise
    records = database.records(crop)
    promoted = select_promotions(records, rung=0,
                                 fraction=float(config["successive_halving"]["promotion_fraction"]))
    for trial_number in promoted:
        if trial_number in _completed_by_rung(database.records(crop), 1):
            continue
        base = _completed_by_rung(database.records(crop), 0)[trial_number]
        sampled, seed = base["config"], int(base["seed"])
        checkpoint = crop_dir / f"trial_{trial_number:03d}" / "checkpoint.pt"
        trainer, history = _train_to_budget(
            root=root, crop=crop, config_payload=sampled, seed=seed,
            budget_steps=budgets[1], checkpoint=checkpoint,
            device=str(resolved_device), resume=True,
            transition_n_jobs=transition_n_jobs,
        )
        metrics, feasible, reasons, pairing = _evaluate(
            root=root, crop=crop, trainer=trainer,
            validation_episode_indices=validation_episode_indices,
            evaluation_seed=int(config["sampler_seed"]), gates=gates,
            max_episode_steps=None if official else 2,
            transition_n_jobs=transition_n_jobs,
        )
        _write(checkpoint.with_suffix(".rung1_metrics.json"),
               {"history": history, "validation": metrics, "test_accessed": False})
        database.upsert(crop=crop, trial_number=trial_number, rung=1, status="complete",
                        config=sampled, seed=seed, budget_steps=budgets[1], metrics=metrics,
                        feasible=feasible, failure_reasons=reasons)
    promoted_records = [record for record in database.records(crop)
                        if record["rung"] == 1 and record["status"] == "complete"
                        and record["feasible"]]
    top_count = int(config["successive_halving"]["top_configurations_for_multi_seed"])
    top_trials = [int(record["trial_number"]) for record in
                  sorted(promoted_records, key=trial_rank_key, reverse=True)[:top_count]]
    replication_seeds = [int(value) for value in
                         config["successive_halving"]["top_configuration_seeds"]]
    for trial_number in top_trials:
        base = _completed_by_rung(database.records(crop), 1)[trial_number]
        for seed_index, seed in enumerate(replication_seeds):
            replication_rung = 10 + seed_index
            if trial_number in _completed_by_rung(database.records(crop), replication_rung):
                continue
            sampled = {**base["config"], "ppo": {**base["config"]["ppo"], "seed": seed}}
            checkpoint = crop_dir / f"trial_{trial_number:03d}" / f"replication_seed_{seed}.pt"
            trainer, history = _train_to_budget(
                root=root, crop=crop, config_payload=sampled, seed=seed,
                budget_steps=budgets[1], checkpoint=checkpoint,
                device=str(resolved_device), resume=resume,
                transition_n_jobs=transition_n_jobs,
            )
            metrics, feasible, reasons, _ = _evaluate(
                root=root, crop=crop, trainer=trainer,
                validation_episode_indices=validation_episode_indices,
                evaluation_seed=int(config["sampler_seed"]), gates=gates,
                max_episode_steps=None if official else 2,
                transition_n_jobs=transition_n_jobs,
            )
            _write(checkpoint.with_suffix(".metrics.json"),
                   {"history": history, "validation": metrics, "test_accessed": False})
            database.upsert(crop=crop, trial_number=trial_number, rung=replication_rung,
                            status="complete", config=sampled, seed=seed,
                            budget_steps=budgets[1], metrics=metrics, feasible=feasible,
                            failure_reasons=reasons)
    result_path = crop_dir / "crop_hpo_status.json"
    records = database.records(crop)
    _write(result_path, {
        "schema_version": STEP20_VERSION, "crop": crop,
        "mode": "official" if official else "smoke",
        "status": "successive_halving_complete",
        "trials_requested": trials, "rung_budgets": budgets,
        "rung0_complete": len(_completed_by_rung(records, 0)),
        "rung1_complete": len(_completed_by_rung(records, 1)),
        "promoted_trial_numbers": promoted, "device": str(resolved_device),
        "transition_n_jobs": transition_n_jobs,
        "multi_seed_trial_numbers": top_trials,
        "multi_seed_replications_complete": sum(
            len(_completed_by_rung(records, 10 + index)) for index in range(len(replication_seeds))
        ),
        "test_accessed": False, "eligible_for_selection": official,
    })
    return result_path


def _write_trial_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for record in records:
        metrics = record.get("metrics") or {}
        rows.append({
            "crop": record["crop"], "trial_number": record["trial_number"],
            "rung": record["rung"], "status": record["status"], "seed": record["seed"],
            "budget_steps": record["budget_steps"], "feasible": record["feasible"],
            "primary_objective": metrics.get("mean_reward_per_scheduled_valid_step"),
            "episodic_return_secondary": metrics.get("mean_episodic_return"),
            "severe_ood_rate": metrics.get("severe_ood_rate"),
            "fallback_rate": metrics.get("fallback_rate"),
            "warning_ood_rate": metrics.get("warning_ood_rate"),
            "action_instability": metrics.get("action_instability"),
            "failure_reasons": "|".join(record.get("failure_reasons", [])),
            "config_json": json.dumps(record["config"], sort_keys=True),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["crop"])
        writer.writeheader(); writer.writerows(rows)


def finalize_official_hpo(*, project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    output = root / "experiments/rl_policy_training/artifacts/step20"
    config = _read(root / "experiments/rl_policy_training/configs/step20_hpo_v1.json")
    database = HpoDatabase(output / "hpo_study.sqlite3", STEP20_VERSION)
    required = int(config["trials_per_crop"])
    all_records: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    completed: dict[str, int] = {}
    for crop in OFFICIAL_CROPS:
        records = database.records(crop)
        for record in records:
            record["crop"] = crop
        all_records.extend(records)
        rung0 = _completed_by_rung(records, 0)
        completed[crop] = len(rung0)
        if len(rung0) != required:
            raise RuntimeError(f"{crop}: requires {required} completed rung-0 trials, found {len(rung0)}")
        replication_seeds = config["successive_halving"]["top_configuration_seeds"]
        candidates: list[dict[str, Any]] = []
        trial_numbers = sorted({record["trial_number"] for record in records if record["rung"] >= 10})
        for trial_number in trial_numbers:
            reps = [record for record in records if record["trial_number"] == trial_number
                    and 10 <= record["rung"] < 10 + len(replication_seeds)
                    and record["status"] == "complete"]
            if len(reps) != len(replication_seeds):
                raise RuntimeError(f"{crop} trial {trial_number}: incomplete multi-seed replication")
            metric_names = reps[0]["metrics"].keys()
            metrics = {name: float(np.mean([float(rep["metrics"][name]) for rep in reps]))
                       for name in metric_names}
            feasible = all(bool(rep["feasible"]) for rep in reps)
            base_config = dict(reps[0]["config"])
            base_config["ppo"] = {**base_config["ppo"], "seed": "assigned_per_replication"}
            candidates.append({"trial_number": trial_number, "feasible": feasible,
                               "metrics": metrics, "config": base_config,
                               "replication_seeds": list(replication_seeds)})
        feasible_candidates = [record for record in candidates if record["feasible"]]
        if not feasible_candidates:
            raise RuntimeError(f"{crop}: no feasible multi-seed candidate")
        selected = max(feasible_candidates, key=trial_rank_key)
        best[crop] = {"trial_number": selected["trial_number"], "config": selected["config"],
                      "metrics": selected["metrics"], "selection_rung": "multi_seed_mean",
                      "replication_seeds": selected["replication_seeds"], "test_accessed": False}
        _write(output / crop / "best_config.json", {"schema_version": STEP20_VERSION, **best[crop]})
    table_path = output / "trial_table.csv"
    best_path = output / "best_configs.json"
    report_path = output / "hpo_report.json"
    status_path = output / "hpo_execution_status.json"
    _write_trial_csv(table_path, all_records)
    _write(best_path, {"schema_version": STEP20_VERSION, "crops": best, "test_accessed": False})
    _write(report_path, {
        "schema_version": STEP20_VERSION, "status": "passed", "crops": best,
        "primary_objective": "mean_reward_per_scheduled_valid_step",
        "episodic_return_role": "secondary_only", "safety_gate_precedes_ranking": True,
        "paired_validation": True, "test_accessed": False,
    })
    _write(status_path, {
        "schema_version": STEP20_VERSION, "status": "passed",
        "official_trials_required_per_crop": required, "completed_trials": completed,
        "best_config_selected": True, "smoke_results_eligible_for_selection": False,
        "test_accessed": False, "step21_started": False,
    })
    manifest_path = output / "step20_integrity_manifest.json"
    manifest = _read(manifest_path)
    manifest["artifacts"]["execution_status"] = {
        "path": status_path.relative_to(root).as_posix(), "sha256": _sha(status_path),
        "size_bytes": status_path.stat().st_size,
    }
    manifest.update({
        "status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count()},
        "final_artifacts": {
            "trial_table": {"path": table_path.relative_to(root).as_posix(), "sha256": _sha(table_path)},
            "best_configs": {"path": best_path.relative_to(root).as_posix(), "sha256": _sha(best_path)},
            "hpo_report": {"path": report_path.relative_to(root).as_posix(), "sha256": _sha(report_path)},
            "execution_status": {"path": status_path.relative_to(root).as_posix(), "sha256": _sha(status_path)},
        },
        "checks": {"test_accessed": False, "step21_started": False,
                   "official_hpo_complete": True, "best_config_selected": True,
                   "length_normalized_objective": True, "safety_gate_applied": True},
    })
    _write(manifest_path, manifest)
    return manifest_path


__all__ = ["finalize_official_hpo", "run_crop_hpo", "select_promotions"]
