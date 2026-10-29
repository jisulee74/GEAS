"""Isolated MLP random-restart orchestration for transition experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from geas35.experiments.transition.config import (
    load_transition_experiment_config,
    read_configured_transition_frames,
)
from geas35.experiments.transition.runner import run_transition_model_experiment
from geas35.models.transition.action_response import evaluate_action_response
from geas35.models.transition.datasets import build_transition_dataset_from_rl_frame
from geas35.models.transition.evaluator import evaluate_transition_model_one_step
from geas35.models.transition.feature_variants import (
    feature_variant_contract,
    observation_columns_for_variant,
    prepare_frame_for_feature_variant,
)
from geas35.models.transition.hyperparameters import (
    transition_target_normalization_scales,
    validation_rollout_weighted_score,
)
from geas35.models.transition.registry import build_transition_model
from geas35.models.transition.resources import benchmark_transition_resources
from geas35.models.transition.trainer import _configure_target_formulation, _evaluate_rollouts
from geas35.rl import MDP_V1_ACTION_COLUMNS, fit_mdp_v1_observation_scaler


SELECTION_RULE = "median_validation_score"
RESTART_MODEL = "mlp"
PROTOCOL_VERSION = "geas35.transition.random_restart.v1.3"


@dataclass(frozen=True)
class RandomRestartConfig:
    source_path: Path
    base_config: Path
    output_root: Path
    seeds: tuple[int, ...] = (42, 43, 44)
    restart_models: tuple[str, ...] = (RESTART_MODEL,)
    selection_rule: str = SELECTION_RULE


def load_random_restart_config(path: str | Path) -> RandomRestartConfig:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Random-restart config must be a mapping.")
    base = _resolve(source.parent, payload.get("base_config"))
    output = _resolve(source.parent, payload.get("output_root"))
    seeds = tuple(int(value) for value in payload.get("seeds", (42, 43, 44)))
    models = tuple(str(value) for value in payload.get("restart_models", (RESTART_MODEL,)))
    rule = str(payload.get("selection_rule", SELECTION_RULE))
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("seeds must be a non-empty unique sequence.")
    if models != (RESTART_MODEL,):
        raise ValueError("v1.3 permits random restarts for MLP only.")
    if rule != SELECTION_RULE:
        raise ValueError(f"Unsupported selection rule: {rule}")
    return RandomRestartConfig(source, base, output, seeds, models, rule)


def run_random_restart_experiment(
    config_path: str | Path,
    *,
    preflight_only: bool = False,
) -> Mapping[str, Any]:
    require_no_legacy_processes()
    config = load_random_restart_config(config_path)
    snapshot = build_snapshot_manifest(_experiment_root())
    loaded = load_transition_experiment_config(config.base_config)
    train, validation, _, rollout = read_configured_transition_frames(
        loaded, include_test=False
    )
    variant = loaded.experiment_config.feature_variant
    if variant is None:
        raise ValueError("v1.3 requires an explicit v1.2 feature variant.")
    train = prepare_frame_for_feature_variant(train, variant)
    validation = prepare_frame_for_feature_variant(validation, variant)
    rollout = validation if rollout is None else prepare_frame_for_feature_variant(rollout, variant)
    observation_columns = observation_columns_for_variant(variant)
    train_build = build_transition_dataset_from_rl_frame(
        train, observation_columns=observation_columns
    )
    validation_build = build_transition_dataset_from_rl_frame(
        validation, observation_columns=observation_columns
    )
    diagnostic = finite_preflight(
        train_build.dataset.x,
        train_build.dataset.y,
        validation_build.dataset.x,
        validation_build.dataset.y,
    )
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_root / "preflight.json", diagnostic)
    if not diagnostic["passed"]:
        raise ValueError(f"Non-finite model input detected; see {config.output_root / 'preflight.json'}")
    if preflight_only:
        return diagnostic

    scaler = fit_mdp_v1_observation_scaler(train, observation_columns=observation_columns)
    scaler_payload = {
        "columns": list(scaler.columns),
        "mean": dict(scaler.means),
        "scale": dict(scaler.scales),
        "fit_split": "train",
        "feature_variant": variant,
    }
    scaler_hash = _sha256_json(scaler_payload)
    _write_json(config.output_root / "observation_scaler.json", scaler_payload)

    base_output = config.output_root / "base_experiment"
    base_experiment_config = replace(
        loaded.experiment_config,
        output_root=base_output,
        resume_completed=True,
        provenance={
            **dict(loaded.experiment_config.provenance),
            "random_restart_protocol": PROTOCOL_VERSION,
            "snapshot_sha256": snapshot["snapshot_sha256"],
        },
    )
    base_result = run_transition_model_experiment(
        config=base_experiment_config,
        train_df=train,
        validation_df=validation,
        rollout_df=rollout,
        observation_scaler=scaler,
        observation_scaler_hash=scaler_hash,
    )
    mlp_result = next(
        result for result in base_result.training_result.candidate_results
        if result.model_name == RESTART_MODEL
    )
    best_params = dict(mlp_result.hpo_result.best_params)
    hpo_dir = config.output_root / RESTART_MODEL / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    _write_json(hpo_dir / "hpo_results.json", mlp_result.hpo_result.to_artifact())
    _write_json(hpo_dir / "best_config.json", {"best_config": best_params})

    normalization = transition_target_normalization_scales(
        validation_build.dataset.y,
        validation_build.dataset.target_columns,
    )
    contract = feature_variant_contract(variant, action_columns=MDP_V1_ACTION_COLUMNS)
    shared = {
        "protocol_version": PROTOCOL_VERSION,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "dataset_hashes": dict(loaded.experiment_config.provenance.get("dataset_sha256", {})),
        "schema_hash": contract["schema_hash"],
        "scaler_hash": scaler_hash,
        "hyperparameter_hash": _sha256_json(best_params),
        "feature_variant": variant,
        "crop": loaded.experiment_config.crop,
        "test_opened": False,
    }
    summaries = []
    for seed in config.seeds:
        summaries.append(
            _run_or_resume_seed(
                output_root=config.output_root,
                seed=seed,
                best_params=best_params,
                shared=shared,
                train_dataset=train_build.dataset,
                validation_dataset=validation_build.dataset,
                rollout_frame=rollout,
                scaler=scaler,
                normalization_scales=normalization,
                experiment_config=base_experiment_config,
            )
        )
    aggregate = aggregate_restart_results(summaries)
    _write_csv(config.output_root / RESTART_MODEL / "restart_summary.csv", summaries)
    _write_json(config.output_root / RESTART_MODEL / "restart_summary.json", aggregate)
    _write_json(config.output_root / RESTART_MODEL / "selected_seed.json", aggregate["selected"])
    _write_json(
        config.output_root / "run_manifest.json",
        {
            **shared,
            "seeds": list(config.seeds),
            "restart_models": list(config.restart_models),
            "selection_rule": config.selection_rule,
            "selected_seed": aggregate["selected"]["seed"],
            "completed_at_utc": _utc_now(),
        },
    )
    return aggregate


def _run_or_resume_seed(
    *, output_root: Path, seed: int, best_params: Mapping[str, Any],
    shared: Mapping[str, Any], train_dataset: Any, validation_dataset: Any,
    rollout_frame: pd.DataFrame, scaler: Any,
    normalization_scales: Mapping[str, float], experiment_config: Any,
) -> Mapping[str, Any]:
    seed_dir = output_root / RESTART_MODEL / "restarts" / f"seed_{seed}"
    expected = {**dict(shared), "seed": int(seed)}
    completed = seed_dir / "completed.json"
    if completed.is_file():
        payload = json.loads(completed.read_text(encoding="utf-8"))
        if all(payload.get(key) == value for key, value in expected.items()):
            return json.loads((seed_dir / "summary.json").read_text(encoding="utf-8"))
        quarantine = seed_dir.with_name(f"{seed_dir.name}.quarantine.{int(time.time())}")
        seed_dir.rename(quarantine)
    elif seed_dir.exists():
        quarantine = seed_dir.with_name(f"{seed_dir.name}.quarantine.{int(time.time())}")
        seed_dir.rename(quarantine)
    seed_dir.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    params = _with_seed(best_params, seed)
    model = build_transition_model(RESTART_MODEL, params=params)
    _configure_target_formulation(model, scaler, experiment_config.target_formulation)
    started = time.perf_counter()
    model.fit(train_dataset, validation_dataset=validation_dataset)
    training_seconds = time.perf_counter() - started
    one_step = evaluate_transition_model_one_step(
        model, validation_dataset, normalization_scales=normalization_scales
    )
    rollout = _evaluate_rollouts(
        model,
        rollout_frame=rollout_frame,
        rollout_horizon_steps=experiment_config.rollout_horizon_steps,
        rollout_step_minutes=experiment_config.rollout_step_minutes,
        action_provider_factory=None,
        observation_scaler=scaler,
        hpo_mode=False,
        normalization_scales=normalization_scales,
    )
    score, components = validation_rollout_weighted_score(
        one_step, rollout,
        objective_weights=experiment_config.hpo_config.objective_weights,
    )
    action = evaluate_action_response(model, validation_dataset)
    safety = all(float(row.get("safety_gate_passed", 0.0)) == 1.0 for row in rollout.values())
    with (seed_dir / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream)
    resources = benchmark_transition_resources(
        model, validation_dataset, artifact_dir=seed_dir,
        model_path=seed_dir / "model.pkl", training_time_seconds=training_seconds,
        hpo_total_time_seconds=0.0,
    )
    summary = {
        "seed": int(seed),
        "validation_rollout_weighted_score": float(score),
        "physical_violation_rate": _max_rollout_metric(rollout, "physical_violation_rate"),
        "cvar90": float(one_step.aggregate_metrics.get("mean_cvar90", np.nan)),
        "safety_passed": bool(safety),
        "action_response_eligible": bool(action.eligible),
        "deployment_eligible": bool(safety and action.eligible and np.isfinite(score)),
        "training_time_seconds": float(training_seconds),
        "one_step": one_step.to_artifact(),
        "rollout": rollout,
        "action_response": action.to_artifact(),
        "resources": dict(resources),
        "objective_components": components,
        "params": params,
    }
    _write_json(seed_dir / "summary.json", summary)
    _write_json(seed_dir / "one_step_metrics.json", one_step.to_artifact())
    _write_json(seed_dir / "rollout_metrics.json", rollout)
    _write_json(seed_dir / "action_response.json", action.to_artifact())
    _write_json(seed_dir / "resource_metrics.json", resources)
    _write_json(seed_dir / "manifest.json", expected)
    _atomic_write_json(seed_dir / "completed.json", {**expected, "completed_at_utc": _utc_now()})
    return summary


def aggregate_restart_results(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [row for row in rows if bool(row["deployment_eligible"])]
    if not eligible:
        raise ValueError("No MLP restart passed safety and action-response gates.")
    ordered = sorted(
        eligible,
        key=lambda row: (
            float(row["validation_rollout_weighted_score"]),
            float(row["physical_violation_rate"]),
            float(row["cvar90"]),
            int(row["seed"]),
        ),
    )
    selected = ordered[(len(ordered) - 1) // 2]
    scores = np.asarray(
        [float(row["validation_rollout_weighted_score"]) for row in rows], dtype=float
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "selection_rule": SELECTION_RULE,
        "seed_count": len(rows),
        "score_statistics": {
            "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=0)),
            "min": float(np.min(scores)), "max": float(np.max(scores)),
        },
        "selected": {
            "seed": int(selected["seed"]),
            "validation_rollout_weighted_score": float(selected["validation_rollout_weighted_score"]),
            "checkpoint": f"restarts/seed_{selected['seed']}/model.pkl",
        },
        "runs": list(rows),
        "test_used": False,
    }



def build_cross_experiment_report(artifacts_root: str | Path) -> Mapping[str, Any]:
    """Aggregate all six runs and pair with/without-flag rollout errors."""
    root = Path(artifacts_root)
    crops = ("strawberry", "melon", "cucumber")
    variants = ("deduplicated_with_quality_flags", "deduplicated_without_quality_flags")
    runs: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for crop in crops:
        crop_runs = {}
        for variant in variants:
            path = root / variant / crop / RESTART_MODEL / "restart_summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed restart summary: {path}")
            crop_runs[variant] = json.loads(path.read_text(encoding="utf-8"))
        runs[crop] = {
            variant: {
                "selected_seed": crop_runs[variant]["selected"]["seed"],
                "score_statistics": crop_runs[variant]["score_statistics"],
            }
            for variant in variants
        }
        comparisons[crop] = _paired_quality_flag_comparison(
            crop_runs[variants[0]]["runs"], crop_runs[variants[1]]["runs"]
        )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": _utc_now(),
        "runs": runs,
        "quality_flag_ablation": comparisons,
        "all_six_completed": True,
        "test_used": False,
    }
    _write_json(root / "ablation_comparison" / "random_restart_comparison.json", report)
    return report


def _paired_quality_flag_comparison(
    with_rows: Sequence[Mapping[str, Any]], without_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    with_by_seed = {int(row["seed"]): row for row in with_rows}
    without_by_seed = {int(row["seed"]): row for row in without_rows}
    if set(with_by_seed) != set(without_by_seed):
        raise ValueError("Quality-flag comparison requires identical restart seeds.")
    horizons: dict[str, Any] = {}
    all_differences = []
    for horizon in ("15min", "30min", "60min"):
        differences = []
        for seed in sorted(with_by_seed):
            left = with_by_seed[seed]["rollout"][horizon]["window_trajectory_nrmse_samples"]
            right = without_by_seed[seed]["rollout"][horizon]["window_trajectory_nrmse_samples"]
            if len(left) != len(right):
                raise ValueError(f"Paired windows differ for seed={seed}, horizon={horizon}.")
            differences.extend(float(a) - float(b) for a, b in zip(left, right, strict=True))
        horizons[horizon] = _paired_bootstrap(differences, seed=42)
        all_differences.extend(differences)
    return {
        "difference": "with_flags_minus_without_flags",
        "horizons": horizons,
        "overall": _paired_bootstrap(all_differences, seed=42),
    }


def _paired_bootstrap(
    differences: Sequence[float], *, seed: int, replicates: int = 5000,
) -> Mapping[str, Any]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"status": "insufficient_support", "count": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    effect = "improved" if high < 0 else "degraded" if low > 0 else "no_significant_difference"
    return {
        "status": "completed", "count": int(len(values)),
        "mean_difference": float(values.mean()),
        "ci95_low": float(low), "ci95_high": float(high),
        "effect": effect, "bootstrap_replicates": int(replicates),
    }


def finite_preflight(*frames: pd.DataFrame) -> Mapping[str, Any]:
    problems = []
    labels = ("train_x", "train_y", "validation_x", "validation_y")
    for label, frame in zip(labels, frames, strict=True):
        array = frame.to_numpy(dtype=float)
        mask = ~np.isfinite(array)
        for column_index in np.flatnonzero(mask.any(axis=0)):
            positions = np.flatnonzero(mask[:, column_index])[:20]
            problems.append({
                "split_component": label,
                "column": str(frame.columns[column_index]),
                "count": int(mask[:, column_index].sum()),
                "row_positions": [int(value) for value in positions],
            })
    return {"passed": not problems, "problems": problems, "test_opened": False}


def require_no_legacy_processes() -> None:
    own_pid = os.getpid()
    offenders = []
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            arguments = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        legacy_runner = "experiments/transition_model_selection/scripts/run.py"
        if legacy_runner in arguments:
            offenders.append({"pid": int(entry.name), "command": " ".join(arguments)})
    if offenders:
        raise RuntimeError(f"Legacy v1.2 transition experiment is still running: {offenders}")


def build_snapshot_manifest(experiment_root: Path) -> Mapping[str, Any]:
    runtime_root = experiment_root / "runtime_src" / "geas35"
    files = []
    for path in sorted(runtime_root.rglob("*.py")):
        files.append({
            "path": str(path.relative_to(experiment_root)),
            "source_path": str(Path("src/geas35") / path.relative_to(runtime_root)),
            "sha256": _sha256_file(path),
        })
    digest = _sha256_json(files)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": _utc_now(),
        "git_commit": _git_commit(experiment_root),
        "snapshot_sha256": digest,
        "files": files,
    }
    _write_json(experiment_root / "snapshot_manifest.json", payload)
    return payload


def _with_seed(value: Any, seed: int) -> Any:
    if not isinstance(value, Mapping):
        raise TypeError("MLP best_params must be a mapping.")
    result = {str(key): item for key, item in value.items()}
    result["random_state"] = int(seed)
    target_overrides = result.get("target_estimator_kwargs")
    if isinstance(target_overrides, Mapping):
        result["target_estimator_kwargs"] = {
            str(target): {**dict(params), "random_state": int(seed)}
            for target, params in target_overrides.items()
        }
    return result


def _max_rollout_metric(rollout: Mapping[str, Mapping[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rollout.values()]
    return float(max(values, default=0.0))


def _experiment_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve(parent: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("Required path is missing.")
    path = Path(str(value)).expanduser()
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flat = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(path, index=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_artifact"):
        return value.to_artifact()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _git_commit(experiment_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=experiment_root,
        check=False, capture_output=True, text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "RandomRestartConfig", "aggregate_restart_results", "build_cross_experiment_report",
    "build_snapshot_manifest",
    "finite_preflight", "load_random_restart_config", "require_no_legacy_processes",
    "run_random_restart_experiment",
]
