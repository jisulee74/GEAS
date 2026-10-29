"""Step 13 integrity and Validation reproducibility review for v1.4 artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from geas35.experiments.transition.config import (
    load_transition_experiment_config,
    read_configured_transition_frames,
)
from geas35.io_utils import write_json
from geas35.models.transition.causal_state import load_frozen_observation_scaler
from geas35.models.transition.datasets import build_transition_dataset_from_rl_frame
from geas35.models.transition.evaluator import evaluate_transition_model_one_step
from geas35.models.transition.feature_variants import (
    observation_columns_for_variant,
    prepare_frame_for_feature_variant,
    project_frozen_observation_scaler,
)
from geas35.models.transition.hyperparameters import transition_target_normalization_scales
from geas35.models.transition.inference import load_transition_model_from_artifact_dir
from geas35.models.transition.trainer import _evaluate_rollouts


STEP13_V1_4_VERSION = "geas35.transition.step13.v1_4"
OFFICIAL_CROPS = ("strawberry", "melon", "cucumber")
OFFICIAL_VARIANTS = (
    "deduplicated_with_quality_flags",
    "deduplicated_without_quality_flags",
)
SELECTED_CANDIDATES = {
    "strawberry": ("deduplicated_without_quality_flags", "lightgbm"),
    "melon": ("deduplicated_without_quality_flags", "xgboost"),
    "cucumber": ("deduplicated_without_quality_flags", "lightgbm"),
}
REQUIRED_CANDIDATE_FILES = (
    "model.pkl", "manifest.json", "feature_schema.json", "one_step_metrics.json",
    "rollout_metrics.json", "resource_metrics.json", "training_summary.json",
    "hpo_results.json", "best_config.json", "action_response.json",
)


class Step13V14ReviewError(ValueError):
    """Raised when a frozen v1.4 artifact violates the Step 13 contract."""


def run_step13_v1_4_reproducibility_review(
    *, project_root: str | Path, official_root: str | Path | None = None,
    output_root: str | Path | None = None, metric_rtol: float = 1e-9,
    metric_atol: float = 1e-11,
) -> Path:
    """Review the six official combinations without opening Test data."""

    root = Path(project_root).expanduser().resolve()
    source = _resolve(official_root, root / "experiments/transition_model_selection/artifacts/step12_v1_4_root_cause/official_rerun")
    destination = _resolve(output_root, root / "experiments/transition_model_selection/artifacts/step13_v1_4")
    _require(source.is_dir(), f"Official v1.4 root not found: {source}")
    combined = _read_json(source / "combined_status.json")
    _require(isinstance(combined, list) and len(combined) == 6, "combined_status.json must contain six combinations")

    combination_reviews: dict[str, Any] = {}
    artifact_hashes: dict[str, Any] = {}
    reviewed = 0
    for variant in OFFICIAL_VARIANTS:
        expected_observations = 57 if variant.endswith("with_quality_flags") else 55
        expected_dimension = expected_observations + 6
        config_suffix = "with_flags" if variant.endswith("with_quality_flags") else "without_flags"
        for crop in OFFICIAL_CROPS:
            key = f"{crop}/{variant}"
            crop_root = source / variant / crop
            _require(crop_root.is_dir(), f"Missing official combination: {key}")
            status = _read_json(crop_root / "official_status.json")
            _require(status.get("status") in {"completed", "no_eligible_seed"}, f"{key}: invalid official status")
            _assert_test_locked(status, crop_root / "official_status.json")
            integrity = _read_json(crop_root / "artifact_integrity.json")
            _require(integrity.get("passed") is True, f"{key}: artifact integrity did not pass")
            output_config = _read_json(crop_root / "config.json")
            contract = _mapping(output_config.get("feature_contract"), f"{key}: feature contract")
            _require(contract.get("feature_variant") == variant, f"{key}: feature variant mismatch")
            _require(int(contract.get("observation_count", -1)) == expected_observations, f"{key}: observation count mismatch")
            _require(int(contract.get("input_dimension", -1)) == expected_dimension, f"{key}: input dimension mismatch")

            config_path = root / "experiments/transition_model_selection/configs" / f"{crop}_v1_4_{config_suffix}.yaml"
            loaded = load_transition_experiment_config(config_path)
            _require(loaded.experiment_config.feature_variant == variant, f"{key}: source config variant mismatch")
            _require(loaded.experiment_config.target_formulation == "residual_physical_delta_v1", f"{key}: target formulation mismatch")
            _require(loaded.experiment_config.provenance.get("test_opened_during_selection") is False, f"{key}: Test-lock provenance mismatch")

            candidates_root = crop_root / crop
            candidates = sorted(path for path in candidates_root.iterdir() if path.is_dir())
            _require(candidates, f"{key}: no candidate artifacts")
            candidate_hashes: dict[str, Any] = {}
            cold_load: dict[str, Any] = {}
            for artifact_dir in candidates:
                model_name = artifact_dir.name
                hashes = {}
                for filename in REQUIRED_CANDIDATE_FILES:
                    path = artifact_dir / filename
                    _require(path.is_file(), f"{key}/{model_name}: missing {filename}")
                    hashes[filename] = _file_record(path, root)
                manifest = _read_json(artifact_dir / "manifest.json")
                schema = _read_json(artifact_dir / "feature_schema.json")
                training = _read_json(artifact_dir / "training_summary.json")
                action = _read_json(artifact_dir / "action_response.json")
                _require(manifest.get("feature_variant") == variant, f"{key}/{model_name}: manifest variant mismatch")
                _require(manifest.get("schema_hash") == contract.get("schema_hash"), f"{key}/{model_name}: schema hash mismatch")
                _require(manifest.get("target_formulation") == "residual_physical_delta_v1", f"{key}/{model_name}: target formulation mismatch")
                _require(tuple(schema.get("input_columns", ())) == tuple(training.get("input_columns", ())), f"{key}/{model_name}: feature order mismatch")
                _assert_test_locked(training, artifact_dir / "training_summary.json")
                _assert_test_locked(action, artifact_dir / "action_response.json")
                bundle = load_transition_model_from_artifact_dir(
                    artifact_dir, expected_target_formulation="residual_physical_delta_v1"
                )
                cold_load[model_name] = {
                    "status": "passed",
                    "model_class": f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}",
                    "deployment_eligible": bool(manifest.get("deployment_eligible", False)),
                }
                candidate_hashes[model_name] = hashes
                reviewed += 1

            selected_variant, selected_model = SELECTED_CANDIDATES[crop]
            recomputation = None
            if variant == selected_variant:
                cache_path = destination / "recomputation_cache" / f"{crop}.json"
                cache_key = {
                    "model_sha256": _sha256(candidates_root / selected_model / "model.pkl"),
                    "validation_sha256": loaded.experiment_config.provenance["dataset_sha256"]["validation"],
                    "projected_scaler_sha256": _canonical_json_sha256(crop_root / "observation_scaler.json"),
                }
                cached = _read_json(cache_path) if cache_path.is_file() else None
                if isinstance(cached, Mapping) and cached.get("cache_key") == cache_key:
                    recomputation = dict(cached["result"])
                    recomputation["resumed_from_cache"] = True
                else:
                    recomputation = _recompute_selected(
                        loaded=loaded, artifact_dir=candidates_root / selected_model,
                        projected_scaler_path=crop_root / "observation_scaler.json",
                        metric_rtol=metric_rtol, metric_atol=metric_atol,
                    )
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(cache_path, {"cache_key": cache_key, "result": recomputation}, convert=True)
            combination_reviews[key] = {
                "status": "passed", "official_status": status.get("status"),
                "artifact_integrity": True, "config": _file_record(config_path, root),
                "schema_hash": contract.get("schema_hash"),
                "observation_count": expected_observations, "input_dimension": expected_dimension,
                "candidate_count": len(candidates), "cold_load": cold_load,
                "selected_candidate_recomputation": recomputation,
                "test_opened": False,
            }
            artifact_hashes[key] = candidate_hashes

    destination.mkdir(parents=True, exist_ok=True)
    dependency_path = destination / "step13_v1_4_dependency_lock.json"
    hash_path = destination / "step13_v1_4_candidate_artifact_hashes.json"
    write_json(dependency_path, _dependency_lock(), convert=True)
    write_json(hash_path, {"schema_version": STEP13_V1_4_VERSION, "artifacts": artifact_hashes}, convert=True)
    manifest = {
        "schema_version": STEP13_V1_4_VERSION, "step": 13, "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "official_root": _relative(source, root),
        "combined_status": _file_record(source / "combined_status.json", root),
        "dependency_lock": _file_record(dependency_path, root),
        "candidate_artifact_hashes": _file_record(hash_path, root),
        "reviewed_combination_count": 6, "reviewed_candidate_count": reviewed,
        "selected_candidates": {crop: {"feature_variant": value[0], "model": value[1]} for crop, value in SELECTED_CANDIDATES.items()},
        "all_candidate_cold_load": "passed", "selected_validation_recomputation": "passed",
        "test_opened": False, "step14_handoff_generated": False,
        "combinations": combination_reviews,
    }
    manifest_path = destination / "step13_v1_4_reproducibility_manifest.json"
    write_json(manifest_path, manifest, convert=True)
    (destination / "step13_v1_4_reproducibility_summary.md").write_text(_summary(manifest), encoding="utf-8")
    return manifest_path


def _recompute_selected(*, loaded: Any, artifact_dir: Path, projected_scaler_path: Path,
                        metric_rtol: float, metric_atol: float) -> dict[str, Any]:
    train, validation, test, rollout = read_configured_transition_frames(loaded, include_test=False)
    _require(test is None, "Test frame was opened during Step 13")
    variant = str(loaded.experiment_config.feature_variant)
    validation = prepare_frame_for_feature_variant(validation, variant)
    rollout = validation if rollout is None else prepare_frame_for_feature_variant(rollout, variant)
    observations = observation_columns_for_variant(variant)
    original_scaler, source_scaler_hash = load_frozen_observation_scaler(loaded.scaler_path)
    scaler = project_frozen_observation_scaler(original_scaler, observations)
    dataset = build_transition_dataset_from_rl_frame(validation, observation_columns=observations).dataset
    scales = transition_target_normalization_scales(dataset.y, dataset.target_columns)
    model = load_transition_model_from_artifact_dir(
        artifact_dir, expected_target_formulation="residual_physical_delta_v1"
    ).model
    one = evaluate_transition_model_one_step(model, dataset, normalization_scales=scales).to_artifact()
    multi = _evaluate_rollouts(
        model, rollout_frame=rollout, rollout_horizon_steps=(3, 6, 12),
        rollout_step_minutes=5, action_provider_factory=None,
        normalization_scales=scales, observation_scaler=scaler, hpo_mode=False,
    )
    stored_one = _read_json(artifact_dir / "one_step_metrics.json")
    stored_multi = _read_json(artifact_dir / "rollout_metrics.json")
    _compare_numeric_tree(stored_one.get("aggregate_metrics"), one.get("aggregate_metrics"), "one_step.aggregate", metric_rtol, metric_atol)
    for horizon in ("15min", "30min", "60min"):
        _compare_numeric_tree(stored_multi[horizon], multi[horizon], f"rollout.{horizon}", metric_rtol, metric_atol)
    training = _read_json(artifact_dir / "training_summary.json")
    _require(training.get("observation_scaler_sha256") == _canonical_json_sha256(projected_scaler_path), "selected model projected scaler hash mismatch")
    return {
        "status": "passed", "model": artifact_dir.name,
        "validation_rows": int(len(dataset.x)),
        "rollout_window_count_60min": int(multi["60min"]["window_count"]),
        "source_scaler_sha256": source_scaler_hash,
        "one_step_aggregate": one["aggregate_metrics"],
        "rollout_summary": {h: {k: multi[h][k] for k in ("trajectory_nrmse", "trajectory_rmse", "trajectory_r2", "physical_violation_rate", "nan_inf_count", "window_count")} for h in ("15min", "30min", "60min")},
        "test_opened": False,
    }


def _compare_numeric_tree(expected: Any, actual: Any, label: str, rtol: float, atol: float) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key, expected_value in expected.items():
            if key not in actual or key in {"window_ids", "window_trajectory_nrmse_samples", "window_drift_nmae_samples"}:
                continue
            _compare_numeric_tree(expected_value, actual[key], f"{label}.{key}", rtol, atol)
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isnan(float(expected)) and math.isnan(float(actual)):
            return
        _require(bool(np.isclose(float(expected), float(actual), rtol=rtol, atol=atol)), f"metric mismatch: {label}: {expected} != {actual}")


def _assert_test_locked(payload: Mapping[str, Any], path: Path) -> None:
    for key, value in _walk(payload):
        normalized = key.lower()
        if ("test_opened" in normalized or "test_used" in normalized) and value is not False:
            raise Step13V14ReviewError(f"Test-lock violation in {path}: {key}={value!r}")


def _walk(value: Any, prefix: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, item
            yield from _walk(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{prefix}[{index}]")


def _dependency_lock() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def _summary(manifest: Mapping[str, Any]) -> str:
    lines = ["# Step 13 v1.4 Reproducibility Review", "", "- Status: passed", "- Test opened: no", f"- Reviewed combinations: {manifest['reviewed_combination_count']}", f"- Reviewed candidate artifacts: {manifest['reviewed_candidate_count']}", "", "## Selected candidates revalidated", ""]
    for crop, choice in manifest["selected_candidates"].items():
        lines.append(f"- {crop}: {choice['model']} ({choice['feature_variant']}) — cold-load and full Validation metric recomputation passed")
    lines += ["", "Step 14 deployment handoff has not been generated.", ""]
    return "\n".join(lines)


def _resolve(value: str | Path | None, default: Path) -> Path:
    return (default if value is None else Path(value)).expanduser().resolve()


def _read_json(path: Path) -> Any:
    _require(path.is_file(), f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _canonical_json_sha256(path: Path) -> str:
    payload = _read_json(path)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": _relative(path, root), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Step13V14ReviewError(message)


__all__ = ["STEP13_V1_4_VERSION", "Step13V14ReviewError", "run_step13_v1_4_reproducibility_review"]
