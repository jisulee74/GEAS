"""Step 12 all-crop real-data transition candidate benchmark."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from geas35.experiments.transition.config import load_transition_experiment_config
from geas35.experiments.transition.cli import run_from_config
from geas35.io_utils import write_json
from geas35.models.transition.comparison import (
    ARTIFACT_INTEGRITY_JSON,
    CANDIDATE_METRICS_CSV,
    CANDIDATE_RESOURCE_METRICS_CSV,
    CANDIDATE_ROLLOUT_METRICS_CSV,
    CANDIDATE_TEST_METRICS_CSV,
    TRANSITION_SUMMARY_MD,
)
from geas35.models.transition.features import OFFICIAL_TRANSITION_TARGET_COLUMNS
from geas35.models.transition.registry import TransitionModelFactory


STEP12_VERSION = "geas35.transition.step12.v1"
STEP12_MANIFEST_NAME = "step12_benchmark_manifest.json"
OFFICIAL_CROPS = ("strawberry", "melon", "cucumber")
OFFICIAL_CANDIDATES = (
    "persistence", "linear_regression", "linear_svr", "knn",
    "extra_trees", "lightgbm", "xgboost", "mlp",
)
OFFICIAL_TARGETS = OFFICIAL_TRANSITION_TARGET_COLUMNS
TABLES = {
    "candidate_metrics": CANDIDATE_METRICS_CSV,
    "rollout_metrics": CANDIDATE_ROLLOUT_METRICS_CSV,
    "resource_metrics": CANDIDATE_RESOURCE_METRICS_CSV,
    "test_metrics": CANDIDATE_TEST_METRICS_CSV,
}


class Step12BenchmarkError(ValueError):
    """Raised when the Step 12 benchmark contract is violated."""


@dataclass(frozen=True)
class Step12BenchmarkResult:
    manifest_path: str
    output_root: str
    crop_output_roots: Mapping[str, str]
    combined_tables: Mapping[str, str]


def run_step12_real_data_benchmark(
    *,
    project_root: str | Path,
    config_paths: Sequence[str | Path] | None = None,
    handoff_path: str | Path | None = None,
    output_root: str | Path | None = None,
    registry: Mapping[str, TransitionModelFactory] | None = None,
) -> Step12BenchmarkResult:
    """Validate Step 11.6 provenance, run all crops, and publish Step 12 tables."""

    root = Path(project_root).expanduser().resolve()
    configs = tuple(config_paths or (
        root / "experiments/transition_model_selection/configs/strawberry.yaml",
        root / "experiments/transition_model_selection/configs/melon.yaml",
        root / "experiments/transition_model_selection/configs/cucumber.yaml",
    ))
    destination = _resolved(
        output_root,
        root / "experiments/transition_model_selection/artifacts/step12",
    )
    handoff_file = _resolved(
        handoff_path,
        root / "offline_dataset_preparation/datasets/5_rl_dataset/step12_handoff.json",
    )
    immutable_path, immutable_hash, immutable = _verify_handoff(root, handoff_file)
    loaded_by_crop = _validate_configs(
        root, configs, immutable_path=immutable_path,
        immutable_hash=immutable_hash, immutable=immutable,
    )
    _validate_runtime_dependencies(registry)

    crop_outputs: dict[str, str] = {}
    for crop in OFFICIAL_CROPS:
        result = run_from_config(loaded_by_crop[crop].source_path, registry=registry)
        crop_root = Path(result.output_root).resolve()
        _validate_crop_outputs(crop_root, crop=crop)
        crop_outputs[crop] = str(crop_root)

    destination.mkdir(parents=True, exist_ok=True)
    combined = _combine_tables(destination, crop_outputs)
    manifest_path = destination / STEP12_MANIFEST_NAME
    manifest = {
        "schema_version": STEP12_VERSION,
        "step": "12",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "automatic_model_selection": False,
        "candidate_order": list(OFFICIAL_CANDIDATES),
        "target_columns": list(OFFICIAL_TARGETS),
        "test_used_for_hpo": False,
        "test_used_for_validation_ranking": False,
        "test_used_for_selection": False,
        "step11_immutable_rl_manifest": {
            "path": _relative_or_absolute(immutable_path, root),
            "sha256": immutable_hash,
        },
        "crops": {
            crop: {
                "config_path": _relative_or_absolute(loaded_by_crop[crop].source_path, root),
                "config_sha256": _sha256(loaded_by_crop[crop].source_path),
                "output_root": _relative_or_absolute(Path(crop_outputs[crop]), root),
                "candidate_count": len(OFFICIAL_CANDIDATES),
                "status": "passed",
            }
            for crop in OFFICIAL_CROPS
        },
        "combined_tables": {
            name: _relative_or_absolute(Path(path), root)
            for name, path in combined.items()
        },
    }
    write_json(manifest_path, manifest, convert=True)
    return Step12BenchmarkResult(
        manifest_path=str(manifest_path), output_root=str(destination),
        crop_output_roots=crop_outputs, combined_tables=combined,
    )


def _verify_handoff(root: Path, path: Path) -> tuple[Path, str, dict[str, Any]]:
    handoff = _read_json(path, "Step 12 handoff")
    _require(handoff.get("status") == "ready", "Step 12 handoff is not ready")
    _require(handoff.get("integrity_gate") == "passed", "Step 11.6 integrity gate is not passed")
    record = _mapping(handoff.get("immutable_rl_manifest"), "immutable_rl_manifest")
    manifest_path = Path(str(record.get("path", "")))
    if not manifest_path.is_absolute():
        manifest_path = (root / manifest_path).resolve()
    expected_hash = str(record.get("sha256", ""))
    _require(expected_hash and _sha256(manifest_path) == expected_hash,
             "Step 11 immutable RL manifest hash does not match the handoff")
    manifest = _read_json(manifest_path, "Step 11.6 immutable manifest")
    _require(manifest.get("status") == "passed", "Step 11.6 manifest is not passed")
    _require(manifest.get("all_crop_integrity_gate") == "passed",
             "Step 11.6 all-crop integrity gate is not passed")
    return manifest_path, expected_hash, manifest


def _validate_runtime_dependencies(
    registry: Mapping[str, TransitionModelFactory] | None,
) -> None:
    """Fail before writing partial crop outputs when optional models are absent."""
    if registry is not None:
        return
    missing = [package for package in ("lightgbm", "xgboost")
               if importlib.util.find_spec(package) is None]
    _require(not missing, "Step 12 requires every official candidate dependency; missing: "
             + ", ".join(missing))


def _validate_configs(root: Path, paths: Sequence[str | Path], *, immutable_path: Path,
                      immutable_hash: str, immutable: Mapping[str, Any]) -> dict[str, Any]:
    _require(len(paths) == len(OFFICIAL_CROPS), "Step 12 requires exactly three crop configs")
    loaded_by_crop: dict[str, Any] = {}
    output_roots: set[Path] = set()
    immutable_crops = _mapping(immutable.get("crops"), "Step 11 crops")
    for raw_path in paths:
        path = _resolved(raw_path, root / str(raw_path))
        loaded = load_transition_experiment_config(path)
        crop = loaded.experiment_config.crop
        _require(crop in OFFICIAL_CROPS and crop not in loaded_by_crop,
                 f"Invalid or duplicate Step 12 crop config: {crop}")
        names = tuple(spec.model_name for spec in loaded.experiment_config.models)
        _require(names == OFFICIAL_CANDIDATES,
                 f"{crop}: candidates must match the official ordered eight candidates")
        cfg = loaded.experiment_config
        _require(cfg.hpo_config.enabled and cfg.hpo_config.budget > 0,
                 f"{crop}: target HPO must be enabled with a positive budget")
        _require(cfg.rollout_enabled and cfg.rollout_horizon_steps == (3, 6, 12)
                 and cfg.rollout_step_minutes == 5,
                 f"{crop}: rollout must be enabled for 15/30/60 minutes at 5-minute steps")
        for candidate in OFFICIAL_CANDIDATES[1:]:
            per_candidate = _mapping(cfg.hpo_config.target_search_spaces.get(candidate),
                                     f"{crop}.{candidate} target_search_spaces")
            _require(set(per_candidate) == set(OFFICIAL_TARGETS),
                     f"{crop}/{candidate}: HPO spaces must cover each official target")
            _require(all(_mapping(per_candidate[target], target) for target in OFFICIAL_TARGETS),
                     f"{crop}/{candidate}: target HPO search spaces cannot be empty")
        provenance = _mapping(loaded.raw_config.get("provenance"), f"{crop}.provenance")
        record = _mapping(provenance.get("immutable_rl_manifest"),
                          f"{crop}.immutable_rl_manifest")
        configured_manifest = Path(str(record.get("path", "")))
        if not configured_manifest.is_absolute():
            configured_manifest = (loaded.source_path.parent / configured_manifest).resolve()
        _require(configured_manifest == immutable_path and record.get("sha256") == immutable_hash,
                 f"{crop}: config Step 11 immutable manifest path/hash mismatch")
        crop_record = _mapping(immutable_crops.get(crop), f"Step 11 {crop}")
        split_records = _mapping(crop_record.get("datasets"), f"Step 11 {crop} datasets")
        for split, dataset_path in (("train", loaded.train_path),
                                    ("validation", loaded.validation_path),
                                    ("test", loaded.test_path)):
            expected = _mapping(_mapping(split_records.get(split), split).get("rl_dataset"),
                                f"{crop}/{split} RL dataset")
            _require(_sha256(dataset_path) == expected.get("sha256"),
                     f"{crop}/{split}: configured dataset differs from Step 11.6")
        out = cfg.output_root.resolve()
        _require(out not in output_roots, "Crop output roots must be independent")
        output_roots.add(out)
        loaded_by_crop[crop] = loaded
    _require(set(loaded_by_crop) == set(OFFICIAL_CROPS), "All official crops are required")
    return loaded_by_crop


def _validate_crop_outputs(root: Path, *, crop: str) -> None:
    required = (*TABLES.values(), TRANSITION_SUMMARY_MD, ARTIFACT_INTEGRITY_JSON,
                "config.json", "experiment_summary.json")
    for name in required:
        _require((root / name).is_file(), f"{crop}: missing output {name}")
    integrity = _read_json(root / ARTIFACT_INTEGRITY_JSON, f"{crop} integrity")
    _require(integrity.get("passed") is True, f"{crop}: no-auto-selection integrity failed")
    for name in TABLES.values():
        frame = pd.read_csv(root / name)
        _require(tuple(frame["candidate_name"]) == OFFICIAL_CANDIDATES,
                 f"{crop}/{name}: incomplete candidate rows")
    metrics = pd.read_csv(root / CANDIDATE_METRICS_CSV)
    tests = pd.read_csv(root / CANDIDATE_TEST_METRICS_CSV)
    for target in OFFICIAL_TARGETS:
        _require(f"validation_{target.removeprefix('obs_')}_rmse" in metrics,
                 f"{crop}: missing target-specific Validation metric for {target}")
        _require(f"test_{target.removeprefix('obs_')}_rmse" in tests,
                 f"{crop}: missing target-specific Test metric for {target}")
    _require(not tests[["test_used_for_hpo", "test_used_for_validation_ranking",
                        "test_used_for_selection"]].astype(bool).any().any(),
             f"{crop}: Test isolation flags are invalid")
    for candidate in OFFICIAL_CANDIDATES:
        artifact = root / crop / candidate
        for name in ("model.pkl", "manifest.json", "feature_schema.json",
                     "one_step_metrics.json", "rollout_metrics.json",
                     "resource_metrics.json", "hpo_results.json",
                     "best_config.json", "training_summary.json", "test_metrics.json"):
            _require((artifact / name).is_file(), f"{crop}/{candidate}: missing {name}")


def _combine_tables(destination: Path, crop_outputs: Mapping[str, str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for label, filename in TABLES.items():
        frames = []
        for crop in OFFICIAL_CROPS:
            frame = pd.read_csv(Path(crop_outputs[crop]) / filename).copy()
            frames.append(pd.concat([pd.Series(crop, index=frame.index, name="crop"), frame], axis=1))
        path = destination / f"all_crop_{filename}"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        paths[label] = str(path)
    return paths


def _resolved(value: str | Path | None, default: Path) -> Path:
    return (default if value is None else Path(value)).expanduser().resolve()


def _sha256(path: Path) -> str:
    _require(path.is_file(), f"Required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Step12BenchmarkError(f"Cannot read {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Step12BenchmarkError(message)


__all__ = [
    "OFFICIAL_CANDIDATES", "OFFICIAL_CROPS", "OFFICIAL_TARGETS",
    "STEP12_MANIFEST_NAME", "STEP12_VERSION", "Step12BenchmarkError",
    "Step12BenchmarkResult", "run_step12_real_data_benchmark",
]
