"""Step 13 reproducibility and integrity review for Step 12 artifacts."""

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

from geas35.experiments.transition.config import load_transition_experiment_config
from geas35.experiments.transition.real_data_benchmark import (
    OFFICIAL_CANDIDATES,
    OFFICIAL_CROPS,
    OFFICIAL_TARGETS,
    STEP12_VERSION,
)
from geas35.io_utils import write_json
from geas35.models.transition.comparison import (
    FORBIDDEN_SELECTION_FILENAMES,
    FORBIDDEN_SELECTION_KEYS,
)
from geas35.models.transition.inference import (
    load_transition_model_from_artifact_dir,
    predict_next_observation_reward,
)
from geas35.models.transition.rollout import RecordedWeatherProvider
from geas35.rl import MDP_V1_ACTION_COLUMNS


STEP13_VERSION = "geas35.transition.step13.v1"
STEP13_MANIFEST_NAME = "step13_reproducibility_manifest.json"
DEPENDENCY_LOCK_NAME = "step13_dependency_lock.json"
ARTIFACT_HASHES_NAME = "step13_candidate_artifact_hashes.json"
REVIEW_SUMMARY_NAME = "step13_reproducibility_summary.md"
REQUIRED_CANDIDATE_FILES = (
    "model.pkl", "manifest.json", "feature_schema.json", "one_step_metrics.json",
    "rollout_metrics.json", "resource_metrics.json", "training_summary.json",
    "hpo_results.json", "best_config.json", "test_metrics.json",
)


class Step13ReviewError(ValueError):
    """Raised when Step 12 reproducibility or integrity review fails."""


def run_step13_reproducibility_review(
    *, project_root: str | Path, step12_manifest_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Path:
    """Review every Step 12 crop/candidate and publish a fail-closed manifest."""

    root = Path(project_root).expanduser().resolve()
    step12_path = _resolved(
        step12_manifest_path,
        root / "experiments/transition_model_selection/artifacts/step12/step12_benchmark_manifest.json",
    )
    destination = _resolved(
        output_root, root / "experiments/transition_model_selection/artifacts/step13",
    )
    step12 = _read_json(step12_path, "Step 12 manifest")
    _require(step12.get("schema_version") == STEP12_VERSION and step12.get("status") == "passed",
             "Step 12 manifest is not passed")
    _require(tuple(step12.get("candidate_order", ())) == OFFICIAL_CANDIDATES,
             "Step 12 candidate order differs from the official contract")
    _require(tuple(step12.get("target_columns", ())) == OFFICIAL_TARGETS,
             "Step 12 target schema differs from the official contract")
    _assert_no_selection(step12, step12_path)

    step11_record = _mapping(step12.get("step11_immutable_rl_manifest"), "Step 11 record")
    step11_path = _project_path(root, step11_record.get("path"))
    _require(_sha256(step11_path) == step11_record.get("sha256"),
             "Step 11 immutable manifest hash changed after Step 12")
    step11 = _read_json(step11_path, "Step 11 immutable manifest")
    provider_path = Path(__file__).resolve().parents[2] / "models/transition/rollout.py"
    provider_hash = _sha256(provider_path)

    crop_reviews: dict[str, Any] = {}
    artifact_hashes: dict[str, Any] = {}
    for crop in OFFICIAL_CROPS:
        record = _mapping(_mapping(step12.get("crops"), "Step 12 crops").get(crop), crop)
        config_path = _project_path(root, record.get("config_path"))
        _require(_sha256(config_path) == record.get("config_sha256"),
                 f"{crop}: config changed after Step 12")
        loaded = load_transition_experiment_config(config_path)
        _require(loaded.experiment_config.random_seed == 42, f"{crop}: seed is not fixed")
        crop_root = _project_path(root, record.get("output_root"))
        scaler_record = _mapping(_mapping(_mapping(step11.get("crops"), "Step 11 crops").get(crop), crop).get("scaler"), f"{crop} scaler")
        scaler_path = _project_path(root, scaler_record.get("path"))
        _require(_sha256(scaler_path) == scaler_record.get("sha256"),
                 f"{crop}: observation scaler hash changed")
        validation = pd.read_parquet(loaded.validation_path)
        candidate_rows = _index_csv(crop_root / "candidate_metrics.csv")
        rollout_rows = _index_csv(crop_root / "candidate_rollout_metrics.csv")
        resource_rows = _index_csv(crop_root / "candidate_resource_metrics.csv")
        test_rows = _index_csv(crop_root / "candidate_test_metrics.csv")
        crop_hashes: dict[str, Any] = {}
        smoke: dict[str, Any] = {}
        dependency_device: dict[str, Any] = {}
        for candidate in OFFICIAL_CANDIDATES:
            artifact_dir = crop_root / crop / candidate
            hashes = {name: _file_record(artifact_dir / name, root) for name in REQUIRED_CANDIDATE_FILES}
            crop_hashes[candidate] = hashes
            payloads = {name: _read_json(artifact_dir / name, f"{crop}/{candidate}/{name}")
                        for name in REQUIRED_CANDIDATE_FILES if name.endswith(".json")}
            for name, payload in payloads.items():
                _assert_no_selection(payload, artifact_dir / name)
            schema = payloads["feature_schema.json"]
            _require(tuple(schema.get("target_columns", ())) == OFFICIAL_TARGETS,
                     f"{crop}/{candidate}: target schema mismatch")
            summary = payloads["training_summary.json"]
            _require(summary.get("offline_exogenous_provider") == "recorded_weather"
                     and summary.get("operating_exogenous_provider_interface") == "forecast_weather",
                     f"{crop}/{candidate}: exogenous provider contract mismatch")
            _compare_report_rows(
                crop, candidate, candidate_rows[candidate], rollout_rows[candidate],
                resource_rows[candidate], test_rows[candidate], payloads,
            )
            resource = payloads["resource_metrics.json"]
            dependency_device[candidate] = {
                "library_versions": resource.get("library_versions"),
                "device": resource.get("device"),
            }
            smoke[candidate] = _cold_load_smoke(
                artifact_dir, validation, tuple(schema["input_columns"])
            )
        crop_reviews[crop] = {
            "status": "passed",
            "config": _file_record(config_path, root),
            "random_seed": loaded.experiment_config.random_seed,
            "hpo_random_seed": loaded.experiment_config.hpo_config.random_seed,
            "observation_scaler": _file_record(scaler_path, root),
            "target_columns": list(OFFICIAL_TARGETS),
            "exogenous_provider": {
                "offline": "RecordedWeatherProvider",
                "operating_interface": "forecast_weather",
                "implementation_path": _relative(provider_path, root),
                "implementation_sha256": provider_hash,
                "class": f"{RecordedWeatherProvider.__module__}.{RecordedWeatherProvider.__qualname__}",
            },
            "candidate_dependency_device": dependency_device,
            "cold_load_smoke": smoke,
        }
        artifact_hashes[crop] = crop_hashes

    _compare_combined_tables(root, step12, crop_reviews)
    _assert_no_selection_tree(step12_path.parent)
    dependencies = _dependency_lock()
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / DEPENDENCY_LOCK_NAME, dependencies, convert=True)
    write_json(destination / ARTIFACT_HASHES_NAME, {
        "schema_version": STEP13_VERSION, "candidates": artifact_hashes,
    }, convert=True)
    manifest = {
        "schema_version": STEP13_VERSION,
        "step": "13",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "step12_manifest": _file_record(step12_path, root),
        "step11_immutable_rl_manifest": _file_record(step11_path, root),
        "dependency_lock": _file_record(destination / DEPENDENCY_LOCK_NAME, root),
        "candidate_artifact_hashes": _file_record(destination / ARTIFACT_HASHES_NAME, root),
        "automatic_model_selection": False,
        "report_raw_artifact_match": True,
        "all_candidate_cold_load_smoke": "passed",
        "crops": crop_reviews,
        "reviewed_candidate_count": len(OFFICIAL_CROPS) * len(OFFICIAL_CANDIDATES),
        "step14_handoff_generated": False,
    }
    manifest_path = destination / STEP13_MANIFEST_NAME
    write_json(manifest_path, manifest, convert=True)
    (destination / REVIEW_SUMMARY_NAME).write_text(_summary(manifest), encoding="utf-8")
    return manifest_path


def _compare_combined_tables(root: Path, step12: Mapping[str, Any], crop_reviews: Mapping[str, Any]) -> None:
    del crop_reviews
    combined = _mapping(step12.get("combined_tables"), "Step 12 combined tables")
    filenames = {
        "candidate_metrics": "candidate_metrics.csv",
        "rollout_metrics": "candidate_rollout_metrics.csv",
        "resource_metrics": "candidate_resource_metrics.csv",
        "test_metrics": "candidate_test_metrics.csv",
    }
    crops = _mapping(step12.get("crops"), "Step 12 crops")
    for label, filename in filenames.items():
        combined_frame = pd.read_csv(_project_path(root, combined.get(label))).reset_index(drop=True)
        expected = []
        for crop in OFFICIAL_CROPS:
            crop_root = _project_path(root, _mapping(crops.get(crop), crop).get("output_root"))
            frame = pd.read_csv(crop_root / filename).copy()
            frame.insert(0, "crop", crop)
            expected.append(frame)
        expected_frame = pd.concat(expected, ignore_index=True)
        _require(list(combined_frame.columns) == list(expected_frame.columns), f"Combined table schema mismatch: {label}")
        try:
            pd.testing.assert_frame_equal(combined_frame, expected_frame, check_dtype=False, rtol=1e-10, atol=1e-12)
        except AssertionError as exc:
            raise Step13ReviewError(f"Combined table differs from crop reports: {label}") from exc


def _cold_load_smoke(artifact_dir: Path, validation: pd.DataFrame,
                     input_columns: tuple[str, ...]) -> dict[str, Any]:
    bundle = load_transition_model_from_artifact_dir(artifact_dir)
    valid = validation.loc[validation.get("rl_valid_transition", 1).astype(bool)]
    _require(not valid.empty, f"{artifact_dir}: no valid smoke-test row")
    row = valid.iloc[0].copy()
    missing = [column for column in input_columns if column not in row.index]
    _require(not missing, f"{artifact_dir}: smoke row missing {missing[0] if missing else ''}")
    action = {column: float(row[column]) for column in MDP_V1_ACTION_COLUMNS}
    result = predict_next_observation_reward(bundle, row, action)
    values = result.prediction.next_observation.loc[:, list(OFFICIAL_TARGETS)].to_numpy(float)
    _require(np.isfinite(values).all() and math.isfinite(result.reward),
             f"{artifact_dir}: cold-load inference/reward is not finite")
    return {
        "status": "passed", "prediction_shape": list(values.shape),
        "target_columns": list(result.prediction.target_columns),
        "reward_finite": True,
    }


def _compare_report_rows(crop: str, candidate: str, candidate_row: Mapping[str, Any],
                         rollout_row: Mapping[str, Any], resource_row: Mapping[str, Any],
                         test_row: Mapping[str, Any], payloads: Mapping[str, Mapping[str, Any]]) -> None:
    one = payloads["one_step_metrics.json"]
    aggregate = _mapping(one.get("aggregate_metrics"), "one-step aggregate")
    mapping = {"validation_one_step_mae": "mean_mae", "validation_one_step_rmse": "mean_rmse",
               "validation_one_step_r2": "mean_r2", "validation_one_step_q90": "mean_q90",
               "validation_one_step_cvar90": "mean_cvar90", "validation_one_step_nrmse": "mean_nrmse"}
    for report_key, raw_key in mapping.items():
        _same(candidate_row.get(report_key), aggregate.get(raw_key), f"{crop}/{candidate}/{report_key}")
    for target, metrics in _mapping(one.get("target_metrics"), "target metrics").items():
        label = target.removeprefix("obs_")
        for metric, value in _mapping(metrics, target).items():
            if metric in ("mae", "rmse", "r2", "q90", "cvar90", "nrmse"):
                _same(candidate_row.get(f"validation_{label}_{metric}"), value,
                      f"{crop}/{candidate}/{target}/{metric}")
    raw_rollout = payloads["rollout_metrics.json"]
    for horizon, metrics in raw_rollout.items():
        for key in ("trajectory_mae", "trajectory_rmse", "trajectory_nrmse",
                    "final_step_rmse", "physical_violation_rate", "nan_inf_count"):
            _same(rollout_row.get(f"{horizon}_{key}"), metrics.get(key),
                  f"{crop}/{candidate}/{horizon}/{key}")
    raw_resource = payloads["resource_metrics.json"]
    for key in ("training_time_seconds", "hpo_total_time_seconds", "inference_latency_median_ms",
                "inference_latency_p95_ms", "peak_cpu_rss_bytes", "serialized_model_size_bytes",
                "artifact_dir_size_bytes"):
        _same(resource_row.get(key), raw_resource.get(key), f"{crop}/{candidate}/resource/{key}")
    raw_test = _mapping(payloads["test_metrics.json"].get("test_one_step_metrics"), "test metrics")
    test_aggregate = _mapping(raw_test.get("aggregate_metrics"), "test aggregate")
    for report_key, raw_key in mapping.items():
        _same(test_row.get(report_key.replace("validation_", "test_")), test_aggregate.get(raw_key),
              f"{crop}/{candidate}/test/{raw_key}")


def _index_csv(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    _require(tuple(frame["candidate_name"]) == OFFICIAL_CANDIDATES,
             f"Candidate rows are incomplete: {path}")
    return {str(row["candidate_name"]): row.to_dict() for _, row in frame.iterrows()}


def _same(left: Any, right: Any, label: str) -> None:
    if left is None and right is None:
        return
    try:
        ok = bool(np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12, equal_nan=True))
    except (TypeError, ValueError):
        ok = left == right
    _require(ok, f"Report/raw artifact mismatch: {label}: {left!r} != {right!r}")


def _assert_no_selection(payload: Mapping[str, Any], path: Path) -> None:
    found = _forbidden_keys(payload)
    _require(not found, f"Forbidden auto-selection field in {path}: {sorted(found)}")


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_SELECTION_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


def _assert_no_selection_tree(root: Path) -> None:
    forbidden = [path for path in root.rglob("*") if path.name in FORBIDDEN_SELECTION_FILENAMES]
    _require(not forbidden, f"Forbidden selected artifact exists: {forbidden[0] if forbidden else ''}")


def _dependency_lock() -> dict[str, Any]:
    packages = ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "xgboost", "joblib")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": STEP13_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def _summary(manifest: Mapping[str, Any]) -> str:
    return "\n".join([
        "# GEAS Transition Step 13 Reproducibility Review", "",
        f"- status: {manifest['status']}",
        f"- reviewed candidates: {manifest['reviewed_candidate_count']}",
        "- report/raw artifact match: passed",
        "- explicit cold-load inference/reward: passed",
        "- automatic model selection: disabled",
        "- Step 14 handoff: not generated", "",
    ])


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Required file missing: {path}")
    return {"path": _relative(path, root), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Step13ReviewError(f"Cannot read {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _project_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolved(value: str | Path | None, default: Path) -> Path:
    return (default if value is None else Path(value)).expanduser().resolve()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Step13ReviewError(message)


__all__ = ["STEP13_MANIFEST_NAME", "STEP13_VERSION", "Step13ReviewError",
           "run_step13_reproducibility_review"]
