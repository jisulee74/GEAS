"""Artifact helpers for GEAS transition models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from geas35.io_utils import jsonable, write_json
from geas35.models.crop_specific import normalize_required_crop
from geas35.models.transition.features import OFFICIAL_TRANSITION_TARGET_COLUMNS

TRANSITION_MODEL_FILENAME = "model.pkl"
TRANSITION_MANIFEST_FILENAME = "manifest.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
ONE_STEP_METRICS_FILENAME = "one_step_metrics.json"
ROLLOUT_METRICS_FILENAME = "rollout_metrics.json"
RESOURCE_METRICS_FILENAME = "resource_metrics.json"
TRAINING_SUMMARY_FILENAME = "training_summary.json"
HPO_RESULTS_FILENAME = "hpo_results.json"
BEST_CONFIG_FILENAME = "best_config.json"
TEST_METRICS_FILENAME = "test_metrics.json"


@dataclass(frozen=True)
class TransitionModelArtifact:
    """Resolved artifact paths for one crop/model transition candidate."""

    crop: str
    model_name: str
    artifact_dir: Path
    model_path: Path
    manifest_path: Path
    feature_schema_path: Path
    one_step_metrics_path: Path
    rollout_metrics_path: Path
    resource_metrics_path: Path
    training_summary_path: Path
    hpo_results_path: Path
    best_config_path: Path
    test_metrics_path: Path


def transition_model_artifact(
    models_root: Path | str,
    crop: str,
    model_name: str,
) -> TransitionModelArtifact:
    """Resolve the artifact directory for one transition model candidate."""

    crop_key = normalize_required_crop(crop)
    name = str(model_name).strip()
    if not name:
        raise ValueError("model_name is required.")
    artifact_dir = Path(models_root) / crop_key / name
    return TransitionModelArtifact(
        crop=crop_key,
        model_name=name,
        artifact_dir=artifact_dir,
        model_path=artifact_dir / TRANSITION_MODEL_FILENAME,
        manifest_path=artifact_dir / TRANSITION_MANIFEST_FILENAME,
        feature_schema_path=artifact_dir / FEATURE_SCHEMA_FILENAME,
        one_step_metrics_path=artifact_dir / ONE_STEP_METRICS_FILENAME,
        rollout_metrics_path=artifact_dir / ROLLOUT_METRICS_FILENAME,
        resource_metrics_path=artifact_dir / RESOURCE_METRICS_FILENAME,
        training_summary_path=artifact_dir / TRAINING_SUMMARY_FILENAME,
        hpo_results_path=artifact_dir / HPO_RESULTS_FILENAME,
        best_config_path=artifact_dir / BEST_CONFIG_FILENAME,
        test_metrics_path=artifact_dir / TEST_METRICS_FILENAME,
    )


def save_transition_model_artifact(
    model: Any,
    models_root: Path | str,
    crop: str,
    *,
    model_name: str,
    feature_schema: Any | None = None,
    one_step_report: Any | None = None,
    rollout_metrics: Mapping[str, Any] | None = None,
    resource_metrics: Mapping[str, Any] | None = None,
    training_summary: Mapping[str, Any] | None = None,
    hpo_results: Any | None = None,
    best_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TransitionModelArtifact:
    """Persist a fitted transition model and its candidate-level artifacts."""

    artifact = transition_model_artifact(models_root, crop, model_name)
    artifact.artifact_dir.mkdir(parents=True, exist_ok=True)
    with artifact.model_path.open("wb") as f:
        pickle.dump(model, f)

    fitted_targets = tuple(getattr(model, "target_columns_", ()) or ())
    if fitted_targets and fitted_targets != OFFICIAL_TRANSITION_TARGET_COLUMNS:
        raise ValueError(
            "Official transition artifact requires exactly temperature, humidity, and CO2 targets."
        )
    manifest = {
        "stage": "transition_model_artifact",
        "schema_version": "geas35.transition.three_target.v1",
        "target_contract": "official_three_target",
        "target_columns": list(OFFICIAL_TRANSITION_TARGET_COLUMNS),
        "multi_output_strategy": getattr(getattr(model, "capabilities", None), "multi_output_strategy", None),
        "crop": artifact.crop,
        "model_name": artifact.model_name,
        "model_filename": artifact.model_path.name,
        "artifact_format": "pickle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
    }
    write_json(artifact.manifest_path, manifest, convert=True)

    if feature_schema is not None:
        write_json(
            artifact.feature_schema_path,
            _artifact_payload(feature_schema),
            convert=True,
        )
    if one_step_report is not None:
        write_json(
            artifact.one_step_metrics_path,
            _artifact_payload(one_step_report),
            convert=True,
        )
    if rollout_metrics is not None:
        write_json(artifact.rollout_metrics_path, rollout_metrics, convert=True)
    if resource_metrics is not None:
        write_json(artifact.resource_metrics_path, resource_metrics, convert=True)
    if training_summary is not None:
        write_json(artifact.training_summary_path, training_summary, convert=True)
    if hpo_results is not None:
        write_json(artifact.hpo_results_path, _artifact_payload(hpo_results), convert=True)
    if best_config is not None:
        write_json(artifact.best_config_path, best_config, convert=True)
    return artifact


def load_transition_model_artifact(
    models_root: Path | str,
    crop: str,
    *,
    model_name: str,
) -> Any:
    """Load a persisted transition model candidate."""

    artifact = transition_model_artifact(models_root, crop, model_name)
    if not artifact.model_path.exists():
        raise FileNotFoundError(
            f"No transition model artifact for crop={artifact.crop!r}, "
            f"model_name={artifact.model_name!r}: {artifact.model_path}"
        )
    with artifact.model_path.open("rb") as f:
        return pickle.load(f)


def save_transition_resource_metrics(
    artifact: TransitionModelArtifact,
    resource_metrics: Mapping[str, Any],
) -> Path:
    """Write candidate-level transition resource metrics."""

    write_json(artifact.resource_metrics_path, resource_metrics, convert=True)
    return artifact.resource_metrics_path


def _artifact_payload(value: Any) -> Any:
    if hasattr(value, "to_artifact"):
        return jsonable(value.to_artifact())
    return jsonable(value)


__all__ = [
    "FEATURE_SCHEMA_FILENAME",
    "BEST_CONFIG_FILENAME",
    "HPO_RESULTS_FILENAME",
    "ONE_STEP_METRICS_FILENAME",
    "RESOURCE_METRICS_FILENAME",
    "ROLLOUT_METRICS_FILENAME",
    "TEST_METRICS_FILENAME",
    "TRAINING_SUMMARY_FILENAME",
    "TRANSITION_MANIFEST_FILENAME",
    "TRANSITION_MODEL_FILENAME",
    "TransitionModelArtifact",
    "load_transition_model_artifact",
    "save_transition_resource_metrics",
    "save_transition_model_artifact",
    "transition_model_artifact",
]
