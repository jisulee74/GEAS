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

TRANSITION_MODEL_FILENAME = "model.pkl"
TRANSITION_MANIFEST_FILENAME = "manifest.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
ONE_STEP_METRICS_FILENAME = "one_step_metrics.json"
ROLLOUT_METRICS_FILENAME = "rollout_metrics.json"
TRAINING_SUMMARY_FILENAME = "training_summary.json"
SELECTED_TRANSITION_MODEL_FILENAME = "selected_transition_model.json"


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
    training_summary_path: Path


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
        training_summary_path=artifact_dir / TRAINING_SUMMARY_FILENAME,
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
    training_summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TransitionModelArtifact:
    """Persist a fitted transition model and its candidate-level artifacts."""

    artifact = transition_model_artifact(models_root, crop, model_name)
    artifact.artifact_dir.mkdir(parents=True, exist_ok=True)
    with artifact.model_path.open("wb") as f:
        pickle.dump(model, f)

    manifest = {
        "stage": "transition_model_artifact",
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
    if training_summary is not None:
        write_json(artifact.training_summary_path, training_summary, convert=True)
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


def save_selected_transition_model(
    models_root: Path | str,
    crop: str,
    selection_result: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write the crop-level selected transition model manifest."""

    crop_key = normalize_required_crop(crop)
    path = Path(models_root) / crop_key / SELECTED_TRANSITION_MODEL_FILENAME
    payload = _artifact_payload(selection_result)
    if not isinstance(payload, dict):
        payload = {"selection_result": payload}
    payload = {
        **payload,
        "crop": crop_key,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
    }
    write_json(path, payload, convert=True)
    return path


def load_selected_transition_model_manifest(
    models_root: Path | str,
    crop: str,
) -> dict[str, Any]:
    """Load the crop-level selected transition model manifest."""

    crop_key = normalize_required_crop(crop)
    path = Path(models_root) / crop_key / SELECTED_TRANSITION_MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No selected transition model manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_payload(value: Any) -> Any:
    if hasattr(value, "to_artifact"):
        return jsonable(value.to_artifact())
    return jsonable(value)


__all__ = [
    "FEATURE_SCHEMA_FILENAME",
    "ONE_STEP_METRICS_FILENAME",
    "ROLLOUT_METRICS_FILENAME",
    "SELECTED_TRANSITION_MODEL_FILENAME",
    "TRAINING_SUMMARY_FILENAME",
    "TRANSITION_MANIFEST_FILENAME",
    "TRANSITION_MODEL_FILENAME",
    "TransitionModelArtifact",
    "load_selected_transition_model_manifest",
    "load_transition_model_artifact",
    "save_selected_transition_model",
    "save_transition_model_artifact",
    "transition_model_artifact",
]
