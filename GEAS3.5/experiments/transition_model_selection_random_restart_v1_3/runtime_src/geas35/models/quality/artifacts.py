"""Artifact loading helpers for applying selected quality models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from joblib.externals import cloudpickle as pickle
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from geas35.io_utils import write_json
from geas35.models.quality.base import BaseQualityModel
from geas35.models.quality.baseline import MedianQualityModel
from geas35.models.quality.rule_only import RuleOnlyQualityModel


QUALITY_MODEL_PICKLE_FILENAME = "quality_model.pkl"
QUALITY_MODEL_APPLICATION_FILENAME = "quality_model_application.json"


@dataclass(frozen=True)
class QualityModelApplication:
    """A selected quality model plus application-time decision parameters."""

    model: BaseQualityModel
    thresholds: object | None
    observation_columns: tuple[str, ...] | None
    imputation_confidence_threshold: float | None
    artifact_path: Path
    raw_payload: dict[str, Any]


def load_quality_model_application(path: str | Path) -> QualityModelApplication:
    """Load a manually selected quality model application artifact.

    Supported JSON payloads:

    * ``{"model_name": "rule_only"}``
    * ``{"model_name": "median", "medians": {...}}``
    * ``{"model_pickle_path": "relative/or/absolute/model.pkl"}``

    The experiment layer intentionally does not auto-select a final model. This
    loader is for a human-selected artifact passed into offline batch refinement.
    """

    artifact_path = Path(path).expanduser().resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Quality model artifact must be a JSON object: {artifact_path}")

    model = _load_pickled_model(payload, artifact_path) or _build_json_model(payload)
    thresholds = payload.get("thresholds", payload.get("threshold"))
    observation_columns = _optional_columns(payload.get("observation_columns"))
    imputation_confidence_threshold = payload.get("imputation_confidence_threshold")
    if imputation_confidence_threshold is not None:
        imputation_confidence_threshold = float(imputation_confidence_threshold)

    return QualityModelApplication(
        model=model,
        thresholds=thresholds,
        observation_columns=observation_columns,
        imputation_confidence_threshold=imputation_confidence_threshold,
        artifact_path=artifact_path,
        raw_payload=payload,
    )


def save_quality_model_application(
    model: BaseQualityModel,
    artifact_dir: str | Path,
    *,
    thresholds: object | None,
    observation_columns: tuple[str, ...],
    imputation_confidence_threshold: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist one fitted candidate as an application-ready quality artifact."""

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = output_dir / QUALITY_MODEL_PICKLE_FILENAME
    with pickle_path.open("wb") as fp:
        pickle.dump(model, fp)

    model_name = str(getattr(model, "model_name", model.__class__.__name__))
    artifact_path = output_dir / QUALITY_MODEL_APPLICATION_FILENAME
    write_json(
        artifact_path,
        {
            "stage": "quality_model_application_artifact",
            "model_name": model_name,
            "model_pickle_path": pickle_path.name,
            "artifact_format": "pickle",
            "threshold": thresholds,
            "observation_columns": list(observation_columns),
            "imputation_confidence_threshold": imputation_confidence_threshold,
            "automatic_best_model_selection": False,
            "test_used_for_selection": False,
            "metadata": dict(metadata or {}),
        },
        convert=True,
    )
    return artifact_path


def _load_pickled_model(
    payload: dict[str, Any],
    artifact_path: Path,
) -> BaseQualityModel | None:
    pickle_path_value = payload.get("model_pickle_path", payload.get("pickle_path"))
    if pickle_path_value is None:
        return None

    pickle_path = Path(str(pickle_path_value)).expanduser()
    if not pickle_path.is_absolute():
        pickle_path = artifact_path.parent / pickle_path
    with pickle_path.open("rb") as fp:
        model = pickle.load(fp)
    if not isinstance(model, BaseQualityModel):
        raise TypeError(f"Pickled quality artifact is not a BaseQualityModel: {pickle_path}")
    return model


def _build_json_model(payload: dict[str, Any]) -> BaseQualityModel:
    model_name = str(payload.get("model_name", "")).strip().lower()
    if model_name == "rule_only":
        model = RuleOnlyQualityModel()
        columns = _optional_columns(payload.get("observation_columns"))
        if columns is not None:
            model.observation_columns_ = columns
        return model

    if model_name == "median":
        model = MedianQualityModel()
        medians = payload.get("medians")
        if not isinstance(medians, dict):
            raise ValueError("Median quality artifact must contain a medians object.")
        columns = _optional_columns(payload.get("observation_columns")) or tuple(medians)
        model.medians_ = pd.Series(
            {str(col): pd.to_numeric(value, errors="coerce") for col, value in medians.items()},
            dtype="float64",
        )
        model.observation_columns_ = columns
        model.rule_model_.observation_columns_ = columns
        return model

    raise ValueError(
        "Unsupported JSON-only quality model artifact. "
        "Use model_name=rule_only, model_name=median with medians, or provide "
        "model_pickle_path for a fitted deep/custom BaseQualityModel."
    )


def _optional_columns(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        columns = (value,)
    else:
        columns = tuple(str(col) for col in value)  # type: ignore[arg-type]
    if not columns:
        return None
    return columns


__all__ = [
    "QUALITY_MODEL_APPLICATION_FILENAME",
    "QUALITY_MODEL_PICKLE_FILENAME",
    "QualityModelApplication",
    "load_quality_model_application",
    "save_quality_model_application",
]
