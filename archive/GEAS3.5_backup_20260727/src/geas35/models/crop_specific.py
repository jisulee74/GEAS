"""Crop-scoped model artifact helpers.

GEAS MDP v1 trains one model per crop. This module keeps that boundary explicit:
training code receives one crop split at a time, and inference loads the artifact
for the active crop only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from geas35.core import normalize_crop

DEFAULT_MODEL_FILENAME = "model.pkl"
DEFAULT_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class CropModelArtifact:
    """Resolved crop-specific model artifact paths."""

    crop: str
    model_name: str
    artifact_dir: Path
    model_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class CropTrainingResult:
    """Summary of one crop-specific training run."""

    crop: str
    model_name: str
    model_path: Path
    train_rows: int
    validation_rows: int


def normalize_required_crop(crop: str | None) -> str:
    """Normalize a crop name or subject code and fail on missing values."""

    normalized = normalize_crop(crop)
    if normalized is None:
        raise ValueError("crop is required.")
    return normalized


def crop_split_path(dataset_root: Path | str, crop: str, split: str = "train") -> Path:
    """Return the parquet path for a crop/split pair."""

    crop_key = normalize_required_crop(crop)
    return Path(dataset_root) / crop_key / f"{split}.parquet"


def discover_dataset_crops(
    dataset_root: Path | str,
    *,
    split: str = "train",
) -> list[str]:
    """Discover crop folders that contain the requested split parquet file."""

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    crops: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        crop = normalize_crop(child.name)
        if crop is None:
            continue
        if (child / f"{split}.parquet").exists():
            crops.append(crop)
    return crops


def load_crop_split(
    dataset_root: Path | str,
    crop: str,
    split: str = "train",
) -> pd.DataFrame:
    """Load a crop-specific parquet split."""

    path = crop_split_path(dataset_root, crop, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing crop split: {path}")
    return pd.read_parquet(path)


def crop_model_artifact(
    models_root: Path | str,
    crop: str,
    model_name: str,
) -> CropModelArtifact:
    """Resolve the artifact directory for a crop/model name pair."""

    crop_key = normalize_required_crop(crop)
    name = str(model_name).strip()
    if not name:
        raise ValueError("model_name is required.")
    artifact_dir = Path(models_root) / crop_key / name
    return CropModelArtifact(
        crop=crop_key,
        model_name=name,
        artifact_dir=artifact_dir,
        model_path=artifact_dir / DEFAULT_MODEL_FILENAME,
        manifest_path=artifact_dir / DEFAULT_MANIFEST_FILENAME,
    )


def save_crop_model(
    model: Any,
    models_root: Path | str,
    crop: str,
    *,
    model_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> CropModelArtifact:
    """Save one crop-specific model artifact with a small manifest."""

    artifact = crop_model_artifact(models_root, crop, model_name)
    artifact.artifact_dir.mkdir(parents=True, exist_ok=True)
    with artifact.model_path.open("wb") as f:
        pickle.dump(model, f)

    manifest = {
        "crop": artifact.crop,
        "model_name": artifact.model_name,
        "model_filename": artifact.model_path.name,
        "artifact_format": "pickle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
    }
    artifact.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact


def load_crop_model(
    models_root: Path | str,
    crop: str,
    *,
    model_name: str,
) -> Any:
    """Load the model artifact for the active crop."""

    artifact = crop_model_artifact(models_root, crop, model_name)
    if not artifact.model_path.exists():
        raise FileNotFoundError(
            f"No model artifact for crop={artifact.crop!r}, "
            f"model_name={artifact.model_name!r}: {artifact.model_path}"
        )
    with artifact.model_path.open("rb") as f:
        return pickle.load(f)


def predict_with_crop_model(
    models_root: Path | str,
    crop: str,
    inputs: Any,
    *,
    model_name: str,
) -> Any:
    """Load the crop-specific model and run inference."""

    model = load_crop_model(models_root, crop, model_name=model_name)
    if hasattr(model, "predict"):
        return model.predict(inputs)
    if callable(model):
        return model(inputs)
    raise TypeError(
        f"Crop model {model_name!r} for crop {normalize_required_crop(crop)!r} "
        "must be callable or expose predict(inputs)."
    )


CropTrainer = Callable[[str, pd.DataFrame, pd.DataFrame | None], Any]


def train_models_by_crop(
    dataset_root: Path | str,
    models_root: Path | str,
    trainer: CropTrainer,
    *,
    model_name: str,
    crops: Iterable[str] | None = None,
    train_split: str = "train",
    validation_split: str = "validation",
    metadata: Mapping[str, Any] | None = None,
) -> list[CropTrainingResult]:
    """Train and save independent model artifacts for each crop.

    ``trainer`` is called as ``trainer(crop, train_df, validation_df)``. It can
    train a transition model, PPO policy, scaler bundle, or any future
    crop-specific artifact.
    """

    crop_names = (
        [normalize_required_crop(crop) for crop in crops]
        if crops is not None
        else discover_dataset_crops(dataset_root, split=train_split)
    )
    results: list[CropTrainingResult] = []
    for crop in crop_names:
        train_df = load_crop_split(dataset_root, crop, train_split)
        validation_path = crop_split_path(dataset_root, crop, validation_split)
        validation_df = (
            pd.read_parquet(validation_path)
            if validation_path.exists()
            else None
        )
        model = trainer(crop, train_df, validation_df)
        artifact = save_crop_model(
            model,
            models_root,
            crop,
            model_name=model_name,
            metadata={
                **dict(metadata or {}),
                "train_split": train_split,
                "validation_split": validation_split,
                "train_rows": int(len(train_df.index)),
                "validation_rows": int(0 if validation_df is None else len(validation_df.index)),
            },
        )
        results.append(
            CropTrainingResult(
                crop=crop,
                model_name=model_name,
                model_path=artifact.model_path,
                train_rows=int(len(train_df.index)),
                validation_rows=int(0 if validation_df is None else len(validation_df.index)),
            )
        )
    return results


__all__ = [
    "CropModelArtifact",
    "CropTrainingResult",
    "CropTrainer",
    "crop_model_artifact",
    "crop_split_path",
    "discover_dataset_crops",
    "load_crop_model",
    "load_crop_split",
    "normalize_required_crop",
    "predict_with_crop_model",
    "save_crop_model",
    "train_models_by_crop",
]
