"""Inference helpers for selected GEAS transition model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from geas35.models.crop_specific import normalize_required_crop
from geas35.models.transition.artifacts import (
    TRANSITION_MANIFEST_FILENAME,
    TRANSITION_MODEL_FILENAME,
    load_selected_transition_model_manifest,
    load_transition_model_artifact,
)
from geas35.models.transition.base import (
    BaseTransitionModel,
    TransitionPrediction,
)
from geas35.rl import (
    MDP_V1_ACTION_COLUMNS,
    MdpV1Config,
    MdpV1RewardNormalizer,
    compute_mdp_v1_reward,
)


@dataclass(frozen=True)
class LoadedTransitionModel:
    """Selected transition model artifact loaded for inference."""

    crop: str
    model_name: str
    model: BaseTransitionModel
    selected_manifest: Mapping[str, Any]
    model_artifact_dir: Path
    model_manifest: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TransitionRewardPrediction:
    """Prediction plus reward computed from the predicted next state."""

    prediction: TransitionPrediction
    next_row: pd.Series
    reward: float
    reward_terms: dict[str, float]


def load_selected_transition_model(
    models_root: Path | str,
    crop: str,
) -> LoadedTransitionModel:
    """Load the selected transition model for a crop."""

    crop_key = normalize_required_crop(crop)
    selected_manifest = load_selected_transition_model_manifest(models_root, crop_key)
    model_name = str(selected_manifest.get("selected_model_name", "")).strip()
    if not model_name:
        raise ValueError("Selected transition manifest is missing selected_model_name.")

    artifact_dir = _selected_artifact_dir(
        models_root,
        crop_key,
        model_name,
        selected_manifest,
    )
    if artifact_dir is None:
        model = load_transition_model_artifact(
            models_root,
            crop_key,
            model_name=model_name,
        )
        artifact_dir = Path(models_root) / crop_key / model_name
    else:
        model_path = artifact_dir / TRANSITION_MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f"No transition model pickle: {model_path}")
        with model_path.open("rb") as f:
            model = pickle.load(f)
    if not isinstance(model, BaseTransitionModel):
        raise TypeError("Loaded artifact is not a BaseTransitionModel.")

    model_manifest_path = artifact_dir / TRANSITION_MANIFEST_FILENAME
    model_manifest = (
        json.loads(model_manifest_path.read_text(encoding="utf-8"))
        if model_manifest_path.exists()
        else None
    )
    return LoadedTransitionModel(
        crop=crop_key,
        model_name=model_name,
        model=model,
        selected_manifest=selected_manifest,
        model_artifact_dir=artifact_dir,
        model_manifest=model_manifest,
    )


def predict_next_observation(
    model_or_bundle: BaseTransitionModel | LoadedTransitionModel,
    current_observation: pd.Series | pd.DataFrame | Mapping[str, Any],
    action: Mapping[str, float] | Sequence[float] | None = None,
) -> TransitionPrediction:
    """Predict the next dynamic observation from current observation and action."""

    model = (
        model_or_bundle.model
        if isinstance(model_or_bundle, LoadedTransitionModel)
        else model_or_bundle
    )
    frame = _inference_frame(current_observation, action)
    return model.predict(frame)


def build_predicted_next_row(
    current_row: pd.Series | Mapping[str, Any],
    prediction: TransitionPrediction,
    *,
    row_index: int = 0,
) -> pd.Series:
    """Restore a row-like next observation by overlaying predicted targets."""

    row = pd.Series(dict(current_row)).copy()
    if prediction.next_observation.empty:
        raise ValueError("Transition prediction is empty.")
    predicted = prediction.next_observation.iloc[row_index]
    for column in prediction.target_columns:
        row[column] = predicted[column]
    return row


def predict_next_observation_reward(
    model_or_bundle: BaseTransitionModel | LoadedTransitionModel,
    current_row: pd.Series | Mapping[str, Any],
    action: Mapping[str, float] | Sequence[float],
    *,
    prev_action: Mapping[str, float] | None = None,
    prev_prev_action: Mapping[str, float] | None = None,
    config: MdpV1Config | None = None,
    reward_normalizer: MdpV1RewardNormalizer | None = None,
) -> TransitionRewardPrediction:
    """Predict next observation and compute reward with the existing MDP v1 reward."""

    prediction = predict_next_observation(model_or_bundle, current_row, action)
    next_row = build_predicted_next_row(current_row, prediction)
    reward, terms = compute_mdp_v1_reward(
        next_row,
        action,
        prev_action=prev_action,
        prev_prev_action=prev_prev_action,
        config=config,
        reward_normalizer=reward_normalizer,
    )
    return TransitionRewardPrediction(
        prediction=prediction,
        next_row=next_row,
        reward=float(reward),
        reward_terms=terms,
    )


def _selected_artifact_dir(
    models_root: Path | str,
    crop: str,
    model_name: str,
    selected_manifest: Mapping[str, Any],
) -> Path | None:
    metadata = selected_manifest.get("metadata", {})
    if isinstance(metadata, Mapping):
        artifacts = metadata.get("candidate_artifacts", {})
        if isinstance(artifacts, Mapping) and model_name in artifacts:
            return Path(str(artifacts[model_name]))
    fallback = Path(models_root) / crop / model_name
    return fallback if fallback.exists() else None


def _inference_frame(
    current_observation: pd.Series | pd.DataFrame | Mapping[str, Any],
    action: Mapping[str, float] | Sequence[float] | None,
) -> pd.DataFrame:
    if isinstance(current_observation, pd.DataFrame):
        frame = current_observation.copy()
        if len(frame.index) != 1 and action is not None and not isinstance(action, Mapping):
            raise ValueError("Sequence action is only supported for single-row inference.")
    else:
        frame = pd.DataFrame([dict(current_observation)])
    if action is None:
        return frame
    if isinstance(action, Mapping):
        for column, value in action.items():
            frame[column] = float(value)
        return frame
    values = list(action)
    if len(frame.index) != 1:
        raise ValueError("Sequence action is only supported for single-row inference.")
    if len(values) != len(MDP_V1_ACTION_COLUMNS):
        raise ValueError(
            f"Sequence action must contain {len(MDP_V1_ACTION_COLUMNS)} values."
        )
    for column, value in zip(MDP_V1_ACTION_COLUMNS, values):
        frame[column] = float(value)
    return frame


__all__ = [
    "LoadedTransitionModel",
    "TransitionRewardPrediction",
    "build_predicted_next_row",
    "load_selected_transition_model",
    "predict_next_observation",
    "predict_next_observation_reward",
]
