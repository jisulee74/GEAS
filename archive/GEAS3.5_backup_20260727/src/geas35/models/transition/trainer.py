"""Training orchestration for GEAS transition model candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from geas35.models.crop_specific import normalize_required_crop
from geas35.models.transition.artifacts import (
    TransitionModelArtifact,
    save_selected_transition_model,
    save_transition_model_artifact,
)
from geas35.models.transition.base import BaseTransitionModel, TransitionDataset
from geas35.models.transition.evaluator import (
    TransitionOneStepEvaluationReport,
    evaluate_transition_model_one_step,
)
from geas35.models.transition.rollout import (
    ActionProvider,
    LoggedActionProvider,
    TransitionRolloutSimulator,
)
from geas35.models.transition.selector import (
    MeanRmseStrategy,
    SelectionMetricStrategy,
    TransitionModelSelectionCandidate,
    TransitionModelSelectionResult,
    select_transition_model,
)

DEFAULT_ROLLOUT_HORIZON_STEPS = (3, 6, 12)
DEFAULT_ROLLOUT_STEP_MINUTES = 5


@dataclass(frozen=True)
class TransitionCandidateSpec:
    """Build recipe for one trainable transition model candidate."""

    model_name: str
    build_model: Callable[[], BaseTransitionModel]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.model_name).strip():
            raise ValueError("model_name is required.")


@dataclass(frozen=True)
class TransitionCandidateTrainingResult:
    """Training, evaluation, and artifact result for one candidate."""

    model_name: str
    model: BaseTransitionModel
    artifact: TransitionModelArtifact
    one_step_report: TransitionOneStepEvaluationReport
    rollout_metrics: dict[str, dict[str, float]]
    selection_candidate: TransitionModelSelectionCandidate


@dataclass(frozen=True)
class TransitionTrainingResult:
    """End-to-end Step 8 transition training result for one crop."""

    crop: str
    selection_result: TransitionModelSelectionResult
    candidate_results: tuple[TransitionCandidateTrainingResult, ...]
    selected_manifest_path: Path

    @property
    def selected_candidate(self) -> TransitionCandidateTrainingResult:
        """Return the candidate result matching the selected model name."""

        selected_name = self.selection_result.selected_model_name
        for candidate in self.candidate_results:
            if candidate.model_name == selected_name:
                return candidate
        raise KeyError(f"Selected candidate is missing: {selected_name}")


def train_transition_candidates(
    train_dataset: TransitionDataset,
    validation_dataset: TransitionDataset,
    candidate_specs: Sequence[TransitionCandidateSpec],
    *,
    crop: str,
    models_root: Path | str,
    selection_strategy: SelectionMetricStrategy | None = None,
    rollout_frame: pd.DataFrame | None = None,
    rollout_horizon_steps: Sequence[int] = DEFAULT_ROLLOUT_HORIZON_STEPS,
    rollout_step_minutes: int = DEFAULT_ROLLOUT_STEP_MINUTES,
    action_provider_factory: Callable[[], ActionProvider] | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
) -> TransitionTrainingResult:
    """Train, evaluate, persist, and select transition model candidates.

    If no explicit selection strategy is provided, rollout metrics are preferred
    when ``rollout_frame`` is present; otherwise one-step mean RMSE is used.
    """

    if not candidate_specs:
        raise ValueError("At least one transition candidate spec is required.")
    crop_key = normalize_required_crop(crop)
    strategy = selection_strategy or (
        None if rollout_frame is not None else MeanRmseStrategy()
    )

    candidate_results = []
    for spec in candidate_specs:
        model = spec.build_model()
        if not isinstance(model, BaseTransitionModel):
            raise TypeError(
                f"Transition candidate {spec.model_name!r} must build a "
                "BaseTransitionModel instance."
            )
        model.fit(train_dataset, validation_dataset=validation_dataset)
        one_step_report = evaluate_transition_model_one_step(
            model,
            validation_dataset,
        )
        rollout_metrics = _evaluate_rollouts(
            model,
            rollout_frame=rollout_frame,
            rollout_horizon_steps=rollout_horizon_steps,
            rollout_step_minutes=rollout_step_minutes,
            action_provider_factory=action_provider_factory,
        )
        selection_candidate = TransitionModelSelectionCandidate(
            model_name=spec.model_name,
            one_step_report=one_step_report,
            rollout_metrics=rollout_metrics,
            metadata=dict(spec.metadata),
        )
        artifact = save_transition_model_artifact(
            model,
            models_root,
            crop_key,
            model_name=spec.model_name,
            feature_schema=_feature_schema_payload(train_dataset),
            one_step_report=one_step_report,
            rollout_metrics=rollout_metrics,
            training_summary={
                "stage": "transition_candidate_training",
                "crop": crop_key,
                "model_name": spec.model_name,
                "train_rows": int(len(train_dataset.x.index)),
                "validation_rows": int(len(validation_dataset.x.index)),
                "input_columns": list(train_dataset.input_columns),
                "target_columns": list(train_dataset.target_columns),
                "rollout_horizon_steps": list(rollout_horizon_steps),
                "rollout_step_minutes": int(rollout_step_minutes),
            },
            metadata={
                **dict(artifact_metadata or {}),
                **dict(spec.metadata),
            },
        )
        candidate_results.append(
            TransitionCandidateTrainingResult(
                model_name=spec.model_name,
                model=model,
                artifact=artifact,
                one_step_report=one_step_report,
                rollout_metrics=rollout_metrics,
                selection_candidate=selection_candidate,
            )
        )

    selection_result = select_transition_model(
        [result.selection_candidate for result in candidate_results],
        strategy=strategy,
    )
    selected_manifest_path = save_selected_transition_model(
        models_root,
        crop_key,
        selection_result,
        metadata={
            **dict(artifact_metadata or {}),
            "candidate_artifacts": {
                result.model_name: str(result.artifact.artifact_dir)
                for result in candidate_results
            },
        },
    )
    return TransitionTrainingResult(
        crop=crop_key,
        selection_result=selection_result,
        candidate_results=tuple(candidate_results),
        selected_manifest_path=selected_manifest_path,
    )


def _evaluate_rollouts(
    model: BaseTransitionModel,
    *,
    rollout_frame: pd.DataFrame | None,
    rollout_horizon_steps: Sequence[int],
    rollout_step_minutes: int,
    action_provider_factory: Callable[[], ActionProvider] | None,
) -> dict[str, dict[str, float]]:
    if rollout_frame is None:
        return {}
    if rollout_step_minutes <= 0:
        raise ValueError("rollout_step_minutes must be positive.")
    metrics: dict[str, dict[str, float]] = {}
    for horizon_steps in rollout_horizon_steps:
        if horizon_steps <= 0:
            raise ValueError("rollout_horizon_steps must contain positive values.")
        provider = (
            action_provider_factory()
            if action_provider_factory is not None
            else LoggedActionProvider()
        )
        result = TransitionRolloutSimulator(model, action_provider=provider).simulate(
            rollout_frame,
            start_index=0,
            horizon_steps=int(horizon_steps),
        )
        horizon_key = f"{int(horizon_steps) * int(rollout_step_minutes)}min"
        metrics[horizon_key] = result.metrics
    return metrics


def _feature_schema_payload(dataset: TransitionDataset) -> dict[str, Any]:
    return {
        "stage": "transition_feature_schema",
        "input_columns": list(dataset.input_columns),
        "target_columns": list(dataset.target_columns),
        "observation_columns": list(dataset.observation_columns),
        "action_columns": list(dataset.action_columns),
    }


__all__ = [
    "DEFAULT_ROLLOUT_HORIZON_STEPS",
    "DEFAULT_ROLLOUT_STEP_MINUTES",
    "TransitionCandidateSpec",
    "TransitionCandidateTrainingResult",
    "TransitionTrainingResult",
    "train_transition_candidates",
]
