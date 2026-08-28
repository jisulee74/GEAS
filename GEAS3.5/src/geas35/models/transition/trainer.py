"""Training orchestration for GEAS transition model candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time

import numpy as np
import pandas as pd

from geas35.models.crop_specific import normalize_required_crop
from geas35.models.transition.artifacts import (
    TransitionModelArtifact,
    save_transition_resource_metrics,
    save_transition_model_artifact,
)
from geas35.models.transition.base import BaseTransitionModel, TransitionDataset
from geas35.models.transition.evaluator import (
    TransitionOneStepEvaluationReport,
    evaluate_transition_model_one_step,
)
from geas35.models.transition.hyperparameters import (
    PERSISTENCE_CANDIDATE_NAME,
    TransitionHPOConfig,
    TransitionHPOResult,
    TransitionHPOTrialResult,
    enrich_rollout_metrics_with_normalized_errors,
    sample_transition_hpo_params,
    transition_target_normalization_scales,
    validation_rollout_weighted_score,
)
from geas35.models.transition.rollout import (
    ActionProvider,
    LoggedActionProvider,
    TransitionRolloutSimulator,
)
from geas35.models.transition.resources import benchmark_transition_resources
from geas35.models.transition.selector import (
    MeanRmseStrategy,
    RankingMetricStrategy,
    TransitionModelRankingCandidate,
    TransitionModelRankingResult,
    rank_transition_models,
)

DEFAULT_ROLLOUT_HORIZON_STEPS = (3, 6, 12)
DEFAULT_ROLLOUT_STEP_MINUTES = 5


@dataclass(frozen=True)
class TransitionCandidateSpec:
    """Build recipe for one trainable transition model candidate."""

    model_name: str
    build_model: Callable[[], BaseTransitionModel]
    build_model_with_params: (
        Callable[[Mapping[str, Any]], BaseTransitionModel] | None
    ) = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.model_name).strip():
            raise ValueError("model_name is required.")

    def build_with_params(self, params: Mapping[str, Any]) -> BaseTransitionModel:
        if self.build_model_with_params is not None:
            return self.build_model_with_params(params)
        if params:
            raise ValueError(
                f"Transition candidate {self.model_name!r} does not support "
                "HPO parameter overrides."
            )
        return self.build_model()


@dataclass(frozen=True)
class TransitionCandidateTrainingResult:
    """Training, evaluation, and artifact result for one candidate."""

    model_name: str
    model: BaseTransitionModel
    artifact: TransitionModelArtifact
    one_step_report: TransitionOneStepEvaluationReport
    rollout_metrics: dict[str, dict[str, float]]
    resource_metrics: Mapping[str, Any]
    hpo_result: TransitionHPOResult
    ranking_candidate: TransitionModelRankingCandidate


@dataclass(frozen=True)
class TransitionTrainingResult:
    """End-to-end Step 8 transition training result for one crop."""

    crop: str
    validation_ranking: TransitionModelRankingResult
    candidate_results: tuple[TransitionCandidateTrainingResult, ...]


def train_transition_candidates(
    train_dataset: TransitionDataset,
    validation_dataset: TransitionDataset,
    candidate_specs: Sequence[TransitionCandidateSpec],
    *,
    crop: str,
    models_root: Path | str,
    ranking_strategy: RankingMetricStrategy | None = None,
    rollout_frame: pd.DataFrame | None = None,
    rollout_horizon_steps: Sequence[int] = DEFAULT_ROLLOUT_HORIZON_STEPS,
    rollout_step_minutes: int = DEFAULT_ROLLOUT_STEP_MINUTES,
    action_provider_factory: Callable[[], ActionProvider] | None = None,
    hpo_config: TransitionHPOConfig | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
) -> TransitionTrainingResult:
    """Train, evaluate, persist, and select transition model candidates.

    If no explicit ranking strategy is provided, rollout metrics are preferred
    when ``rollout_frame`` is present; otherwise one-step mean RMSE is used.
    """

    if not candidate_specs:
        raise ValueError("At least one transition candidate spec is required.")
    crop_key = normalize_required_crop(crop)
    strategy = ranking_strategy or (
        None if rollout_frame is not None else MeanRmseStrategy()
    )
    hpo = hpo_config or TransitionHPOConfig()
    normalization_scales = transition_target_normalization_scales(
        validation_dataset.y,
        validation_dataset.target_columns,
    )

    candidate_results = []
    for spec in candidate_specs:
        base_params = _candidate_base_params(spec)
        hpo_started = time.perf_counter()
        hpo_result = _run_candidate_hpo(
            spec,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            rollout_frame=rollout_frame,
            rollout_horizon_steps=rollout_horizon_steps,
            rollout_step_minutes=rollout_step_minutes,
            action_provider_factory=action_provider_factory,
            normalization_scales=normalization_scales,
            hpo_config=hpo,
            base_params=base_params,
        )
        hpo_total_time_seconds = time.perf_counter() - hpo_started
        model = spec.build_with_params(hpo_result.best_params)
        _require_transition_model(model, spec.model_name)
        training_started = time.perf_counter()
        model.fit(train_dataset, validation_dataset=validation_dataset)
        training_time_seconds = time.perf_counter() - training_started
        one_step_report = evaluate_transition_model_one_step(
            model,
            validation_dataset,
            normalization_scales=normalization_scales,
        )
        rollout_metrics = _evaluate_rollouts(
            model,
            rollout_frame=rollout_frame,
            rollout_horizon_steps=rollout_horizon_steps,
            rollout_step_minutes=rollout_step_minutes,
            action_provider_factory=action_provider_factory,
            normalization_scales=normalization_scales,
        )
        comparison_score, comparison_components = validation_rollout_weighted_score(
            one_step_report,
            rollout_metrics,
            objective_weights=hpo.objective_weights,
        )
        hpo_artifact = hpo_result.to_artifact()
        hpo_artifact["final_validation_rollout_weighted_score"] = comparison_score
        hpo_artifact["final_objective_components"] = comparison_components
        best_config_artifact = hpo_result.best_config_artifact()
        best_config_artifact["validation_rollout_weighted_score"] = comparison_score
        best_config_artifact["objective_components"] = comparison_components
        ranking_candidate = TransitionModelRankingCandidate(
            model_name=spec.model_name,
            one_step_report=one_step_report,
            rollout_metrics=rollout_metrics,
            metadata={
                **dict(spec.metadata),
                "best_config": dict(hpo_result.best_params),
                "hpo_status": hpo_result.status,
                "validation_rollout_weighted_score": comparison_score,
                "validation_rollout_weighted_score_components": comparison_components,
                "test_used_for_hpo": False,
                "test_used_for_validation_ranking": False,
                "test_used_for_selection": False,
            },
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
                "offline_exogenous_provider": "recorded_weather",
                "operating_exogenous_provider_interface": "forecast_weather",
                "training_time_seconds": training_time_seconds,
                "hpo_total_time_seconds": hpo_total_time_seconds,
                "hpo_status": hpo_result.status,
                "best_config": dict(hpo_result.best_params),
                "validation_rollout_weighted_score": comparison_score,
                "validation_rollout_weighted_score_components": comparison_components,
                "test_used_for_hpo": False,
                "test_used_for_validation_ranking": False,
                "test_used_for_selection": False,
            },
            hpo_results=hpo_artifact,
            best_config=best_config_artifact,
            metadata={
                **dict(artifact_metadata or {}),
                **dict(spec.metadata),
                "best_config": dict(hpo_result.best_params),
                "hpo_status": hpo_result.status,
            },
        )
        resource_metrics = benchmark_transition_resources(
            model,
            validation_dataset,
            artifact_dir=artifact.artifact_dir,
            model_path=artifact.model_path,
            training_time_seconds=training_time_seconds,
            hpo_total_time_seconds=hpo_total_time_seconds,
        )
        save_transition_resource_metrics(artifact, resource_metrics)
        candidate_results.append(
            TransitionCandidateTrainingResult(
                model_name=spec.model_name,
                model=model,
                artifact=artifact,
                one_step_report=one_step_report,
                rollout_metrics=rollout_metrics,
                resource_metrics=resource_metrics,
                hpo_result=hpo_result,
                ranking_candidate=ranking_candidate,
            )
        )

    validation_ranking = rank_transition_models(
        [result.ranking_candidate for result in candidate_results],
        strategy=strategy,
    )
    return TransitionTrainingResult(
        crop=crop_key,
        validation_ranking=validation_ranking,
        candidate_results=tuple(candidate_results),
    )


def _evaluate_rollouts(
    model: BaseTransitionModel,
    *,
    rollout_frame: pd.DataFrame | None,
    rollout_horizon_steps: Sequence[int],
    rollout_step_minutes: int,
    action_provider_factory: Callable[[], ActionProvider] | None,
    normalization_scales: Mapping[str, float] | None = None,
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
        horizon_metrics = dict(result.metrics)
        if normalization_scales is not None:
            horizon_metrics = enrich_rollout_metrics_with_normalized_errors(
                horizon_metrics,
                result.errors,
                normalization_scales,
            )
        metrics[horizon_key] = horizon_metrics
    return metrics


def _run_candidate_hpo(
    spec: TransitionCandidateSpec,
    *,
    train_dataset: TransitionDataset,
    validation_dataset: TransitionDataset,
    rollout_frame: pd.DataFrame | None,
    rollout_horizon_steps: Sequence[int],
    rollout_step_minutes: int,
    action_provider_factory: Callable[[], ActionProvider] | None,
    normalization_scales: Mapping[str, float],
    hpo_config: TransitionHPOConfig,
    base_params: Mapping[str, Any],
) -> TransitionHPOResult:
    if not hpo_config.enabled:
        return TransitionHPOResult.not_applicable(
            model_name=spec.model_name,
            config=hpo_config,
            params=base_params,
            reason="hpo_disabled",
        )
    if spec.model_name == PERSISTENCE_CANDIDATE_NAME:
        return TransitionHPOResult.not_applicable(
            model_name=spec.model_name,
            config=hpo_config,
            params=base_params,
            reason="persistence_baseline_has_no_trainable_hyperparameters",
        )

    hpo_train_dataset = _deterministic_dataset_subset(train_dataset, hpo_config.max_train_rows)
    hpo_validation_dataset = _deterministic_dataset_subset(validation_dataset, hpo_config.max_validation_rows)
    trials = []
    best_model_score = float("inf")
    best_trial_index: int | None = None
    best_params: Mapping[str, Any] = dict(base_params)
    sampled_params = sample_transition_hpo_params(
        spec.model_name,
        base_params,
        hpo_config,
        target_columns=train_dataset.target_columns,
    )
    for trial_index, params in enumerate(sampled_params, start=1):
        model = spec.build_with_params(params)
        _require_transition_model(model, spec.model_name)
        model.fit(hpo_train_dataset, validation_dataset=hpo_validation_dataset)
        one_step_report = evaluate_transition_model_one_step(
            model,
            hpo_validation_dataset,
            normalization_scales=normalization_scales,
        )
        rollout_metrics = _evaluate_rollouts(
            model,
            rollout_frame=rollout_frame,
            rollout_horizon_steps=rollout_horizon_steps,
            rollout_step_minutes=rollout_step_minutes,
            action_provider_factory=action_provider_factory,
            normalization_scales=normalization_scales,
        )
        score, components = validation_rollout_weighted_score(
            one_step_report,
            rollout_metrics,
            objective_weights=hpo_config.objective_weights,
        )
        trials.append(
            TransitionHPOTrialResult(
                trial_index=trial_index,
                model_name=spec.model_name,
                params=dict(params),
                validation_one_step_metrics=one_step_report.to_artifact(),
                validation_rollout_metrics=rollout_metrics,
                objective_components=components,
                validation_rollout_weighted_score=score,
            )
        )
        if best_trial_index is None or score < best_model_score:
            best_model_score = score
            best_trial_index = trial_index
            best_params = dict(params)

    return TransitionHPOResult(
        model_name=spec.model_name,
        status="completed",
        best_trial_index=best_trial_index,
        best_params=best_params,
        best_score=best_model_score,
        trials=tuple(trials),
        config=hpo_config,
    )


def _deterministic_dataset_subset(
    dataset: TransitionDataset, max_rows: int | None
) -> TransitionDataset:
    """Return an evenly spaced HPO-only subset without changing final evaluation."""
    row_count = len(dataset.x.index)
    if max_rows is None or row_count <= int(max_rows):
        return dataset
    positions = np.linspace(0, row_count - 1, num=int(max_rows), dtype=int)
    return TransitionDataset(
        x=dataset.x.iloc[positions].reset_index(drop=True),
        y=dataset.y.iloc[positions].reset_index(drop=True),
        metadata=(None if dataset.metadata is None else dataset.metadata.iloc[positions].reset_index(drop=True)),
        input_columns=dataset.input_columns,
        target_columns=dataset.target_columns,
        observation_columns=dataset.observation_columns,
        action_columns=dataset.action_columns,
    )


def _candidate_base_params(spec: TransitionCandidateSpec) -> Mapping[str, Any]:
    params = spec.metadata.get("params", {})
    return params if isinstance(params, Mapping) else {}


def _require_transition_model(model: BaseTransitionModel, model_name: str) -> None:
    if not isinstance(model, BaseTransitionModel):
        raise TypeError(
            f"Transition candidate {model_name!r} must build a "
            "BaseTransitionModel instance."
        )


def _feature_schema_payload(dataset: TransitionDataset) -> dict[str, Any]:
    return {
        "stage": "transition_feature_schema",
        "schema_version": "geas35.transition.three_target.v1",
        "target_contract": "official_three_target",
        "estimator_contract": "independent_per_target_default",
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
