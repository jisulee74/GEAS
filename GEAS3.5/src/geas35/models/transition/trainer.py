"""Training orchestration for GEAS transition model candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any
import time

import numpy as np
import pandas as pd

from geas35.models.crop_specific import normalize_required_crop
from geas35.models.transition.action_response import ActionResponseReport, evaluate_action_response
from geas35.models.transition.artifacts import (
    TransitionModelArtifact,
    load_transition_model_artifact,
    transition_model_artifact,
    save_transition_resource_metrics,
    save_transition_model_artifact,
)
from geas35.models.transition.base import (
    BaseTransitionModel,
    TransitionDataset,
    TransitionTargetFormulation,
)
from geas35.models.transition.causal_state import ROLLOUT_PROTOCOL_VERSION, build_rollout_windows
from geas35.models.transition.evaluator import (
    TransitionOneStepEvaluationReport,
    evaluate_transition_model_one_step,
)
from geas35.models.transition.hpo_staged import build_local_hpo_params
from geas35.models.transition.hyperparameters import (
    PERSISTENCE_CANDIDATE_NAME,
    STAGED_NONLINEAR_CANDIDATES,
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

from geas35.rl import MdpV1ObservationScaler
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
    action_response_report: ActionResponseReport
    deployment_eligible: bool
    ranking_candidate: TransitionModelRankingCandidate


@dataclass(frozen=True)
class TransitionTrainingResult:
    """End-to-end Step 8 transition training result for one crop."""

    crop: str
    validation_ranking: TransitionModelRankingResult
    candidate_results: tuple[TransitionCandidateTrainingResult, ...]
    candidate_failures: tuple[Mapping[str, Any], ...] = ()


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
    observation_scaler: MdpV1ObservationScaler | None = None,
    observation_scaler_hash: str | None = None,
    hpo_config: TransitionHPOConfig | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    target_formulation: TransitionTargetFormulation | str = TransitionTargetFormulation.ABSOLUTE_STATE,
    resume_completed: bool = False,
    feature_contract: Mapping[str, Any] | None = None,
) -> TransitionTrainingResult:
    """Train, evaluate, persist, and select transition model candidates.

    If no explicit ranking strategy is provided, rollout metrics are preferred
    when ``rollout_frame`` is present; otherwise one-step mean RMSE is used.
    """

    if rollout_frame is not None and (observation_scaler is None or not observation_scaler_hash):
        raise ValueError("Rollout requires a frozen Train observation scaler and hash.")
    if not candidate_specs:
        raise ValueError("At least one transition candidate spec is required.")
    crop_key = normalize_required_crop(crop)
    strategy = ranking_strategy or (
        None if rollout_frame is not None else MeanRmseStrategy()
    )
    hpo = hpo_config or TransitionHPOConfig()
    formulation = TransitionTargetFormulation(target_formulation)
    normalization_scales = transition_target_normalization_scales(
        validation_dataset.y,
        validation_dataset.target_columns,
    )

    candidate_results = []
    candidate_failures: list[dict[str, Any]] = []
    for spec in candidate_specs:
        base_params = _candidate_base_params(spec)
        if resume_completed:
            cached = _load_completed_candidate_result(
                models_root=models_root, crop=crop_key, spec=spec,
                hpo_config=hpo, target_formulation=formulation,
                observation_scaler_hash=observation_scaler_hash,
                artifact_metadata=artifact_metadata,
                feature_contract=feature_contract,
            )
            if cached is not None:
                candidate_results.append(cached)
                continue
        hpo_started = time.perf_counter()
        try:
            hpo_result = _run_candidate_hpo(
                spec, train_dataset=train_dataset,
                validation_dataset=validation_dataset, rollout_frame=rollout_frame,
                rollout_horizon_steps=rollout_horizon_steps,
                rollout_step_minutes=rollout_step_minutes,
                action_provider_factory=action_provider_factory,
                normalization_scales=normalization_scales, hpo_config=hpo,
                base_params=base_params, observation_scaler=observation_scaler,
                target_formulation=formulation,
            )
        except Exception as exc:
            candidate_failures.append({
                "model_name": spec.model_name, "status": "execution_error",
                "stage": "hpo", "error_type": type(exc).__name__,
                "message": str(exc),
            })
            continue
        hpo_total_time_seconds = time.perf_counter() - hpo_started
        if not hpo_result.has_eligible_trial and hpo_result.status != "not_applicable":
            candidate_failures.append({
                "model_name": spec.model_name, "status": hpo_result.status,
                "stage": "hpo", "reason": hpo_result.reason,
                "hpo_results": hpo_result.to_artifact(),
            })
            continue
        model = spec.build_with_params(hpo_result.best_params)
        _configure_target_formulation(model, observation_scaler, formulation)
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
            observation_scaler=observation_scaler,
            hpo_mode=False,
            normalization_scales=normalization_scales,
        )
        comparison_score, comparison_components = validation_rollout_weighted_score(
            one_step_report,
            rollout_metrics,
            objective_weights=hpo.objective_weights,
        )
        action_response_report = evaluate_action_response(model, validation_dataset)
        safety_passed = all(
            float(metrics.get("safety_gate_passed", 0.0)) == 1.0
            for metrics in rollout_metrics.values()
        ) if rollout_metrics else True
        deployment_eligible = (
            spec.model_name != PERSISTENCE_CANDIDATE_NAME
            and action_response_report.eligible
            and safety_passed
            and np.isfinite(comparison_score)
            and np.isfinite(one_step_report.aggregate_metrics.get("mean_rmse", np.nan))
        )
        model.deployment_eligible = deployment_eligible
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
            feature_schema=_feature_schema_payload(
                train_dataset, formulation, feature_contract=feature_contract
            ),
            one_step_report=one_step_report,
            rollout_metrics=rollout_metrics,
            training_summary={
                "stage": "transition_candidate_training",
                "crop": crop_key,
                "model_name": spec.model_name,
                "target_formulation": formulation.value,
                "feature_contract": dict(feature_contract or {}),
                "train_rows": int(len(train_dataset.x.index)),
                "validation_rows": int(len(validation_dataset.x.index)),
                "input_columns": list(train_dataset.input_columns),
                "target_columns": list(train_dataset.target_columns),
                "rollout_horizon_steps": list(rollout_horizon_steps),
                "rollout_step_minutes": int(rollout_step_minutes),
                "rollout_protocol_version": ROLLOUT_PROTOCOL_VERSION,
                "observation_scaler_sha256": observation_scaler_hash,
                "offline_exogenous_provider": "recorded_weather",
                "operating_exogenous_provider_interface": "forecast_weather",
                "training_time_seconds": training_time_seconds,
                "hpo_total_time_seconds": hpo_total_time_seconds,
                "hpo_status": hpo_result.status,
                "convergence": _model_convergence_payload(model),
                "best_config": dict(hpo_result.best_params),
                "validation_rollout_weighted_score": comparison_score,
                "validation_rollout_weighted_score_components": comparison_components,
                "test_used_for_hpo": False,
                "test_used_for_validation_ranking": False,
                "test_used_for_selection": False,
            },
            hpo_results=hpo_artifact,
            best_config=best_config_artifact,
            action_response=action_response_report,
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
                action_response_report=action_response_report,
                deployment_eligible=deployment_eligible,
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
        candidate_failures=tuple(candidate_failures),
    )



def _load_completed_candidate_result(
    *,
    models_root: Path | str,
    crop: str,
    spec: TransitionCandidateSpec,
    hpo_config: TransitionHPOConfig,
    target_formulation: TransitionTargetFormulation,
    observation_scaler_hash: str | None,
    artifact_metadata: Mapping[str, Any] | None,
    feature_contract: Mapping[str, Any] | None,
) -> TransitionCandidateTrainingResult | None:
    """Cold-load a complete compatible candidate or return None for recomputation."""

    artifact = transition_model_artifact(models_root, crop, spec.model_name)
    required = (
        artifact.model_path,
        artifact.manifest_path,
        artifact.feature_schema_path,
        artifact.one_step_metrics_path,
        artifact.rollout_metrics_path,
        artifact.resource_metrics_path,
        artifact.training_summary_path,
        artifact.hpo_results_path,
        artifact.best_config_path,
        artifact.action_response_path,
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        training = json.loads(artifact.training_summary_path.read_text(encoding="utf-8"))
        expected = target_formulation.value
        if manifest.get("target_formulation") != expected:
            return None
        if training.get("target_formulation") != expected:
            return None
        if training.get("observation_scaler_sha256") != observation_scaler_hash:
            return None
        stored_contract = dict(training.get("feature_contract", {}))
        if dict(feature_contract or {}) != stored_contract:
            return None
        expected_provenance = dict((artifact_metadata or {}).get("provenance", {}))
        stored_provenance = dict(manifest.get("metadata", {}).get("provenance", {}))
        if expected_provenance and stored_provenance != expected_provenance:
            return None
        model = load_transition_model_artifact(
            models_root,
            crop,
            model_name=spec.model_name,
            expected_target_formulation=expected,
        )
        one = json.loads(artifact.one_step_metrics_path.read_text(encoding="utf-8"))
        one_step_report = TransitionOneStepEvaluationReport(
            row_count=int(one["row_count"]),
            target_columns=tuple(one["target_columns"]),
            target_metrics=dict(one["target_metrics"]),
            aggregate_metrics=dict(one["aggregate_metrics"]),
            target_group_metrics=dict(one["target_group_metrics"]),
        )
        rollout_metrics = json.loads(artifact.rollout_metrics_path.read_text(encoding="utf-8"))
        resource_metrics = json.loads(artifact.resource_metrics_path.read_text(encoding="utf-8"))
        hpo = json.loads(artifact.hpo_results_path.read_text(encoding="utf-8"))
        best = json.loads(artifact.best_config_path.read_text(encoding="utf-8"))
        hpo_result = TransitionHPOResult(
            model_name=spec.model_name,
            status=str(hpo.get("status", "completed")),
            best_trial_index=hpo.get("best_trial_index"),
            best_params=dict(best.get("best_config", {})),
            best_score=hpo.get("best_validation_rollout_weighted_score"),
            trials=(),
            config=hpo_config,
            reason=hpo.get("reason"),
        )
        action = json.loads(artifact.action_response_path.read_text(encoding="utf-8"))
        action_response_report = ActionResponseReport(
            actions=dict(action.get("actions", {})),
            eligible=bool(action.get("eligible", False)),
            fully_action_insensitive=bool(action.get("fully_action_insensitive", False)),
            minimum_support=int(action.get("minimum_support", 100)),
        )
        score = training.get("validation_rollout_weighted_score")
        ranking_candidate = TransitionModelRankingCandidate(
            model_name=spec.model_name,
            one_step_report=one_step_report,
            rollout_metrics=rollout_metrics,
            metadata={
                **dict(spec.metadata),
                "best_config": dict(hpo_result.best_params),
                "hpo_status": hpo_result.status,
                "validation_rollout_weighted_score": score,
                "validation_rollout_weighted_score_components": training.get(
                    "validation_rollout_weighted_score_components", {}
                ),
                "resumed_from_complete_artifact": True,
                "test_used_for_hpo": False,
                "test_used_for_validation_ranking": False,
                "test_used_for_selection": False,
            },
        )
        return TransitionCandidateTrainingResult(
            model_name=spec.model_name,
            model=model,
            artifact=artifact,
            one_step_report=one_step_report,
            rollout_metrics=rollout_metrics,
            resource_metrics=resource_metrics,
            hpo_result=hpo_result,
            action_response_report=action_response_report,
            deployment_eligible=bool(manifest.get("deployment_eligible", False)),
            ranking_candidate=ranking_candidate,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

def _evaluate_rollouts(
    model: BaseTransitionModel,
    *,
    rollout_frame: pd.DataFrame | None,
    rollout_horizon_steps: Sequence[int],
    rollout_step_minutes: int,
    action_provider_factory: Callable[[], ActionProvider] | None,
    normalization_scales: Mapping[str, float] | None = None,
    observation_scaler: MdpV1ObservationScaler | None = None,
    hpo_mode: bool = False,
) -> dict[str, dict[str, float]]:
    if rollout_frame is None:
        return {}
    if rollout_step_minutes <= 0 or observation_scaler is None:
        raise ValueError("Rollout requires a positive step and frozen Train scaler.")
    horizons = tuple(sorted(set(int(value) for value in rollout_horizon_steps)))
    if not horizons or horizons[-1] != 12:
        raise ValueError("Official rollout protocol requires a 12-step maximum horizon.")
    windows = build_rollout_windows(
        rollout_frame, horizon_steps=12, stride_steps=(1 if hpo_mode else 12),
        max_windows=(32 if hpo_mode else None), episode_balanced=hpo_mode,
    )
    if not windows:
        raise ValueError("No valid causal rollout windows were found.")
    by_horizon: dict[int, list[dict[str, float]]] = {h: [] for h in horizons}
    pooled_predictions: dict[int, list[pd.DataFrame]] = {h: [] for h in horizons}
    pooled_errors: dict[int, list[pd.DataFrame]] = {h: [] for h in horizons}
    window_ids: dict[int, list[str]] = {h: [] for h in horizons}
    failure = False
    for window in windows:
        provider = action_provider_factory() if action_provider_factory else LoggedActionProvider()
        result = TransitionRolloutSimulator(
            model, scaler=observation_scaler, action_provider=provider
        ).simulate(rollout_frame, start_index=window.start_position, horizon_steps=12)
        for horizon in horizons:
            predictions = result.predictions.iloc[:horizon].reset_index(drop=True)
            errors = result.errors.iloc[:horizon].reset_index(drop=True)
            pooled_predictions[horizon].append(predictions)
            pooled_errors[horizon].append(errors)
            item = dict(result.metrics) if horizon == 12 else _rollout_metrics_for_slice(predictions, errors)
            if normalization_scales is not None:
                item = enrich_rollout_metrics_with_normalized_errors(
                    item, errors, normalization_scales
                )
            failure |= bool(item.get("nan_inf_count", 0.0)) or bool(
                item.get("physical_violation_rate", 0.0)
            )
            by_horizon[horizon].append(item)
            window_ids[horizon].append(
                f"{window.crop}|{window.episode_id}|{window.start_position}"
            )
    metrics: dict[str, dict[str, float]] = {}
    for horizon, rows in by_horizon.items():
        keys = sorted(set().union(*(row.keys() for row in rows)))
        aggregate = {}
        for key in keys:
            values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                aggregate[key] = float("nan")
                continue
            aggregate[key] = float(np.mean(finite))
            aggregate[f"{key}_median"] = float(np.median(finite))
            aggregate[f"{key}_q90"] = float(np.quantile(finite, 0.90))
            cutoff = float(np.quantile(finite, 0.90))
            aggregate[f"{key}_cvar90"] = float(np.mean(finite[finite >= cutoff]))
        pooled_prediction = pd.concat(pooled_predictions[horizon], ignore_index=True)
        pooled_error = pd.concat(pooled_errors[horizon], ignore_index=True)
        pooled = _rollout_metrics_for_slice(pooled_prediction, pooled_error)
        if normalization_scales is not None:
            pooled = enrich_rollout_metrics_with_normalized_errors(
                pooled, pooled_error, normalization_scales
            )
        for key, value in pooled.items():
            aggregate[f"pooled_{key}"] = float(value)
            if (
                key in {"trajectory_mae", "trajectory_rmse", "trajectory_nrmse", "trajectory_r2"}
                or "_trajectory_" in key
            ):
                aggregate[key] = float(value)
        aggregate["window_ids"] = list(window_ids[horizon])
        aggregate["window_trajectory_nrmse_samples"] = [
            float(row.get("trajectory_nrmse", np.nan)) for row in rows
        ]
        aggregate["window_drift_nmae_samples"] = [
            float(row.get("drift_slope_nmae", np.nan)) for row in rows
        ]
        aggregate["r2_aggregation"] = "pooled_all_valid_windows"
        aggregate["window_count"] = float(len(rows))
        aggregate["safety_gate_passed"] = float(not failure)
        aggregate["protocol_version_code"] = 2.0
        metrics[f"{horizon * rollout_step_minutes}min"] = aggregate
    return metrics

def _rollout_metrics_for_slice(predictions: pd.DataFrame, errors: pd.DataFrame) -> dict[str, float]:
    from geas35.models.transition.rollout import _rollout_metrics
    return _rollout_metrics(predictions, errors)


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
    observation_scaler: MdpV1ObservationScaler | None,
    target_formulation: TransitionTargetFormulation,
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
    local_params = None
    for trial_index, params in enumerate(sampled_params, start=1):
        if (
            hpo_config.staged_nonlinear
            and spec.model_name in STAGED_NONLINEAR_CANDIDATES
            and trial_index == 21
        ):
            eligible_broad = [trial for trial in trials if trial.eligible]
            top_three = [
                trial.params for trial in sorted(
                    eligible_broad, key=lambda trial: trial.validation_rollout_weighted_score
                )[:3]
            ]
            if not top_three:
                break
            local_params = build_local_hpo_params(
                top_three, count=max(0, len(sampled_params) - 20),
                seed=43, forbidden=sampled_params[:20],
            )
        if local_params is not None and trial_index > 20:
            params = local_params[trial_index - 21]
        error_type = error_message = None
        try:
            model = spec.build_with_params(params)
            _configure_target_formulation(model, observation_scaler, target_formulation)
            _require_transition_model(model, spec.model_name)
            model.fit(hpo_train_dataset, validation_dataset=hpo_validation_dataset)
            one_step_report = evaluate_transition_model_one_step(
                model, hpo_validation_dataset, normalization_scales=normalization_scales,
            )
            rollout_metrics = _evaluate_rollouts(
                model, rollout_frame=rollout_frame,
                rollout_horizon_steps=rollout_horizon_steps,
                observation_scaler=observation_scaler, hpo_mode=True,
                rollout_step_minutes=rollout_step_minutes,
                action_provider_factory=action_provider_factory,
                normalization_scales=normalization_scales,
            )
            score, components = validation_rollout_weighted_score(
                one_step_report, rollout_metrics,
                objective_weights=hpo_config.objective_weights,
            )
            action_report = evaluate_action_response(model, hpo_validation_dataset)
            if hpo_config.strict_eligibility:
                eligible, reasons = _hpo_trial_eligibility(
                    one_step_report, rollout_metrics, action_report, score
                )
            else:
                eligible = bool(np.isfinite(score))
                reasons = () if eligible else ("non_finite_objective",)
            one_step_artifact = one_step_report.to_artifact()
        except Exception as exc:
            score = float("inf")
            components = {}
            rollout_metrics = {}
            one_step_artifact = {}
            eligible = False
            reasons = ("trial_execution_error",)
            error_type, error_message = type(exc).__name__, str(exc)
        trials.append(
            TransitionHPOTrialResult(
                trial_index=trial_index,
                model_name=spec.model_name,
                params=dict(params),
                validation_one_step_metrics=one_step_artifact,
                validation_rollout_metrics=rollout_metrics,
                objective_components=components,
                validation_rollout_weighted_score=score,
                search_stage=(
                    "broad" if hpo_config.staged_nonlinear and spec.model_name in STAGED_NONLINEAR_CANDIDATES and trial_index <= 20
                    else "local" if hpo_config.staged_nonlinear and spec.model_name in STAGED_NONLINEAR_CANDIDATES
                    else "single"
                ),
                search_seed=(43 if hpo_config.staged_nonlinear and spec.model_name in STAGED_NONLINEAR_CANDIDATES and trial_index > 20 else hpo_config.random_seed),
                eligible=eligible, ineligibility_reasons=reasons,
                error_type=error_type, error_message=error_message,
            )
        )
        if eligible and (best_trial_index is None or score < best_model_score):
            best_model_score = score
            best_trial_index = trial_index
            best_params = dict(params)

    if best_trial_index is None:
        return TransitionHPOResult(
            model_name=spec.model_name, status="no_eligible_hpo_trial",
            best_trial_index=None, best_params={}, best_score=None,
            trials=tuple(trials), config=hpo_config,
            reason="All HPO trials failed finite, rollout safety, physical, or action-response gates.",
        )
    return TransitionHPOResult(
        model_name=spec.model_name, status="completed",
        best_trial_index=best_trial_index, best_params=best_params,
        best_score=best_model_score, trials=tuple(trials), config=hpo_config,
    )


def _hpo_trial_eligibility(one_step_report, rollout_metrics, action_report, score):
    reasons: list[str] = []
    if not np.isfinite(float(score)):
        reasons.append("non_finite_objective")
    one_values = list(one_step_report.aggregate_metrics.values())
    if any(isinstance(v, (int, float)) and not np.isfinite(float(v)) for v in one_values):
        reasons.append("non_finite_one_step")
    for horizon, metrics in rollout_metrics.items():
        numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
        if any(not np.isfinite(float(v)) for v in numeric):
            reasons.append(f"{horizon}:non_finite")
        if float(metrics.get("nan_inf_count", 0.0)) != 0.0:
            reasons.append(f"{horizon}:nan_inf")
        if float(metrics.get("physical_violation_rate", 0.0)) != 0.0:
            reasons.append(f"{horizon}:physical_violation")
        if float(metrics.get("safety_gate_passed", 0.0)) != 1.0:
            reasons.append(f"{horizon}:safety_gate")
    if not action_report.eligible:
        reasons.append("action_response_gate")
    return not reasons, tuple(dict.fromkeys(reasons))


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


def _model_convergence_payload(model: BaseTransitionModel) -> dict[str, Any]:
    estimators = getattr(model, "estimators_", None) or {}
    targets = {}
    for target, estimator in estimators.items():
        n_iter = getattr(estimator, "n_iter_", None)
        max_iter = getattr(estimator, "max_iter", None)
        converged = None
        if n_iter is not None and max_iter is not None:
            value = int(np.max(np.asarray(n_iter)))
            converged = value < int(max_iter)
            n_iter = value
        targets[str(target)] = {
            "n_iter": n_iter,
            "max_iter": max_iter,
            "converged_before_max_iter": converged,
        }
    return {"available": bool(targets), "targets": targets}


def _configure_target_formulation(
    model: BaseTransitionModel,
    observation_scaler: MdpV1ObservationScaler | None,
    formulation: TransitionTargetFormulation,
) -> None:
    configure = getattr(model, "configure_target_formulation", None)
    if callable(configure):
        configure(formulation, observation_scaler=observation_scaler)
        return
    model.target_formulation = formulation.value
    if hasattr(model, "observation_scaler"):
        model.observation_scaler = observation_scaler


def _candidate_base_params(spec: TransitionCandidateSpec) -> Mapping[str, Any]:
    params = spec.metadata.get("params", {})
    return params if isinstance(params, Mapping) else {}


def _require_transition_model(model: BaseTransitionModel, model_name: str) -> None:
    if not isinstance(model, BaseTransitionModel):
        raise TypeError(
            f"Transition candidate {model_name!r} must build a "
            "BaseTransitionModel instance."
        )


def _feature_schema_payload(
    dataset: TransitionDataset, formulation: TransitionTargetFormulation,
    *, feature_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": "transition_feature_schema",
        "schema_version": "geas35.transition.three_target.v2",
        **dict(feature_contract or {}),
        "target_contract": "official_three_target",
        "target_formulation": formulation.value,
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
