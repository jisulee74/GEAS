"""Threshold calibration runner for quality-model experiments.

This module implements Experiment Plan v1.1 Step 3 only. It consumes Step 2
HPO results, fixes each model's best hyperparameter configuration, trains that
configuration, and calibrates anomaly-score thresholds on validation data.
Validation/test evaluation, online benchmarking, comparison reports, and
visualization are intentionally left to later experiment steps.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from geas35.experiments.quality.hpo_runner import ModelHPOResult, QualityHPOResult
from geas35.experiments.quality.runner import (
    ModelImplementation,
    default_quality_model_registry,
)
from geas35.models.quality import (
    BaseQualityModel,
    EarlyStoppingConfig,
    GridSearchThresholdOptimizer,
    ThresholdOptimizationResult,
)


@dataclass(frozen=True)
class QualityThresholdCalibrationConfig:
    """Configuration required by the Step 3 threshold calibration runner."""

    output_root: Path
    threshold_candidates: tuple[float, ...]
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    objective_metric: str = "f1_score"
    mask_fraction: float = 0.1
    anomaly_fraction: float = 0.1
    anomaly_scale: float = 8.0
    random_state: int | None = 0
    include_masking: bool = True


@dataclass(frozen=True)
class ModelThresholdCalibrationResult:
    """Fitted best-config model and threshold calibration artifacts."""

    model_name: str
    model: BaseQualityModel
    hpo_result: Any
    threshold_result: ThresholdOptimizationResult
    artifact_dir: str


@dataclass(frozen=True)
class QualityThresholdCalibrationResult:
    """Full Step 3 threshold calibration result."""

    output_root: str
    model_results: list[ModelThresholdCalibrationResult]
    threshold_calibration_path: str


def run_quality_threshold_calibration(
    *,
    config: QualityThresholdCalibrationConfig,
    hpo_result: QualityHPOResult,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    observation_columns: Iterable[str],
    registry: Mapping[str, ModelImplementation] | None = None,
) -> QualityThresholdCalibrationResult:
    """Run threshold calibration for each Step 2 HPO-selected model."""

    columns = tuple(observation_columns)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_registry = default_quality_model_registry() if registry is None else dict(registry)
    _write_json(output_root / "threshold_config.json", _config_payload(config, columns))

    model_results: list[ModelThresholdCalibrationResult] = []
    for model_hpo_result in hpo_result.model_results:
        implementation = _implementation_for(model_hpo_result.model_name, model_registry)
        model_results.append(
            run_model_threshold_calibration(
                config=config,
                model_hpo_result=model_hpo_result,
                implementation=implementation,
                train_df=train_df,
                validation_df=validation_df,
                observation_columns=columns,
                output_root=output_root,
            )
        )

    threshold_calibration_path = output_root / "threshold_calibration.json"
    _write_json(
        threshold_calibration_path,
        _aggregate_threshold_payload(model_results),
    )
    return QualityThresholdCalibrationResult(
        output_root=str(output_root),
        model_results=model_results,
        threshold_calibration_path=str(threshold_calibration_path),
    )


def run_model_threshold_calibration(
    *,
    config: QualityThresholdCalibrationConfig,
    model_hpo_result: ModelHPOResult,
    implementation: ModelImplementation,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    observation_columns: Iterable[str],
    output_root: str | Path,
) -> ModelThresholdCalibrationResult:
    """Calibrate one model's threshold after fixing its HPO best config."""

    columns = tuple(observation_columns)
    model_dir = Path(output_root) / model_hpo_result.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model = implementation.build_model(model_hpo_result.hpo_result.best_candidate)
    _fit_final_model(
        model,
        train_df,
        validation_df,
        columns,
        config.early_stopping,
    )
    threshold_result = GridSearchThresholdOptimizer(
        config.threshold_candidates,
        objective_metric=config.objective_metric,
        anomaly_fraction=config.anomaly_fraction,
        anomaly_scale=config.anomaly_scale,
        mask_fraction=config.mask_fraction,
        random_state=config.random_state,
        include_masking=config.include_masking,
    ).optimize(model, validation_df, columns)
    _write_json(
        model_dir / "threshold_calibration.json",
        _threshold_payload(threshold_result),
    )
    return ModelThresholdCalibrationResult(
        model_name=model_hpo_result.model_name,
        model=model,
        hpo_result=model_hpo_result.hpo_result,
        threshold_result=threshold_result,
        artifact_dir=str(model_dir),
    )


def quality_threshold_config_from_experiment_config(
    config,
) -> QualityThresholdCalibrationConfig:
    """Create a Step 3 calibration config from the broader experiment config."""

    return QualityThresholdCalibrationConfig(
        output_root=Path(config.output_root),
        threshold_candidates=tuple(config.threshold_candidates),
        early_stopping=config.early_stopping,
        mask_fraction=config.mask_fraction,
        anomaly_fraction=config.anomaly_fraction,
        anomaly_scale=config.anomaly_scale,
        random_state=config.evaluation_random_seed,
    )


def _fit_final_model(
    model: BaseQualityModel,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    columns: tuple[str, ...],
    early_stopping: EarlyStoppingConfig,
) -> BaseQualityModel:
    signature = inspect.signature(model.fit)
    kwargs: dict[str, Any] = {}
    if "validation_df" in signature.parameters:
        kwargs["validation_df"] = validation_df
    if "early_stopping" in signature.parameters:
        kwargs["early_stopping"] = early_stopping
    return model.fit(train_df, columns, **kwargs)


def _implementation_for(
    model_name: str,
    registry: Mapping[str, ModelImplementation],
) -> ModelImplementation:
    if model_name not in registry:
        raise ValueError(f"Unknown experiment model: {model_name}")
    return registry[model_name]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _config_payload(
    config: QualityThresholdCalibrationConfig,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "stage": "quality_model_threshold_calibration",
        "observation_columns": list(columns),
        "threshold_candidates": list(config.threshold_candidates),
        "early_stopping": config.early_stopping.to_artifact(),
        "objective_metric": config.objective_metric,
        "mask_fraction": config.mask_fraction,
        "anomaly_fraction": config.anomaly_fraction,
        "anomaly_scale": config.anomaly_scale,
        "random_state": config.random_state,
        "include_masking": config.include_masking,
        "threshold_calibration_after_hpo": True,
        "threshold_is_hpo_parameter": False,
    }


def _aggregate_threshold_payload(
    model_results: Iterable[ModelThresholdCalibrationResult],
) -> dict[str, Any]:
    return {
        "stage": "threshold_calibration_summary",
        "threshold_calibration_after_hpo": True,
        "threshold_is_hpo_parameter": False,
        "models": {
            result.model_name: _threshold_payload(result.threshold_result)
            for result in model_results
        },
    }


def _threshold_payload(result: ThresholdOptimizationResult) -> dict[str, Any]:
    metrics = result.best_detection_result.metrics
    cm = metrics.confusion_matrix
    return {
        "best_threshold": result.best_threshold,
        "best_objective_value": result.best_objective_value,
        "objective_metric": result.objective_metric,
        "threshold_calibration_after_hpo": True,
        "threshold_is_hpo_parameter": False,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "confusion_matrix": {
            "true_positive": cm.true_positive,
            "false_positive": cm.false_positive,
            "true_negative": cm.true_negative,
            "false_negative": cm.false_negative,
        },
        "candidate_results": [
            _threshold_candidate_payload(detection)
            for detection in result.detection_results
        ],
    }


def _threshold_candidate_payload(detection) -> dict[str, Any]:
    metrics = detection.metrics
    cm = metrics.confusion_matrix
    return {
        "threshold": detection.threshold,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "confusion_matrix": {
            "true_positive": cm.true_positive,
            "false_positive": cm.false_positive,
            "true_negative": cm.true_negative,
            "false_negative": cm.false_negative,
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


__all__ = [
    "ModelThresholdCalibrationResult",
    "QualityThresholdCalibrationConfig",
    "QualityThresholdCalibrationResult",
    "quality_threshold_config_from_experiment_config",
    "run_model_threshold_calibration",
    "run_quality_threshold_calibration",
]
