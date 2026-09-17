"""Validation/test evaluation runner for quality-model experiments.

This module implements Experiment Plan v1.1 Step 4 only. It evaluates each
HPO-selected and threshold-calibrated model on validation and test splits for
reconstruction and anomaly-detection metrics. Online benchmarking, comparison
tables, markdown reports, and visualization are intentionally left to later
experiment steps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from geas35.experiments.quality.calibration_runner import (
    ModelThresholdCalibrationResult,
    QualityThresholdCalibrationResult,
)
from geas35.models.quality import (
    QualityModelEvaluationReport,
    evaluate_quality_model_for_report,
)


@dataclass(frozen=True)
class QualityEvaluationConfig:
    """Configuration required by the Step 4 validation/test evaluation runner."""

    output_root: Path
    mask_fraction: float = 0.1
    anomaly_fraction: float = 0.1
    anomaly_scale: float = 8.0
    random_state: int | None = 0
    include_efficiency: bool = False


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Validation/test evaluation reports for one calibrated model."""

    model_name: str
    validation_report: QualityModelEvaluationReport
    test_report: QualityModelEvaluationReport
    artifact_dir: str


@dataclass(frozen=True)
class QualityEvaluationResult:
    """Full Step 4 validation/test evaluation result."""

    output_root: str
    model_results: list[ModelEvaluationResult]
    evaluation_path: str


def run_quality_validation_test_evaluation(
    *,
    config: QualityEvaluationConfig,
    calibration_result: QualityThresholdCalibrationResult,
    reference_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    observation_columns: Iterable[str],
) -> QualityEvaluationResult:
    """Evaluate calibrated models on validation and test splits."""

    columns = tuple(observation_columns)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "evaluation_config.json", _config_payload(config, columns))

    model_results = [
        run_model_validation_test_evaluation(
            config=config,
            model_result=model_result,
            reference_df=reference_df,
            validation_df=validation_df,
            test_df=test_df,
            observation_columns=columns,
            output_root=output_root,
        )
        for model_result in calibration_result.model_results
    ]

    evaluation_path = output_root / "evaluation.json"
    _write_json(evaluation_path, _aggregate_evaluation_payload(model_results))
    return QualityEvaluationResult(
        output_root=str(output_root),
        model_results=model_results,
        evaluation_path=str(evaluation_path),
    )


def run_model_validation_test_evaluation(
    *,
    config: QualityEvaluationConfig,
    model_result: ModelThresholdCalibrationResult,
    reference_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    observation_columns: Iterable[str],
    output_root: str | Path,
) -> ModelEvaluationResult:
    """Evaluate one calibrated model on validation and test splits."""

    columns = tuple(observation_columns)
    crop_name = Path(output_root).name
    print(
        f"[{crop_name}][{model_result.model_name}] Validation/Test 평가 시작",
        flush=True,
    )
    model_dir = Path(output_root) / model_result.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    threshold = model_result.threshold_result.best_threshold
    validation_report = evaluate_quality_model_for_report(
        model_result.model,
        validation_df,
        columns,
        split="validation",
        calibrated_threshold=threshold,
        mask_fraction=config.mask_fraction,
        anomaly_fraction=config.anomaly_fraction,
        anomaly_scale=config.anomaly_scale,
        random_state=config.random_state,
        include_efficiency=False,
        reference_df=reference_df,
    )
    test_report = evaluate_quality_model_for_report(
        model_result.model,
        test_df,
        columns,
        split="test",
        calibrated_threshold=threshold,
        mask_fraction=config.mask_fraction,
        anomaly_fraction=config.anomaly_fraction,
        anomaly_scale=config.anomaly_scale,
        random_state=config.random_state,
        include_efficiency=False,
        reference_df=reference_df,
    )
    result = ModelEvaluationResult(
        model_name=model_result.model_name,
        validation_report=validation_report,
        test_report=test_report,
        artifact_dir=str(model_dir),
    )
    _write_json(model_dir / "evaluation.json", _model_evaluation_payload(result))
    print(
        f"[{crop_name}][{model_result.model_name}] Validation/Test 평가 완료",
        flush=True,
    )
    return result


def quality_evaluation_config_from_experiment_config(config) -> QualityEvaluationConfig:
    """Create a Step 4 evaluation config from the broader experiment config."""

    return QualityEvaluationConfig(
        output_root=Path(config.output_root),
        mask_fraction=config.mask_fraction,
        anomaly_fraction=config.anomaly_fraction,
        anomaly_scale=config.anomaly_scale,
        random_state=config.evaluation_random_seed,
        include_efficiency=False,
    )


def _config_payload(
    config: QualityEvaluationConfig,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "stage": "quality_model_validation_test_evaluation",
        "observation_columns": list(columns),
        "mask_fraction": config.mask_fraction,
        "anomaly_fraction": config.anomaly_fraction,
        "anomaly_scale": config.anomaly_scale,
        "random_state": config.random_state,
        "include_efficiency": False,
        "online_benchmark_included": False,
        "automatic_best_model_selection": False,
        "test_used_for_selection": False,
    }


def _aggregate_evaluation_payload(
    model_results: Iterable[ModelEvaluationResult],
) -> dict[str, Any]:
    return {
        "stage": "quality_model_validation_test_evaluation",
        "automatic_best_model_selection": False,
        "test_used_for_selection": False,
        "online_benchmark_included": False,
        "models": {
            result.model_name: _model_evaluation_payload(result)
            for result in model_results
        },
    }


def _model_evaluation_payload(result: ModelEvaluationResult) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "validation": _report_payload(result.validation_report),
        "test": _report_payload(result.test_report),
        "automatic_best_model_selection": False,
        "test_used_for_selection": False,
        "online_benchmark_included": False,
    }


def _report_payload(report: QualityModelEvaluationReport) -> dict[str, Any]:
    return asdict(report)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    "ModelEvaluationResult",
    "QualityEvaluationConfig",
    "QualityEvaluationResult",
    "quality_evaluation_config_from_experiment_config",
    "run_model_validation_test_evaluation",
    "run_quality_validation_test_evaluation",
]
