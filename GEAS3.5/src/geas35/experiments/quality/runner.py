"""Experiment-layer runner for GEAS AI Quality Model comparisons."""

from __future__ import annotations

import csv
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from geas35.experiments.quality.search import (
    RandomSearchCandidateGenerator,
    SearchSpace,
)
from geas35.models.quality import (
    BaseQualityModel,
    EarlyStoppingConfig,
    GridSearchHyperparameterOptimizer,
    GridSearchThresholdOptimizer,
    HyperparameterCandidate,
    HyperparameterOptimizationResult,
    ModernTCNConfig,
    ModernTCNQualityModel,
    PatchTSTConfig,
    PatchTSTQualityModel,
    QualityModelEvaluationReport,
    ThresholdOptimizationResult,
    TimesNetConfig,
    TimesNetQualityModel,
    evaluate_quality_model_for_report,
    train_modern_tcn_candidate,
    train_patch_tst_candidate,
    train_timesnet_candidate,
)
from geas35.models.quality.hyperparameters import CandidateTrainingResult


TrainCandidateFn = Callable[
    [
        HyperparameterCandidate,
        pd.DataFrame,
        pd.DataFrame,
        tuple[str, ...],
        EarlyStoppingConfig,
    ],
    CandidateTrainingResult,
]
BuildModelFn = Callable[[HyperparameterCandidate], BaseQualityModel]


@dataclass(frozen=True)
class ModelImplementation:
    """Experiment-layer binding between a model name and train/build functions."""

    train_candidate: TrainCandidateFn
    build_model: BuildModelFn


@dataclass(frozen=True)
class ModelExperimentSpec:
    """Experiment config for one model."""

    model_name: str
    search_space: Mapping[str, Any]
    hpo_budget: int
    random_seed: int = 0


@dataclass(frozen=True)
class QualityExperimentConfig:
    """End-to-end experiment config excluding dataset payloads."""

    output_root: Path
    models: tuple[ModelExperimentSpec, ...]
    threshold_candidate_count: int
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    mask_fraction: float = 0.1
    anomaly_fraction: float = 0.1
    anomaly_scale: float = 8.0
    evaluation_random_seed: int = 0
    efficiency_repeats: int = 3


@dataclass(frozen=True)
class ModelExperimentResult:
    """Artifacts and metrics produced for one model experiment."""

    model_name: str
    hpo_result: HyperparameterOptimizationResult
    threshold_result: ThresholdOptimizationResult
    validation_report: QualityModelEvaluationReport
    test_report: QualityModelEvaluationReport
    artifact_dir: str


@dataclass(frozen=True)
class QualityExperimentResult:
    """Full no-selection quality-model experiment result."""

    output_root: str
    model_results: list[ModelExperimentResult]
    hpo_results_path: str
    best_config_path: str
    threshold_calibration_path: str
    evaluation_path: str
    training_history_path: str
    comparison_json_path: str
    comparison_csv_path: str
    markdown_summary_path: str


def default_quality_model_registry() -> dict[str, ModelImplementation]:
    """Return default experiment registry for v1.4 deep quality models."""

    return {
        "modern_tcn": ModelImplementation(
            train_candidate=train_modern_tcn_candidate,
            build_model=lambda candidate: ModernTCNQualityModel(
                ModernTCNConfig.from_candidate(candidate)
            ),
        ),
        "timesnet": ModelImplementation(
            train_candidate=train_timesnet_candidate,
            build_model=lambda candidate: TimesNetQualityModel(
                TimesNetConfig.from_candidate(candidate)
            ),
        ),
        "patch_tst": ModelImplementation(
            train_candidate=train_patch_tst_candidate,
            build_model=lambda candidate: PatchTSTQualityModel(
                PatchTSTConfig.from_candidate(candidate)
            ),
        ),
    }


def run_quality_model_experiment(
    *,
    config: QualityExperimentConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    observation_columns: Iterable[str],
    registry: Mapping[str, ModelImplementation] | None = None,
) -> QualityExperimentResult:
    """Run Random Search HPO, threshold calibration, and no-selection reporting."""

    columns = tuple(observation_columns)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_registry = default_quality_model_registry() if registry is None else dict(registry)
    write_json(output_root / "config.json", _config_payload(config, columns))

    model_results: list[ModelExperimentResult] = []
    comparison_rows: list[dict[str, Any]] = []
    for model_spec in config.models:
        if model_spec.model_name not in model_registry:
            raise ValueError(f"Unknown experiment model: {model_spec.model_name}")
        implementation = model_registry[model_spec.model_name]
        model_dir = output_root / model_spec.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        candidates = RandomSearchCandidateGenerator(
            model_name=model_spec.model_name,
            search_space=SearchSpace.from_config(model_spec.search_space),
            budget=model_spec.hpo_budget,
            random_seed=model_spec.random_seed,
        ).generate()
        hpo_result = GridSearchHyperparameterOptimizer(
            candidates,
            implementation.train_candidate,
            early_stopping=config.early_stopping,
        ).optimize(train_df, validation_df, columns)
        write_json(model_dir / "hpo_results.json", hpo_result.to_artifact())
        write_json(model_dir / "best_config.json", hpo_result.best_candidate.to_artifact())
        write_training_history_csv(
            model_dir / "training_history.csv",
            hpo_result.best_result.training_history,
        )

        final_model = implementation.build_model(hpo_result.best_candidate)
        _fit_final_model(
            final_model,
            train_df,
            validation_df,
            columns,
            config.early_stopping,
        )
        threshold_result = GridSearchThresholdOptimizer(
            candidate_count=config.threshold_candidate_count,
            anomaly_fraction=config.anomaly_fraction,
            anomaly_scale=config.anomaly_scale,
            mask_fraction=config.mask_fraction,
            random_state=config.evaluation_random_seed,
            progress_context=f"{Path(config.output_root).name}][{model_spec.model_name}",
        ).optimize(final_model, validation_df, columns)
        write_json(
            model_dir / "threshold_calibration.json",
            _threshold_payload(threshold_result),
        )

        validation_report = evaluate_quality_model_for_report(
            final_model,
            validation_df,
            columns,
            split="validation",
            calibrated_threshold=threshold_result.best_threshold,
            mask_fraction=config.mask_fraction,
            anomaly_fraction=config.anomaly_fraction,
            anomaly_scale=config.anomaly_scale,
            random_state=config.evaluation_random_seed,
            efficiency_repeats=config.efficiency_repeats,
        )
        test_report = evaluate_quality_model_for_report(
            final_model,
            test_df,
            columns,
            split="test",
            calibrated_threshold=threshold_result.best_threshold,
            mask_fraction=config.mask_fraction,
            anomaly_fraction=config.anomaly_fraction,
            anomaly_scale=config.anomaly_scale,
            random_state=config.evaluation_random_seed,
            efficiency_repeats=config.efficiency_repeats,
        )
        write_json(
            model_dir / "evaluation.json",
            {
                "model_name": model_spec.model_name,
                "validation": _report_payload(validation_report),
                "test": _report_payload(test_report),
                "test_used_for_selection": False,
                "automatic_best_model_selection": False,
            },
        )

        result = ModelExperimentResult(
            model_name=model_spec.model_name,
            hpo_result=hpo_result,
            threshold_result=threshold_result,
            validation_report=validation_report,
            test_report=test_report,
            artifact_dir=str(model_dir),
        )
        model_results.append(result)
        comparison_rows.append(_comparison_row(result))

    comparison_json_path = output_root / "model_comparison.json"
    comparison_csv_path = output_root / "model_comparison.csv"
    markdown_summary_path = output_root / "markdown_summary.md"
    hpo_results_path = output_root / "hpo_results.json"
    best_config_path = output_root / "best_config.json"
    threshold_calibration_path = output_root / "threshold_calibration.json"
    evaluation_path = output_root / "evaluation.json"
    training_history_path = output_root / "training_history.csv"
    write_json(hpo_results_path, _aggregate_hpo_payload(model_results))
    write_json(best_config_path, _aggregate_best_config_payload(model_results))
    write_json(
        threshold_calibration_path,
        _aggregate_threshold_payload(model_results),
    )
    write_json(evaluation_path, _aggregate_evaluation_payload(model_results))
    write_aggregate_training_history_csv(training_history_path, model_results)
    write_json(
        comparison_json_path,
        {
            "stage": "quality_model_experiment_comparison",
            "automatic_best_model_selection": False,
            "test_used_for_selection": False,
            "models": comparison_rows,
        },
    )
    write_comparison_csv(comparison_csv_path, comparison_rows)
    write_markdown_summary(markdown_summary_path, comparison_rows)
    return QualityExperimentResult(
        output_root=str(output_root),
        model_results=model_results,
        hpo_results_path=str(hpo_results_path),
        best_config_path=str(best_config_path),
        threshold_calibration_path=str(threshold_calibration_path),
        evaluation_path=str(evaluation_path),
        training_history_path=str(training_history_path),
        comparison_json_path=str(comparison_json_path),
        comparison_csv_path=str(comparison_csv_path),
        markdown_summary_path=str(markdown_summary_path),
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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


def write_training_history_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    row_list = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in row_list for key in row})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def write_aggregate_training_history_csv(
    path: Path,
    model_results: Iterable[ModelExperimentResult],
) -> None:
    """Write all best-candidate training histories into one root CSV artifact."""

    rows: list[dict[str, Any]] = []
    for result in model_results:
        candidate_name = result.hpo_result.best_candidate.name
        for row in result.hpo_result.best_result.training_history:
            rows.append(
                {
                    "model_name": result.model_name,
                    "candidate_name": candidate_name,
                    **dict(row),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# GEAS AI Quality Model Experiment Summary",
        "",
        "This report does not automatically select a best model.",
        "",
        "| Model | Validation RMSE | Validation F1 | Test RMSE | Test F1 | Latency ms/row | Model Size bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model_name} | {validation_rmse} | {validation_f1} | "
            "{test_rmse} | {test_f1} | {validation_latency_ms_per_row} | "
            "{model_size_bytes} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config_payload(config: QualityExperimentConfig, columns: tuple[str, ...]) -> dict[str, Any]:
    return {
        "observation_columns": list(columns),
        "models": [asdict(model) for model in config.models],
        "threshold_candidate_generation": "validation_reconstruction_error_linspace",
        "threshold_candidate_count": config.threshold_candidate_count,
        "early_stopping": config.early_stopping.to_artifact(),
        "mask_fraction": config.mask_fraction,
        "anomaly_fraction": config.anomaly_fraction,
        "anomaly_scale": config.anomaly_scale,
        "evaluation_random_seed": config.evaluation_random_seed,
        "efficiency_repeats": config.efficiency_repeats,
        "threshold_is_hpo_parameter": False,
        "automatic_best_model_selection": False,
    }


def _threshold_payload(result: ThresholdOptimizationResult) -> dict[str, Any]:
    metrics = result.best_detection_result.metrics
    cm = metrics.confusion_matrix
    return {
        "best_threshold": result.best_threshold,
        "best_objective_value": result.best_objective_value,
        "objective_metric": result.objective_metric,
        "candidate_generation": result.candidate_generation,
        "validation_error_min": result.validation_error_min,
        "validation_error_max": result.validation_error_max,
        "requested_candidate_count": result.requested_candidate_count,
        "actual_candidate_count": result.actual_candidate_count,
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


def _report_payload(report: QualityModelEvaluationReport) -> dict[str, Any]:
    return asdict(report)


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


def _aggregate_hpo_payload(
    model_results: Iterable[ModelExperimentResult],
) -> dict[str, Any]:
    return {
        "stage": "hyperparameter_optimization_summary",
        "threshold_calibration_included": False,
        "threshold_is_hpo_parameter": False,
        "models": {
            result.model_name: result.hpo_result.to_artifact()
            for result in model_results
        },
    }


def _aggregate_best_config_payload(
    model_results: Iterable[ModelExperimentResult],
) -> dict[str, Any]:
    return {
        "stage": "best_hyperparameter_configurations",
        "threshold_calibration_included": False,
        "models": {
            result.model_name: result.hpo_result.best_candidate.to_artifact()
            for result in model_results
        },
    }


def _aggregate_threshold_payload(
    model_results: Iterable[ModelExperimentResult],
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


def _aggregate_evaluation_payload(
    model_results: Iterable[ModelExperimentResult],
) -> dict[str, Any]:
    return {
        "stage": "quality_model_evaluation_summary",
        "automatic_best_model_selection": False,
        "test_used_for_selection": False,
        "models": {
            result.model_name: {
                "validation": _report_payload(result.validation_report),
                "test": _report_payload(result.test_report),
            }
            for result in model_results
        },
    }


def _comparison_row(result: ModelExperimentResult) -> dict[str, Any]:
    validation = result.validation_report
    test = result.test_report
    efficiency = validation.online_efficiency
    return {
        "model_name": result.model_name,
        "best_config_name": result.hpo_result.best_candidate.name,
        "calibrated_threshold": result.threshold_result.best_threshold,
        "validation_rmse": validation.reconstruction.rmse,
        "validation_mae": validation.reconstruction.mae,
        "validation_precision": validation.anomaly_detection.precision,
        "validation_recall": validation.anomaly_detection.recall,
        "validation_f1": validation.anomaly_detection.f1_score,
        "validation_roc_auc": validation.anomaly_detection.roc_auc,
        "validation_pr_auc": validation.anomaly_detection.pr_auc,
        "test_rmse": test.reconstruction.rmse,
        "test_mae": test.reconstruction.mae,
        "test_precision": test.anomaly_detection.precision,
        "test_recall": test.anomaly_detection.recall,
        "test_f1": test.anomaly_detection.f1_score,
        "test_roc_auc": test.anomaly_detection.roc_auc,
        "test_pr_auc": test.anomaly_detection.pr_auc,
        "validation_latency_ms_per_row": None
        if efficiency is None
        else efficiency.inference_latency_ms_per_row,
        "validation_peak_memory_bytes": None if efficiency is None else efficiency.peak_memory_bytes,
        "model_size_bytes": None if efficiency is None else efficiency.model_size_bytes,
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
    "ModelExperimentSpec",
    "ModelImplementation",
    "QualityExperimentConfig",
    "QualityExperimentResult",
    "default_quality_model_registry",
    "run_quality_model_experiment",
    "write_aggregate_training_history_csv",
]
