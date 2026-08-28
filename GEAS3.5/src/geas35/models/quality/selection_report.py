"""Model comparison and final selection reports for quality models."""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from geas35.models.quality.base import BaseQualityModel
from geas35.models.quality.evaluation import (
    AnomalyDetectionMetrics,
    SyntheticAnomalyDetectionResult,
    SyntheticMaskingResult,
    evaluate_synthetic_anomaly_detection,
    evaluate_synthetic_masking,
)
from geas35.models.quality.hyperparameters import HyperparameterOptimizationResult


@dataclass(frozen=True)
class QualityModelSelectionCandidate:
    """A trained model with fixed best config and calibrated threshold."""

    model_name: str
    model: BaseQualityModel
    calibrated_threshold: float
    hyperparameter_result: HyperparameterOptimizationResult | None = None
    hyperparameter_artifact: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ReconstructionPerformance:
    """Synthetic masking reconstruction metrics."""

    masked_cells: int
    mae: float
    rmse: float
    per_column: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ConfusionMatrixReport:
    """JSON-safe binary confusion matrix."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True)
class AnomalyDetectionPerformance:
    """Synthetic anomaly detection metrics."""

    injected_cells: int
    threshold: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: ConfusionMatrixReport
    per_column: dict[str, dict[str, object]]
    injection_stats: dict[str, dict[str, object]]


@dataclass(frozen=True)
class OnlineEfficiencyMetrics:
    """Approximate online inference efficiency metrics."""

    inference_latency_ms_per_row: float
    peak_memory_bytes: int
    model_size_bytes: int
    measured_rows: int
    repeats: int


@dataclass(frozen=True)
class QualityModelEvaluationReport:
    """One model evaluation report for one split."""

    split: str
    reconstruction: ReconstructionPerformance
    anomaly_detection: AnomalyDetectionPerformance
    online_efficiency: OnlineEfficiencyMetrics | None = None


@dataclass(frozen=True)
class QualityModelComparisonEntry:
    """Full comparison entry for one candidate model."""

    model_name: str
    calibrated_threshold: float
    hyperparameter_selection: dict[str, object] | None
    validation: QualityModelEvaluationReport
    test: QualityModelEvaluationReport | None
    selected: bool = False


@dataclass(frozen=True)
class QualityModelComparisonReport:
    """Final model/config/threshold comparison report."""

    stage: str
    selection_split: str
    test_split: str | None
    test_used_for_selection: bool
    selected_model: str
    selected_threshold: float
    selection_policy: dict[str, object]
    entries: list[QualityModelComparisonEntry]

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))


def evaluate_quality_model_for_report(
    model: BaseQualityModel,
    df: pd.DataFrame,
    observation_columns: Iterable[str],
    *,
    split: str,
    calibrated_threshold: float,
    mask_fraction: float = 0.1,
    anomaly_fraction: float = 0.1,
    anomaly_scale: float = 8.0,
    random_state: int | None = 0,
    include_efficiency: bool = True,
    efficiency_repeats: int = 3,
    reference_df: pd.DataFrame | None = None,
) -> QualityModelEvaluationReport:
    """Evaluate one fixed model/config/threshold on one split."""

    masking = evaluate_synthetic_masking(
        model,
        df,
        observation_columns,
        mask_fraction=mask_fraction,
        random_state=random_state,
    )
    detection = evaluate_synthetic_anomaly_detection(
        model,
        df,
        observation_columns,
        threshold=calibrated_threshold,
        anomaly_fraction=anomaly_fraction,
        anomaly_scale=anomaly_scale,
        random_state=random_state,
        reference_df=reference_df,
    )
    efficiency = (
        measure_online_inference_efficiency(
            model,
            df,
            observation_columns,
            threshold=calibrated_threshold,
            repeats=efficiency_repeats,
        )
        if include_efficiency
        else None
    )
    return QualityModelEvaluationReport(
        split=split,
        reconstruction=_reconstruction_report(masking),
        anomaly_detection=_anomaly_report(detection),
        online_efficiency=efficiency,
    )


def build_quality_model_comparison_report(
    candidates: Iterable[QualityModelSelectionCandidate],
    validation_df: pd.DataFrame,
    observation_columns: Iterable[str],
    *,
    test_df: pd.DataFrame | None = None,
    mask_fraction: float = 0.1,
    anomaly_fraction: float = 0.1,
    anomaly_scale: float = 8.0,
    random_state: int | None = 0,
    include_efficiency: bool = True,
    efficiency_repeats: int = 3,
) -> QualityModelComparisonReport:
    """Build a final comparison report using validation for selection only."""

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidates must not be empty.")

    entries_without_selection: list[QualityModelComparisonEntry] = []
    for candidate in candidate_list:
        validation = evaluate_quality_model_for_report(
            candidate.model,
            validation_df,
            observation_columns,
            split="validation",
            calibrated_threshold=candidate.calibrated_threshold,
            mask_fraction=mask_fraction,
            anomaly_fraction=anomaly_fraction,
            anomaly_scale=anomaly_scale,
            random_state=random_state,
            include_efficiency=include_efficiency,
            efficiency_repeats=efficiency_repeats,
        )
        test = (
            evaluate_quality_model_for_report(
                candidate.model,
                test_df,
                observation_columns,
                split="test",
                calibrated_threshold=candidate.calibrated_threshold,
                mask_fraction=mask_fraction,
                anomaly_fraction=anomaly_fraction,
                anomaly_scale=anomaly_scale,
                random_state=random_state,
                include_efficiency=include_efficiency,
                efficiency_repeats=efficiency_repeats,
            )
            if test_df is not None
            else None
        )
        entries_without_selection.append(
            QualityModelComparisonEntry(
                model_name=candidate.model_name,
                calibrated_threshold=float(candidate.calibrated_threshold),
                hyperparameter_selection=_hyperparameter_selection(candidate),
                validation=validation,
                test=test,
                selected=False,
            )
        )

    selected_entry = max(entries_without_selection, key=_selection_key)
    entries = [
        QualityModelComparisonEntry(
            model_name=entry.model_name,
            calibrated_threshold=entry.calibrated_threshold,
            hyperparameter_selection=entry.hyperparameter_selection,
            validation=entry.validation,
            test=entry.test,
            selected=entry.model_name == selected_entry.model_name,
        )
        for entry in entries_without_selection
    ]
    return QualityModelComparisonReport(
        stage="quality_model_final_selection",
        selection_split="validation",
        test_split="test" if test_df is not None else None,
        test_used_for_selection=False,
        selected_model=selected_entry.model_name,
        selected_threshold=selected_entry.calibrated_threshold,
        selection_policy={
            "uses_validation_only": True,
            "test_used_for_selection": False,
            "criteria": [
                "lower_validation_reconstruction_rmse",
                "lower_validation_reconstruction_mae",
                "higher_validation_f1",
                "higher_validation_pr_auc",
                "higher_validation_roc_auc",
                "lower_online_latency",
                "lower_peak_memory",
                "lower_model_size",
            ],
        },
        entries=entries,
    )


def write_quality_model_comparison_report(
    path: Path,
    report: QualityModelComparisonReport,
) -> None:
    """Write the model comparison report artifact as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def measure_online_inference_efficiency(
    model: BaseQualityModel,
    df: pd.DataFrame,
    observation_columns: Iterable[str],
    *,
    threshold: float,
    repeats: int = 3,
) -> OnlineEfficiencyMetrics:
    """Measure approximate predict_outlier latency, peak memory, and model size."""

    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    measured_rows = len(df)
    tracemalloc.start()
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict_outlier(df, threshold, observation_columns)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    denominator = max(1, measured_rows * repeats)
    return OnlineEfficiencyMetrics(
        inference_latency_ms_per_row=float((elapsed * 1000.0) / denominator),
        peak_memory_bytes=int(peak),
        model_size_bytes=int(estimate_model_size_bytes(model)),
        measured_rows=int(measured_rows),
        repeats=int(repeats),
    )


def estimate_model_size_bytes(model: BaseQualityModel) -> int:
    """Estimate model size from parameters and JSON-safe artifact payload."""

    model_obj = getattr(model, "model_", None)
    if model_obj is not None and hasattr(model_obj, "parameters"):
        try:
            return int(
                sum(
                    param.numel() * param.element_size()
                    for param in model_obj.parameters()
                )
            )
        except Exception:
            pass

    if hasattr(model, "to_artifact"):
        try:
            payload = model.to_artifact()
            text = json.dumps(_jsonable(payload), ensure_ascii=False)
            return len(text.encode("utf-8"))
        except Exception:
            pass
    return int(sys.getsizeof(model))


def _selection_key(entry: QualityModelComparisonEntry) -> tuple[float, ...]:
    validation = entry.validation
    reconstruction = validation.reconstruction
    anomaly = validation.anomaly_detection
    efficiency = validation.online_efficiency
    latency = (
        efficiency.inference_latency_ms_per_row
        if efficiency is not None
        else float("inf")
    )
    memory = efficiency.peak_memory_bytes if efficiency is not None else float("inf")
    model_size = efficiency.model_size_bytes if efficiency is not None else float("inf")
    return (
        _negative_finite(reconstruction.rmse),
        _negative_finite(reconstruction.mae),
        _finite(anomaly.f1_score),
        _finite(anomaly.pr_auc),
        _finite(anomaly.roc_auc),
        _negative_finite(latency),
        _negative_finite(memory),
        _negative_finite(model_size),
    )


def _reconstruction_report(result: SyntheticMaskingResult) -> ReconstructionPerformance:
    return ReconstructionPerformance(
        masked_cells=result.masked_cells,
        mae=float(result.mae),
        rmse=float(result.rmse),
        per_column=result.per_column,
    )


def _anomaly_report(
    result: SyntheticAnomalyDetectionResult,
) -> AnomalyDetectionPerformance:
    metrics = result.metrics
    return AnomalyDetectionPerformance(
        injected_cells=result.injected_cells,
        threshold=result.threshold,
        precision=float(metrics.precision),
        recall=float(metrics.recall),
        f1_score=float(metrics.f1_score),
        roc_auc=float(metrics.roc_auc),
        pr_auc=float(metrics.pr_auc),
        confusion_matrix=_confusion_report(metrics),
        per_column={
            col: _metrics_dict(per_col_metrics)
            for col, per_col_metrics in result.per_column.items()
        },
        injection_stats=result.injection_stats,
    )


def _confusion_report(metrics: AnomalyDetectionMetrics) -> ConfusionMatrixReport:
    cm = metrics.confusion_matrix
    return ConfusionMatrixReport(
        true_positive=cm.true_positive,
        false_positive=cm.false_positive,
        true_negative=cm.true_negative,
        false_negative=cm.false_negative,
    )


def _metrics_dict(metrics: AnomalyDetectionMetrics) -> dict[str, object]:
    cm = metrics.confusion_matrix
    return {
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1_score": float(metrics.f1_score),
        "roc_auc": float(metrics.roc_auc),
        "pr_auc": float(metrics.pr_auc),
        "confusion_matrix": {
            "true_positive": cm.true_positive,
            "false_positive": cm.false_positive,
            "true_negative": cm.true_negative,
            "false_negative": cm.false_negative,
        },
    }


def _hyperparameter_selection(
    candidate: QualityModelSelectionCandidate,
) -> dict[str, object] | None:
    if candidate.hyperparameter_result is not None:
        artifact = candidate.hyperparameter_result.to_artifact()
        return {
            "best_candidate": artifact["best_candidate"],
            "best_metrics": artifact["best_metrics"],
            "threshold_calibration_included": artifact[
                "threshold_calibration_included"
            ],
            "objective": artifact["objective"],
        }
    if candidate.hyperparameter_artifact is not None:
        return dict(candidate.hyperparameter_artifact)
    return None


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else float("-inf")


def _negative_finite(value: float) -> float:
    return -float(value) if np.isfinite(value) else float("-inf")


def _jsonable(value):
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "AnomalyDetectionPerformance",
    "ConfusionMatrixReport",
    "OnlineEfficiencyMetrics",
    "QualityModelComparisonEntry",
    "QualityModelComparisonReport",
    "QualityModelEvaluationReport",
    "QualityModelSelectionCandidate",
    "ReconstructionPerformance",
    "build_quality_model_comparison_report",
    "estimate_model_size_bytes",
    "evaluate_quality_model_for_report",
    "measure_online_inference_efficiency",
    "write_quality_model_comparison_report",
]
