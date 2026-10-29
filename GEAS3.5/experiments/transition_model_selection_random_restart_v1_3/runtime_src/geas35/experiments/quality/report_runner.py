"""CSV/JSON/Markdown report runner for quality-model experiments.

This module implements Experiment Plan v1.1 Step 6 only. It converts Step 4
evaluation results and Step 5 online benchmark results into no-selection report
artifacts. Visualization and artifact integrity checks are intentionally left
to later experiment steps.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from geas35.experiments.quality.benchmark_runner import (
    ModelOnlineBenchmarkResult,
    QualityOnlineBenchmarkResult,
)
from geas35.experiments.quality.evaluation_runner import (
    ModelEvaluationResult,
    QualityEvaluationResult,
)
from geas35.models.quality import OnlineEfficiencyMetrics


@dataclass(frozen=True)
class QualityReportConfig:
    """Configuration required by the Step 6 report generator."""

    output_root: Path


@dataclass(frozen=True)
class QualityReportResult:
    """Report artifact paths produced by Step 6."""

    output_root: str
    model_comparison_json_path: str
    model_comparison_csv_path: str
    markdown_summary_path: str
    reconstruction_metric_table_path: str
    anomaly_detection_metric_table_path: str
    online_benchmark_table_path: str


def run_quality_report_generation(
    *,
    config: QualityReportConfig,
    evaluation_result: QualityEvaluationResult,
    online_benchmark_result: QualityOnlineBenchmarkResult | None = None,
) -> QualityReportResult:
    """Generate no-selection CSV, JSON, and Markdown report artifacts."""

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark_by_model = _benchmark_map(online_benchmark_result)
    comparison_rows = [
        _comparison_row(model_result, benchmark_by_model.get(model_result.model_name))
        for model_result in evaluation_result.model_results
    ]
    reconstruction_rows = list(_reconstruction_rows(evaluation_result.model_results))
    anomaly_rows = list(_anomaly_rows(evaluation_result.model_results))
    benchmark_rows = list(_benchmark_rows(benchmark_by_model.values()))

    model_comparison_json_path = output_root / "model_comparison.json"
    model_comparison_csv_path = output_root / "model_comparison.csv"
    markdown_summary_path = output_root / "markdown_summary.md"
    reconstruction_metric_table_path = output_root / "reconstruction_metric_table.csv"
    anomaly_detection_metric_table_path = output_root / "anomaly_detection_metric_table.csv"
    online_benchmark_table_path = output_root / "online_benchmark_table.csv"

    _write_json(
        model_comparison_json_path,
        {
            "stage": "quality_model_experiment_report",
            "automatic_best_model_selection": False,
            "test_used_for_selection": False,
            "online_benchmark_included": online_benchmark_result is not None,
            "models": comparison_rows,
        },
    )
    _write_csv(model_comparison_csv_path, comparison_rows)
    _write_csv(reconstruction_metric_table_path, reconstruction_rows)
    _write_csv(anomaly_detection_metric_table_path, anomaly_rows)
    _write_csv(online_benchmark_table_path, benchmark_rows)
    _write_markdown_summary(
        markdown_summary_path,
        comparison_rows,
        online_benchmark_included=online_benchmark_result is not None,
    )
    return QualityReportResult(
        output_root=str(output_root),
        model_comparison_json_path=str(model_comparison_json_path),
        model_comparison_csv_path=str(model_comparison_csv_path),
        markdown_summary_path=str(markdown_summary_path),
        reconstruction_metric_table_path=str(reconstruction_metric_table_path),
        anomaly_detection_metric_table_path=str(anomaly_detection_metric_table_path),
        online_benchmark_table_path=str(online_benchmark_table_path),
    )


def quality_report_config_from_experiment_config(config) -> QualityReportConfig:
    """Create a Step 6 report config from the broader experiment config."""

    return QualityReportConfig(output_root=Path(config.output_root))


def _benchmark_map(
    result: QualityOnlineBenchmarkResult | None,
) -> dict[str, ModelOnlineBenchmarkResult]:
    if result is None:
        return {}
    return {model.model_name: model for model in result.model_results}


def _comparison_row(
    evaluation: ModelEvaluationResult,
    benchmark: ModelOnlineBenchmarkResult | None,
) -> dict[str, Any]:
    validation = evaluation.validation_report
    test = evaluation.test_report
    validation_benchmark = None if benchmark is None else benchmark.validation
    test_benchmark = None if benchmark is None else benchmark.test
    return {
        "model_name": evaluation.model_name,
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
        "validation_latency_ms_per_row": _efficiency_value(
            validation_benchmark,
            "inference_latency_ms_per_row",
        ),
        "validation_peak_memory_bytes": _efficiency_value(
            validation_benchmark,
            "peak_memory_bytes",
        ),
        "validation_model_size_bytes": _efficiency_value(
            validation_benchmark,
            "model_size_bytes",
        ),
        "test_latency_ms_per_row": _efficiency_value(
            test_benchmark,
            "inference_latency_ms_per_row",
        ),
        "test_peak_memory_bytes": _efficiency_value(
            test_benchmark,
            "peak_memory_bytes",
        ),
        "test_model_size_bytes": _efficiency_value(
            test_benchmark,
            "model_size_bytes",
        ),
    }


def _reconstruction_rows(
    model_results: Iterable[ModelEvaluationResult],
) -> Iterable[dict[str, Any]]:
    for model_result in model_results:
        for split, report in (
            ("validation", model_result.validation_report),
            ("test", model_result.test_report),
        ):
            base = {
                "model_name": model_result.model_name,
                "split": split,
                "masked_cells": report.reconstruction.masked_cells,
                "mae": report.reconstruction.mae,
                "rmse": report.reconstruction.rmse,
            }
            yield {**base, "column": "__all__"}
            for column, metrics in report.reconstruction.per_column.items():
                yield {
                    **base,
                    "column": column,
                    **_flatten_mapping(metrics),
                }


def _anomaly_rows(
    model_results: Iterable[ModelEvaluationResult],
) -> Iterable[dict[str, Any]]:
    for model_result in model_results:
        for split, report in (
            ("validation", model_result.validation_report),
            ("test", model_result.test_report),
        ):
            anomaly = report.anomaly_detection
            base = {
                "model_name": model_result.model_name,
                "split": split,
                "injected_cells": anomaly.injected_cells,
                "threshold": anomaly.threshold,
                "precision": anomaly.precision,
                "recall": anomaly.recall,
                "f1_score": anomaly.f1_score,
                "roc_auc": anomaly.roc_auc,
                "pr_auc": anomaly.pr_auc,
                "true_positive": anomaly.confusion_matrix.true_positive,
                "false_positive": anomaly.confusion_matrix.false_positive,
                "true_negative": anomaly.confusion_matrix.true_negative,
                "false_negative": anomaly.confusion_matrix.false_negative,
            }
            yield {**base, "column": "__all__"}
            for column, metrics in anomaly.per_column.items():
                yield {
                    **base,
                    "column": column,
                    **_flatten_mapping(metrics),
                }


def _benchmark_rows(
    model_results: Iterable[ModelOnlineBenchmarkResult],
) -> Iterable[dict[str, Any]]:
    for model_result in model_results:
        for split, metrics in (
            ("validation", model_result.validation),
            ("test", model_result.test),
        ):
            if metrics is None:
                continue
            yield {
                "model_name": model_result.model_name,
                "split": split,
                "inference_latency_ms_per_row": metrics.inference_latency_ms_per_row,
                "peak_memory_bytes": metrics.peak_memory_bytes,
                "model_size_bytes": metrics.model_size_bytes,
                "measured_rows": metrics.measured_rows,
                "repeats": metrics.repeats,
            }


def _efficiency_value(metrics: OnlineEfficiencyMetrics | None, name: str) -> Any:
    if metrics is None:
        return None
    return getattr(metrics, name)


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
        if isinstance(item, Mapping):
            output.update(_flatten_mapping(item, name))
        else:
            output[name] = item
    return output


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_summary(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    online_benchmark_included: bool,
) -> None:
    lines = [
        "# GEAS AI Quality Model Experiment Summary",
        "",
        "This report does not automatically select a best model.",
        "",
        f"Online benchmark included: {online_benchmark_included}",
        "",
        "| Model | Validation RMSE | Validation MAE | Validation F1 | Validation PR-AUC | Test RMSE | Test F1 | Validation Latency ms/row |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model_name} | {validation_rmse} | {validation_mae} | "
            "{validation_f1} | {validation_pr_auc} | {test_rmse} | {test_f1} | "
            "{validation_latency_ms_per_row} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    "QualityReportConfig",
    "QualityReportResult",
    "quality_report_config_from_experiment_config",
    "run_quality_report_generation",
]
