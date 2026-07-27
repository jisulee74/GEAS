"""Online benchmark runner for quality-model experiments.

This module implements Experiment Plan v1.1 Step 5 only. It measures online
deployment efficiency metrics for calibrated models and writes benchmark
artifacts. CSV/Markdown comparison reports and figures are intentionally left
to later experiment steps.
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
from geas35.models.quality import OnlineEfficiencyMetrics, measure_online_inference_efficiency


@dataclass(frozen=True)
class QualityOnlineBenchmarkConfig:
    """Configuration required by the Step 5 online benchmark runner."""

    output_root: Path
    repeats: int = 3
    benchmark_splits: tuple[str, ...] = ("validation", "test")


@dataclass(frozen=True)
class ModelOnlineBenchmarkResult:
    """Online benchmark metrics for one calibrated model."""

    model_name: str
    validation: OnlineEfficiencyMetrics | None
    test: OnlineEfficiencyMetrics | None
    artifact_dir: str


@dataclass(frozen=True)
class QualityOnlineBenchmarkResult:
    """Full Step 5 online benchmark result."""

    output_root: str
    model_results: list[ModelOnlineBenchmarkResult]
    online_benchmark_path: str


def run_quality_online_benchmark(
    *,
    config: QualityOnlineBenchmarkConfig,
    calibration_result: QualityThresholdCalibrationResult,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    observation_columns: Iterable[str],
) -> QualityOnlineBenchmarkResult:
    """Measure online benchmark metrics for calibrated models."""

    columns = tuple(observation_columns)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "online_benchmark_config.json", _config_payload(config, columns))

    model_results = [
        run_model_online_benchmark(
            config=config,
            model_result=model_result,
            validation_df=validation_df,
            test_df=test_df,
            observation_columns=columns,
            output_root=output_root,
        )
        for model_result in calibration_result.model_results
    ]

    online_benchmark_path = output_root / "online_benchmark.json"
    _write_json(
        online_benchmark_path,
        _aggregate_benchmark_payload(model_results),
    )
    return QualityOnlineBenchmarkResult(
        output_root=str(output_root),
        model_results=model_results,
        online_benchmark_path=str(online_benchmark_path),
    )


def run_model_online_benchmark(
    *,
    config: QualityOnlineBenchmarkConfig,
    model_result: ModelThresholdCalibrationResult,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    observation_columns: Iterable[str],
    output_root: str | Path,
) -> ModelOnlineBenchmarkResult:
    """Measure online benchmark metrics for one calibrated model."""

    columns = tuple(observation_columns)
    model_dir = Path(output_root) / model_result.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    threshold = model_result.threshold_result.best_threshold
    split_set = {str(split) for split in config.benchmark_splits}
    validation = (
        measure_online_inference_efficiency(
            model_result.model,
            validation_df,
            columns,
            threshold=threshold,
            repeats=config.repeats,
        )
        if "validation" in split_set
        else None
    )
    test = (
        measure_online_inference_efficiency(
            model_result.model,
            test_df,
            columns,
            threshold=threshold,
            repeats=config.repeats,
        )
        if "test" in split_set
        else None
    )
    result = ModelOnlineBenchmarkResult(
        model_name=model_result.model_name,
        validation=validation,
        test=test,
        artifact_dir=str(model_dir),
    )
    _write_json(model_dir / "online_benchmark.json", _model_benchmark_payload(result))
    return result


def quality_online_benchmark_config_from_experiment_config(
    config,
) -> QualityOnlineBenchmarkConfig:
    """Create a Step 5 benchmark config from the broader experiment config."""

    return QualityOnlineBenchmarkConfig(
        output_root=Path(config.output_root),
        repeats=config.efficiency_repeats,
    )


def _config_payload(
    config: QualityOnlineBenchmarkConfig,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "stage": "quality_model_online_benchmark",
        "observation_columns": list(columns),
        "repeats": config.repeats,
        "benchmark_splits": list(config.benchmark_splits),
        "online_benchmark_included": True,
        "automatic_best_model_selection": False,
        "test_used_for_selection": False,
    }


def _aggregate_benchmark_payload(
    model_results: Iterable[ModelOnlineBenchmarkResult],
) -> dict[str, Any]:
    return {
        "stage": "quality_model_online_benchmark",
        "online_benchmark_included": True,
        "automatic_best_model_selection": False,
        "test_used_for_selection": False,
        "models": {
            result.model_name: _model_benchmark_payload(result)
            for result in model_results
        },
    }


def _model_benchmark_payload(result: ModelOnlineBenchmarkResult) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "validation": None if result.validation is None else asdict(result.validation),
        "test": None if result.test is None else asdict(result.test),
        "online_benchmark_included": True,
        "automatic_best_model_selection": False,
        "test_used_for_selection": False,
    }


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
    "ModelOnlineBenchmarkResult",
    "QualityOnlineBenchmarkConfig",
    "QualityOnlineBenchmarkResult",
    "quality_online_benchmark_config_from_experiment_config",
    "run_model_online_benchmark",
    "run_quality_online_benchmark",
]
