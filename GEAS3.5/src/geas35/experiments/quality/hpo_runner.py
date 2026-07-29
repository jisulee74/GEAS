"""HPO-only experiment runner for quality models.

This module implements Experiment Plan v1.1 Step 2 only: model-wise Random
Search HPO and HPO artifacts. Threshold calibration, validation/test final
evaluation, online benchmarking, reports, and visualization are intentionally
handled by later experiment steps.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from geas35.experiments.quality.runner import (
    ModelExperimentSpec,
    ModelImplementation,
    default_quality_model_registry,
)
from geas35.experiments.quality.search import (
    RandomSearchCandidateGenerator,
    SearchSpace,
)
from geas35.models.quality import (
    EarlyStoppingConfig,
    GridSearchHyperparameterOptimizer,
    HyperparameterOptimizationResult,
)


@dataclass(frozen=True)
class QualityHPOConfig:
    """Configuration required by the Step 2 HPO-only runner."""

    output_root: Path
    models: tuple[ModelExperimentSpec, ...]
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(frozen=True)
class ModelHPOResult:
    """Artifacts and HPO result for one model."""

    model_name: str
    hpo_result: HyperparameterOptimizationResult
    artifact_dir: str


@dataclass(frozen=True)
class QualityHPOResult:
    """Full HPO-only experiment result."""

    output_root: str
    model_results: list[ModelHPOResult]
    hpo_results_path: str
    best_config_path: str
    training_history_path: str


def run_quality_hpo(
    *,
    config: QualityHPOConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    observation_columns: Iterable[str],
    registry: Mapping[str, ModelImplementation] | None = None,
) -> QualityHPOResult:
    """Run model-wise Random Search HPO and write Step 2 artifacts only."""

    columns = tuple(observation_columns)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_registry = default_quality_model_registry() if registry is None else dict(registry)
    _write_json(output_root / "hpo_config.json", _config_payload(config, columns))

    model_results: list[ModelHPOResult] = []
    for model_spec in config.models:
        model_results.append(
            run_model_hpo(
                model_spec=model_spec,
                implementation=_implementation_for(model_spec.model_name, model_registry),
                train_df=train_df,
                validation_df=validation_df,
                observation_columns=columns,
                early_stopping=config.early_stopping,
                output_root=output_root,
            )
        )

    hpo_results_path = output_root / "hpo_results.json"
    best_config_path = output_root / "best_config.json"
    training_history_path = output_root / "training_history.csv"
    _write_json(hpo_results_path, _aggregate_hpo_payload(model_results))
    _write_json(best_config_path, _aggregate_best_config_payload(model_results))
    _write_aggregate_training_history_csv(training_history_path, model_results)
    return QualityHPOResult(
        output_root=str(output_root),
        model_results=model_results,
        hpo_results_path=str(hpo_results_path),
        best_config_path=str(best_config_path),
        training_history_path=str(training_history_path),
    )


def run_model_hpo(
    *,
    model_spec: ModelExperimentSpec,
    implementation: ModelImplementation,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    observation_columns: Iterable[str],
    early_stopping: EarlyStoppingConfig,
    output_root: str | Path,
) -> ModelHPOResult:
    """Run Random Search HPO for one model and write model-level artifacts."""

    columns = tuple(observation_columns)
    model_dir = Path(output_root) / model_spec.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    crop_name = Path(output_root).name
    print(
        f"[{crop_name}][{model_spec.model_name}] HPO 시작 "
        f"({model_spec.hpo_budget} trials)",
        flush=True,
    )
    candidates = RandomSearchCandidateGenerator(
        model_name=model_spec.model_name,
        search_space=SearchSpace.from_config(model_spec.search_space),
        budget=model_spec.hpo_budget,
        random_seed=model_spec.random_seed,
    ).generate()
    hpo_result = GridSearchHyperparameterOptimizer(
        candidates,
        implementation.train_candidate,
        early_stopping=early_stopping,
        progress_context=f"{crop_name}][{model_spec.model_name}",
    ).optimize(train_df, validation_df, columns)
    _write_json(model_dir / "hpo_results.json", hpo_result.to_artifact())
    _write_json(model_dir / "best_config.json", hpo_result.best_candidate.to_artifact())
    _write_training_history_csv(
        model_dir / "training_history.csv",
        hpo_result.best_result.training_history,
    )
    print(
        f"[{crop_name}][{model_spec.model_name}] HPO 완료: "
        f"best={hpo_result.best_candidate.name}",
        flush=True,
    )
    return ModelHPOResult(
        model_name=model_spec.model_name,
        hpo_result=hpo_result,
        artifact_dir=str(model_dir),
    )


def quality_hpo_config_from_experiment_config(config) -> QualityHPOConfig:
    """Create a Step 2 HPO-only config from the broader experiment config."""

    return QualityHPOConfig(
        output_root=Path(config.output_root),
        models=tuple(config.models),
        early_stopping=config.early_stopping,
    )


def _implementation_for(
    model_name: str,
    registry: Mapping[str, ModelImplementation],
) -> ModelImplementation:
    if model_name not in registry:
        raise ValueError(f"Unknown experiment model: {model_name}")
    return registry[model_name]


def _write_training_history_csv(
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


def _write_aggregate_training_history_csv(
    path: Path,
    model_results: Iterable[ModelHPOResult],
) -> None:
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _config_payload(config: QualityHPOConfig, columns: tuple[str, ...]) -> dict[str, Any]:
    return {
        "stage": "quality_model_hpo",
        "observation_columns": list(columns),
        "models": [asdict(model) for model in config.models],
        "early_stopping": config.early_stopping.to_artifact(),
        "threshold_calibration_included": False,
        "threshold_is_hpo_parameter": False,
    }


def _aggregate_hpo_payload(
    model_results: Iterable[ModelHPOResult],
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
    model_results: Iterable[ModelHPOResult],
) -> dict[str, Any]:
    return {
        "stage": "best_hyperparameter_configurations",
        "threshold_calibration_included": False,
        "models": {
            result.model_name: result.hpo_result.best_candidate.to_artifact()
            for result in model_results
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
    "ModelHPOResult",
    "QualityHPOConfig",
    "QualityHPOResult",
    "quality_hpo_config_from_experiment_config",
    "run_model_hpo",
    "run_quality_hpo",
]
