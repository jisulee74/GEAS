"""Hyperparameter optimization framework for AI quality models.

This module intentionally does not calibrate anomaly thresholds. Hyperparameter
optimization selects model/training configurations by validation reconstruction
metrics, then threshold calibration is performed separately by ThresholdOptimizer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from geas35.models.quality.base import validate_observation_columns


DEFAULT_PRIMARY_HPO_METRIC = "validation_synthetic_masking_rmse"
DEFAULT_SECONDARY_HPO_METRICS = (
    "validation_synthetic_masking_mae",
    "validation_anomaly_pr_auc",
)


@dataclass(frozen=True)
class MetricGoal:
    """Metric direction used for candidate ranking."""

    name: str
    mode: str = "min"

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("MetricGoal.mode must be 'min' or 'max'.")
        if not self.name:
            raise ValueError("MetricGoal.name must not be empty.")

    def score(self, metrics: Mapping[str, Any]) -> float:
        value = pd.to_numeric(metrics.get(self.name), errors="coerce")
        if pd.isna(value):
            return float("-inf")
        numeric = float(value)
        if not np.isfinite(numeric):
            return float("-inf")
        return -numeric if self.mode == "min" else numeric

    def to_artifact(self) -> dict[str, str]:
        return {"name": self.name, "mode": self.mode}


@dataclass(frozen=True)
class HyperparameterObjective:
    """Validation objective for hyperparameter selection."""

    primary: MetricGoal = field(
        default_factory=lambda: MetricGoal(DEFAULT_PRIMARY_HPO_METRIC, "min")
    )
    secondary: tuple[MetricGoal, ...] = field(
        default_factory=lambda: (
            MetricGoal("validation_synthetic_masking_mae", "min"),
            MetricGoal("validation_anomaly_pr_auc", "max"),
        )
    )

    def ranking_key(self, metrics: Mapping[str, Any]) -> tuple[float, ...]:
        goals = (self.primary, *self.secondary)
        return tuple(goal.score(metrics) for goal in goals)

    def to_artifact(self) -> dict[str, object]:
        return {
            "primary": self.primary.to_artifact(),
            "secondary": [goal.to_artifact() for goal in self.secondary],
            "threshold_dependent_primary_metrics_allowed": False,
        }


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Common early stopping training strategy for AI quality models."""

    monitor: str = "validation_reconstruction_loss"
    mode: str = "min"
    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("EarlyStoppingConfig.mode must be 'min' or 'max'.")
        if self.patience < 0:
            raise ValueError("EarlyStoppingConfig.patience must be >= 0.")
        if self.min_delta < 0.0:
            raise ValueError("EarlyStoppingConfig.min_delta must be >= 0.")
        if not self.monitor:
            raise ValueError("EarlyStoppingConfig.monitor must not be empty.")

    def to_artifact(self) -> dict[str, object]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }


class EarlyStoppingTracker:
    """Track validation reconstruction loss and decide when to stop training."""

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config
        self.best_epoch: int | None = None
        self.best_value: float | None = None
        self.wait_count = 0
        self.stopped_epoch: int | None = None

    def update(self, epoch: int, metrics: Mapping[str, Any]) -> bool:
        """Update tracker and return True when training should stop."""

        value = pd.to_numeric(metrics.get(self.config.monitor), errors="coerce")
        if pd.isna(value):
            return False
        current = float(value)
        if self._is_improvement(current):
            self.best_epoch = int(epoch)
            self.best_value = current
            self.wait_count = 0
            return False

        self.wait_count += 1
        if self.wait_count > self.config.patience:
            self.stopped_epoch = int(epoch)
            return True
        return False

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.config.mode == "min":
            return value < self.best_value - self.config.min_delta
        return value > self.best_value + self.config.min_delta

    def to_artifact(self) -> dict[str, object]:
        return {
            "monitor": self.config.monitor,
            "mode": self.config.mode,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "stopped_epoch": self.stopped_epoch,
            "wait_count": self.wait_count,
        }


@dataclass(frozen=True)
class HyperparameterCandidate:
    """One model hyperparameter candidate excluding anomaly thresholds."""

    name: str
    common: Mapping[str, Any] = field(default_factory=dict)
    model_specific: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HyperparameterCandidate.name must not be empty.")
        _reject_threshold_keys(self.common, path=f"{self.name}.common")
        _reject_threshold_keys(self.model_specific, path=f"{self.name}.model_specific")

    def merged_config(self) -> dict[str, Any]:
        return {
            "common": dict(self.common),
            "model_specific": dict(self.model_specific),
        }

    def to_artifact(self) -> dict[str, object]:
        return {
            "name": self.name,
            "common": _jsonable_mapping(self.common),
            "model_specific": _jsonable_mapping(self.model_specific),
        }


@dataclass(frozen=True)
class CandidateTrainingResult:
    """Validation result for one trained hyperparameter candidate."""

    candidate: HyperparameterCandidate
    validation_metrics: Mapping[str, float]
    training_history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    model_artifact: Mapping[str, Any] | None = None

    def to_artifact(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_artifact(),
            "validation_metrics": _jsonable_mapping(self.validation_metrics),
            "training_history": [_jsonable_mapping(row) for row in self.training_history],
            "model_artifact": None
            if self.model_artifact is None
            else _jsonable_mapping(self.model_artifact),
        }


@dataclass(frozen=True)
class HyperparameterOptimizationResult:
    """Best hyperparameter configuration and all candidate validation reports."""

    best_result: CandidateTrainingResult
    candidate_results: list[CandidateTrainingResult]
    objective: HyperparameterObjective
    early_stopping: EarlyStoppingConfig

    @property
    def best_candidate(self) -> HyperparameterCandidate:
        return self.best_result.candidate

    @property
    def best_metrics(self) -> Mapping[str, float]:
        return self.best_result.validation_metrics

    def to_artifact(self) -> dict[str, object]:
        return {
            "stage": "hyperparameter_optimization",
            "threshold_calibration_included": False,
            "threshold_fields_allowed_in_candidates": False,
            "objective": self.objective.to_artifact(),
            "early_stopping": self.early_stopping.to_artifact(),
            "best_candidate": self.best_candidate.to_artifact(),
            "best_metrics": _jsonable_mapping(self.best_metrics),
            "candidate_results": [
                result.to_artifact() for result in self.candidate_results
            ],
        }


class HyperparameterOptimizer(ABC):
    """Base interface for validation-based hyperparameter search."""

    @abstractmethod
    def optimize(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        observation_columns: Iterable[str],
    ) -> HyperparameterOptimizationResult:
        """Return the best hyperparameter candidate and validation reports."""


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


class GridSearchHyperparameterOptimizer(HyperparameterOptimizer):
    """Evaluate explicit candidate configurations and select the best one."""

    def __init__(
        self,
        candidates: Sequence[HyperparameterCandidate],
        train_candidate: TrainCandidateFn,
        *,
        objective: HyperparameterObjective | None = None,
        early_stopping: EarlyStoppingConfig | None = None,
        progress_context: str | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("candidates must not be empty.")
        self.candidates = list(candidates)
        self.train_candidate = train_candidate
        self.objective = objective or HyperparameterObjective()
        self.early_stopping = early_stopping or EarlyStoppingConfig()
        self.progress_context = progress_context

    def optimize(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        observation_columns: Iterable[str],
    ) -> HyperparameterOptimizationResult:
        columns = validate_observation_columns(observation_columns)
        results = []
        total = len(self.candidates)
        for index, candidate in enumerate(self.candidates, start=1):
            if self.progress_context is not None:
                print(
                    f"[{self.progress_context}] HPO trial 시작 "
                    f"({index}/{total}): {candidate.name}",
                    flush=True,
                )
            result = self.train_candidate(
                candidate,
                train_df,
                validation_df,
                columns,
                self.early_stopping,
            )
            results.append(result)
            if self.progress_context is not None:
                rmse = result.validation_metrics.get(
                    "validation_synthetic_masking_rmse"
                )
                print(
                    f"[{self.progress_context}] HPO trial 완료 "
                    f"({index}/{total}), validation_rmse={rmse}",
                    flush=True,
                )
        best = max(
            results,
            key=lambda result: self.objective.ranking_key(result.validation_metrics),
        )
        return HyperparameterOptimizationResult(
            best_result=best,
            candidate_results=results,
            objective=self.objective,
            early_stopping=self.early_stopping,
        )


def write_hyperparameter_artifact(
    path: Path,
    result: HyperparameterOptimizationResult,
) -> None:
    """Write a selected configuration artifact without threshold calibration."""

    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_artifact(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _reject_threshold_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if "threshold" in key_text:
                raise ValueError(
                    "Threshold calibration is separate from hyperparameter "
                    f"optimization; threshold key is not allowed: {path}.{key}"
                )
            _reject_threshold_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _reject_threshold_keys(item, path=f"{path}[{idx}]")


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(item) for key, item in value.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return str(value)


__all__ = [
    "CandidateTrainingResult",
    "DEFAULT_PRIMARY_HPO_METRIC",
    "DEFAULT_SECONDARY_HPO_METRICS",
    "EarlyStoppingConfig",
    "EarlyStoppingTracker",
    "GridSearchHyperparameterOptimizer",
    "HyperparameterCandidate",
    "HyperparameterObjective",
    "HyperparameterOptimizationResult",
    "HyperparameterOptimizer",
    "MetricGoal",
    "write_hyperparameter_artifact",
]
