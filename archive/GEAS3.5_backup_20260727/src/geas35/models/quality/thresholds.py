"""Threshold optimization for quality model anomaly scores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from geas35.models.quality.base import BaseQualityModel
from geas35.models.quality.evaluation import (
    SyntheticAnomalyDetectionResult,
    SyntheticMaskingResult,
    evaluate_synthetic_anomaly_detection,
    evaluate_synthetic_masking,
)


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    """Result of validation-based threshold optimization."""

    best_threshold: float
    best_objective_value: float
    objective_metric: str
    detection_results: list[SyntheticAnomalyDetectionResult]
    best_detection_result: SyntheticAnomalyDetectionResult
    masking_result: SyntheticMaskingResult | None = None


class ThresholdOptimizer(ABC):
    """Base interface for validation-based threshold selection."""

    @abstractmethod
    def optimize(
        self,
        model: BaseQualityModel,
        validation_df: pd.DataFrame,
        observation_columns: Iterable[str],
    ) -> ThresholdOptimizationResult:
        """Return the best threshold and validation evaluation report."""


class GridSearchThresholdOptimizer(ThresholdOptimizer):
    """Initial threshold optimizer using explicit candidate thresholds."""

    def __init__(
        self,
        candidate_thresholds: Sequence[float],
        *,
        objective_metric: str = "f1_score",
        anomaly_fraction: float = 0.1,
        anomaly_scale: float = 8.0,
        mask_fraction: float = 0.1,
        random_state: int | None = 0,
        include_masking: bool = True,
    ) -> None:
        if not candidate_thresholds:
            raise ValueError("candidate_thresholds must not be empty.")
        self.candidate_thresholds = [float(value) for value in candidate_thresholds]
        self.objective_metric = objective_metric
        self.anomaly_fraction = anomaly_fraction
        self.anomaly_scale = anomaly_scale
        self.mask_fraction = mask_fraction
        self.random_state = random_state
        self.include_masking = include_masking

    def optimize(
        self,
        model: BaseQualityModel,
        validation_df: pd.DataFrame,
        observation_columns: Iterable[str],
    ) -> ThresholdOptimizationResult:
        columns = model._validate_columns(observation_columns)
        detection_results = [
            evaluate_synthetic_anomaly_detection(
                model,
                validation_df,
                columns,
                threshold=threshold,
                anomaly_fraction=self.anomaly_fraction,
                anomaly_scale=self.anomaly_scale,
                random_state=self.random_state,
            )
            for threshold in self.candidate_thresholds
        ]

        best_result = max(
            detection_results,
            key=lambda result: self._objective_value(result),
        )
        masking_result = (
            evaluate_synthetic_masking(
                model,
                validation_df,
                columns,
                mask_fraction=self.mask_fraction,
                random_state=self.random_state,
            )
            if self.include_masking
            else None
        )
        return ThresholdOptimizationResult(
            best_threshold=best_result.threshold,
            best_objective_value=self._objective_value(best_result),
            objective_metric=self.objective_metric,
            detection_results=detection_results,
            best_detection_result=best_result,
            masking_result=masking_result,
        )

    def _objective_value(self, result: SyntheticAnomalyDetectionResult) -> float:
        value = getattr(result.metrics, self.objective_metric)
        if value is None or not np.isfinite(value):
            return float("-inf")
        return float(value)


__all__ = [
    "GridSearchThresholdOptimizer",
    "ThresholdOptimizationResult",
    "ThresholdOptimizer",
]
