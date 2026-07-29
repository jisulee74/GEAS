"""Threshold optimization for quality model anomaly scores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

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
    candidate_generation: str
    validation_error_min: float
    validation_error_max: float
    requested_candidate_count: int
    actual_candidate_count: int
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
    """Validation-error grid optimizer using evenly spaced candidate thresholds."""

    def __init__(
        self,
        *,
        candidate_count: int = 100,
        objective_metric: str = "f1_score",
        anomaly_fraction: float = 0.1,
        anomaly_scale: float = 8.0,
        mask_fraction: float = 0.1,
        random_state: int | None = 0,
        include_masking: bool = True,
        progress_context: str | None = None,
    ) -> None:
        if candidate_count < 2:
            raise ValueError("candidate_count must be >= 2.")
        self.candidate_count = int(candidate_count)
        self.objective_metric = objective_metric
        self.anomaly_fraction = anomaly_fraction
        self.anomaly_scale = anomaly_scale
        self.mask_fraction = mask_fraction
        self.random_state = random_state
        self.include_masking = include_masking
        self.progress_context = progress_context

    def optimize(
        self,
        model: BaseQualityModel,
        validation_df: pd.DataFrame,
        observation_columns: Iterable[str],
    ) -> ThresholdOptimizationResult:
        columns = model._validate_columns(observation_columns)
        validation_prediction = model.reconstruct(validation_df, columns)
        validation_errors = model.anomaly_score(
            validation_df, validation_prediction, columns
        ).to_numpy(dtype=float)
        finite_errors = validation_errors[np.isfinite(validation_errors)]
        if finite_errors.size == 0:
            raise ValueError(
                "Threshold calibration requires at least one finite validation "
                "reconstruction error."
            )
        error_min = float(finite_errors.min())
        error_max = float(finite_errors.max())
        candidate_thresholds = np.linspace(
            error_min, error_max, self.candidate_count
        ).tolist()
        detection_results = []
        total_candidates = len(candidate_thresholds)
        for index, threshold in enumerate(candidate_thresholds, start=1):
            detection_results.append(
                evaluate_synthetic_anomaly_detection(
                    model,
                    validation_df,
                    columns,
                    threshold=threshold,
                    anomaly_fraction=self.anomaly_fraction,
                    anomaly_scale=self.anomaly_scale,
                    random_state=self.random_state,
                )
            )
            if (
                self.progress_context is not None
                and (index == 1 or index % 10 == 0 or index == total_candidates)
            ):
                print(
                    f"[{self.progress_context}] Threshold 후보 평가 "
                    f"({index}/{total_candidates})",
                    flush=True,
                )

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
            candidate_generation="validation_reconstruction_error_linspace",
            validation_error_min=error_min,
            validation_error_max=error_max,
            requested_candidate_count=self.candidate_count,
            actual_candidate_count=len(candidate_thresholds),
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
