"""Per-variable threshold optimization for quality model anomaly scores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from geas35.models.quality.base import BaseQualityModel
from geas35.models.quality.evaluation import (
    SyntheticAnomalyDetectionResult, SyntheticMaskingResult, _binary_metrics,
    _binary_metrics_from_predictions, _normal_candidate_mask,
    evaluate_synthetic_masking, make_synthetic_anomaly_injection,
)


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    """Validation F1-optimal threshold for every modeled variable."""

    best_threshold: dict[str, float]
    best_objective_value: float
    objective_metric: str
    candidate_generation: str
    validation_error_min: dict[str, float]
    validation_error_max: dict[str, float]
    requested_candidate_count: int
    actual_candidate_count: int
    detection_results: list[SyntheticAnomalyDetectionResult]
    best_detection_result: SyntheticAnomalyDetectionResult
    per_column_calibration: dict[str, dict[str, object]]
    masking_result: SyntheticMaskingResult | None = None


class ThresholdOptimizer(ABC):
    @abstractmethod
    def optimize(self, model: BaseQualityModel, validation_df: pd.DataFrame,
                 observation_columns: Iterable[str]) -> ThresholdOptimizationResult:
        """Return variable-specific validation thresholds."""


class GridSearchThresholdOptimizer(ThresholdOptimizer):
    """Select thresholds from injected-validation score ranges per variable."""

    def __init__(self, *, candidate_count: int = 100, objective_metric: str = "f1_score",
                 anomaly_fraction: float = 0.1, anomaly_scale: float = 8.0,
                 mask_fraction: float = 0.1, random_state: int | None = 0,
                 include_masking: bool = True, progress_context: str | None = None,
                 reference_df: pd.DataFrame | None = None) -> None:
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
        self.reference_df = reference_df

    def optimize(self, model: BaseQualityModel, validation_df: pd.DataFrame,
                 observation_columns: Iterable[str]) -> ThresholdOptimizationResult:
        columns = model._validate_columns(observation_columns)
        clean_prediction = model.reconstruct(validation_df, columns)
        clean_scores = model.anomaly_score(validation_df, clean_prediction, columns)
        injection = make_synthetic_anomaly_injection(
            validation_df, columns, anomaly_fraction=self.anomaly_fraction,
            anomaly_scale=self.anomaly_scale, random_state=self.random_state,
            reference_df=self.reference_df)
        injected_prediction = model.reconstruct(injection.frame, columns)
        scores = model.anomaly_score(injection.frame, injected_prediction, columns)
        eval_mask = _normal_candidate_mask(validation_df, columns)

        thresholds: dict[str, float] = {}
        mins: dict[str, float] = {}
        maxs: dict[str, float] = {}
        calibration: dict[str, dict[str, object]] = {}
        per_column_metrics = {}
        predicted_mask = pd.DataFrame(False, index=scores.index, columns=list(columns))

        for col_index, col in enumerate(columns, start=1):
            finite_clean = pd.to_numeric(
                clean_scores.loc[eval_mask[col], col], errors="coerce"
            ).dropna().to_numpy(float)
            if not len(finite_clean):
                raise ValueError(f"{col}: no finite validation reconstruction error.")
            col_eval = eval_mask[col].to_numpy(bool)
            y_true = injection.anomaly_mask[col].to_numpy(bool)[col_eval]
            y_score = scores[col].to_numpy(float)[col_eval]
            finite_injected = y_score[np.isfinite(y_score)]
            if not len(finite_injected):
                raise ValueError(f"{col}: no finite injected-validation anomaly score.")
            clean_lo, clean_hi = float(finite_clean.min()), float(finite_clean.max())
            lo, hi = float(finite_injected.min()), float(finite_injected.max())
            candidates = np.linspace(lo, hi, self.candidate_count).tolist()
            rows = []
            best_threshold, best_metric, best_value = candidates[0], None, float("-inf")
            best_key = (float("-inf"), float("-inf"), float("-inf"))
            for candidate in candidates:
                metric = _binary_metrics(y_true, y_score, threshold=float(candidate))
                value = getattr(metric, self.objective_metric)
                value = float(value) if value is not None and np.isfinite(value) else float("-inf")
                rows.append({"threshold": float(candidate), "precision": metric.precision,
                             "recall": metric.recall, "f1_score": metric.f1_score,
                             "roc_auc": metric.roc_auc, "pr_auc": metric.pr_auc})
                candidate_key = (value, float(metric.precision), float(candidate))
                if candidate_key > best_key:
                    best_threshold, best_metric, best_value = float(candidate), metric, value
                    best_key = candidate_key
            thresholds[col], mins[col], maxs[col] = best_threshold, lo, hi
            per_column_metrics[col] = best_metric
            predicted_mask[col] = scores[col] >= best_threshold
            calibration[col] = {"best_threshold": best_threshold,
                                "best_objective_value": best_value,
                                "validation_error_min": lo, "validation_error_max": hi,
                                "clean_validation_error_min": clean_lo,
                                "clean_validation_error_max": clean_hi,
                                "candidate_range_upper_expansion": hi - clean_hi,
                                "tie_breaker": ["precision", "higher_threshold"],
                                "requested_candidate_count": self.candidate_count,
                                "actual_candidate_count": len(candidates),
                                "candidate_results": rows}
            if self.progress_context:
                print(f"[{self.progress_context}] {col} Threshold Calibration 완료 "
                      f"({col_index}/{len(columns)}): threshold={best_threshold}, "
                      f"{self.objective_metric}={best_value}", flush=True)

        selected = eval_mask.to_numpy(bool)
        overall = _binary_metrics_from_predictions(
            injection.anomaly_mask.to_numpy(bool)[selected],
            scores.to_numpy(float)[selected], predicted_mask.to_numpy(bool)[selected])
        best_result = SyntheticAnomalyDetectionResult(
            injected_cells=injection.injected_cells, threshold=thresholds, metrics=overall,
            per_column=per_column_metrics, scores=scores, predicted_mask=predicted_mask,
            ground_truth_mask=injection.anomaly_mask,
            injection_stats=injection.per_column_stats)
        masking_result = evaluate_synthetic_masking(
            model, validation_df, columns, mask_fraction=self.mask_fraction,
            random_state=self.random_state) if self.include_masking else None
        values = [float(v["best_objective_value"]) for v in calibration.values()]
        return ThresholdOptimizationResult(
            best_threshold=thresholds, best_objective_value=float(np.mean(values)),
            objective_metric=self.objective_metric,
            candidate_generation="per_column_injected_validation_score_linspace",
            validation_error_min=mins, validation_error_max=maxs,
            requested_candidate_count=self.candidate_count,
            actual_candidate_count=self.candidate_count * len(columns),
            detection_results=[best_result], best_detection_result=best_result,
            per_column_calibration=calibration, masking_result=masking_result)


__all__ = ["GridSearchThresholdOptimizer", "ThresholdOptimizationResult", "ThresholdOptimizer"]
