"""Evaluation helpers for quality model baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from geas35.models.quality.base import BaseQualityModel


@dataclass(frozen=True)
class SyntheticMaskingResult:
    """Summary of synthetic masking imputation evaluation."""

    masked_cells: int
    mae: float
    rmse: float
    per_column: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary anomaly-detection confusion matrix."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True)
class AnomalyDetectionMetrics:
    """Binary anomaly-detection metrics for synthetic anomaly injection."""

    precision: float
    recall: float
    f1_score: float
    confusion_matrix: ConfusionMatrix
    roc_auc: float
    pr_auc: float


@dataclass(frozen=True)
class SyntheticAnomalyInjection:
    """Synthetic anomaly injection output."""

    frame: pd.DataFrame
    anomaly_mask: pd.DataFrame
    injected_cells: int


@dataclass(frozen=True)
class SyntheticAnomalyDetectionResult:
    """Quality model evaluation on synthetic anomaly injection."""

    injected_cells: int
    threshold: float
    metrics: AnomalyDetectionMetrics
    per_column: dict[str, AnomalyDetectionMetrics]
    scores: pd.DataFrame
    predicted_mask: pd.DataFrame
    ground_truth_mask: pd.DataFrame


def _normal_candidate_mask(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    mask = pd.DataFrame(True, index=df.index, columns=list(columns))
    for col in columns:
        mask[col] &= pd.to_numeric(df[col], errors="coerce").notna()
        for suffix in ("_missing_flag", "_rule_outlier_flag", "_ai_outlier_flag"):
            flag_col = f"{col}{suffix}"
            if flag_col in df.columns:
                mask[col] &= ~(
                    pd.to_numeric(df[flag_col], errors="coerce")
                    .fillna(0)
                    .astype(bool)
                )
    return mask


def _numeric_observation_frame(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Copy a frame with model observation columns normalized to float dtype."""

    result = df.copy()
    for col in columns:
        result[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return result


def _binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> AnomalyDetectionMetrics:
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    y_pred = y_score >= threshold

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return AnomalyDetectionMetrics(
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        confusion_matrix=ConfusionMatrix(
            true_positive=tp,
            false_positive=fp,
            true_negative=tn,
            false_negative=fn,
        ),
        roc_auc=_roc_auc(y_true, y_score),
        pr_auc=_pr_auc(y_true, y_score),
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true.astype(bool)
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = _average_ranks(y_score)
    rank_sum_pos = float(ranks[positives].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true.astype(bool)
    n_pos = int(positives.sum())
    if n_pos == 0 or len(y_true) == 0:
        return np.nan

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = positives[order]
    tp = np.cumsum(sorted_true.astype(float))
    fp = np.cumsum((~sorted_true).astype(float))
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    area = np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2.0)
    return float(area)


def make_synthetic_mask(
    df: pd.DataFrame,
    observation_columns: Iterable[str],
    *,
    mask_fraction: float = 0.1,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Return a reproducible mask over normal observed cells."""

    columns = tuple(observation_columns)
    if not 0.0 < mask_fraction <= 1.0:
        raise ValueError("mask_fraction must be in the interval (0, 1].")

    candidates = _normal_candidate_mask(df, columns)
    candidate_positions = np.argwhere(candidates.to_numpy(dtype=bool))
    mask = pd.DataFrame(False, index=df.index, columns=list(columns))
    if len(candidate_positions) == 0:
        return mask

    n_mask = max(1, int(round(len(candidate_positions) * mask_fraction)))
    rng = np.random.default_rng(random_state)
    selected_idx = rng.choice(len(candidate_positions), size=n_mask, replace=False)
    selected = candidate_positions[selected_idx]
    values = mask.to_numpy(dtype=bool).copy()
    values[selected[:, 0], selected[:, 1]] = True
    return pd.DataFrame(values, index=df.index, columns=list(columns))


def make_synthetic_anomaly_injection(
    df: pd.DataFrame,
    observation_columns: Iterable[str],
    *,
    anomaly_fraction: float = 0.1,
    anomaly_scale: float = 8.0,
    random_state: int | None = 0,
) -> SyntheticAnomalyInjection:
    """Inject synthetic numeric anomalies into normal observed cells."""

    columns = tuple(observation_columns)
    if not 0.0 < anomaly_fraction <= 1.0:
        raise ValueError("anomaly_fraction must be in the interval (0, 1].")
    if anomaly_scale <= 0.0:
        raise ValueError("anomaly_scale must be positive.")

    candidates = _normal_candidate_mask(df, columns)
    candidate_positions = np.argwhere(candidates.to_numpy(dtype=bool))
    anomaly_mask = pd.DataFrame(False, index=df.index, columns=list(columns))
    injected = _numeric_observation_frame(df, columns)
    if len(candidate_positions) == 0:
        return SyntheticAnomalyInjection(
            frame=injected,
            anomaly_mask=anomaly_mask,
            injected_cells=0,
        )

    n_inject = max(1, int(round(len(candidate_positions) * anomaly_fraction)))
    rng = np.random.default_rng(random_state)
    selected_idx = rng.choice(len(candidate_positions), size=n_inject, replace=False)
    selected = candidate_positions[selected_idx]
    mask_values = anomaly_mask.to_numpy(dtype=bool).copy()
    mask_values[selected[:, 0], selected[:, 1]] = True
    anomaly_mask = pd.DataFrame(mask_values, index=df.index, columns=list(columns))

    for col_idx, col in enumerate(columns):
        col_mask = anomaly_mask[col].astype(bool)
        if not bool(col_mask.any()):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        scale = float(values.std(skipna=True, ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            median = float(values.median(skipna=True))
            scale = max(abs(median) * 0.1, 1.0) if np.isfinite(median) else 1.0
        direction = 1.0 if col_idx % 2 == 0 else -1.0
        injected.loc[col_mask, col] = values.loc[col_mask] + direction * anomaly_scale * scale

    return SyntheticAnomalyInjection(
        frame=injected,
        anomaly_mask=anomaly_mask,
        injected_cells=int(anomaly_mask.to_numpy(dtype=bool).sum()),
    )


def evaluate_synthetic_masking(
    model: BaseQualityModel,
    validation_df: pd.DataFrame,
    observation_columns: Iterable[str],
    *,
    mask_fraction: float = 0.1,
    random_state: int | None = 0,
) -> SyntheticMaskingResult:
    """Mask normal validation cells and evaluate model imputation MAE/RMSE."""

    columns = model._validate_columns(observation_columns)
    synthetic_mask = make_synthetic_mask(
        validation_df,
        columns,
        mask_fraction=mask_fraction,
        random_state=random_state,
    )

    masked_df = _numeric_observation_frame(validation_df, columns)
    for col in columns:
        masked_df.loc[synthetic_mask[col], col] = np.nan

    prediction = model.reconstruct(masked_df, columns)
    imputed = model.impute(masked_df, synthetic_mask, prediction, columns)

    errors: list[float] = []
    per_column: dict[str, dict[str, float]] = {}
    for col in columns:
        col_mask = synthetic_mask[col].astype(bool)
        if not bool(col_mask.any()):
            per_column[col] = {"masked_cells": 0.0, "mae": np.nan, "rmse": np.nan}
            continue
        actual = pd.to_numeric(validation_df.loc[col_mask, col], errors="coerce")
        predicted = pd.to_numeric(imputed.loc[col_mask, col], errors="coerce")
        diff = (predicted - actual).dropna()
        errors.extend(diff.tolist())
        per_column[col] = {
            "masked_cells": float(int(col_mask.sum())),
            "mae": float(diff.abs().mean()) if len(diff) else np.nan,
            "rmse": float(np.sqrt((diff**2).mean())) if len(diff) else np.nan,
        }

    if errors:
        error_array = np.asarray(errors, dtype=float)
        mae = float(np.mean(np.abs(error_array)))
        rmse = float(np.sqrt(np.mean(error_array**2)))
    else:
        mae = np.nan
        rmse = np.nan

    return SyntheticMaskingResult(
        masked_cells=int(synthetic_mask.to_numpy(dtype=bool).sum()),
        mae=mae,
        rmse=rmse,
        per_column=per_column,
    )


def evaluate_synthetic_anomaly_detection(
    model: BaseQualityModel,
    validation_df: pd.DataFrame,
    observation_columns: Iterable[str],
    *,
    threshold: float,
    anomaly_fraction: float = 0.1,
    anomaly_scale: float = 8.0,
    random_state: int | None = 0,
) -> SyntheticAnomalyDetectionResult:
    """Evaluate anomaly detection using synthetic anomaly injection."""

    columns = model._validate_columns(observation_columns)
    injection = make_synthetic_anomaly_injection(
        validation_df,
        columns,
        anomaly_fraction=anomaly_fraction,
        anomaly_scale=anomaly_scale,
        random_state=random_state,
    )
    prediction = model.reconstruct(injection.frame, columns)
    scores = model.anomaly_score(injection.frame, prediction, columns)
    predicted_mask = scores >= threshold
    eval_mask = _normal_candidate_mask(validation_df, columns)

    all_true = injection.anomaly_mask.to_numpy(dtype=bool)[
        eval_mask.to_numpy(dtype=bool)
    ]
    all_scores = scores.to_numpy(dtype=float)[eval_mask.to_numpy(dtype=bool)]
    metrics = _binary_metrics(all_true, all_scores, threshold=threshold)

    per_column: dict[str, AnomalyDetectionMetrics] = {}
    for col in columns:
        col_eval = eval_mask[col].to_numpy(dtype=bool)
        per_column[col] = _binary_metrics(
            injection.anomaly_mask[col].to_numpy(dtype=bool)[col_eval],
            scores[col].to_numpy(dtype=float)[col_eval],
            threshold=threshold,
        )

    return SyntheticAnomalyDetectionResult(
        injected_cells=injection.injected_cells,
        threshold=float(threshold),
        metrics=metrics,
        per_column=per_column,
        scores=scores,
        predicted_mask=predicted_mask,
        ground_truth_mask=injection.anomaly_mask,
    )


__all__ = [
    "SyntheticMaskingResult",
    "AnomalyDetectionMetrics",
    "ConfusionMatrix",
    "SyntheticAnomalyDetectionResult",
    "SyntheticAnomalyInjection",
    "evaluate_synthetic_anomaly_detection",
    "evaluate_synthetic_masking",
    "make_synthetic_anomaly_injection",
    "make_synthetic_mask",
]
