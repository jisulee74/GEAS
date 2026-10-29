"""Evaluation helpers for quality model baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

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
    per_column_stats: dict[str, dict[str, object]]


@dataclass(frozen=True)
class SyntheticAnomalyDetectionResult:
    """Quality model evaluation on synthetic anomaly injection."""

    injected_cells: int
    threshold: float | dict[str, float]
    metrics: AnomalyDetectionMetrics
    per_column: dict[str, AnomalyDetectionMetrics]
    scores: pd.DataFrame
    predicted_mask: pd.DataFrame
    ground_truth_mask: pd.DataFrame
    injection_stats: dict[str, dict[str, object]]


def _domain_constraints(
    columns: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Load canonical rule bounds used to keep AI injections rule-normal."""

    from geas35.preprocessing import load_domain_range_rules

    constraints: dict[str, dict[str, object]] = {}
    for rule in load_domain_range_rules():
        if rule.column not in columns:
            continue
        constraints[rule.column] = {
            "lower": rule.lower,
            "upper": rule.upper,
            "lower_inclusive": rule.lower_inclusive,
            "upper_inclusive": rule.upper_inclusive,
            "allowed_values": tuple(rule.allowed_values),
            "integer_only": rule.integer_only,
        }
    return constraints


def _rule_normal_mask(
    values: pd.Series,
    constraint: Mapping[str, object] | None,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    if not constraint:
        return valid
    lower = constraint.get("lower")
    upper = constraint.get("upper")
    if lower is not None:
        valid &= numeric.ge(float(lower)) if constraint.get("lower_inclusive", True) else numeric.gt(float(lower))
    if upper is not None:
        valid &= numeric.le(float(upper)) if constraint.get("upper_inclusive", True) else numeric.lt(float(upper))
    allowed = tuple(constraint.get("allowed_values", ()))
    if allowed:
        valid &= numeric.isin(allowed)
    if constraint.get("integer_only", False):
        valid &= np.isclose(numeric, numeric.round(), equal_nan=False)
    return valid


def _normal_candidate_mask(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    mask = pd.DataFrame(True, index=df.index, columns=list(columns))
    constraints = _domain_constraints(columns)
    for col in columns:
        mask[col] &= _rule_normal_mask(df[col], constraints.get(col))
        for suffix in ("_missing_flag", "_rule_outlier_flag", "_ai_outlier_flag", "_ffill_flag"):
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


def _binary_metrics_from_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> AnomalyDetectionMetrics:
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred, dtype=bool)
    finite = np.isfinite(y_score)
    y_true, y_score, y_pred = y_true[finite], y_score[finite], y_pred[finite]
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return AnomalyDetectionMetrics(
        precision=float(precision), recall=float(recall), f1_score=float(f1),
        confusion_matrix=ConfusionMatrix(tp, fp, tn, fn),
        roc_auc=_roc_auc(y_true, y_score), pr_auc=_pr_auc(y_true, y_score),
    )


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
    reference_df: pd.DataFrame | None = None,
) -> SyntheticAnomalyInjection:
    """Inject seeded, direction-balanced anomalies inside canonical rule bounds.

    Variable scales are estimated once from ``reference_df`` (Train in the
    experiment pipeline) and reused for Validation/Test. When a full ``8σ``
    move would leave the rule-normal interval, its magnitude is capped at the
    nearest valid boundary and recorded in ``per_column_stats``.
    """

    columns = tuple(observation_columns)
    if not 0.0 < anomaly_fraction <= 1.0:
        raise ValueError("anomaly_fraction must be in the interval (0, 1].")
    if anomaly_scale <= 0.0:
        raise ValueError("anomaly_scale must be positive.")

    candidates = _normal_candidate_mask(df, columns)
    candidate_positions = np.argwhere(candidates.to_numpy(dtype=bool))
    anomaly_mask = pd.DataFrame(False, index=df.index, columns=list(columns))
    injected = _numeric_observation_frame(df, columns)
    stats: dict[str, dict[str, object]] = {}
    if len(candidate_positions) == 0:
        return SyntheticAnomalyInjection(
            frame=injected,
            anomaly_mask=anomaly_mask,
            injected_cells=0,
            per_column_stats=stats,
        )

    n_inject = max(1, int(round(len(candidate_positions) * anomaly_fraction)))
    rng = np.random.default_rng(random_state)
    selected_idx = rng.choice(len(candidate_positions), size=n_inject, replace=False)
    selected = candidate_positions[selected_idx]
    mask_values = anomaly_mask.to_numpy(dtype=bool).copy()
    mask_values[selected[:, 0], selected[:, 1]] = True
    anomaly_mask = pd.DataFrame(mask_values, index=df.index, columns=list(columns))

    reference = df if reference_df is None else reference_df
    constraints = _domain_constraints(columns)
    for col in columns:
        selected_rows = list(df.index[anomaly_mask[col].astype(bool)])
        reference_values = pd.to_numeric(reference[col], errors="coerce")
        reference_values = reference_values[_rule_normal_mask(reference_values, constraints.get(col))]
        scale = float(reference_values.std(skipna=True, ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            median = float(reference_values.median(skipna=True))
            scale = max(abs(median) * 0.1, 1.0) if np.isfinite(median) else 1.0
        target_magnitude = float(anomaly_scale * scale)
        constraint = constraints.get(col, {})
        lower = constraint.get("lower")
        upper = constraint.get("upper")
        if lower is not None and not constraint.get("lower_inclusive", True):
            lower = float(np.nextafter(float(lower), np.inf))
        if upper is not None and not constraint.get("upper_inclusive", True):
            upper = float(np.nextafter(float(upper), -np.inf))

        rng.shuffle(selected_rows)
        positive_count = 0
        negative_count = 0
        capped_count = 0
        excluded_count = 0
        deltas: list[float] = []
        for row in selected_rows:
            value = float(pd.to_numeric(df.at[row, col], errors="coerce"))
            positive_room = float("inf") if upper is None else max(0.0, float(upper) - value)
            negative_room = float("inf") if lower is None else max(0.0, value - float(lower))
            positive_ok = positive_room > 0.0
            negative_ok = negative_room > 0.0
            if not positive_ok and not negative_ok:
                anomaly_mask.at[row, col] = False
                excluded_count += 1
                continue
            if positive_ok and negative_ok:
                if positive_count == negative_count:
                    direction = 1.0 if bool(rng.integers(0, 2)) else -1.0
                else:
                    direction = 1.0 if positive_count < negative_count else -1.0
            else:
                direction = 1.0 if positive_ok else -1.0
            room = positive_room if direction > 0 else negative_room
            magnitude = min(target_magnitude, room)
            if magnitude < target_magnitude:
                capped_count += 1
            if magnitude <= 0.0 or not np.isfinite(magnitude):
                anomaly_mask.at[row, col] = False
                excluded_count += 1
                continue
            injected_value = value + direction * magnitude
            if lower is not None:
                injected_value = max(injected_value, float(lower))
            if upper is not None:
                injected_value = min(injected_value, float(upper))
            actual_delta = injected_value - value
            if actual_delta == 0.0:
                anomaly_mask.at[row, col] = False
                excluded_count += 1
                continue
            injected.at[row, col] = injected_value
            deltas.append(actual_delta)
            if direction > 0:
                positive_count += 1
            else:
                negative_count += 1

        actual = positive_count + negative_count
        stats[col] = {
            "candidate_cells": int(candidates[col].sum()),
            "requested_injected_cells": len(selected_rows),
            "injected_cells": actual,
            "excluded_cells": excluded_count,
            "positive_injections": positive_count,
            "negative_injections": negative_count,
            "capped_to_rule_range_cells": capped_count,
            "actual_injection_rate": float(actual / max(1, int(candidates[col].sum()))),
            "reference_scale_source": "train" if reference_df is not None else "current_split",
            "reference_std": scale,
            "target_magnitude": target_magnitude,
            "actual_abs_delta_min": float(min(map(abs, deltas))) if deltas else None,
            "actual_abs_delta_mean": float(np.mean(np.abs(deltas))) if deltas else None,
            "actual_abs_delta_max": float(max(map(abs, deltas))) if deltas else None,
        }

    return SyntheticAnomalyInjection(
        frame=injected,
        anomaly_mask=anomaly_mask,
        injected_cells=int(anomaly_mask.to_numpy(dtype=bool).sum()),
        per_column_stats=stats,
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
    threshold: float | Mapping[str, float],
    anomaly_fraction: float = 0.1,
    anomaly_scale: float = 8.0,
    random_state: int | None = 0,
    reference_df: pd.DataFrame | None = None,
) -> SyntheticAnomalyDetectionResult:
    """Evaluate anomaly detection using synthetic anomaly injection."""

    columns = model._validate_columns(observation_columns)
    injection = make_synthetic_anomaly_injection(
        validation_df,
        columns,
        anomaly_fraction=anomaly_fraction,
        anomaly_scale=anomaly_scale,
        random_state=random_state,
        reference_df=reference_df,
    )
    prediction = model.reconstruct(injection.frame, columns)
    scores = model.anomaly_score(injection.frame, prediction, columns)
    threshold_map = (
        {col: float(threshold[col]) for col in columns}
        if isinstance(threshold, Mapping)
        else {col: float(threshold) for col in columns}
    )
    predicted_mask = pd.DataFrame(
        {col: scores[col] >= threshold_map[col] for col in columns},
        index=scores.index,
    )
    eval_mask = _normal_candidate_mask(validation_df, columns)

    selected = eval_mask.to_numpy(dtype=bool)
    all_true = injection.anomaly_mask.to_numpy(dtype=bool)[selected]
    all_scores = scores.to_numpy(dtype=float)[selected]
    all_pred = predicted_mask.to_numpy(dtype=bool)[selected]
    metrics = _binary_metrics_from_predictions(all_true, all_scores, all_pred)

    per_column: dict[str, AnomalyDetectionMetrics] = {}
    for col in columns:
        col_eval = eval_mask[col].to_numpy(dtype=bool)
        per_column[col] = _binary_metrics(
            injection.anomaly_mask[col].to_numpy(dtype=bool)[col_eval],
            scores[col].to_numpy(dtype=float)[col_eval],
            threshold=threshold_map[col],
        )

    return SyntheticAnomalyDetectionResult(
        injected_cells=injection.injected_cells,
        threshold=threshold_map if isinstance(threshold, Mapping) else float(threshold),
        metrics=metrics,
        per_column=per_column,
        scores=scores,
        predicted_mask=predicted_mask,
        ground_truth_mask=injection.anomaly_mask,
        injection_stats=injection.per_column_stats,
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
