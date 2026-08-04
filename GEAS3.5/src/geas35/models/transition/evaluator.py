"""One-step evaluation helpers for GEAS transition models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from geas35.models.transition.base import (
    BaseTransitionModel,
    TransitionDataset,
    TransitionPrediction,
)
from geas35.rl import evaluate_mdp_v1_transition_predictions

DEFAULT_TARGET_GROUPS = {
    "indoor_environment": (
        "obs_indoor_temp_c",
        "obs_indoor_humidity_pct",
    ),
    "outdoor_environment": (
        "obs_outdoor_temp_c",
        "obs_outdoor_humidity_pct",
        "obs_outdoor_light",
        "obs_outdoor_wind_speed",
        "obs_rain_flag",
    ),
    "target_temperature": (
        "obs_current_target_temp_min_c",
        "obs_current_target_temp_max_c",
        "obs_current_target_temp_c",
    ),
    "derived_environment": (
        "obs_current_vpd_kpa",
        "obs_current_dewpoint_c",
        "obs_current_condensation_margin_c",
    ),
}


@dataclass(frozen=True)
class TransitionOneStepEvaluationReport:
    """One-step transition prediction metrics and aggregate summaries."""

    row_count: int
    target_columns: tuple[str, ...]
    target_metrics: dict[str, dict[str, float]]
    aggregate_metrics: dict[str, float]
    target_group_metrics: dict[str, dict[str, float]]

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-safe artifact payload."""

        return {
            "stage": "transition_one_step_evaluation",
            "row_count": int(self.row_count),
            "target_columns": list(self.target_columns),
            "target_metrics": _jsonable_metrics(self.target_metrics),
            "aggregate_metrics": _jsonable_metrics(self.aggregate_metrics),
            "target_group_metrics": _jsonable_metrics(self.target_group_metrics),
        }


def evaluate_transition_predictions(
    y_true: pd.DataFrame | Mapping[str, Sequence[float]] | np.ndarray,
    y_pred: pd.DataFrame | Mapping[str, Sequence[float]] | np.ndarray | TransitionPrediction,
    *,
    target_columns: Sequence[str] | None = None,
    normalization_scales: Mapping[str, float] | None = None,
    target_groups: Mapping[str, Sequence[str]] | None = None,
) -> TransitionOneStepEvaluationReport:
    """Evaluate one-step next-observation predictions."""

    true_df = pd.DataFrame(y_true)
    pred_df = _prediction_frame(y_pred)
    columns = _resolve_target_columns(true_df, pred_df, target_columns)
    target_metrics = evaluate_mdp_v1_transition_predictions(
        true_df,
        pred_df,
        target_columns=columns,
    )
    target_metrics = _with_normalized_target_metrics(
        true_df,
        target_metrics,
        columns,
        normalization_scales=normalization_scales,
    )
    aggregate_metrics = _aggregate_metrics(
        true_df,
        target_metrics,
        columns,
        normalization_scales=normalization_scales,
    )
    group_metrics = _target_group_metrics(
        target_metrics,
        target_groups or DEFAULT_TARGET_GROUPS,
    )
    return TransitionOneStepEvaluationReport(
        row_count=int(len(true_df.index)),
        target_columns=tuple(columns),
        target_metrics=target_metrics,
        aggregate_metrics=aggregate_metrics,
        target_group_metrics=group_metrics,
    )


def evaluate_transition_model_one_step(
    model: BaseTransitionModel,
    dataset: TransitionDataset,
    *,
    normalization_scales: Mapping[str, float] | None = None,
    target_groups: Mapping[str, Sequence[str]] | None = None,
) -> TransitionOneStepEvaluationReport:
    """Run model inference on a dataset and evaluate one-step predictions."""

    prediction = model.predict(dataset)
    return evaluate_transition_predictions(
        dataset.y,
        prediction,
        target_columns=dataset.target_columns,
        normalization_scales=normalization_scales,
        target_groups=target_groups,
    )


def _prediction_frame(
    prediction: pd.DataFrame | Mapping[str, Sequence[float]] | np.ndarray | TransitionPrediction,
) -> pd.DataFrame:
    if isinstance(prediction, TransitionPrediction):
        return prediction.next_observation
    return pd.DataFrame(prediction)


def _resolve_target_columns(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    columns = (
        tuple(str(column) for column in target_columns)
        if target_columns is not None
        else tuple(column for column in true_df.columns if column in pred_df.columns)
    )
    if not columns:
        raise ValueError("No transition target columns are available for evaluation.")
    missing_true = [column for column in columns if column not in true_df.columns]
    missing_pred = [column for column in columns if column not in pred_df.columns]
    if missing_true:
        raise KeyError(f"Missing true target column: {missing_true[0]}")
    if missing_pred:
        raise KeyError(f"Missing predicted target column: {missing_pred[0]}")
    return columns


def _aggregate_metrics(
    true_df: pd.DataFrame,
    target_metrics: Mapping[str, Mapping[str, float]],
    target_columns: Sequence[str],
    *,
    normalization_scales: Mapping[str, float] | None,
) -> dict[str, float]:
    aggregate = {
        f"mean_{metric_key}": _mean_metric(target_metrics, metric_key, target_columns)
        for metric_key in ("r2", "mae", "rmse", "q90", "cvar90", "nrmse")
    }
    normalized_rmse_values = []
    for column in target_columns:
        rmse = float(target_metrics[column]["rmse"])
        scale = (
            float(normalization_scales[column])
            if normalization_scales is not None and column in normalization_scales
            else _target_scale(true_df[column])
        )
        if np.isfinite(rmse) and np.isfinite(scale) and scale > 0.0:
            normalized_rmse_values.append(rmse / scale)
    aggregate["normalized_mean_rmse"] = _finite_mean(normalized_rmse_values)
    aggregate["mean_nrmse"] = aggregate["normalized_mean_rmse"]

    if "obs_indoor_temp_c" in target_metrics:
        aggregate["indoor_temp_rmse"] = float(
            target_metrics["obs_indoor_temp_c"]["rmse"]
        )
        aggregate["indoor_temp_mae"] = float(
            target_metrics["obs_indoor_temp_c"]["mae"]
        )
    if "obs_indoor_humidity_pct" in target_metrics:
        aggregate["indoor_humidity_rmse"] = float(
            target_metrics["obs_indoor_humidity_pct"]["rmse"]
        )
        aggregate["indoor_humidity_mae"] = float(
            target_metrics["obs_indoor_humidity_pct"]["mae"]
        )
    return aggregate


def _target_group_metrics(
    target_metrics: Mapping[str, Mapping[str, float]],
    target_groups: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for group_name, columns in target_groups.items():
        present = [column for column in columns if column in target_metrics]
        if not present:
            continue
        out[group_name] = {
            f"mean_{metric_key}": _mean_metric(target_metrics, metric_key, present)
            for metric_key in ("r2", "mae", "rmse", "q90", "cvar90", "nrmse")
        }
    return out


def _with_normalized_target_metrics(
    true_df: pd.DataFrame,
    target_metrics: Mapping[str, Mapping[str, float]],
    target_columns: Sequence[str],
    *,
    normalization_scales: Mapping[str, float] | None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for column in target_columns:
        metrics = dict(target_metrics[column])
        scale = (
            float(normalization_scales[column])
            if normalization_scales is not None and column in normalization_scales
            else _target_scale(true_df[column])
        )
        rmse = float(metrics["rmse"])
        metrics["nrmse"] = (
            rmse / scale
            if np.isfinite(rmse) and np.isfinite(scale) and scale > 0.0
            else float("nan")
        )
        out[str(column)] = metrics
    return out


def _mean_metric(
    target_metrics: Mapping[str, Mapping[str, float]],
    metric_key: str,
    columns: Sequence[str],
) -> float:
    return _finite_mean(float(target_metrics[column][metric_key]) for column in columns)


def _finite_mean(values: Sequence[float] | Any) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def _target_scale(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return 1.0
    std = float(np.std(finite, ddof=0))
    if np.isfinite(std) and std > 0.0:
        return std
    value_range = float(np.max(finite) - np.min(finite))
    if np.isfinite(value_range) and value_range > 0.0:
        return value_range
    return 1.0


def _jsonable_metrics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_metrics(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable_metrics(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


__all__ = [
    "DEFAULT_TARGET_GROUPS",
    "TransitionOneStepEvaluationReport",
    "evaluate_transition_model_one_step",
    "evaluate_transition_predictions",
]
