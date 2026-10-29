"""Validation-only action-response diagnostics for transition candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from geas35.models.transition.base import BaseTransitionModel, TransitionDataset

CONTINUOUS_ACTIONS = ("vent_pct", "shade_curtain_pct", "thermal_curtain_pct")
BINARY_ACTIONS = ("heat_run", "cool_run", "fan_run")


@dataclass(frozen=True)
class ActionResponseReport:
    actions: Mapping[str, Mapping[str, Any]]
    eligible: bool
    fully_action_insensitive: bool
    minimum_support: int = 100

    def to_artifact(self) -> dict[str, Any]:
        return {
            "stage": "transition_action_response_validation",
            "minimum_support": int(self.minimum_support),
            "eligible": bool(self.eligible),
            "fully_action_insensitive": bool(self.fully_action_insensitive),
            "actions": {name: dict(values) for name, values in self.actions.items()},
            "test_used": False,
        }


def evaluate_action_response(
    model: BaseTransitionModel,
    dataset: TransitionDataset,
    *,
    minimum_support: int = 100,
    continuous_change_threshold: float = 0.10,
    probe_count: int = 256,
) -> ActionResponseReport:
    """Evaluate logged action changes and paired in-support counterfactual probes."""

    x = dataset.x.reset_index(drop=True)
    y = dataset.y.reset_index(drop=True)
    prediction = model.predict(x).next_observation.reset_index(drop=True)
    current = _current_physical_targets(model, x, dataset.target_columns)
    true_delta = y.loc[:, list(dataset.target_columns)] - current
    pred_delta = prediction.loc[:, list(dataset.target_columns)] - current
    reports: dict[str, dict[str, Any]] = {}
    sufficient_sensitivities = []

    for action in dataset.action_columns:
        values = pd.to_numeric(x[action], errors="coerce")
        previous = values.shift(1)
        same_trajectory = pd.Series(True, index=x.index)
        if dataset.metadata is not None:
            metadata = dataset.metadata.reset_index(drop=True)
            for boundary_column in ("episode_id", "segment_id"):
                if boundary_column in metadata.columns:
                    same_trajectory &= metadata[boundary_column].eq(metadata[boundary_column].shift(1))
        if action in BINARY_ACTIONS:
            changed = same_trajectory & values.notna() & previous.notna() & (values != previous)
        else:
            changed = same_trajectory & values.notna() & previous.notna() & (
                (values - previous).abs() >= float(continuous_change_threshold)
            )
        indices = np.flatnonzero(changed.to_numpy())
        support = int(len(indices))
        status = "eligible" if support >= minimum_support else "insufficient_support"
        target_metrics = {
            target: _metrics(
                true_delta.loc[indices, target].to_numpy(dtype=float),
                pred_delta.loc[indices, target].to_numpy(dtype=float),
            )
            for target in dataset.target_columns
        } if support else {}

        finite_values = values[np.isfinite(values)]
        sensitivity = float("nan")
        low = high = float("nan")
        if len(finite_values):
            low, high = (
                (0.0, 1.0)
                if action in BINARY_ACTIONS
                else (float(finite_values.quantile(0.10)), float(finite_values.quantile(0.90)))
            )
            if high > low:
                probe_positions = np.linspace(
                    0, len(x.index) - 1, num=min(probe_count, len(x.index)), dtype=int
                )
                low_x = x.iloc[probe_positions].copy()
                high_x = low_x.copy()
                low_x[action] = low
                high_x[action] = high
                low_prediction = model.predict(low_x).next_observation
                high_prediction = model.predict(high_x).next_observation
                sensitivity = float(np.nanmean(np.abs(
                    high_prediction.to_numpy(dtype=float)
                    - low_prediction.to_numpy(dtype=float)
                )))
        insensitive = bool(np.isfinite(sensitivity) and sensitivity <= 1e-12)
        if support >= minimum_support:
            sufficient_sensitivities.append(not insensitive)
        reports[str(action)] = {
            "support_count": support,
            "status": status,
            "change_threshold": 1.0 if action in BINARY_ACTIONS else float(continuous_change_threshold),
            "target_delta_metrics": target_metrics,
            "counterfactual_low": low,
            "counterfactual_high": high,
            "counterfactual_mean_absolute_sensitivity": sensitivity,
            "action_insensitive": insensitive,
        }

    fully_insensitive = bool(sufficient_sensitivities) and not any(sufficient_sensitivities)
    return ActionResponseReport(
        actions=reports,
        eligible=bool(sufficient_sensitivities) and not fully_insensitive,
        fully_action_insensitive=fully_insensitive,
        minimum_support=minimum_support,
    )


def _current_physical_targets(
    model: BaseTransitionModel,
    x: pd.DataFrame,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    scaler = getattr(model, "observation_scaler", None)
    if scaler is None:
        raise ValueError("Action-response validation requires the frozen Train scaler.")
    return scaler.inverse_transform_frame(x).loc[:, list(target_columns)].reset_index(drop=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    true = y_true[finite]
    pred = y_pred[finite]
    error = pred - true
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else float("nan"),
    }


__all__ = ["ActionResponseReport", "evaluate_action_response"]
