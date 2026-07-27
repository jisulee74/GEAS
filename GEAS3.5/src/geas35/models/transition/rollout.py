"""Rollout helpers for learned GEAS transition models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from geas35.models.transition.base import BaseTransitionModel, TransitionPrediction
from geas35.models.transition.features import (
    NEXT_OBSERVATION_PREFIX,
    infer_action_columns,
)
from geas35.rl import MDP_V1_ACTION_COLUMNS


@dataclass(frozen=True)
class RolloutContext:
    """Shared rollout context passed to action providers."""

    frame: pd.DataFrame
    action_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class RolloutStepContext:
    """Current rollout step context for action selection."""

    frame: pd.DataFrame
    source_position: int
    step: int
    current_observation: pd.Series
    action_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


class ActionProvider(ABC):
    """Interface for rollout action sources."""

    def reset(self, context: RolloutContext) -> None:
        """Reset provider state before a rollout."""

    @abstractmethod
    def action(self, context: RolloutStepContext) -> dict[str, float]:
        """Return the action to apply at the current rollout step."""


class LoggedActionProvider(ActionProvider):
    """Action provider that replays logged action columns from the source frame."""

    def __init__(self, action_columns: Sequence[str] | None = None) -> None:
        self.requested_action_columns = (
            None if action_columns is None else tuple(action_columns)
        )
        self.action_columns_: tuple[str, ...] | None = None

    def reset(self, context: RolloutContext) -> None:
        self.action_columns_ = self.requested_action_columns or tuple(
            column for column in context.action_columns if column in context.frame.columns
        )
        if not self.action_columns_:
            inferred = infer_action_columns(context.frame)
            self.action_columns_ = inferred
        if not self.action_columns_:
            raise ValueError("LoggedActionProvider could not resolve action columns.")

    def action(self, context: RolloutStepContext) -> dict[str, float]:
        action_columns = self.action_columns_
        if action_columns is None:
            action_columns = self.requested_action_columns or context.action_columns
        missing = [column for column in action_columns if column not in context.frame.columns]
        if missing:
            raise KeyError(f"Missing logged action column: {missing[0]}")
        row = context.frame.iloc[context.source_position]
        return {
            column: _numeric_value(row[column], default=0.0)
            for column in action_columns
        }


class PolicyActionProvider(ActionProvider, ABC):
    """Interface placeholder for future RL policy-backed rollout actions."""


@dataclass(frozen=True)
class TransitionRolloutResult:
    """One rollout trajectory and drift metrics."""

    start_index: int
    horizon_steps: int
    target_columns: tuple[str, ...]
    predictions: pd.DataFrame
    truth: pd.DataFrame
    actions: pd.DataFrame
    errors: pd.DataFrame
    metrics: dict[str, float]


class TransitionRolloutSimulator:
    """Simulate multi-step next-state rollouts with a learned transition model."""

    def __init__(
        self,
        model: BaseTransitionModel,
        action_provider: ActionProvider | None = None,
    ) -> None:
        self.model = model
        self.action_provider = action_provider or LoggedActionProvider()

    def simulate(
        self,
        frame: pd.DataFrame,
        *,
        start_index: int = 0,
        horizon_steps: int,
        action_columns: Sequence[str] | None = None,
        target_columns: Sequence[str] | None = None,
    ) -> TransitionRolloutResult:
        """Run one closed-loop rollout over a source trajectory."""

        if frame is None or frame.empty:
            raise ValueError("Rollout source frame must not be empty.")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")
        if start_index < 0:
            raise ValueError("start_index must be non-negative.")
        if start_index + horizon_steps > len(frame.index):
            raise ValueError("Source frame does not contain enough rows for rollout.")

        input_columns, fitted_targets = self.model._require_fitted()
        targets = tuple(target_columns or fitted_targets)
        actions = tuple(action_columns or _model_action_columns(input_columns, frame))
        rollout_context = RolloutContext(
            frame=frame,
            action_columns=actions,
            target_columns=targets,
        )
        self.action_provider.reset(rollout_context)

        current_row = frame.iloc[start_index].copy()
        prediction_rows: list[dict[str, float]] = []
        truth_rows: list[dict[str, float]] = []
        action_rows: list[dict[str, float]] = []

        for step in range(horizon_steps):
            source_position = start_index + step
            step_context = RolloutStepContext(
                frame=frame,
                source_position=source_position,
                step=step,
                current_observation=current_row.copy(),
                action_columns=actions,
                target_columns=targets,
            )
            action = self.action_provider.action(step_context)
            current_input = current_row.copy()
            for column, value in action.items():
                current_input[column] = value
            prediction = self.model.predict(current_input.to_frame().T)
            predicted_next = _prediction_row(prediction, targets)
            true_next = _true_next_row(frame.iloc[source_position], targets)

            prediction_rows.append(predicted_next)
            truth_rows.append(true_next)
            action_rows.append({column: float(action.get(column, np.nan)) for column in actions})

            if step < horizon_steps - 1:
                next_source_row = frame.iloc[source_position + 1].copy()
                for column, value in predicted_next.items():
                    next_source_row[column] = value
                _update_previous_action_observations(next_source_row, action)
                current_row = next_source_row

        predictions = pd.DataFrame(prediction_rows, columns=targets)
        truth = pd.DataFrame(truth_rows, columns=targets)
        errors = predictions - truth
        actions_frame = pd.DataFrame(action_rows, columns=actions)
        return TransitionRolloutResult(
            start_index=int(start_index),
            horizon_steps=int(horizon_steps),
            target_columns=targets,
            predictions=predictions,
            truth=truth,
            actions=actions_frame,
            errors=errors,
            metrics=_rollout_metrics(errors),
        )


def _model_action_columns(
    input_columns: Sequence[str],
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    actions = tuple(
        column
        for column in MDP_V1_ACTION_COLUMNS
        if column in input_columns and column in frame.columns
    )
    if actions:
        return actions
    return infer_action_columns(frame)


def _prediction_row(
    prediction: TransitionPrediction,
    target_columns: Sequence[str],
) -> dict[str, float]:
    frame = prediction.next_observation
    if frame.empty:
        raise ValueError("Transition model returned an empty prediction.")
    row = frame.iloc[0]
    return {column: _numeric_value(row[column], default=np.nan) for column in target_columns}


def _true_next_row(row: pd.Series, target_columns: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for column in target_columns:
        next_column = f"{NEXT_OBSERVATION_PREFIX}{column}"
        if next_column not in row.index:
            raise KeyError(f"Missing rollout truth column: {next_column}")
        out[column] = _numeric_value(row[next_column], default=np.nan)
    return out


def _update_previous_action_observations(
    row: pd.Series,
    action: Mapping[str, float],
) -> None:
    for column, value in action.items():
        obs_column = f"obs_prev_{column}"
        if obs_column in row.index:
            row[obs_column] = float(value)


def _rollout_metrics(errors: pd.DataFrame) -> dict[str, float]:
    values = errors.to_numpy(dtype=float)
    abs_values = np.abs(values)
    final_abs = abs_values[-1] if len(abs_values) else np.asarray([], dtype=float)
    step_mae = np.nanmean(abs_values, axis=1) if len(abs_values) else np.asarray([])
    metrics = {
        "trajectory_mae": _finite_mean(abs_values.reshape(-1)),
        "trajectory_rmse": _finite_rmse(values.reshape(-1)),
        "final_step_mae": _finite_mean(final_abs),
        "final_step_rmse": _finite_rmse(values[-1] if len(values) else []),
        "drift_slope_mae": _drift_slope(step_mae),
    }
    for column in errors.columns:
        col_values = errors[column].to_numpy(dtype=float)
        metrics[f"{column}_trajectory_mae"] = _finite_mean(np.abs(col_values))
        metrics[f"{column}_trajectory_rmse"] = _finite_rmse(col_values)
        metrics[f"{column}_final_abs_error"] = (
            float(abs(col_values[-1])) if len(col_values) and np.isfinite(col_values[-1]) else float("nan")
        )
    return metrics


def _numeric_value(value: object, *, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _finite_mean(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def _finite_rmse(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def _drift_slope(step_mae: np.ndarray) -> float:
    finite = np.asarray(step_mae, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return 0.0
    steps = np.arange(len(finite), dtype=float)
    slope, _ = np.polyfit(steps, finite, deg=1)
    return float(slope)


__all__ = [
    "ActionProvider",
    "LoggedActionProvider",
    "PolicyActionProvider",
    "RolloutContext",
    "RolloutStepContext",
    "TransitionRolloutResult",
    "TransitionRolloutSimulator",
]
