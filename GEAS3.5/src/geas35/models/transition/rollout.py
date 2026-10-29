"""Rollout helpers for learned GEAS transition models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from geas35.models.transition.base import BaseTransitionModel, TransitionPrediction
from geas35.models.transition.causal_state import (
    CausalStateUpdater, ROLLOUT_PROTOCOL_VERSION, RolloutEvaluationProtocol, RolloutWindow, build_rollout_windows)
from geas35.models.transition.features import (
    NEXT_OBSERVATION_PREFIX,
    infer_action_columns,
)
from geas35.rl import MDP_V1_ACTION_COLUMNS
from geas35.rl import MdpV1ObservationScaler

EXOGENOUS_WEATHER_COLUMNS = (
    "obs_outdoor_temp_c",
    "obs_outdoor_humidity_pct",
    "obs_outdoor_light",
    "obs_outdoor_wind_speed",
    "obs_rain_flag",
)


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


class ExogenousProvider(ABC):
    """Supply weather observations without asking the transition model to predict them."""

    mode: str

    def reset(self, context: RolloutContext) -> None:
        """Validate provider inputs before a rollout."""

    @abstractmethod
    def next_row(
        self,
        *,
        source_frame: pd.DataFrame,
        next_source_position: int,
        recorded_next_row: pd.Series,
    ) -> pd.Series:
        """Return the next row with exogenous weather supplied."""


class RecordedWeatherProvider(ExogenousProvider):
    """Offline provider that uses timestamp-aligned weather recorded in the frame."""

    mode = "recorded_weather"

    def next_row(self, *, source_frame: pd.DataFrame, next_source_position: int,
                 recorded_next_row: pd.Series) -> pd.Series:
        del source_frame, next_source_position
        return recorded_next_row.copy()


class ForecastWeatherProvider(ExogenousProvider):
    """Operating provider using an exact-timestamp short-term weather forecast."""

    mode = "forecast_weather"

    def __init__(self, forecast_frame: pd.DataFrame, *, time_column: str = "reg_date") -> None:
        self.forecast_frame = forecast_frame.copy()
        self.time_column = time_column
        self._indexed: pd.DataFrame | None = None

    def reset(self, context: RolloutContext) -> None:
        del context
        if self.time_column not in self.forecast_frame.columns:
            raise KeyError(f"Forecast frame is missing {self.time_column}.")
        missing = [column for column in EXOGENOUS_WEATHER_COLUMNS
                   if column not in self.forecast_frame.columns]
        if missing:
            raise KeyError(f"Forecast frame is missing exogenous columns: {missing}")
        timestamps = pd.to_datetime(self.forecast_frame[self.time_column], errors="coerce")
        if timestamps.isna().any() or timestamps.duplicated().any():
            raise ValueError("Forecast timestamps must be valid and unique.")
        indexed = self.forecast_frame.copy()
        indexed.index = timestamps
        self._indexed = indexed

    def next_row(self, *, source_frame: pd.DataFrame, next_source_position: int,
                 recorded_next_row: pd.Series) -> pd.Series:
        del source_frame, next_source_position
        if self._indexed is None:
            raise RuntimeError("ForecastWeatherProvider.reset must be called before rollout.")
        timestamp = pd.to_datetime(recorded_next_row.get(self.time_column), errors="coerce")
        if pd.isna(timestamp) or timestamp not in self._indexed.index:
            raise KeyError(f"No timestamp-aligned forecast for {timestamp}.")
        forecast = self._indexed.loc[timestamp]
        out = recorded_next_row.copy()
        for column in EXOGENOUS_WEATHER_COLUMNS:
            out[column] = _numeric_value(forecast[column], default=np.nan)
        return out


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
    exogenous_provider_mode: str
    trace: pd.DataFrame


class TransitionRolloutSimulator:
    """Simulate multi-step next-state rollouts with a learned transition model."""

    def __init__(
        self,
        model: BaseTransitionModel,
        action_provider: ActionProvider | None = None,
        *,
        scaler: MdpV1ObservationScaler,
        state_updater: CausalStateUpdater | None = None,
        exogenous_provider: ExogenousProvider | None = None,
    ) -> None:
        self.model = model
        self.action_provider = action_provider or LoggedActionProvider()
        if scaler is None:
            raise ValueError("A frozen Train observation scaler is required.")
        self.scaler = scaler
        self.state_updater = state_updater or CausalStateUpdater(scaler)
        self.exogenous_provider = exogenous_provider or RecordedWeatherProvider()

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
        self.exogenous_provider.reset(rollout_context)

        current_row = frame.iloc[start_index].copy()
        self.state_updater.reset(current_row)
        prediction_rows: list[dict[str, float]] = []
        truth_rows: list[dict[str, float]] = []
        action_rows: list[dict[str, float]] = []
        trace_rows: list[dict[str, float | int]] = []

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
            current_input_frame = pd.DataFrame(
                [current_input.reindex(input_columns).to_dict()],
                columns=input_columns,
            ).apply(pd.to_numeric, errors="coerce")
            prediction = self.model.predict(current_input_frame)
            predicted_next = _prediction_row(prediction, targets)
            true_next = _true_next_row(frame.iloc[source_position], targets)
            current_physical = self.scaler.inverse_transform_frame(current_input_frame).iloc[0]
            trace_row: dict[str, float | int] = {
                "step": int(step + 1),
                "source_position": int(source_position),
                "max_abs_scaled_input": float(np.nanmax(np.abs(current_input_frame.to_numpy(dtype=float)))),
            }
            for column in targets:
                trace_row[f"current_{column}"] = _numeric_value(current_physical.get(column), default=np.nan)
                trace_row[f"predicted_{column}"] = float(predicted_next[column])
                trace_row[f"truth_{column}"] = float(true_next[column])
            if prediction.raw_prediction is not None and not prediction.raw_prediction.empty:
                for column in targets:
                    if column in prediction.raw_prediction:
                        trace_row[f"residual_{column}"] = _numeric_value(prediction.raw_prediction.iloc[0][column], default=np.nan)
            for column in (
                "obs_current_vpd_kpa", "obs_current_dewpoint_c",
                "obs_current_condensation_margin_c",
                "obs_derived_condensation_margin_pred_10m",
                "obs_derived_condensation_risk_10m",
            ):
                if column in current_physical:
                    trace_row[f"input_{column}"] = _numeric_value(current_physical[column], default=np.nan)
            trace_rows.append(trace_row)

            prediction_rows.append(predicted_next)
            truth_rows.append(true_next)
            action_rows.append({column: float(action.get(column, np.nan)) for column in actions})

            if step < horizon_steps - 1:
                recorded_next_scaled = frame.iloc[source_position + 1].copy()
                recorded_next_physical = self.scaler.inverse_transform_frame(
                    pd.DataFrame([recorded_next_scaled])
                ).iloc[0]
                next_source_physical = self.exogenous_provider.next_row(
                    source_frame=frame,
                    next_source_position=source_position + 1,
                    recorded_next_row=recorded_next_physical,
                )
                next_source_scaled = self.scaler.transform_frame(pd.DataFrame([next_source_physical])).iloc[0]
                current_row = self.state_updater.advance(
                    current_scaled=current_row,
                    recorded_next_scaled=next_source_scaled,
                    prediction_physical=predicted_next,
                    action=action,
                )

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
            metrics=_rollout_metrics(predictions, errors),
            exogenous_provider_mode=self.exogenous_provider.mode,
            trace=pd.DataFrame(trace_rows),
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


def _rollout_metrics(predictions: pd.DataFrame, errors: pd.DataFrame) -> dict[str, float]:
    values = errors.to_numpy(dtype=float)
    prediction_values = predictions.to_numpy(dtype=float)
    abs_values = np.abs(values)
    final_abs = abs_values[-1] if len(abs_values) else np.asarray([], dtype=float)
    step_mae = np.nanmean(abs_values, axis=1) if len(abs_values) else np.asarray([])
    nan_inf_count = int(np.size(prediction_values) - np.isfinite(prediction_values).sum())
    total_prediction_values = int(np.size(prediction_values))
    metrics = {
        "trajectory_mae": _finite_mean(abs_values.reshape(-1)),
        "trajectory_rmse": _finite_rmse(values.reshape(-1)),
        "final_step_mae": _finite_mean(final_abs),
        "final_step_rmse": _finite_rmse(values[-1] if len(values) else []),
        "drift_slope_mae": _drift_slope(step_mae),
        "physical_violation_rate": _physical_violation_rate(predictions),
        "nan_inf_count": float(nan_inf_count),
        "nan_inf_rate": (
            float(nan_inf_count / total_prediction_values)
            if total_prediction_values
            else 0.0
        ),
    }
    truth = predictions.subtract(errors)
    target_r2_values = []
    for column in errors.columns:
        col_values = errors[column].to_numpy(dtype=float)
        true_values = truth[column].to_numpy(dtype=float)
        finite_pair = np.isfinite(col_values) & np.isfinite(true_values)
        if finite_pair.sum() >= 2:
            denominator = float(np.sum((true_values[finite_pair] - np.mean(true_values[finite_pair])) ** 2))
            r2 = (float(1.0 - np.sum(col_values[finite_pair] ** 2) / denominator)
                  if denominator > 0.0 else float("nan"))
        else:
            r2 = float("nan")
        metrics[f"{column}_trajectory_r2"] = r2
        if np.isfinite(r2):
            target_r2_values.append(r2)
        metrics[f"{column}_trajectory_mae"] = _finite_mean(np.abs(col_values))
        metrics[f"{column}_trajectory_rmse"] = _finite_rmse(col_values)
        metrics[f"{column}_final_abs_error"] = (
            float(abs(col_values[-1])) if len(col_values) and np.isfinite(col_values[-1]) else float("nan")
        )
    metrics["trajectory_r2"] = _finite_mean(target_r2_values)
    return metrics


def _physical_violation_rate(predictions: pd.DataFrame) -> float:
    if predictions.empty:
        return 0.0
    checks = []
    for column in predictions.columns:
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        violation = ~finite
        name = str(column).lower()
        if "humidity" in name or "hum" in name:
            violation |= finite & ((values < 0.0) | (values > 100.0))
        elif "rain_flag" in name:
            violation |= finite & ((values < 0.0) | (values > 1.0))
        elif "temp" in name or "dewpoint" in name:
            violation |= finite & ((values < -60.0) | (values > 80.0))
        elif "co2" in name:
            violation |= finite & ((values < 0.0) | (values > 10000.0))
        elif "vpd" in name:
            violation |= finite & (values < 0.0)
        checks.append(violation)
    if not checks:
        return 0.0
    stacked = np.vstack(checks)
    return float(np.mean(stacked))


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
    "EXOGENOUS_WEATHER_COLUMNS",
    "ExogenousProvider",
    "ForecastWeatherProvider",
    "LoggedActionProvider",
    "PolicyActionProvider",
    "RecordedWeatherProvider",
    "RolloutContext",
    "RolloutStepContext",
    "TransitionRolloutResult",
    "TransitionRolloutSimulator",
]
