"""Unit-safe causal state updates and deterministic rollout window banks."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
from geas35.features import MDP_DERIVED_OBSERVATION_COLUMNS, add_mdp_derived_features
from geas35.rl import MdpV1ObservationScaler

ROLLOUT_PROTOCOL_VERSION = "geas35.transition.causal_rollout.v2"
EXOGENOUS_COLUMNS = ("obs_outdoor_temp_c", "obs_outdoor_humidity_pct", "obs_outdoor_light", "obs_outdoor_wind_speed", "obs_rain_flag")
BOUNDARY_COLUMNS = ("crop", "series_id", "segment_id", "episode_id", "rollout_id")

def load_frozen_observation_scaler(path: str | Path) -> tuple[MdpV1ObservationScaler, str]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    columns = tuple(payload.get("columns", ()))
    means, scales = payload.get("mean"), payload.get("scale")
    if not columns or not isinstance(means, dict) or not isinstance(scales, dict):
        raise ValueError("Invalid frozen Train observation scaler artifact.")
    if set(columns) - set(means) or set(columns) - set(scales):
        raise ValueError("Scaler columns/mean/scale contract mismatch.")
    scaler = MdpV1ObservationScaler(columns=columns, means=means, scales=scales)
    return scaler, hashlib.sha256(source.read_bytes()).hexdigest()

@dataclass(frozen=True)
class RolloutWindow:
    crop: str
    episode_id: str
    start_position: int
    horizon_steps: int

@dataclass(frozen=True)
class RolloutEvaluationProtocol:
    horizon_steps: tuple[int, ...] = (3, 6, 12)
    step_minutes: int = 5
    stride_steps: int = 12
    max_hpo_windows: int = 32
    version: str = ROLLOUT_PROTOCOL_VERSION

@dataclass
class CausalStateUpdater:
    scaler: MdpV1ObservationScaler
    history_limit: int = 288
    _history: list[pd.Series] = field(default_factory=list, init=False)

    def reset(self, initial_scaled: pd.Series) -> None:
        self._history = [self._inverse(initial_scaled)]

    def advance(self, *, current_scaled: pd.Series, recorded_next_scaled: pd.Series,
                prediction_physical: Mapping[str, float], action: Mapping[str, float]) -> pd.Series:
        current = self._inverse(current_scaled)
        recorded = self._inverse(recorded_next_scaled)
        next_row = current.copy()
        for column in recorded.index:
            if self._recorded_column_allowed(column):
                next_row[column] = recorded[column]
        for column, value in prediction_physical.items():
            next_row[column] = float(value)
        for column, value in action.items():
            next_row[column] = float(value)
            previous = f"obs_prev_{column}"
            if previous in next_row.index:
                next_row[previous] = float(value)
        if "vent_pct" in action:
            next_row["_mdp_v1_logged_vent_pct"] = float(action["vent_pct"])
        history = [*self._history, next_row][-self.history_limit:]
        physical = pd.DataFrame(history)
        required = {"reg_date", "obs_indoor_temp_c", "obs_indoor_humidity_pct",
                    "obs_outdoor_temp_c", "obs_outdoor_light", "obs_current_target_temp_c"}
        if required.issubset(physical.columns):
            recomputed = add_mdp_derived_features(physical).iloc[-1]
            for column in MDP_DERIVED_OBSERVATION_COLUMNS:
                if column in next_row.index and column in recomputed.index:
                    next_row[column] = recomputed[column]
        self._history = history
        return self.scaler.transform_frame(pd.DataFrame([next_row])).iloc[0]

    def _inverse(self, row: pd.Series) -> pd.Series:
        return self.scaler.inverse_transform_frame(pd.DataFrame([row])).iloc[0]

    @property
    def current_physical(self) -> pd.Series:
        return self._history[-1].copy()

    @property
    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._history).reset_index(drop=True)

    @staticmethod
    def _recorded_column_allowed(column: str) -> bool:
        return (column in ("reg_date", "timestamp", *BOUNDARY_COLUMNS, *EXOGENOUS_COLUMNS)
                or column.startswith(("obs_hour_", "obs_is_daytime", "obs_growth_",
                    "obs_target_", "obs_current_target_", "obs_solar_period_", "obs_daylight_")))

def build_rollout_windows(frame: pd.DataFrame, *, horizon_steps: int = 12,
                          stride_steps: int = 12, max_windows: int | None = None,
                          episode_balanced: bool = False) -> tuple[RolloutWindow, ...]:
    if horizon_steps <= 0 or stride_steps <= 0:
        raise ValueError("Rollout horizon and stride must be positive.")
    time_column = "reg_date" if "reg_date" in frame else "timestamp" if "timestamp" in frame else None
    if time_column is None:
        raise ValueError("Rollout frame requires a timestamp column.")
    timestamps = pd.to_datetime(frame[time_column], errors="coerce")
    group_columns = [column for column in BOUNDARY_COLUMNS if column in frame]
    group_key = (frame[group_columns].astype(str).agg("|".join, axis=1)
                 if group_columns else pd.Series("all", index=frame.index))
    group_key = group_key + "|" + timestamps.dt.date.astype(str)
    by_group: list[list[RolloutWindow]] = []
    for key in group_key.drop_duplicates():
        positions = np.flatnonzero(group_key.to_numpy() == key)
        windows = []
        for offset in range(0, len(positions) - horizon_steps + 1, stride_steps):
            block = positions[offset:offset + horizon_steps]
            if not np.all(np.diff(block) == 1):
                continue
            times = timestamps.iloc[block]
            if times.isna().any() or not (times.diff().dropna() == pd.Timedelta(minutes=5)).all():
                continue
            row = frame.iloc[block[0]]
            windows.append(RolloutWindow(str(row.get("crop", "unknown")),
                str(row.get("episode_id", key)), int(block[0]), horizon_steps))
        if windows:
            by_group.append(windows)
    if not episode_balanced:
        result = [window for group in by_group for window in group]
        return tuple(result if max_windows is None else result[:max_windows])
    result: list[RolloutWindow] = []
    groups = [list(group) for group in by_group]
    while groups and (max_windows is None or len(result) < max_windows):
        remaining = []
        for group in groups:
            if group and (max_windows is None or len(result) < max_windows):
                result.append(group.pop(0))
            if group:
                remaining.append(group)
        groups = remaining
    return tuple(result)

__all__ = ["BOUNDARY_COLUMNS", "CausalStateUpdater", "EXOGENOUS_COLUMNS",
    "ROLLOUT_PROTOCOL_VERSION", "RolloutEvaluationProtocol", "RolloutWindow",
    "build_rollout_windows", "load_frozen_observation_scaler"]
