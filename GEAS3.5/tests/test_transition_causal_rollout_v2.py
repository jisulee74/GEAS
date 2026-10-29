from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.models.transition.causal_state import CausalStateUpdater, build_rollout_windows
from geas35.models.transition.sklearn_models import PersistenceTransitionModel
from geas35.models.transition.base import TransitionDataset
from geas35.rl import MdpV1ObservationScaler


def _scaler() -> MdpV1ObservationScaler:
    columns = ("obs_indoor_temp_c", "obs_indoor_humidity_pct", "obs_indoor_co2_ppm",
               "obs_outdoor_temp_c", "obs_outdoor_light")
    return MdpV1ObservationScaler(columns=columns,
        means={column: 10.0 for column in columns},
        scales={column: 2.0 for column in columns})


def test_causal_update_rescales_physical_prediction_and_ignores_future_endogenous() -> None:
    scaler = _scaler()
    current_physical = pd.Series({
        "reg_date": pd.Timestamp("2024-01-01 00:00"), "crop": "cucumber",
        "episode_id": "e1", "segment_id": "s1", "obs_indoor_temp_c": 20.0,
        "obs_indoor_humidity_pct": 60.0, "obs_indoor_co2_ppm": 500.0,
        "obs_outdoor_temp_c": 12.0, "obs_outdoor_light": 0.0,
    })
    current = scaler.transform_frame(pd.DataFrame([current_physical])).iloc[0]
    recorded_a = current_physical.copy()
    recorded_a["reg_date"] = pd.Timestamp("2024-01-01 00:05")
    recorded_a["obs_indoor_temp_c"] = 70.0
    recorded_b = recorded_a.copy()
    recorded_b["obs_indoor_temp_c"] = -40.0
    updater_a, updater_b = CausalStateUpdater(scaler), CausalStateUpdater(scaler)
    updater_a.reset(current); updater_b.reset(current)
    kwargs = {"current_scaled": current, "prediction_physical": {
        "obs_indoor_temp_c": 24.0, "obs_indoor_humidity_pct": 65.0,
        "obs_indoor_co2_ppm": 520.0}, "action": {}}
    next_a = updater_a.advance(recorded_next_scaled=scaler.transform_frame(pd.DataFrame([recorded_a])).iloc[0], **kwargs)
    next_b = updater_b.advance(recorded_next_scaled=scaler.transform_frame(pd.DataFrame([recorded_b])).iloc[0], **kwargs)
    assert next_a["obs_indoor_temp_c"] == 7.0
    assert next_a["obs_indoor_temp_c"] == next_b["obs_indoor_temp_c"]


def test_persistence_returns_physical_targets_under_nonidentity_scaler() -> None:
    scaler = _scaler()
    targets = ("obs_indoor_temp_c", "obs_indoor_humidity_pct", "obs_indoor_co2_ppm")
    x = pd.DataFrame({column: [1.0] for column in targets})
    dataset = TransitionDataset(x=x, y=pd.DataFrame({column: [0.0] for column in targets}),
        input_columns=targets, target_columns=targets)
    prediction = PersistenceTransitionModel(observation_scaler=scaler).fit(dataset).predict(dataset)
    assert prediction.next_observation.iloc[0].to_dict() == {column: 12.0 for column in targets}


def test_window_bank_excludes_boundaries_and_is_deterministic() -> None:
    frame = pd.DataFrame({
        "reg_date": pd.date_range("2024-01-01", periods=30, freq="5min"),
        "crop": "cucumber", "episode_id": ["a"] * 15 + ["b"] * 15,
        "segment_id": ["s1"] * 15 + ["s2"] * 15,
    })
    first = build_rollout_windows(frame, horizon_steps=12, stride_steps=12,
                                  max_windows=32, episode_balanced=True)
    second = build_rollout_windows(frame, horizon_steps=12, stride_steps=12,
                                   max_windows=32, episode_balanced=True)
    assert first == second
    assert [window.start_position for window in first] == [0, 15]
    assert all(window.horizon_steps == 12 for window in first)
