from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.rl import MdpV1ObservationScaler
from geas35.rl.model_driven_env import load_model_driven_env_from_step15


@pytest.fixture(scope="module")
def melon_env():
    return load_model_driven_env_from_step15(
        project_root=PROJECT_ROOT, crop="melon", split="train"
    )


def _in_support_episode(env) -> int:
    thresholds = env.support.config["thresholds"]
    for index in range(len(env.episodes)):
        env.reset(seed=42, options={"episode_index": index})
        action = {
            column: float(env._current_scaled[column])
            for column in env.action_columns
        }
        state, action_frame = env._support_frames(action)
        scores = env.support.scores(state, action_frame)
        if (
            scores["state"][0] <= thresholds["state_warning"]
            and scores["joint"][0] <= thresholds["joint_warning"]
        ):
            return index
    raise AssertionError("No in-support Melon episode was found.")


def test_observation_scaler_inverse_round_trip() -> None:
    scaler = MdpV1ObservationScaler(
        columns=("a", "b"), means={"a": 10.0, "b": -2.0},
        scales={"a": 2.0, "b": 4.0},
    )
    physical = pd.DataFrame({"a": [12.0, 8.0], "b": [2.0, -6.0], "id": [1, 2]})
    scaled = scaler.transform_frame(physical)
    restored = scaler.inverse_transform_frame(scaled)
    np.testing.assert_allclose(restored[["a", "b"]], physical[["a", "b"]])
    assert restored["id"].tolist() == [1, 2]


def test_test_split_is_locked_without_final_test_authorization() -> None:
    with pytest.raises(ValueError, match="locked"):
        load_model_driven_env_from_step15(
            project_root=PROJECT_ROOT, crop="melon", split="test"
        )


def test_model_step_has_prediction_parity_and_deterministic_replay(melon_env) -> None:
    episode_index = _in_support_episode(melon_env)
    observation, reset_info = melon_env.reset(
        seed=42, options={"episode_index": episode_index}
    )
    assert observation.shape == melon_env.observation_shape == (59,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert reset_info["scheduled_valid_steps"] > 0

    action = {
        column: float(melon_env._current_scaled[column])
        for column in melon_env.action_columns
    }
    before = melon_env._current_scaled.copy()
    next_observation, reward, terminated, truncated, info = melon_env.step(action)
    model_input = before.copy()
    for column, value in info["executed_action"].items():
        model_input[column] = value
    expected = melon_env.model.predict(model_input.to_frame().T).next_observation.iloc[0]
    assert set(info["transition_prediction_physical"]) == set(
        melon_env.model.target_columns_
    )
    for column in melon_env.model.target_columns_:
        assert info["transition_prediction_physical"][column] == pytest.approx(
            expected[column], abs=1e-12
        )
    assert np.isfinite(next_observation).all()
    assert np.isfinite(reward)
    assert truncated is False
    assert info["executed_steps"] == 1
    assert info["episode_id"] == reset_info["episode_id"]

    melon_env.reset(seed=42, options={"episode_index": episode_index})
    replay = melon_env.step(action)
    np.testing.assert_array_equal(replay[0], next_observation)
    assert replay[1] == pytest.approx(reward, abs=1e-12)
    assert replay[2] is terminated


def test_severe_state_ood_terminates_with_scheduled_step_cost(melon_env) -> None:
    episode_index = _in_support_episode(melon_env)
    _, reset_info = melon_env.reset(seed=42, options={"episode_index": episode_index})
    continuous = melon_env.support.config["continuous_state_columns"][0]
    melon_env._current_scaled[continuous] = 1_000_000.0
    action = {
        column: float(melon_env._current_scaled[column])
        for column in melon_env.action_columns
    }
    observation, reward, terminated, truncated, info = melon_env.step(action)
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "severe_state_ood"
    assert info["executed_action"] is None
    assert info["remaining_scheduled_steps_charged"] == reset_info["scheduled_valid_steps"]
    assert reward == pytest.approx(
        melon_env.env_config.worst_step_reward * reset_info["scheduled_valid_steps"]
    )
    assert np.isfinite(observation).all()


def test_episode_registry_and_official_step16_artifacts() -> None:
    root = PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step16"
    integrity = json.loads((root / "step16_integrity_manifest.json").read_text())
    manifest = json.loads((root / "step16_environment_manifest.json").read_text())
    trace = json.loads((root / "step16_trajectory_trace.json").read_text())
    assert integrity["status"] == manifest["status"] == trace["status"] == "passed"
    assert integrity["checks"]["step17_implemented"] is False
    assert set(manifest["crops"]) == {"strawberry", "melon", "cucumber"}
    assert manifest["contracts"]["step_minutes"] == 5
    assert manifest["contracts"]["episode_crosses_date_or_series_boundary"] is False
    for crop, record in manifest["crops"].items():
        assert record["candidate_name"] == "extra_trees"
        assert record["observation_shape"] == [59]
        assert record["action_shape"] == [6]
        assert record["one_step_inference_parity"] is True
        assert record["deterministic_replay"] is True
        assert set(trace["crops"][crop]["prediction_physical"]) == {
            "obs_indoor_temp_c", "obs_indoor_humidity_pct", "obs_indoor_co2_ppm"
        }
