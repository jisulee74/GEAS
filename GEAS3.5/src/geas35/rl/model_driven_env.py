"""Support-aware ExtraTrees model-driven environment for GEAS RL Step 16."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from geas35.features import add_mdp_context_features, add_mdp_derived_features
from geas35.models.transition.inference import load_transition_model_from_artifact_dir
from geas35.models.transition.rollout import (
    ExogenousProvider,
    ForecastWeatherProvider,
    RecordedWeatherProvider,
    RolloutContext,
)
from geas35.preprocessing import TIME_COLUMN
from geas35.rl.mdp_v1 import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_BINARY_ACTION_COLUMNS,
    MDP_V1_CONTINUOUS_ACTION_COLUMNS,
    MDP_V1_STEP_MINUTES,
    MdpV1Config,
    MdpV1ObservationScaler,
    compute_mdp_v1_reward,
    normalize_mdp_v1_action,
    project_mdp_v1_action_constraints,
)
from geas35.rl.support_v1 import StateActionSupportModel

STEP15_VERSION = "geas35.rl.step15.v1"
STEP16_ENV_VERSION = "geas35.rl.model_driven_env.step16.v1"
_EXOGENOUS_DERIVED_COLUMNS = (
    "obs_derived_dli_mol_m2",
    "obs_derived_light_sum",
    "obs_derived_clear_sky_sum",
    "obs_derived_light_sum_ratio",
    "obs_derived_light_eta_minutes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve_record(root: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"])).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if _sha256(path) != record.get("sha256"):
        raise ValueError(f"Frozen artifact hash mismatch: {path}")
    return path


def _load_scaler(path: Path) -> MdpV1ObservationScaler:
    payload = _read_json(path)
    if payload.get("fit_split") != "train":
        raise ValueError("Model-driven environment requires the frozen Train scaler.")
    return MdpV1ObservationScaler(
        columns=tuple(payload["columns"]),
        means=payload["mean"],
        scales=payload["scale"],
    )


def _episode_registry(frame: pd.DataFrame) -> list[dict[str, Any]]:
    timestamps = pd.to_datetime(frame[TIME_COLUMN], errors="coerce")
    valid = frame["rl_valid_transition"].astype(bool).to_numpy()
    episodes: list[dict[str, Any]] = []
    start = 0
    for position in range(1, len(frame)):
        previous, current = position - 1, position
        boundary = (
            not valid[previous]
            or pd.isna(timestamps.iloc[previous])
            or pd.isna(timestamps.iloc[current])
            or (timestamps.iloc[current] - timestamps.iloc[previous]).total_seconds() / 60.0
            != MDP_V1_STEP_MINUTES
            or timestamps.iloc[current].date() != timestamps.iloc[previous].date()
        )
        for column in ("crop", "series_id", "segment_id", "episode_id", "mdp_v1_rollout_id"):
            if column in frame.columns and frame.iloc[current][column] != frame.iloc[previous][column]:
                boundary = True
        if boundary:
            if position - start >= 2:
                episodes.append(_episode_record(frame, start, position - 1))
            start = position
    if len(frame) - start >= 2:
        episodes.append(_episode_record(frame, start, len(frame) - 1))
    if not episodes:
        raise ValueError("No model-driven episode has at least two contiguous rows.")
    return episodes


def _episode_record(frame: pd.DataFrame, start: int, final_observation: int) -> dict[str, Any]:
    row = frame.iloc[start]
    return {
        "start": start,
        "final_observation": final_observation,
        "scheduled_valid_steps": final_observation - start,
        "episode_id": str(row.get("episode_id", "")),
        "series_id": str(row.get("series_id", "")),
        "segment_id": str(row.get("segment_id", "")),
        "rollout_id": str(row.get("mdp_v1_rollout_id", "")),
        "date": str(pd.to_datetime(row[TIME_COLUMN]).date()),
    }


def _action_mapping(action: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(action, Mapping):
        values = {column: float(action.get(column, 0.0)) for column in MDP_V1_ACTION_COLUMNS}
    else:
        if len(action) != len(MDP_V1_ACTION_COLUMNS):
            raise ValueError(f"Action must contain {len(MDP_V1_ACTION_COLUMNS)} values.")
        values = dict(zip(MDP_V1_ACTION_COLUMNS, map(float, action)))
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Action values must be finite.")
    return values


def _restore_context_codes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    solar_columns = {
        "day": "obs_solar_period_day",
        "early_night": "obs_solar_period_early_night",
        "late_night": "obs_solar_period_late_night",
    }
    daylight_columns = {
        "sunny": "obs_daylight_condition_sunny",
        "partly_cloudy": "obs_daylight_condition_partly_cloudy",
        "cloudy": "obs_daylight_condition_cloudy",
        "unknown": "obs_daylight_condition_unknown",
    }
    out["solar_period"] = [
        max(solar_columns, key=lambda key: float(row.get(solar_columns[key], 0.0)))
        for _, row in out.iterrows()
    ]
    out["daylight_condition"] = [
        max(daylight_columns, key=lambda key: float(row.get(daylight_columns[key], 0.0)))
        for _, row in out.iterrows()
    ]
    return out


@dataclass(frozen=True)
class ModelDrivenEnvConfig:
    warning_penalty_max: float = 1.0
    worst_step_reward: float = -1.0
    repeated_severe_limit: int = 3
    max_episode_steps: int | None = None

    def __post_init__(self) -> None:
        if self.repeated_severe_limit < 1:
            raise ValueError("repeated_severe_limit must be positive.")
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive when configured.")


class GeasModelDrivenEnv:
    """Gymnasium-signature, dependency-light GEAS model-driven environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        crop: str,
        split: str,
        frame: pd.DataFrame,
        transition_bundle: Any,
        observation_scaler: MdpV1ObservationScaler,
        support_model: StateActionSupportModel,
        exogenous_provider: ExogenousProvider | None = None,
        mdp_config: MdpV1Config | None = None,
        env_config: ModelDrivenEnvConfig | None = None,
    ) -> None:
        self.crop = crop
        self.split = split
        self.frame = frame.reset_index(drop=True).copy()
        self.transition_bundle = transition_bundle
        self.model = transition_bundle.model
        self.scaler = observation_scaler
        self.support = support_model
        self.exogenous_provider = exogenous_provider or RecordedWeatherProvider()
        self.mdp_config = mdp_config or MdpV1Config()
        self.env_config = env_config or ModelDrivenEnvConfig()
        self.observation_columns = tuple(self.support.config["state_columns"])
        self.action_columns = tuple(MDP_V1_ACTION_COLUMNS)
        self.observation_shape = (len(self.observation_columns),)
        self.action_shape = (len(self.action_columns),)
        self.observation_dtype = np.dtype("float32")
        self.action_dtype = np.dtype("float32")
        self.episodes = _episode_registry(self.frame)
        self._rng = np.random.default_rng(0)
        self._done = True
        self._current_scaled = pd.Series(dtype=float)
        self._current_physical = pd.Series(dtype=float)
        self._physical_history = pd.DataFrame()
        self._last_action: dict[str, float] | None = None
        self._last_last_action: dict[str, float] | None = None
        self._consecutive_severe = 0

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})
        episode_index = int(options.get("episode_index", self._rng.integers(len(self.episodes))))
        if not 0 <= episode_index < len(self.episodes):
            raise ValueError("episode_index is outside the episode registry.")
        self._episode_index = episode_index
        self._episode = self.episodes[episode_index]
        self._position = int(self._episode["start"])
        self._executed_steps = 0
        self._done = False
        self._consecutive_severe = 0
        self._current_scaled = self.frame.iloc[self._position].copy()
        self._current_physical = self.scaler.inverse_transform_frame(
            pd.DataFrame([self._current_scaled])
        ).iloc[0]
        self._physical_history = pd.DataFrame([self._current_physical])
        logged = {
            column: float(self._current_scaled[column]) for column in self.action_columns
        }
        self._last_action = logged
        self._last_last_action = dict(logged)
        self.exogenous_provider.reset(RolloutContext(
            frame=self.frame,
            action_columns=self.action_columns,
            target_columns=tuple(self.model.target_columns_ or ()),
        ))
        observation = self._observation()
        return observation, self._base_info("reset")

    def _observation(self) -> np.ndarray:
        values = self._current_scaled.loc[list(self.observation_columns)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError("Model-driven observation contains NaN or Inf.")
        return values.astype(self.observation_dtype, copy=False)

    def _base_info(self, event: str) -> dict[str, Any]:
        return {
            "environment_version": STEP16_ENV_VERSION,
            "event": event,
            "crop": self.crop,
            "split": self.split,
            "episode_index": self._episode_index,
            "episode_id": self._episode["episode_id"],
            "series_id": self._episode["series_id"],
            "segment_id": self._episode["segment_id"],
            "rollout_id": self._episode["rollout_id"],
            "episode_date": self._episode["date"],
            "scheduled_valid_steps": self._episode["scheduled_valid_steps"],
            "executed_steps": self._executed_steps,
            "exogenous_provider_mode": self.exogenous_provider.mode,
        }

    def _support_frames(self, action: Mapping[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
        state = pd.DataFrame([self._current_scaled.loc[list(self.observation_columns)]])
        action_frame = pd.DataFrame([action], columns=self.action_columns)
        return state, action_frame

    def _project_local_support(self, action: Mapping[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
        state, _ = self._support_frames(action)
        local = self.support.local_action_support(state)[0]
        projected = dict(action)
        changed: dict[str, Any] = {}
        for column in MDP_V1_CONTINUOUS_ACTION_COLUMNS:
            lower, upper = local["continuous"][column]
            value = float(np.clip(projected[column], lower, upper))
            if value != projected[column]:
                changed[column] = {"from": projected[column], "to": value, "bounds": [lower, upper]}
            projected[column] = value
        for column in MDP_V1_BINARY_ACTION_COLUMNS:
            allowed = local["binary"][column]
            if projected[column] not in allowed:
                if allowed:
                    value = float(min(allowed, key=lambda candidate: abs(candidate - projected[column])))
                    changed[column] = {"from": projected[column], "to": value, "allowed": allowed}
                    projected[column] = value
                else:
                    changed[column] = {"from": projected[column], "to": None, "allowed": []}
        return projected, {"local_support": local, "changes": changed}

    def _safe_fallback_action(self) -> dict[str, float]:
        row = self._current_physical
        temp = float(row["obs_indoor_temp_c"])
        humidity = float(row["obs_indoor_humidity_pct"])
        vpd = float(row["obs_current_vpd_kpa"])
        low = float(row["obs_current_target_temp_min_c"])
        high = float(row["obs_current_target_temp_max_c"])
        daytime = float(row.get("obs_is_daytime", 0.0)) > 0.5
        fallback = {
            "vent_pct": 0.2 if (temp > high or humidity > self.mdp_config.rh_max_pct) else 0.0,
            "shade_curtain_pct": 1.0 if daytime and temp > high else 0.0,
            "thermal_curtain_pct": 1.0 if not daytime and temp < low else 0.0,
            "heat_run": float(temp < low),
            "cool_run": float(temp > high),
            "fan_run": float(humidity > self.mdp_config.rh_max_pct or vpd < self.mdp_config.vpd_min_kpa),
        }
        projected, _ = self._project_local_support(fallback)
        for column in MDP_V1_BINARY_ACTION_COLUMNS:
            if projected[column] is None:
                projected[column] = 0.0
        return projected

    def _warning_penalty(self, joint_score: float) -> float:
        thresholds = self.support.config["thresholds"]
        warning = float(thresholds["joint_warning"])
        severe = float(thresholds["joint_severe"])
        if joint_score <= warning:
            return 0.0
        denominator = max(severe - warning, np.finfo(float).eps)
        return float(np.clip((joint_score - warning) / denominator, 0.0, 1.0)
                     * self.env_config.warning_penalty_max)

    def _remaining_steps(self) -> int:
        return int(self._episode["scheduled_valid_steps"] - self._executed_steps)

    def step(
        self, action: Mapping[str, float] | Sequence[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")
        raw_action = _action_mapping(action)
        proposed_action = normalize_mdp_v1_action(raw_action)
        state_frame, action_frame = self._support_frames(proposed_action)
        proposed_scores = self.support.scores(state_frame, action_frame)
        state_score = float(proposed_scores["state"][0])
        joint_score = float(proposed_scores["joint"][0])
        thresholds = self.support.config["thresholds"]
        state_severe = state_score > float(thresholds["state_severe"])
        joint_warning = joint_score > float(thresholds["joint_warning"])
        joint_severe = joint_score > float(thresholds["joint_severe"])
        info = self._base_info("step")
        info.update({
            "raw_action": raw_action,
            "proposed_action": proposed_action,
            "state_ood_score": state_score,
            "joint_ood_score": joint_score,
            "support_thresholds": thresholds,
            "state_ood_level": "severe" if state_severe else (
                "warning" if state_score > float(thresholds["state_warning"]) else "in_support"
            ),
            "joint_ood_level": "severe" if joint_severe else (
                "warning" if joint_warning else "in_support"
            ),
        })
        if state_severe:
            reward = self.env_config.worst_step_reward * self._remaining_steps()
            self._done = True
            info.update({
                "termination_reason": "severe_state_ood",
                "truncation_reason": None,
                "pessimistic_terminal_cost": reward,
                "remaining_scheduled_steps_charged": self._remaining_steps(),
                "executed_action": None,
                "safety_fallback_applied": False,
            })
            return self._observation(), float(reward), True, False, info

        support_projection = {"changes": {}}
        if joint_warning:
            proposed_action, support_projection = self._project_local_support(proposed_action)
        if joint_severe:
            executed_action = self._safe_fallback_action()
            self._consecutive_severe += 1
            fallback_applied = True
        else:
            executed_action = proposed_action
            self._consecutive_severe = 0
            fallback_applied = False
        executed_action, weather_info = project_mdp_v1_action_constraints(
            executed_action, self._current_physical, self.mdp_config
        )
        _, executed_action_frame = self._support_frames(executed_action)
        executed_scores = self.support.scores(state_frame, executed_action_frame)

        current_input = self._current_scaled.copy()
        for column, value in executed_action.items():
            current_input[column] = value
        input_columns, targets = self.model._require_fitted()
        model_frame = pd.DataFrame([
            current_input.reindex(input_columns).to_dict()
        ], columns=input_columns).apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(model_frame.to_numpy(dtype=float)).all():
            raise RuntimeError("Transition input contains NaN or Inf.")
        prediction = self.model.predict(model_frame)
        if tuple(prediction.target_columns) != tuple(targets):
            raise RuntimeError("Transition prediction schema differs from fitted targets.")
        # Transition inputs are scaled observations, while official target outputs
        # remain in physical temperature/humidity/CO2 units.
        predicted_physical = prediction.next_observation.iloc[0].copy()

        next_position = self._position + 1
        recorded_next_scaled = self.frame.iloc[next_position].copy()
        recorded_next_physical = self.scaler.inverse_transform_frame(
            pd.DataFrame([recorded_next_scaled])
        ).iloc[0]
        next_physical = self.exogenous_provider.next_row(
            source_frame=self.frame,
            next_source_position=next_position,
            recorded_next_row=recorded_next_physical,
        )
        for column in targets:
            next_physical[column] = float(predicted_physical[column])
        for column, value in executed_action.items():
            next_physical[f"obs_prev_{column}"] = value
        next_physical["_mdp_v1_logged_vent_pct"] = executed_action["vent_pct"]
        history = pd.concat([
            self._physical_history,
            pd.DataFrame([next_physical]),
        ], ignore_index=True)
        history = self._recompute_history(history)
        next_physical = history.iloc[-1].copy()
        next_scaled = self.scaler.transform_frame(pd.DataFrame([next_physical])).iloc[0]
        observation_values = next_scaled.loc[list(self.observation_columns)].to_numpy(dtype=float)
        if not np.isfinite(observation_values).all():
            raise RuntimeError("Predicted next observation contains NaN or Inf.")

        base_reward, reward_terms = compute_mdp_v1_reward(
            next_physical,
            executed_action,
            prev_action=self._last_action,
            prev_prev_action=self._last_last_action,
            config=self.mdp_config,
        )
        warning_penalty = self._warning_penalty(joint_score)
        reward = max(self.env_config.worst_step_reward, base_reward - warning_penalty)
        if joint_severe:
            reward = self.env_config.worst_step_reward

        self._position = next_position
        self._executed_steps += 1
        self._physical_history = history
        self._current_physical = next_physical
        self._current_scaled = next_scaled
        self._last_last_action = self._last_action
        self._last_action = dict(executed_action)
        natural_end = next_position >= int(self._episode["final_observation"])
        repeated_severe = self._consecutive_severe >= self.env_config.repeated_severe_limit
        terminated = natural_end or repeated_severe
        time_limit = (
            not terminated
            and self.env_config.max_episode_steps is not None
            and self._executed_steps >= self.env_config.max_episode_steps
        )
        pessimistic_terminal_cost = 0.0
        if repeated_severe and not natural_end:
            future_steps = self._remaining_steps()
            pessimistic_terminal_cost = self.env_config.worst_step_reward * future_steps
            reward += pessimistic_terminal_cost
        self._done = terminated or time_limit
        info.update({
            "executed_action": executed_action,
            "support_projection": support_projection,
            "weather_constraint": weather_info,
            "safety_fallback_applied": fallback_applied,
            "executed_joint_ood_score": float(executed_scores["joint"][0]),
            "warning_ood_penalty": warning_penalty,
            "base_reward": float(base_reward),
            "reward_terms": reward_terms,
            "transition_prediction_physical": {
                column: float(predicted_physical[column]) for column in targets
            },
            "next_timestamp": str(next_physical[TIME_COLUMN]),
            "termination_reason": (
                "episode_boundary" if natural_end else
                "repeated_severe_action_ood" if repeated_severe else None
            ),
            "truncation_reason": "training_time_limit" if time_limit else None,
            "pessimistic_terminal_cost": pessimistic_terminal_cost,
            "remaining_scheduled_steps_charged": (
                self._remaining_steps() if repeated_severe and not natural_end else 0
            ),
            "executed_steps": self._executed_steps,
        })
        return self._observation(), float(reward), terminated, time_limit, info

    def _recompute_history(self, history: pd.DataFrame) -> pd.DataFrame:
        history = _restore_context_codes(history)
        timestamps = pd.to_datetime(history[TIME_COLUMN], errors="coerce")
        hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
        history["obs_hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
        history["obs_hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
        light = pd.to_numeric(history["obs_outdoor_light"], errors="coerce")
        history["obs_is_daytime"] = (light >= self.mdp_config.daylight_light_threshold).astype(float)
        history = add_mdp_context_features(
            history, time_col=TIME_COLUMN, lat=self.mdp_config.latitude,
            lon=self.mdp_config.longitude, tz=self.mdp_config.timezone,
            early_night_hours=self.mdp_config.early_night_hours, prefix="obs",
        )
        provider_values = history.loc[:, [
            column for column in _EXOGENOUS_DERIVED_COLUMNS if column in history.columns
        ]].copy()
        history = add_mdp_derived_features(
            history, time_col=TIME_COLUMN,
            condensation_margin_min_c=self.mdp_config.condensation_margin_min_c,
            rh_high_pct=self.mdp_config.rh_max_pct,
            ramp_limit_pct=self.mdp_config.ramp_limit_pct,
            latitude=self.mdp_config.latitude, longitude=self.mdp_config.longitude,
            timezone=self.mdp_config.timezone,
        )
        for column in provider_values.columns:
            history[column] = provider_values[column]
        history["obs_current_vpd_kpa"] = history["obs_derived_vpd_kpa"]
        history["obs_current_dewpoint_c"] = history["obs_derived_dewpoint_c"]
        history["obs_current_condensation_margin_c"] = history[
            "obs_derived_condensation_margin_c"
        ]
        return history


def load_model_driven_env_from_step15(
    *,
    project_root: str | Path,
    crop: str,
    split: str,
    protocol_manifest_path: str | Path | None = None,
    test_authorization_path: str | Path | None = None,
    exogenous_provider: ExogenousProvider | None = None,
    env_config: ModelDrivenEnvConfig | None = None,
) -> GeasModelDrivenEnv:
    root = Path(project_root).resolve()
    protocol_path = Path(protocol_manifest_path).resolve() if protocol_manifest_path else (
        root / "experiments/rl_policy_training/artifacts/step15/rl_protocol_manifest.json"
    )
    protocol = _read_json(protocol_path)
    if protocol.get("schema_version") != STEP15_VERSION or protocol.get("status") != "passed":
        raise ValueError("Step 16 requires a passed frozen Step 15 protocol.")
    if crop not in protocol.get("crops", {}):
        raise ValueError(f"Unsupported crop: {crop}")
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test.")
    split_policy_path = _resolve_record(root, protocol["split_access_policy"])
    split_policy = _read_json(split_policy_path)
    dataset_record = split_policy["datasets"][crop][split]
    if split == "test":
        if test_authorization_path is None:
            raise ValueError("Test environment is locked without final-test authorization.")
        from geas35.experiments.rl.step15_contract import authorize_test_access
        dataset_path = authorize_test_access(protocol_path, test_authorization_path)
        expected = Path(dataset_record["path"]).resolve()
        if dataset_path != expected:
            raise ValueError("Test authorization crop/path differs from requested environment.")
    else:
        dataset_path = Path(dataset_record["path"]).resolve()
    if _sha256(dataset_path) != dataset_record["sha256"]:
        raise ValueError("RL dataset hash differs from the frozen Step 15 split policy.")
    crop_record = protocol["crops"][crop]
    model_path = _resolve_record(root, crop_record["candidate_model"])
    feature_schema_path = _resolve_record(root, crop_record["feature_schema"])
    scaler_path = _resolve_record(root, crop_record["observation_scaler"])
    support_path = _resolve_record(root, crop_record["state_action_support"])
    _resolve_record(root, crop_record["support_reference"])
    feature_schema = _read_json(feature_schema_path)
    bundle = load_transition_model_from_artifact_dir(model_path.parent)
    if tuple(bundle.model.input_columns_ or ()) != tuple(feature_schema["input_columns"]):
        raise ValueError("ExtraTrees input schema differs from the frozen feature schema.")
    frame = pd.read_parquet(dataset_path)
    return GeasModelDrivenEnv(
        crop=crop, split=split, frame=frame,
        transition_bundle=bundle, observation_scaler=_load_scaler(scaler_path),
        support_model=StateActionSupportModel(support_path),
        exogenous_provider=exogenous_provider,
        env_config=env_config,
    )


__all__ = [
    "ForecastWeatherProvider", "GeasModelDrivenEnv", "ModelDrivenEnvConfig",
    "RecordedWeatherProvider", "STEP16_ENV_VERSION",
    "load_model_driven_env_from_step15",
]
