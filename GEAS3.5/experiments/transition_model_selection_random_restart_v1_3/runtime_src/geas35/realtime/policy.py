"""Deterministic PPO policy and support-aware online safety helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from geas35.rl import (
    MDP_V1_ACTION_COLUMNS, MDP_V1_BINARY_ACTION_COLUMNS,
    MDP_V1_CONTINUOUS_ACTION_COLUMNS, MdpV1Config,
)
from geas35.rl.hybrid_ppo import FeasibleActionSpec, load_ppo_checkpoint


def support_level(support, score: float, *, kind: str) -> str:
    if kind not in {"state", "joint"}:
        raise ValueError("kind must be state or joint")
    thresholds = support.config["thresholds"]
    if score > float(thresholds[f"{kind}_severe"]): return "severe"
    if score > float(thresholds[f"{kind}_warning"]): return "warning"
    return "in_support"


def state_support_score(support, state: pd.DataFrame) -> float:
    raw = state.loc[:, support.config["state_columns"]].to_numpy(dtype=float)
    distances = support._state_nn.kneighbors(raw * support.state_weights, return_distance=True)[0]
    return float(distances[0, -1])


def build_feasible_action_spec(
    support, scaled_observation: pd.Series, physical_observation: pd.Series,
    *, config: MdpV1Config | None = None,
    fallback_action: Mapping[str, float] | None = None,
) -> FeasibleActionSpec:
    config = config or MdpV1Config()
    local = support.local_action_support(scaled_observation.to_frame().T)[0]
    fallback = dict(fallback_action or {column: 0.0 for column in MDP_V1_ACTION_COLUMNS})
    low, high = [], []
    for column in MDP_V1_CONTINUOUS_ACTION_COLUMNS:
        lower, upper = local["continuous"][column]
        low.append(float(lower)); high.append(float(upper))
    rain = float(physical_observation.get("obs_rain_flag", 0.0)) > 0.0
    wind = float(physical_observation.get("obs_outdoor_wind_speed", 0.0))
    cap = 1.0
    if rain: cap = min(cap, config.rain_vent_open_cap)
    if np.isfinite(wind) and wind > config.wind_cap_threshold:
        cap = min(cap, config.wind_vent_open_cap)
    high[0] = min(high[0], cap); low[0] = min(low[0], high[0])
    allow_zero, allow_one = [], []
    for column in MDP_V1_BINARY_ACTION_COLUMNS:
        allowed = {int(value) for value in local["binary"][column]}
        if not allowed: allowed = {int(float(fallback[column]) > 0.5)}
        allow_zero.append(0 in allowed); allow_one.append(1 in allowed)
    return FeasibleActionSpec(
        continuous_low=np.asarray(low, dtype=np.float32),
        continuous_high=np.asarray(high, dtype=np.float32),
        binary_allow_zero=np.asarray(allow_zero, dtype=bool),
        binary_allow_one=np.asarray(allow_one, dtype=bool),
    )


@dataclass(frozen=True)
class SafetyResult:
    action: dict[str, float]
    projected: bool
    reason: str | None = None


def apply_action_safety(action: Mapping[str, float], feasible: FeasibleActionSpec) -> SafetyResult:
    values = {column: float(action[column]) for column in MDP_V1_ACTION_COLUMNS}
    projected = False
    for index, column in enumerate(MDP_V1_CONTINUOUS_ACTION_COLUMNS):
        safe = float(np.clip(values[column], feasible.continuous_low[index], feasible.continuous_high[index]))
        projected |= not np.isclose(safe, values[column]); values[column] = safe
    for index, column in enumerate(MDP_V1_BINARY_ACTION_COLUMNS):
        proposed = int(values[column] >= 0.5)
        if proposed == 0 and not feasible.binary_allow_zero[index]: proposed = 1
        if proposed == 1 and not feasible.binary_allow_one[index]: proposed = 0
        projected |= float(proposed) != values[column]; values[column] = float(proposed)
    if values["heat_run"] > 0.5 and values["cool_run"] > 0.5:
        values["heat_run"] = 0.0; values["cool_run"] = 0.0
        return SafetyResult(values, projected=True, reason="heat_cool_conflict")
    return SafetyResult(values, projected=projected)


class PPOCheckpointPolicy:
    def __init__(self, checkpoint: str | Path, *, expected_artifact_hashes=None, device="cpu"):
        self.trainer, self.payload = load_ppo_checkpoint(
            checkpoint, expected_artifact_hashes=expected_artifact_hashes,
            device=device, restore_rng=False,
        )
        self.trainer.policy.eval()

    def action(self, observation: pd.Series, feasible: FeasibleActionSpec) -> dict[str, float]:
        array = observation.to_numpy(dtype=np.float32)
        if not np.isfinite(array).all(): raise ValueError("Policy observation contains NaN or Inf")
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.trainer.device).unsqueeze(0)
        with torch.no_grad():
            result = self.trainer.policy.act(
                tensor, feasible.batched(1, device=self.trainer.device), deterministic=True
            )["action"][0].cpu().numpy()
        return {column: float(value) for column, value in zip(MDP_V1_ACTION_COLUMNS, result)}


__all__ = [
    "PPOCheckpointPolicy", "SafetyResult", "apply_action_safety",
    "build_feasible_action_spec", "state_support_score", "support_level",
]
