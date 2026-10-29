"""Train-only state/action support contracts for GEAS MDP v1."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from geas35.rl.mdp_v1 import (
    MDP_V1_BINARY_ACTION_COLUMNS,
    MDP_V1_CONTINUOUS_ACTION_COLUMNS,
)

SUPPORT_SCHEMA_VERSION = "geas35.rl.state_action_support.v1"


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Conformal scores must be non-empty and finite.")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one.")
    rank = min(values.size, math.ceil((values.size + 1) * coverage))
    return float(np.partition(values, rank - 1)[rank - 1])


def _weights(columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty(0, dtype=float)
    return np.full(len(columns), 1.0 / math.sqrt(len(columns)), dtype=float)


def _binary_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    result = []
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if np.isin(values, (0.0, 1.0)).all():
            result.append(column)
    return result


def _nearest(reference: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(reference) < k:
        raise ValueError(f"Support reference has {len(reference)} rows but k={k}.")
    model = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=1)
    model.fit(reference)
    return model.kneighbors(query, return_distance=True)


@dataclass(frozen=True)
class SupportBuildConfig:
    seed: int = 42
    calibration_episode_fraction: float = 0.20
    warning_coverage: float = 0.95
    severe_coverage: float = 0.99
    neighbors: int = 50
    max_reference_rows: int = 10000
    max_calibration_rows: int = 5000


def build_support_artifact(
    frame: pd.DataFrame,
    *,
    crop: str,
    state_columns: Sequence[str],
    action_columns: Sequence[str],
    episode_column: str,
    output_dir: str | Path,
    config: SupportBuildConfig = SupportBuildConfig(),
) -> tuple[Path, Path]:
    required = [*state_columns, *action_columns, episode_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Support frame is missing columns: " + ", ".join(missing))
    values = frame.loc[:, [*state_columns, *action_columns]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Support fitting requires finite Train state/action rows.")
    episodes = np.asarray(sorted(frame[episode_column].astype(str).unique()))
    if episodes.size < 2:
        raise ValueError("Episode-disjoint support calibration requires at least two episodes.")

    rng = np.random.default_rng(config.seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    calibration_count = min(
        len(shuffled) - 1,
        max(1, int(round(len(shuffled) * config.calibration_episode_fraction))),
    )
    calibration_episodes = set(shuffled[:calibration_count])
    calibration_mask = frame[episode_column].astype(str).isin(calibration_episodes).to_numpy()
    reference_positions = np.flatnonzero(~calibration_mask)
    calibration_positions = np.flatnonzero(calibration_mask)
    if len(reference_positions) > config.max_reference_rows:
        reference_positions = np.sort(rng.choice(
            reference_positions, config.max_reference_rows, replace=False
        ))
    if len(calibration_positions) > config.max_calibration_rows:
        calibration_positions = np.sort(rng.choice(
            calibration_positions, config.max_calibration_rows, replace=False
        ))

    state_columns = list(state_columns)
    action_columns = list(action_columns)
    binary_state = _binary_columns(frame.iloc[reference_positions], state_columns)
    continuous_state = [c for c in state_columns if c not in binary_state]
    continuous_actions = [c for c in MDP_V1_CONTINUOUS_ACTION_COLUMNS if c in action_columns]
    binary_actions = [c for c in MDP_V1_BINARY_ACTION_COLUMNS if c in action_columns]
    if set(action_columns) != set(continuous_actions + binary_actions):
        raise ValueError("Support action schema differs from the official MDP v1 actions.")

    state_weight_map = {
        **dict(zip(continuous_state, _weights(continuous_state))),
        **dict(zip(binary_state, _weights(binary_state))),
    }
    state_weights = np.asarray([state_weight_map[c] for c in state_columns])
    action_weight_map = {
        **dict(zip(continuous_actions, _weights(continuous_actions))),
        **dict(zip(binary_actions, _weights(binary_actions))),
    }
    action_weights = np.asarray([action_weight_map[c] for c in action_columns])

    all_state = frame.loc[:, state_columns].to_numpy(dtype=float)
    all_action = frame.loc[:, action_columns].to_numpy(dtype=float)
    reference_state_raw = all_state[reference_positions]
    reference_action_raw = all_action[reference_positions]
    calibration_state_raw = all_state[calibration_positions]
    calibration_action_raw = all_action[calibration_positions]
    reference_state = reference_state_raw * state_weights
    calibration_state = calibration_state_raw * state_weights
    reference_joint = np.concatenate([
        reference_state / math.sqrt(2.0),
        reference_action_raw * action_weights / math.sqrt(2.0),
    ], axis=1)
    calibration_joint = np.concatenate([
        calibration_state / math.sqrt(2.0),
        calibration_action_raw * action_weights / math.sqrt(2.0),
    ], axis=1)

    state_distances, state_neighbors = _nearest(
        reference_state, calibration_state, config.neighbors
    )
    joint_distances, _ = _nearest(reference_joint, calibration_joint, config.neighbors)
    state_scores = state_distances[:, -1]
    joint_scores = joint_distances[:, -1]
    thresholds = {
        "state_warning": conformal_quantile(state_scores, config.warning_coverage),
        "state_severe": conformal_quantile(state_scores, config.severe_coverage),
        "joint_warning": conformal_quantile(joint_scores, config.warning_coverage),
        "joint_severe": conformal_quantile(joint_scores, config.severe_coverage),
    }

    continuous_adjustments: dict[str, float] = {}
    alpha = 1.0 - config.warning_coverage
    for column in continuous_actions:
        index = action_columns.index(column)
        neighbor_values = reference_action_raw[state_neighbors, index]
        lower = np.quantile(neighbor_values, alpha / 2.0, axis=1)
        upper = np.quantile(neighbor_values, 1.0 - alpha / 2.0, axis=1)
        observed = calibration_action_raw[:, index]
        scores = np.maximum.reduce([lower - observed, observed - upper, np.zeros_like(observed)])
        continuous_adjustments[column] = conformal_quantile(
            scores, config.warning_coverage
        )
    minimum_binary_neighbor_count = max(
        1, math.ceil((1.0 - config.warning_coverage) * config.neighbors)
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / "support_reference.npz"
    np.savez_compressed(
        arrays_path,
        reference_state=reference_state_raw,
        reference_action=reference_action_raw,
    )
    artifact = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "crop": crop,
        "fit_split": "train_only",
        "episode_disjoint_calibration": True,
        "episode_column": episode_column,
        "seed": config.seed,
        "state_columns": state_columns,
        "action_columns": action_columns,
        "continuous_state_columns": continuous_state,
        "binary_state_columns": binary_state,
        "continuous_action_columns": continuous_actions,
        "binary_action_columns": binary_actions,
        "state_weights": state_weights.tolist(),
        "action_weights": action_weights.tolist(),
        "neighbors": config.neighbors,
        "reference_rows": int(len(reference_positions)),
        "calibration_rows": int(len(calibration_positions)),
        "reference_episode_count": int(len(episodes) - calibration_count),
        "calibration_episode_count": int(calibration_count),
        "warning_coverage": config.warning_coverage,
        "severe_coverage": config.severe_coverage,
        "thresholds": thresholds,
        "empirical_calibration_coverage": {
            "state_warning": float(np.mean(state_scores <= thresholds["state_warning"])),
            "state_severe": float(np.mean(state_scores <= thresholds["state_severe"])),
            "joint_warning": float(np.mean(joint_scores <= thresholds["joint_warning"])),
            "joint_severe": float(np.mean(joint_scores <= thresholds["joint_severe"])),
        },
        "local_action_support": {
            "continuous_quantile_alpha": alpha,
            "continuous_conformal_adjustments": continuous_adjustments,
            "binary_minimum_neighbor_count": minimum_binary_neighbor_count,
            "insufficient_local_evidence_policy": "rule_based_safe_fallback",
        },
        "arrays_file": arrays_path.name,
    }
    artifact_path = destination / "state_action_support.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    report_path = destination / "conformal_calibration_report.json"
    report_path.write_text(json.dumps({
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "crop": crop,
        "status": "passed",
        "fit_split": "train_only",
        "thresholds": thresholds,
        "empirical_coverage": artifact["empirical_calibration_coverage"],
        "reference_rows": artifact["reference_rows"],
        "calibration_rows": artifact["calibration_rows"],
        "validation_rows_used": 0,
        "test_rows_used": 0,
    }, indent=2) + "\n", encoding="utf-8")
    return artifact_path, report_path


class StateActionSupportModel:
    """Load and query a frozen Step 15 Train-only support artifact."""

    def __init__(self, artifact_path: str | Path):
        self.artifact_path = Path(artifact_path)
        self.config: Mapping[str, Any] = json.loads(self.artifact_path.read_text())
        if self.config.get("schema_version") != SUPPORT_SCHEMA_VERSION:
            raise ValueError("Unsupported state/action support schema.")
        arrays = np.load(self.artifact_path.parent / self.config["arrays_file"])
        self.reference_state_raw = arrays["reference_state"]
        self.reference_action_raw = arrays["reference_action"]
        self.state_weights = np.asarray(self.config["state_weights"], dtype=float)
        self.action_weights = np.asarray(self.config["action_weights"], dtype=float)
        self.k = int(self.config["neighbors"])
        self._state_reference = self.reference_state_raw * self.state_weights
        self._joint_reference = np.concatenate([
            self._state_reference / math.sqrt(2.0),
            self.reference_action_raw * self.action_weights / math.sqrt(2.0),
        ], axis=1)
        self._state_nn = NearestNeighbors(n_neighbors=self.k, n_jobs=1).fit(self._state_reference)
        self._joint_nn = NearestNeighbors(n_neighbors=self.k, n_jobs=1).fit(self._joint_reference)

    def scores(self, state: pd.DataFrame, action: pd.DataFrame) -> dict[str, np.ndarray]:
        state_raw = state.loc[:, self.config["state_columns"]].to_numpy(dtype=float)
        action_raw = action.loc[:, self.config["action_columns"]].to_numpy(dtype=float)
        weighted_state = state_raw * self.state_weights
        weighted_joint = np.concatenate([
            weighted_state / math.sqrt(2.0),
            action_raw * self.action_weights / math.sqrt(2.0),
        ], axis=1)
        state_distances = self._state_nn.kneighbors(weighted_state, return_distance=True)[0]
        joint_distances = self._joint_nn.kneighbors(weighted_joint, return_distance=True)[0]
        return {"state": state_distances[:, -1], "joint": joint_distances[:, -1]}

    def local_action_support(self, state: pd.DataFrame) -> list[dict[str, Any]]:
        state_raw = state.loc[:, self.config["state_columns"]].to_numpy(dtype=float)
        neighbors = self._state_nn.kneighbors(
            state_raw * self.state_weights, return_distance=False
        )
        spec = self.config["local_action_support"]
        alpha = float(spec["continuous_quantile_alpha"])
        results = []
        action_columns = list(self.config["action_columns"])
        for row_neighbors in neighbors:
            local = self.reference_action_raw[row_neighbors]
            continuous = {}
            for column in self.config["continuous_action_columns"]:
                values = local[:, action_columns.index(column)]
                adjustment = float(spec["continuous_conformal_adjustments"][column])
                continuous[column] = [
                    max(0.0, float(np.quantile(values, alpha / 2.0) - adjustment)),
                    min(1.0, float(np.quantile(values, 1.0 - alpha / 2.0) + adjustment)),
                ]
            binary = {}
            minimum = int(spec["binary_minimum_neighbor_count"])
            for column in self.config["binary_action_columns"]:
                values = local[:, action_columns.index(column)]
                binary[column] = [value for value in (0, 1)
                                  if int(np.sum(values == value)) >= minimum]
            results.append({"continuous": continuous, "binary": binary})
        return results
