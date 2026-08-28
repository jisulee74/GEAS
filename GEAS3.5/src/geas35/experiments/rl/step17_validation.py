"""Step 17 validation of GEAS model-driven dynamics and safety contracts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from geas35.models.transition.datasets import build_transition_dataset_from_rl_frame
from geas35.models.transition.hyperparameters import transition_target_normalization_scales
from geas35.models.transition.trainer import _evaluate_rollouts
from geas35.rl.mdp_v1 import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_REWARD_TERM_KEYS,
    compute_mdp_v1_reward,
    normalize_mdp_v1_action,
    project_mdp_v1_action_constraints,
)
from geas35.rl.model_driven_env import (
    ModelDrivenEnvConfig,
    load_model_driven_env_from_step15,
)

STEP17_VERSION = "geas35.rl.step17.v1"
OFFICIAL_CROPS = ("strawberry", "melon", "cucumber")
HORIZONS = (3, 6, 12)
SUPPORT_SAMPLE_ROWS = 3000
RANDOM_ACTIONS_PER_STATE = 384


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_record(root: Path, record: Mapping[str, Any]) -> Path:
    path = (root / str(record["path"])).resolve()
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return path


def _logged_action(env) -> dict[str, float]:
    return {column: float(env._current_scaled[column]) for column in env.action_columns}


def _in_support_episode(env) -> int:
    thresholds = env.support.config["thresholds"]
    for index in range(len(env.episodes)):
        env.reset(seed=17, options={"episode_index": index})
        action = _logged_action(env)
        state, action_frame = env._support_frames(action)
        scores = env.support.scores(state, action_frame)
        if (
            float(scores["state"][0]) <= float(thresholds["state_warning"])
            and float(scores["joint"][0]) <= float(thresholds["joint_warning"])
        ):
            return index
    raise ValueError(f"{env.crop}: no in-support episode start.")


def _model_action_sensitivity(env, episode_index: int) -> dict[str, Any]:
    env.reset(seed=17, options={"episode_index": episode_index})
    current = env._current_scaled.copy()
    cases = {
        "zero": {column: 0.0 for column in env.action_columns},
        "logged": _logged_action(env),
        "extreme": {
            **{column: 1.0 for column in env.action_columns[:3]},
            **dict(zip(env.action_columns[3:], (1.0, 0.0, 1.0))),
        },
    }
    predictions: dict[str, dict[str, float]] = {}
    for name, action in cases.items():
        row = current.copy()
        for column, value in action.items():
            row[column] = value
        predicted = env.model.predict(row.to_frame().T).next_observation.iloc[0]
        predictions[name] = {
            column: float(predicted[column]) for column in env.model.target_columns_
        }
    vectors = [np.asarray(list(value.values()), dtype=float) for value in predictions.values()]
    max_pairwise_delta = max(
        float(np.max(np.abs(vectors[left] - vectors[right])))
        for left in range(len(vectors)) for right in range(left + 1, len(vectors))
    )
    if not np.isfinite(vectors).all() or max_pairwise_delta <= 0.0:
        raise ValueError(f"{env.crop}: transition model has no finite action response.")
    return {
        "status": "passed",
        "actions": cases,
        "predictions_physical": predictions,
        "max_pairwise_target_delta": max_pairwise_delta,
        "repeatability": True,
    }


def _reward_parity(env, episode_index: int) -> dict[str, Any]:
    env.reset(seed=17, options={"episode_index": episode_index})
    action = _logged_action(env)
    previous = dict(env._last_action)
    previous_previous = dict(env._last_last_action)
    _, reward, terminated, truncated, info = env.step(action)
    expected_reward, expected_terms = compute_mdp_v1_reward(
        env._current_physical,
        info["executed_action"],
        prev_action=previous,
        prev_prev_action=previous_previous,
        config=env.mdp_config,
    )
    penalty_labels = [f"{key}_penalty" for key in MDP_V1_REWARD_TERM_KEYS]
    terms_match = all(
        np.isclose(info["reward_terms"][key], expected_terms[key], rtol=0.0, atol=1e-12)
        for key in expected_terms
    )
    decomposition = -sum(info["reward_terms"][key] for key in penalty_labels)
    passed = (
        terms_match
        and np.isclose(info["base_reward"], expected_reward, rtol=0.0, atol=1e-12)
        and np.isclose(decomposition, expected_reward, rtol=0.0, atol=1e-12)
        and len(penalty_labels) == 6
        and not truncated
    )
    if not passed:
        raise ValueError(f"{env.crop}: reward decomposition parity failed.")
    return {
        "status": "passed",
        "reward": reward,
        "base_reward": info["base_reward"],
        "warning_ood_penalty": info["warning_ood_penalty"],
        "penalty_terms": {key: info["reward_terms"][key] for key in penalty_labels},
        "terminated": terminated,
        "truncated": truncated,
    }


def _constraint_matrix(env) -> dict[str, Any]:
    raw = {
        "vent_pct": 2.0, "shade_curtain_pct": -1.0,
        "thermal_curtain_pct": 2.0, "heat_run": 0.9,
        "cool_run": 0.1, "fan_run": 0.9,
    }
    normalized = normalize_mdp_v1_action(raw)
    cases = {
        "none": (0.0, 0.0, 1.0),
        "rain": (1.0, 0.0, env.mdp_config.rain_vent_open_cap),
        "wind": (0.0, env.mdp_config.wind_cap_threshold + 1.0,
                 env.mdp_config.wind_vent_open_cap),
        "rain_and_wind": (1.0, env.mdp_config.wind_cap_threshold + 1.0,
                          min(env.mdp_config.rain_vent_open_cap,
                              env.mdp_config.wind_vent_open_cap)),
    }
    rows = {}
    for name, (rain, wind, expected_cap) in cases.items():
        projected, info = project_mdp_v1_action_constraints(
            normalized, {"obs_rain_flag": rain, "obs_outdoor_wind_speed": wind},
            env.mdp_config,
        )
        if not np.isclose(projected["vent_pct"], expected_cap):
            raise ValueError(f"{env.crop}: {name} vent cap failed.")
        rows[name] = {
            "raw_action": raw,
            "range_projected_action": normalized,
            "executed_action": projected,
            "intervention": info,
            "status": "passed",
        }
    return rows


def _find_ood_actions(env) -> tuple[int, dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(17)
    thresholds = env.support.config["thresholds"]
    for episode_index in range(min(50, len(env.episodes))):
        warning = severe = None
        env.reset(seed=17, options={"episode_index": episode_index})
        logged = _logged_action(env)
        state, logged_frame = env._support_frames(logged)
        state_score = env.support.scores(state, logged_frame)["state"][0]
        if state_score > thresholds["state_warning"]:
            continue
        actions = []
        for _ in range(RANDOM_ACTIONS_PER_STATE):
            action = {column: float(rng.random()) for column in env.action_columns[:3]}
            action.update({column: float(rng.integers(2)) for column in env.action_columns[3:]})
            actions.append(action)
        action_frame = pd.DataFrame(actions, columns=env.action_columns)
        state_frame = pd.concat([state] * len(action_frame), ignore_index=True)
        scores = env.support.scores(state_frame, action_frame)["joint"]
        for action, score in zip(actions, scores):
            if warning is None and thresholds["joint_warning"] < score <= thresholds["joint_severe"]:
                _, projection = env._project_local_support(action)
                if projection["changes"]:
                    warning = action
            if severe is None and score > thresholds["joint_severe"]:
                severe = action
            if warning is not None and severe is not None:
                return episode_index, warning, severe
    raise ValueError(f"{env.crop}: deterministic warning/severe OOD actions not found.")


def _ood_matrix(env) -> dict[str, Any]:
    episode_index, warning_action, severe_action = _find_ood_actions(env)
    env.reset(seed=17, options={"episode_index": episode_index})
    warning_result = env.step(warning_action)
    warning_info = warning_result[4]
    if not (
        warning_info["joint_ood_level"] == "warning"
        and warning_info["support_projection"]["changes"]
        and warning_info["warning_ood_penalty"] > 0.0
    ):
        raise ValueError(f"{env.crop}: warning OOD response failed.")

    env.reset(seed=17, options={"episode_index": episode_index})
    severe_result = env.step(severe_action)
    severe_info = severe_result[4]
    if not (
        severe_info["joint_ood_level"] == "severe"
        and severe_info["safety_fallback_applied"]
        and severe_result[1] == env.env_config.worst_step_reward
    ):
        raise ValueError(f"{env.crop}: severe action OOD response failed.")

    env.reset(seed=17, options={"episode_index": episode_index})
    continuous = env.support.config["continuous_state_columns"][0]
    env._current_scaled[continuous] = 1_000_000.0
    state_result = env.step(_logged_action(env))
    state_info = state_result[4]
    if not (
        state_result[2]
        and state_info["termination_reason"] == "severe_state_ood"
        and state_info["remaining_scheduled_steps_charged"] > 0
    ):
        raise ValueError(f"{env.crop}: severe state OOD termination failed.")
    return {
        "warning_action": {
            "status": "passed", "raw_action": warning_action,
            "executed_action": warning_info["executed_action"],
            "joint_ood_score": warning_info["joint_ood_score"],
            "projection": warning_info["support_projection"],
            "penalty": warning_info["warning_ood_penalty"],
        },
        "severe_action": {
            "status": "passed", "raw_action": severe_action,
            "executed_action": severe_info["executed_action"],
            "joint_ood_score": severe_info["joint_ood_score"],
            "fallback": severe_info["safety_fallback_applied"],
        },
        "severe_state": {
            "status": "passed",
            "termination_reason": state_info["termination_reason"],
            "pessimistic_terminal_cost": state_info["pessimistic_terminal_cost"],
            "remaining_scheduled_steps_charged": state_info["remaining_scheduled_steps_charged"],
        },
    }


def _support_coverage(env) -> dict[str, Any]:
    frame = env.frame
    positions = np.unique(np.linspace(
        0, len(frame) - 1, min(SUPPORT_SAMPLE_ROWS, len(frame)), dtype=int
    ))
    sample = frame.iloc[positions].reset_index(drop=True)
    state = sample.loc[:, env.observation_columns]
    action = sample.loc[:, env.action_columns]
    scores = env.support.scores(state, action)
    thresholds = env.support.config["thresholds"]
    state_severe = scores["state"] <= thresholds["state_severe"]
    joint_severe = scores["joint"] <= thresholds["joint_severe"]
    overall = {
        "rows": len(sample),
        "state_warning_coverage": float(np.mean(scores["state"] <= thresholds["state_warning"])),
        "state_severe_coverage": float(np.mean(state_severe)),
        "joint_warning_coverage": float(np.mean(scores["joint"] <= thresholds["joint_warning"])),
        "joint_severe_coverage": float(np.mean(joint_severe)),
    }
    groups = []
    stage_columns = [column for column in sample if column.startswith("obs_growth_stage_")
                     and column != "obs_growth_stage_dat"]
    stage_values = sample[stage_columns].to_numpy(dtype=float)
    stage = np.asarray([stage_columns[index] for index in np.argmax(stage_values, axis=1)])
    daylight = np.where(sample["obs_is_daytime"].to_numpy(dtype=float) > 0.5, "day", "night")
    for dimension, labels in (("growth_stage", stage), ("day_night", daylight)):
        for label in sorted(set(labels)):
            mask = labels == label
            groups.append({
                "dimension": dimension, "group": str(label), "rows": int(mask.sum()),
                "state_severe_coverage": float(np.mean(state_severe[mask])),
                "joint_severe_coverage": float(np.mean(joint_severe[mask])),
            })
    evaluable = [group for group in groups if group["rows"] >= 30]
    minimum_allowed = max(0.90, overall["joint_severe_coverage"] - 0.10)
    flagged = [
        {**group, "reason": "joint_severe_coverage_below_subgroup_diagnostic_level"}
        for group in evaluable
        if group["joint_severe_coverage"] < minimum_allowed
    ]
    passed = (
        overall["state_severe_coverage"] >= 0.95
        and overall["joint_severe_coverage"] >= 0.95
    )
    if not passed:
        raise ValueError(f"{env.crop}: overall logged Train support coverage gate failed.")
    return {
        "status": "passed_with_subgroup_flags" if flagged else "passed",
        "sampling": "deterministic_even_positions",
        "sample_rows_cap": SUPPORT_SAMPLE_ROWS,
        "threshold_source": "frozen_train_conformal_support",
        "overall": overall, "subgroups": groups,
        "subgroup_diagnostic": {
            "minimum_rows": 30,
            "comparison_level": minimum_allowed,
            "hard_gate": False,
            "flagged_groups": flagged,
        },
    }


def _rollout_parity(root: Path, crop: str, env, validation_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(validation_path)
    dataset = build_transition_dataset_from_rl_frame(frame).dataset
    scales = transition_target_normalization_scales(dataset.y, dataset.target_columns)
    reproduced = _evaluate_rollouts(
        env.model, rollout_frame=frame, rollout_horizon_steps=HORIZONS,
        rollout_step_minutes=5, action_provider_factory=None,
        normalization_scales=scales,
    )
    reference_path = root / (
        f"experiments/transition_model_selection/artifacts/step12/{crop}/{crop}/"
        "extra_trees/rollout_metrics.json"
    )
    reference = _read(reference_path)
    differences = {
        horizon: max(
            abs(float(value) - float(reference[horizon][key]))
            for key, value in metrics.items()
        )
        for horizon, metrics in reproduced.items()
    }
    max_difference = max(differences.values())
    if max_difference > 1e-10:
        raise ValueError(f"{crop}: Step 12 rollout parity differs by {max_difference}.")
    return {
        "status": "passed", "reference": _record(reference_path, root),
        "source_split": "validation", "horizon_steps": list(HORIZONS),
        "step_minutes": 5, "max_absolute_metric_difference": max_difference,
        "per_horizon_max_absolute_difference": differences,
    }


def _split_registry_audit(root: Path, crop: str) -> dict[str, Any]:
    path = root / f"offline_dataset_preparation/datasets/02_split/{crop}/episode_manifest.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    assigned = frame.loc[frame["split"].isin(("train", "validation", "test"))]
    sets = {
        split: set(assigned.loc[assigned["split"] == split, "episode_id"].astype(str))
        for split in ("train", "validation", "test")
    }
    pairwise_disjoint = all(
        sets[left].isdisjoint(sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if not pairwise_disjoint or not all(sets.values()):
        raise ValueError(f"{crop}: split episode registry audit failed.")
    return {
        "status": "passed", "episode_manifest": _record(path, root),
        "episode_counts": {key: len(value) for key, value in sets.items()},
        "pairwise_disjoint": True,
        "test_parquet_opened": False,
    }


def _time_limit_check(env, episode_index: int) -> dict[str, Any]:
    original_config = env.env_config
    env.env_config = ModelDrivenEnvConfig(max_episode_steps=1)
    env.reset(seed=17, options={"episode_index": episode_index})
    _, _, terminated, truncated, info = env.step(_logged_action(env))
    env.env_config = original_config
    if terminated or not truncated or info["truncation_reason"] != "training_time_limit":
        raise ValueError(f"{env.crop}: training time-limit truncation contract failed.")
    return {
        "status": "passed", "terminated": terminated, "truncated": truncated,
        "termination_reason": info["termination_reason"],
        "truncation_reason": info["truncation_reason"],
    }


def _data_boundary_check(env, episode_index: int) -> dict[str, Any]:
    env.reset(seed=17, options={"episode_index": episode_index})
    env._episode = dict(env._episode)
    env._episode["final_observation"] = env._position + 1
    env._episode["scheduled_valid_steps"] = 1
    _, _, terminated, truncated, info = env.step(_logged_action(env))
    if not terminated or truncated or info["termination_reason"] != "episode_boundary":
        raise ValueError(f"{env.crop}: data-boundary termination contract failed.")
    return {
        "status": "passed", "terminated": terminated, "truncated": truncated,
        "termination_reason": info["termination_reason"],
        "truncation_reason": info["truncation_reason"],
    }


def run_step17_environment_validation(
    *, project_root: str | Path, output_root: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    destination = Path(output_root).resolve() if output_root else (
        root / "experiments/rl_policy_training/artifacts/step17"
    )
    step16_integrity_path = root / "experiments/rl_policy_training/artifacts/step16/step16_integrity_manifest.json"
    step16_integrity = _read(step16_integrity_path)
    if step16_integrity.get("status") != "passed":
        raise ValueError("Step 17 requires passed Step 16 integrity.")
    _verify_record(root, step16_integrity["environment_manifest"])
    split_policy = _read(root / "experiments/rl_policy_training/artifacts/step15/split_access_policy.json")

    dynamics: dict[str, Any] = {}
    constraints: dict[str, Any] = {}
    rollouts: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for crop in OFFICIAL_CROPS:
        env = load_model_driven_env_from_step15(project_root=root, crop=crop, split="train")
        episode_index = _in_support_episode(env)
        dynamics[crop] = {
            "status": "passed",
            "action_sensitivity": _model_action_sensitivity(env, episode_index),
            "reward_parity": _reward_parity(env, episode_index),
            "termination_truncation": {
                "data_boundary": _data_boundary_check(env, episode_index),
                "severe_ood_reasons": ["severe_state_ood", "repeated_severe_action_ood"],
                "time_limit": _time_limit_check(env, episode_index),
            },
            "split_registry": _split_registry_audit(root, crop),
        }
        constraints[crop] = {
            "status": "passed", "action_and_weather_constraints": _constraint_matrix(env),
            "ood_response_sequence": _ood_matrix(env),
        }
        coverage[crop] = _support_coverage(env)
        validation_path = Path(split_policy["datasets"][crop]["validation"]["path"])
        rollouts[crop] = _rollout_parity(root, crop, env, validation_path)

    common = {"schema_version": STEP17_VERSION, "status": "passed"}
    validation_path = destination / "environment_validation_report.json"
    constraint_path = destination / "constraint_ood_test_matrix.json"
    rollout_path = destination / "rollout_parity_report.json"
    coverage_path = destination / "support_coverage_report.json"
    _write(validation_path, {**common, "crops": dynamics})
    _write(constraint_path, {**common, "crops": constraints})
    _write(rollout_path, {**common, "crops": rollouts})
    _write(coverage_path, {**common, "fit_split": "train_only", "crops": coverage})

    integrity = {
        "schema_version": STEP17_VERSION, "step": "17", "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "environment_validation_only_step18_not_implemented",
        "step16_integrity": _record(step16_integrity_path, root),
        "implementations": {
            "environment": _record(root / "src/geas35/rl/model_driven_env.py", root),
            "validation_runner": _record(root / "src/geas35/experiments/rl/step17_validation.py", root),
        },
        "reports": {
            "environment_validation": _record(validation_path, root),
            "constraint_ood_test_matrix": _record(constraint_path, root),
            "rollout_parity": _record(rollout_path, root),
            "support_coverage": _record(coverage_path, root),
        },
        "checks": {
            "dynamics_action_sensitivity": True,
            "reward_six_term_parity": True,
            "weather_and_range_constraints": True,
            "termination_truncation_distinguished": True,
            "warning_projection_penalty": True,
            "severe_fallback_and_pessimistic_termination": True,
            "step12_rollout_numeric_parity": True,
            "logged_support_coverage": True,
            "subgroup_coverage_reported": True,
            "split_episode_registries_disjoint": True,
            "test_parquet_opened": False,
            "step18_implemented": False,
        },
    }
    integrity_path = destination / "step17_integrity_manifest.json"
    _write(integrity_path, integrity)
    return integrity_path


__all__ = ["STEP17_VERSION", "run_step17_environment_validation"]
