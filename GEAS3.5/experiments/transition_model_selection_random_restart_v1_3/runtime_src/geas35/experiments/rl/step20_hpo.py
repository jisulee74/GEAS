"""Reproducible, resumable Step 20 HPO protocol for GEAS Hybrid PPO.

The module deliberately separates metric aggregation/ranking from the expensive
environment runner.  This makes the length-normalisation and safety gates
independently auditable and prevents a smoke run from becoming an official HPO
result accidentally.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from geas35.rl.hybrid_ppo import (HybridPPOConfig, HybridPPOTrainer, SupportAwareActionAdapter)

STEP20_VERSION = "geas35.rl.step20.v1"
OFFICIAL_CROPS = ("strawberry", "melon", "cucumber")


@dataclass(frozen=True)
class EpisodeEvaluation:
    episode_id: str
    scheduled_valid_steps: int
    executed_steps: int
    reward_sum: float
    discounted_return: float
    hard_constraint_violations: int
    severe_ood_steps: int
    fallback_steps: int
    warning_ood_steps: int
    action_total_variation: float

    def __post_init__(self) -> None:
        if self.scheduled_valid_steps <= 0:
            raise ValueError("scheduled_valid_steps must be positive")
        if not 0 <= self.executed_steps <= self.scheduled_valid_steps:
            raise ValueError("executed_steps must be within the scheduled horizon")
        counts = (self.hard_constraint_violations, self.severe_ood_steps,
                  self.fallback_steps, self.warning_ood_steps)
        if any(value < 0 or value > self.executed_steps for value in counts):
            raise ValueError("event counts must be within executed_steps")


def aggregate_validation_metrics(episodes: Sequence[EpisodeEvaluation]) -> dict[str, float | int]:
    """Micro-average on scheduled steps; early termination cannot improve score."""
    if not episodes:
        raise ValueError("At least one validation episode is required")
    scheduled = sum(item.scheduled_valid_steps for item in episodes)
    executed = sum(item.executed_steps for item in episodes)
    reward_sum = float(sum(item.reward_sum for item in episodes))
    denom = float(scheduled)
    executed_denom = float(max(executed, 1))
    return {
        "mean_reward_per_scheduled_valid_step": reward_sum / denom,
        "mean_reward_per_executed_step": reward_sum / executed_denom,
        "mean_episodic_return": reward_sum / len(episodes),
        "mean_discounted_return": float(np.mean([item.discounted_return for item in episodes])),
        "scheduled_valid_steps": scheduled,
        "executed_steps": executed,
        "early_termination_missing_steps": scheduled - executed,
        "hard_constraint_violations": sum(item.hard_constraint_violations for item in episodes),
        "severe_ood_rate": sum(item.severe_ood_steps for item in episodes) / executed_denom,
        "fallback_rate": sum(item.fallback_steps for item in episodes) / executed_denom,
        "warning_ood_rate": sum(item.warning_ood_steps for item in episodes) / executed_denom,
        "action_instability": float(np.average(
            [item.action_total_variation for item in episodes],
            weights=[item.executed_steps or 1 for item in episodes],
        )),
    }


def apply_safety_gate(
    metrics: Mapping[str, float | int], *, severe_ood_rate_max: float,
    fallback_rate_max: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(metrics["hard_constraint_violations"]) != 0:
        reasons.append("hard_constraint_violation")
    if float(metrics["severe_ood_rate"]) > severe_ood_rate_max:
        reasons.append("severe_ood_rate_above_train_frozen_limit")
    if float(metrics["fallback_rate"]) > fallback_rate_max:
        reasons.append("fallback_rate_above_train_frozen_limit")
    return not reasons, reasons


def trial_rank_key(record: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    """Feasible first; objective primary; OOD/instability are tie breakers only."""
    feasible = bool(record["feasible"])
    metrics = record["metrics"]
    return (
        1.0 if feasible else 0.0,
        float(metrics["mean_reward_per_scheduled_valid_step"]) if feasible else -np.inf,
        -float(metrics["warning_ood_rate"]),
        -float(metrics["action_instability"]),
        -int(record["trial_number"]),
    )


class DeterministicSearchSpace:
    """Seeded sampler equivalent suitable for an SQLite-backed resumable study."""

    def __init__(self, specification: Mapping[str, Any], seed: int):
        self.specification = dict(specification)
        self.seed = int(seed)

    def sample_trial(self, trial_number: int, observation_dim: int) -> dict[str, Any]:
        rng = random.Random(self.seed + 1_000_003 * int(trial_number))
        spec = self.specification
        def choice(name: str):
            return rng.choice(spec[name])
        learning_rate = float(np.exp(rng.uniform(
            np.log(spec["learning_rate"][0]), np.log(spec["learning_rate"][1])
        )))
        ppo = HybridPPOConfig(
            observation_dim=observation_dim,
            hidden_sizes=tuple([int(choice("network_width"))] * int(choice("network_depth"))),
            beta_concentration_floor=float(choice("beta_concentration_floor")),
            learning_rate=learning_rate, gamma=float(choice("gamma")),
            gae_lambda=float(choice("gae_lambda")), clip_range=float(choice("clip_range")),
            value_coefficient=float(choice("value_coefficient")),
            entropy_coefficient=float(choice("entropy_coefficient")),
            update_epochs=int(choice("update_epochs")),
            minibatch_size=int(choice("minibatch_size")),
            target_kl=float(choice("target_kl")), seed=self.seed + trial_number,
        )
        return {"ppo": json.loads(json.dumps(asdict(ppo))), "rollout_length": int(choice("rollout_length"))}

    def sample(self, trial_number: int, observation_dim: int) -> HybridPPOConfig:
        payload = self.sample_trial(trial_number, observation_dim)
        values = dict(payload["ppo"])
        values["hidden_sizes"] = tuple(values["hidden_sizes"])
        return HybridPPOConfig(**values)


class HpoDatabase:
    """Small append-safe SQLite trial store with explicit study identity."""

    def __init__(self, path: str | Path, study_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.study_id = study_id
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS trials (
                study_id TEXT NOT NULL, crop TEXT NOT NULL, trial_number INTEGER NOT NULL,
                rung INTEGER NOT NULL, status TEXT NOT NULL, config_json TEXT NOT NULL,
                metrics_json TEXT, feasible INTEGER, failure_reasons_json TEXT,
                seed INTEGER NOT NULL, budget_steps INTEGER NOT NULL,
                PRIMARY KEY (study_id, crop, trial_number, rung))""")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def upsert(self, *, crop: str, trial_number: int, rung: int, status: str,
               config: Mapping[str, Any], seed: int, budget_steps: int,
               metrics: Mapping[str, Any] | None = None, feasible: bool | None = None,
               failure_reasons: Sequence[str] = ()) -> None:
        with self._connect() as db:
            db.execute("""INSERT OR REPLACE INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                self.study_id, crop, trial_number, rung, status,
                json.dumps(dict(config), sort_keys=True),
                None if metrics is None else json.dumps(dict(metrics), sort_keys=True),
                None if feasible is None else int(feasible), json.dumps(list(failure_reasons)),
                seed, budget_steps,
            ))

    def records(self, crop: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""SELECT trial_number, rung, status, config_json,
                metrics_json, feasible, failure_reasons_json, seed, budget_steps
                FROM trials WHERE study_id=? AND crop=? ORDER BY trial_number, rung""",
                (self.study_id, crop)).fetchall()
        return [{
            "trial_number": row[0], "rung": row[1], "status": row[2],
            "config": json.loads(row[3]),
            "metrics": None if row[4] is None else json.loads(row[4]),
            "feasible": None if row[5] is None else bool(row[5]),
            "failure_reasons": json.loads(row[6]), "seed": row[7], "budget_steps": row[8],
        } for row in rows]


def evaluate_paired_validation_episodes(
    env: Any, trainer: HybridPPOTrainer, *, episode_indices: Sequence[int], seed: int
) -> tuple[list[EpisodeEvaluation], dict[str, Any]]:
    """Deterministically evaluate one policy on a frozen ordered episode registry."""
    import torch
    if getattr(env, "split", None) != "validation":
        raise ValueError("HPO objectives may only be evaluated on Validation")
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("episode_indices must be non-empty and unique")
    results: list[EpisodeEvaluation] = []
    episode_manifest = []
    for ordinal, episode_index in enumerate(episode_indices):
        observation, reset_info = env.reset(seed=seed + ordinal, options={"episode_index": episode_index})
        scheduled = int(reset_info["scheduled_valid_steps"])
        rewards: list[float] = []
        actions: list[np.ndarray] = []
        hard = severe = fallback = warning = 0
        terminated = truncated = False
        adapter = SupportAwareActionAdapter(env)
        while not (terminated or truncated):
            spec = adapter.feasible_spec()
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=trainer.device).unsqueeze(0)
            with torch.no_grad():
                output = trainer.policy.act(tensor, spec.batched(1, device=trainer.device), deterministic=True)
            action = output["action"][0].cpu().numpy().astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            severe += int(info.get("joint_ood_level") == "severe" or info.get("state_ood_level") == "severe")
            warning += int(info.get("joint_ood_level") == "warning")
            fallback += int(info.get("safety_fallback_applied", False))
            executed = info.get("executed_action")
            if executed is not None:
                values = np.asarray(list(executed.values()), dtype=float)
                weather = info.get("weather_constraint", {})
                binary_valid = all(float(executed[name]) in (0.0, 1.0)
                                   for name in ("heat_run", "cool_run", "fan_run"))
                vent_valid = float(executed["vent_pct"]) <= float(weather.get("vent_open_cap", 1.0)) + 1e-7
                hard += int(not np.isfinite(values).all() or np.any(values < 0.0)
                            or np.any(values > 1.0) or not binary_valid or not vent_valid)
                actions.append(values)
        discounted = sum((trainer.config.gamma ** index) * value for index, value in enumerate(rewards))
        variation = (float(np.abs(np.diff(np.asarray(actions), axis=0)).sum(axis=1).mean())
                     if len(actions) > 1 else 0.0)
        results.append(EpisodeEvaluation(
            episode_id=str(reset_info["episode_id"]), scheduled_valid_steps=scheduled,
            executed_steps=len(rewards), reward_sum=float(sum(rewards)),
            discounted_return=float(discounted), hard_constraint_violations=hard,
            severe_ood_steps=severe, fallback_steps=fallback, warning_ood_steps=warning,
            action_total_variation=variation,
        ))
        episode_manifest.append({"episode_index": episode_index, "episode_id": reset_info["episode_id"],
                                 "series_id": reset_info["series_id"], "rollout_id": reset_info["rollout_id"],
                                 "scheduled_valid_steps": scheduled, "seed": seed + ordinal})
    return results, {"split": "validation", "paired": True, "episodes": episode_manifest,
                     "test_accessed": False}


def validate_step20_config(config: Mapping[str, Any], step15_hpo: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "geas35.rl.step20.config.v1":
        raise ValueError("Unsupported Step 20 config schema")
    if config["trials_per_crop"] != step15_hpo["trials_per_crop"]:
        raise ValueError("trials_per_crop differs from the frozen Step 15 budget")
    if config["sampler_seed"] != step15_hpo["sampler_seed"]:
        raise ValueError("sampler_seed differs from the frozen Step 15 budget")
    if config["objective"]["primary"] != "mean_reward_per_scheduled_valid_step":
        raise ValueError("Step 20 primary objective must be length-normalized")
    if config["objective"]["aggregation"] != "scheduled_step_micro_average":
        raise ValueError("Step 20 aggregation must be frozen to scheduled-step micro average")
    if config["split_policy"] != {"fit": "train", "objective": "validation", "test": "locked"}:
        raise ValueError("Step 20 split policy must keep Test locked")
    if config["execution"]["official_trials_required"] != 30:
        raise ValueError("Official completion requires 30 trials per crop")


def normalized_objective_audit() -> dict[str, Any]:
    """Executable proof that early termination cannot improve equal accumulated reward."""
    complete = EpisodeEvaluation("complete", 10, 10, -10.0, -9.0, 0, 0, 0, 0, 0.1)
    early = EpisodeEvaluation("early", 10, 2, -10.0, -9.0, 0, 0, 0, 0, 0.1)
    complete_metrics = aggregate_validation_metrics([complete])
    early_metrics = aggregate_validation_metrics([early])
    return {
        "denominator": "scheduled_valid_steps",
        "complete": complete_metrics,
        "early_terminated": early_metrics,
        "same_primary_for_same_reward_and_schedule": (
            complete_metrics["mean_reward_per_scheduled_valid_step"]
            == early_metrics["mean_reward_per_scheduled_valid_step"]
        ),
        "executed_step_metric_not_used_for_ranking": True,
    }


__all__ = [
    "STEP20_VERSION", "EpisodeEvaluation", "HpoDatabase", "DeterministicSearchSpace",
    "aggregate_validation_metrics", "apply_safety_gate", "evaluate_paired_validation_episodes", "normalized_objective_audit",
    "trial_rank_key", "validate_step20_config",
]
