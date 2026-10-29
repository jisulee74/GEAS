"""Step 19 trainability-only pilot for the GEAS Hybrid PPO implementation."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from geas35.rl.hybrid_ppo import (
    FeasibleActionSpec,
    HybridActorCritic,
    HybridPPOConfig,
    HybridPPOTrainer,
    collect_rollout,
    compute_gae,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
    seed_everything,
)
from geas35.rl.model_driven_env import load_model_driven_env_from_step15

STEP19_VERSION = "geas35.rl.step19.v1"
OFFICIAL_CROPS = ("strawberry", "melon", "cucumber")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _artifact_hashes(root: Path, protocol: Mapping[str, Any], crop: str) -> dict[str, str]:
    record = protocol["crops"][crop]
    return {
        "observation_scaler": record["observation_scaler"]["sha256"],
        "environment": _sha256(root / "src/geas35/rl/model_driven_env.py"),
        "transition_model": record["candidate_model"]["sha256"],
        "support": record["state_action_support"]["sha256"],
    }


def _raw_signal(buffer, config: HybridPPOConfig) -> dict[str, float]:
    advantages, returns = compute_gae(
        np.asarray(buffer.rewards), np.asarray(buffer.values), np.asarray(buffer.next_values),
        np.asarray(buffer.terminated), gamma=config.gamma, gae_lambda=config.gae_lambda,
    )
    return {
        "reward_mean": float(np.mean(buffer.rewards)),
        "reward_std": float(np.std(buffer.rewards)),
        "reward_min": float(np.min(buffer.rewards)),
        "reward_max": float(np.max(buffer.rewards)),
        "raw_advantage_mean": float(np.mean(advantages)),
        "raw_advantage_std": float(np.std(advantages)),
        "return_std": float(np.std(returns)),
    }


def _support_restriction_rate(buffer) -> float:
    restricted = []
    for spec in buffer.feasible_specs:
        restricted.append(bool(
            np.any(spec.continuous_low > 0.0)
            or np.any(spec.continuous_high < 1.0)
            or np.any(~(spec.binary_allow_zero & spec.binary_allow_one))
        ))
    return float(np.mean(restricted)) if restricted else 0.0


def _invalid_action_count(buffer) -> int:
    actions = np.asarray(buffer.policy_actions, dtype=float)
    if actions.shape != (len(buffer), 6):
        return len(buffer)
    invalid = ~np.isfinite(actions).all(axis=1)
    invalid |= np.any((actions[:, :3] < 0.0) | (actions[:, :3] > 1.0), axis=1)
    invalid |= ~np.isin(actions[:, 3:], (0.0, 1.0)).all(axis=1)
    for index, spec in enumerate(buffer.feasible_specs):
        invalid[index] |= np.any(actions[index, :3] < spec.continuous_low - 1e-7)
        invalid[index] |= np.any(actions[index, :3] > spec.continuous_high + 1e-7)
        for binary_index, value in enumerate(actions[index, 3:]):
            invalid[index] |= bool(
                (value == 0.0 and not spec.binary_allow_zero[binary_index])
                or (value == 1.0 and not spec.binary_allow_one[binary_index])
            )
    return int(invalid.sum())


def _state_response(env, policy: HybridActorCritic, count: int) -> dict[str, float]:
    positions = np.unique(np.linspace(0, len(env.frame) - 1, count, dtype=int))
    observations = env.frame.iloc[positions].loc[:, env.observation_columns].to_numpy(np.float32)
    tensor = torch.as_tensor(observations)
    feasible = FeasibleActionSpec.unconstrained().batched(len(tensor))
    with torch.no_grad():
        alpha, beta, logits, values = policy(tensor)
        deterministic = policy.act(tensor, feasible, deterministic=True)["action"]
    parameters = torch.cat((alpha, beta, logits), dim=1).cpu().numpy()
    actions = deterministic.cpu().numpy()
    return {
        "representative_states": int(len(positions)),
        "distribution_parameter_max_range": float(np.max(np.ptp(parameters, axis=0))),
        "deterministic_action_max_range": float(np.max(np.ptp(actions, axis=0))),
        "value_range": float(np.ptp(values.cpu().numpy())),
    }


def _seed_reproducibility(config: HybridPPOConfig) -> bool:
    observation = torch.linspace(-1.0, 1.0, config.observation_dim).unsqueeze(0)
    feasible = FeasibleActionSpec.unconstrained().batched(1)
    seed_everything(config.seed)
    first = HybridActorCritic(config)
    with torch.no_grad():
        first_output = first.act(observation, feasible, deterministic=True)
    seed_everything(config.seed)
    second = HybridActorCritic(config)
    with torch.no_grad():
        second_output = second.act(observation, feasible, deterministic=True)
    return bool(
        torch.equal(first_output["action"], second_output["action"])
        and torch.equal(first_output["value"], second_output["value"])
        and all(torch.equal(left, right) for left, right in zip(first.parameters(), second.parameters()))
    )


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    return all(
        isinstance(value, (bool, int)) or np.isfinite(float(value))
        for value in metrics.values()
    )


def _gate_crop(
    *, signal: Mapping[str, float], updates: list[Mapping[str, Any]],
    state_response: Mapping[str, float], aggregate_ood: Mapping[str, float],
    invalid_actions: int, resume_passed: bool, seed_reproducible: bool,
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    gradients = [float(update["gradient_norm"]) for update in updates]
    parameter_updates = [float(update["parameter_update_l2"]) for update in updates]
    kls = [abs(float(update["approx_kl"])) for update in updates]
    ratios = [float(update["ratio_mean"]) for update in updates]
    finite = _finite_metrics(signal) and all(_finite_metrics(update) for update in updates)
    return {
        "finite_no_nan_inf": finite,
        "reward_learning_signal": signal["reward_std"] >= gates["minimum_reward_std"],
        "advantage_learning_signal": signal["raw_advantage_std"] >= gates["minimum_raw_advantage_std"],
        "gradient_nonvanishing": min(gradients) >= gates["minimum_gradient_norm"],
        "gradient_nonexploding": max(gradients) <= gates["maximum_gradient_norm"],
        "parameter_update_observed": min(parameter_updates) >= gates["minimum_parameter_update_l2"],
        "kl_sane": max(kls) <= gates["maximum_absolute_approx_kl"],
        "ratio_sane": all(gates["ratio_mean_range"][0] <= ratio <= gates["ratio_mean_range"][1] for ratio in ratios),
        "state_dependent_distribution": state_response["distribution_parameter_max_range"] >= gates["minimum_state_action_response"],
        "state_dependent_action": state_response["deterministic_action_max_range"] >= gates["minimum_state_action_response"],
        "state_dependent_value": state_response["value_range"] >= gates["minimum_state_value_response"],
        "valid_actions": invalid_actions == 0,
        "support_not_blocking_all": (
            aggregate_ood["projection_rate"] <= gates["maximum_projection_rate"]
            and aggregate_ood["fallback_rate"] <= gates["maximum_fallback_rate"]
        ),
        "support_layer_active": aggregate_ood["support_restriction_rate"] >= gates["minimum_support_restriction_rate"],
        "trajectory_mostly_in_support": aggregate_ood["in_support_rate"] >= gates["minimum_in_support_rate"],
        "severe_ood_sane": aggregate_ood["severe_ood_rate"] <= gates["maximum_severe_ood_rate"],
        "checkpoint_resume": resume_passed,
        "seed_reproducibility": seed_reproducible,
    }


def run_step19_trainability_pilot(
    *, project_root: str | Path, output_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    destination = Path(output_root).resolve() if output_root else (
        root / "experiments/rl_policy_training/artifacts/step19"
    )
    config_path = Path(config_path).resolve() if config_path else (
        root / "experiments/rl_policy_training/configs/step19_pilot_v1.json"
    )
    pilot_config = _read(config_path)
    step18_path = root / "experiments/rl_policy_training/artifacts/step18/step18_integrity_manifest.json"
    step18 = _read(step18_path)
    if step18.get("status") != "passed":
        raise ValueError("Step 19 requires passed Step 18 integrity.")
    hybrid_record = step18["implementations"]["hybrid_ppo"]
    if _sha256(root / hybrid_record["path"]) != hybrid_record["sha256"]:
        raise ValueError("Step 18 Hybrid PPO implementation hash is stale.")
    protocol = _read(root / "experiments/rl_policy_training/artifacts/step15/rl_protocol_manifest.json")

    diagnostics: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for crop in OFFICIAL_CROPS:
        seed = int(pilot_config["seed"])
        seed_everything(seed)
        env = load_model_driven_env_from_step15(project_root=root, crop=crop, split="train")
        base_config = HybridPPOConfig(
            observation_dim=env.observation_shape[0], seed=seed,
            update_epochs=int(pilot_config["update_epochs"]),
            minibatch_size=int(pilot_config["minibatch_size"]),
        )
        trainer = HybridPPOTrainer(HybridActorCritic(base_config), base_config)
        initial_buffer = collect_rollout(
            env, trainer, steps=int(pilot_config["initial_rollout_steps"]), seed=seed,
        )
        initial_signal = _raw_signal(initial_buffer, base_config)
        initial_metrics = trainer.update(initial_buffer)
        hashes = _artifact_hashes(root, protocol, crop)
        checkpoint_path = destination / crop / "pilot_checkpoint.pt"
        save_ppo_checkpoint(
            checkpoint_path, trainer, artifact_hashes=hashes,
            extra_state={"crop": crop, "training_steps": len(initial_buffer), "pilot": True},
        )
        probe = torch.as_tensor(initial_buffer.observations[0]).unsqueeze(0)
        probe_spec = initial_buffer.feasible_specs[0].batched(1)
        with torch.no_grad():
            before_resume = trainer.policy.act(probe, probe_spec, deterministic=True)
        resumed, payload = load_ppo_checkpoint(
            checkpoint_path, expected_artifact_hashes=hashes, restore_rng=True,
        )
        with torch.no_grad():
            after_resume = resumed.policy.act(probe, probe_spec, deterministic=True)
        roundtrip = bool(
            torch.equal(before_resume["action"], after_resume["action"])
            and torch.equal(before_resume["value"], after_resume["value"])
            and int(payload["update_count"]) == 1
        )
        resume_buffer = collect_rollout(
            env, resumed, steps=int(pilot_config["resume_rollout_steps"]), seed=seed + 1,
        )
        resume_signal = _raw_signal(resume_buffer, base_config)
        resume_metrics = resumed.update(resume_buffer)
        resume_passed = roundtrip and resumed.update_count == 2
        total_steps = len(initial_buffer) + len(resume_buffer)
        save_ppo_checkpoint(
            checkpoint_path, resumed, artifact_hashes=hashes,
            extra_state={"crop": crop, "training_steps": total_steps, "pilot": True},
        )

        all_buffers = (initial_buffer, resume_buffer)
        intervention = [buffer.intervention_metrics() for buffer in all_buffers]
        weights = np.asarray([len(buffer) for buffer in all_buffers], dtype=float)
        aggregate_ood = {
            key: float(np.average([item[key] for item in intervention], weights=weights))
            for key in intervention[0]
        }
        aggregate_ood["support_restriction_rate"] = float(np.average(
            [_support_restriction_rate(buffer) for buffer in all_buffers], weights=weights
        ))
        invalid_actions = sum(_invalid_action_count(buffer) for buffer in all_buffers)
        state_response = _state_response(
            env, resumed.policy, int(pilot_config["representative_state_count"])
        )
        combined_signal = {
            key: float(np.average([initial_signal[key], resume_signal[key]], weights=weights))
            for key in initial_signal
        }
        updates = [initial_metrics, resume_metrics]
        gates = _gate_crop(
            signal=combined_signal, updates=updates, state_response=state_response,
            aggregate_ood=aggregate_ood, invalid_actions=invalid_actions,
            resume_passed=resume_passed, seed_reproducible=_seed_reproducibility(base_config),
            gates=pilot_config["gates"],
        )
        passed = all(gates.values())
        diagnostics[crop] = {
            "status": "passed" if passed else "failed", "training_steps": total_steps,
            "updates": 2, "signal": combined_signal, "update_metrics": updates,
            "ood_summary": aggregate_ood,
            "state_response": state_response, "invalid_action_count": invalid_actions,
            "checkpoint_resume_passed": resume_passed,
            "seed_reproducible": gates["seed_reproducibility"],
            "checkpoint": _record(checkpoint_path, root), "gates": gates,
        }
        curves[crop] = {
            "steps": [len(initial_buffer), total_steps],
            "reward_mean": [initial_signal["reward_mean"], resume_signal["reward_mean"]],
            "raw_advantage_std": [initial_signal["raw_advantage_std"], resume_signal["raw_advantage_std"]],
            "projection_rate": [item["projection_rate"] for item in intervention],
            "fallback_rate": [item["fallback_rate"] for item in intervention],
            "warning_ood_rate": [item["warning_ood_rate"] for item in intervention],
            "severe_ood_rate": [item["severe_ood_rate"] for item in intervention],
            "in_support_rate": [item["in_support_rate"] for item in intervention],
            "support_restriction_rate": [
                _support_restriction_rate(buffer) for buffer in all_buffers
            ],
            "approx_kl": [item["approx_kl"] for item in updates],
            "entropy": [item["entropy"] for item in updates],
            "gradient_norm": [item["gradient_norm"] for item in updates],
        }
        decisions[crop] = {
            "decision": "continue_to_step20_hpo" if passed else "implementation_investigation_required",
            "trainable": passed,
            "baseline_comparison_performed": False,
            "reward_performance_gate_applied": False,
            "failed_gates": [name for name, value in gates.items() if not value],
        }

    all_passed = all(record["trainable"] for record in decisions.values())
    diagnostics_path = destination / "pilot_learning_diagnostics.json"
    curves_path = destination / "ood_intervention_curves.json"
    decision_path = destination / "trainability_decision.json"
    _write(diagnostics_path, {"schema_version": STEP19_VERSION, "status": "passed" if all_passed else "failed", "crops": diagnostics})
    _write(curves_path, {"schema_version": STEP19_VERSION, "status": "passed" if all_passed else "failed", "crops": curves})
    _write(decision_path, {
        "schema_version": STEP19_VERSION, "status": "passed" if all_passed else "failed",
        "overall_decision": "continue_to_step20_hpo" if all_passed else "implementation_investigation_required",
        "scope": "trainability_only_no_baseline_performance_judgment",
        "crops": decisions, "test_accessed": False, "hpo_started": False,
    })
    integrity = {
        "schema_version": STEP19_VERSION, "step": "19",
        "status": "passed" if all_passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "ppo_trainability_pilot_only_step20_not_implemented",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"},
        "step18_integrity": _record(step18_path, root), "pilot_config": _record(config_path, root),
        "implementation": _record(root / "src/geas35/experiments/rl/step19_pilot.py", root),
        "artifacts": {
            "diagnostics": _record(diagnostics_path, root),
            "ood_intervention_curves": _record(curves_path, root),
            "trainability_decision": _record(decision_path, root),
        },
        "checks": {
            "all_crop_updates_finite": all_passed,
            "reward_and_advantage_signal": all_passed,
            "kl_entropy_gradient_sane": all_passed,
            "state_dependent_action_and_value": all_passed,
            "support_layer_not_blocking_or_inactive": all_passed,
            "checkpoint_resume_and_seed_reproducibility": all_passed,
            "baseline_performance_gate_used": False,
            "test_accessed": False, "hpo_started": False, "step20_implemented": False,
        },
    }
    integrity_path = destination / "step19_integrity_manifest.json"
    _write(integrity_path, integrity)
    if not all_passed:
        failures = {crop: record["failed_gates"] for crop, record in decisions.items() if record["failed_gates"]}
        raise RuntimeError(f"Step 19 trainability gates failed: {failures}")
    return integrity_path


__all__ = ["STEP19_VERSION", "run_step19_trainability_pilot"]
