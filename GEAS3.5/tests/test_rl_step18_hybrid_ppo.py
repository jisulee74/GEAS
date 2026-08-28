from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.rl.hybrid_ppo import (
    FeasibleActionSpec,
    HybridActorCritic,
    HybridPPOConfig,
    HybridPPOTrainer,
    RolloutBuffer,
    SupportAwareActionAdapter,
    collect_rollout,
    compute_gae,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
    seed_everything,
)
from geas35.rl.model_driven_env import load_model_driven_env_from_step15


def _config(**overrides) -> HybridPPOConfig:
    values = {
        "observation_dim": 59, "hidden_sizes": (32, 16),
        "update_epochs": 2, "minibatch_size": 8, "seed": 17,
    }
    values.update(overrides)
    return HybridPPOConfig(**values)


def test_hybrid_distribution_respects_bounds_masks_and_log_prob_parity() -> None:
    seed_everything(17)
    policy = HybridActorCritic(_config())
    observations = torch.randn(32, 59)
    spec = FeasibleActionSpec(
        continuous_low=np.array([0.2, 0.0, 0.7], dtype=np.float32),
        continuous_high=np.array([0.4, 1.0, 0.7], dtype=np.float32),
        binary_allow_zero=np.array([True, False, True]),
        binary_allow_one=np.array([False, True, True]),
    )
    feasible = spec.batched(len(observations))
    sampled = policy.act(observations, feasible)
    actions = sampled["action"]
    assert torch.all(actions[:, :3] >= torch.tensor(spec.continuous_low))
    assert torch.all(actions[:, :3] <= torch.tensor(spec.continuous_high))
    assert torch.all(actions[:, 3] == 0.0)
    assert torch.all(actions[:, 4] == 1.0)
    assert torch.all(torch.isin(actions[:, 5], torch.tensor([0.0, 1.0])))
    evaluated = policy.evaluate_actions(observations, actions, feasible)
    torch.testing.assert_close(evaluated["log_prob"], sampled["log_prob"])
    assert torch.isfinite(evaluated["entropy"]).all()
    deterministic_one = policy.act(observations, feasible, deterministic=True)
    deterministic_two = policy.act(observations, feasible, deterministic=True)
    torch.testing.assert_close(deterministic_one["action"], deterministic_two["action"])
    torch.testing.assert_close(deterministic_one["value"], deterministic_two["value"])


def test_gae_termination_bootstrap_and_ood_reward_propagation() -> None:
    rewards = np.array([1.0, -0.4, -3.0], dtype=np.float32)
    values = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    next_values = np.array([0.3, 0.4, 99.0], dtype=np.float32)
    advantages, returns = compute_gae(
        rewards, values, next_values, np.array([False, False, True]),
        gamma=0.9, gae_lambda=0.8,
    )
    expected_last = rewards[-1] - values[-1]
    assert advantages[-1] == pytest.approx(expected_last)
    assert returns[-1] == pytest.approx(rewards[-1])
    assert advantages[1] < 0.0
    truncated_advantage, _ = compute_gae(
        np.array([0.0]), np.array([0.0]), np.array([2.0]), np.array([False]),
        gamma=0.9, gae_lambda=0.8,
    )
    assert truncated_advantage[0] == pytest.approx(1.8)


def test_ppo_update_is_finite_and_changes_parameters() -> None:
    seed_everything(17)
    config = _config(target_kl=None)
    policy = HybridActorCritic(config)
    trainer = HybridPPOTrainer(policy, config)
    spec = FeasibleActionSpec.unconstrained()
    buffer = RolloutBuffer()
    for index in range(24):
        observation = np.random.default_rng(index).normal(size=59).astype(np.float32)
        tensor = torch.as_tensor(observation).unsqueeze(0)
        with torch.no_grad():
            output = policy.act(tensor, spec.batched(1))
            next_value = float(policy(torch.as_tensor(observation * 0.9).unsqueeze(0))[-1][0])
        action = output["action"][0].numpy()
        penalty = 0.2 if index % 5 == 0 else 0.0
        buffer.add(
            observations=observation, policy_actions=action,
            executed_actions=np.clip(action + 0.01, 0.0, 1.0),
            log_probs=float(output["log_prob"][0]), values=float(output["value"][0]),
            next_values=next_value, rewards=float(0.5 - penalty),
            terminated=index == 23, truncated=False, feasible_specs=spec,
            warning_ood_penalties=penalty, projected=index % 3 == 0, fallback=False,
        )
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = trainer.update(buffer)
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))
    assert metrics["projection_rate"] > 0.0
    assert metrics["mean_warning_ood_penalty"] > 0.0


def test_target_kl_stops_update_and_ratio_is_clipped() -> None:
    seed_everything(17)
    config = _config(target_kl=1e-4, update_epochs=5)
    policy = HybridActorCritic(config)
    trainer = HybridPPOTrainer(policy, config)
    spec = FeasibleActionSpec.unconstrained()
    buffer = RolloutBuffer()
    for index in range(8):
        observation = np.full(59, index / 10.0, dtype=np.float32)
        with torch.no_grad():
            output = policy.act(torch.as_tensor(observation).unsqueeze(0), spec.batched(1))
        action = output["action"][0].numpy()
        buffer.add(
            observations=observation, policy_actions=action, executed_actions=action,
            log_probs=float(output["log_prob"][0]) + 2.0,
            values=float(output["value"][0]), next_values=0.0, rewards=1.0,
            terminated=index == 7, truncated=False, feasible_specs=spec,
            warning_ood_penalties=0.0, projected=False, fallback=False,
        )
    metrics = trainer.update(buffer)
    assert metrics["target_kl_early_stop"] is True
    assert metrics["epochs_completed"] == 1
    assert metrics["clip_fraction"] > 0.0


def test_checkpoint_roundtrip_preserves_action_value_and_hash_contract(tmp_path: Path) -> None:
    seed_everything(17)
    config = _config()
    trainer = HybridPPOTrainer(HybridActorCritic(config), config)
    observation = torch.randn(1, 59)
    feasible = FeasibleActionSpec.unconstrained().batched(1)
    with torch.no_grad():
        expected = trainer.policy.act(observation, feasible, deterministic=True)
    hashes = {
        "observation_scaler": "scaler", "environment": "environment",
        "transition_model": "transition", "support": "support",
    }
    path = save_ppo_checkpoint(tmp_path / "policy.pt", trainer, artifact_hashes=hashes)
    restored, payload = load_ppo_checkpoint(
        path, expected_artifact_hashes=hashes, restore_rng=False
    )
    with torch.no_grad():
        actual = restored.policy.act(observation, feasible, deterministic=True)
    torch.testing.assert_close(actual["action"], expected["action"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual["value"], expected["value"], rtol=0.0, atol=0.0)
    assert payload["artifact_hashes"] == hashes
    assert payload["scheduler_state"] is not None
    with pytest.raises(ValueError, match="artifact hashes"):
        load_ppo_checkpoint(path, expected_artifact_hashes={**hashes, "support": "wrong"})


@pytest.fixture(scope="module")
def melon_env():
    return load_model_driven_env_from_step15(
        project_root=PROJECT_ROOT, crop="melon", split="train"
    )


def test_real_support_adapter_and_rollout_buffer_capture_interventions(melon_env) -> None:
    observation, _ = melon_env.reset(seed=17, options={"episode_index": 0})
    adapter = SupportAwareActionAdapter(melon_env)
    spec = adapter.feasible_spec()
    policy = HybridActorCritic(_config(observation_dim=len(observation)))
    output = policy.act(
        torch.as_tensor(observation).unsqueeze(0), spec.batched(1), deterministic=False
    )
    action = output["action"][0].detach().numpy()
    assert np.all(action[:3] >= spec.continuous_low)
    assert np.all(action[:3] <= spec.continuous_high)
    for index, value in enumerate(action[3:]):
        assert (value == 0.0 and spec.binary_allow_zero[index]) or (
            value == 1.0 and spec.binary_allow_one[index]
        )
    trainer = HybridPPOTrainer(policy, policy.config)
    buffer = collect_rollout(melon_env, trainer, steps=3, seed=17)
    assert len(buffer) == 3
    assert np.asarray(buffer.policy_actions).shape == (3, 6)
    assert np.asarray(buffer.executed_actions).shape == (3, 6)
    assert np.isfinite(buffer.rewards).all()


def test_official_step18_artifacts_are_frozen_and_step19_is_absent() -> None:
    import hashlib
    import json

    root = PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step18"
    integrity = json.loads((root / "step18_integrity_manifest.json").read_text())
    smoke = json.loads((root / "checkpoint_smoke_report.json").read_text())
    schema = json.loads((root / "hybrid_action_schema.json").read_text())
    assert integrity["status"] == smoke["status"] == "passed"
    assert integrity["checks"]["step19_implemented"] is False
    assert integrity["checks"]["test_accessed"] is False
    assert schema["joint_log_probability"] == "sum_of_six_component_log_probabilities"
    assert len(schema["continuous"]["columns"]) == len(schema["binary"]["columns"]) == 3
    for record in [*integrity["implementations"].values(), *integrity["artifacts"].values()]:
        path = PROJECT_ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    for crop in ("strawberry", "melon", "cucumber"):
        assert smoke["crops"][crop]["deterministic_action_equal"] is True
        assert smoke["crops"][crop]["value_equal"] is True
        assert smoke["crops"][crop]["training_steps"] == 0
