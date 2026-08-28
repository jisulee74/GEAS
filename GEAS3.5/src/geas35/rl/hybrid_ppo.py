"""Support-aware hybrid-action PPO primitives for GEAS RL Step 18."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Beta

from geas35.rl.mdp_v1 import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_BINARY_ACTION_COLUMNS,
    MDP_V1_CONTINUOUS_ACTION_COLUMNS,
)

STEP18_PPO_VERSION = "geas35.rl.hybrid_ppo.step18.v1"


@dataclass(frozen=True)
class HybridPPOConfig:
    observation_dim: int = 59
    continuous_action_dim: int = 3
    binary_action_dim: int = 3
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    beta_concentration_floor: float = 1.01
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 64
    target_kl: float | None = 0.03
    normalize_advantages: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.continuous_action_dim != 3 or self.binary_action_dim != 3:
            raise ValueError("GEAS Hybrid PPO requires positive observations and 3+3 actions.")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values.")
        if self.activation not in ("tanh", "relu"):
            raise ValueError("activation must be tanh or relu.")
        if self.beta_concentration_floor <= 0.0:
            raise ValueError("beta_concentration_floor must be positive.")
        if self.minibatch_size <= 0 or self.update_epochs <= 0:
            raise ValueError("PPO batch and epoch settings must be positive.")


@dataclass(frozen=True)
class FeasibleActionSpec:
    continuous_low: np.ndarray
    continuous_high: np.ndarray
    binary_allow_zero: np.ndarray
    binary_allow_one: np.ndarray

    def __post_init__(self) -> None:
        low = np.asarray(self.continuous_low, dtype=np.float32)
        high = np.asarray(self.continuous_high, dtype=np.float32)
        allow_zero = np.asarray(self.binary_allow_zero, dtype=bool)
        allow_one = np.asarray(self.binary_allow_one, dtype=bool)
        if low.shape != (3,) or high.shape != (3,):
            raise ValueError("Continuous feasible bounds must have shape (3,).")
        if allow_zero.shape != (3,) or allow_one.shape != (3,):
            raise ValueError("Binary feasible masks must have shape (3,).")
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("Continuous feasible bounds must be finite.")
        if np.any(low < 0.0) or np.any(high > 1.0) or np.any(low > high):
            raise ValueError("Continuous feasible bounds must satisfy 0 <= low <= high <= 1.")
        if np.any(~(allow_zero | allow_one)):
            raise ValueError("Every binary action must allow at least one value.")

    @classmethod
    def unconstrained(cls) -> "FeasibleActionSpec":
        return cls(
            continuous_low=np.zeros(3, dtype=np.float32),
            continuous_high=np.ones(3, dtype=np.float32),
            binary_allow_zero=np.ones(3, dtype=bool),
            binary_allow_one=np.ones(3, dtype=bool),
        )

    def batched(self, batch_size: int, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        return {
            "continuous_low": torch.as_tensor(self.continuous_low, dtype=torch.float32, device=device).expand(batch_size, -1),
            "continuous_high": torch.as_tensor(self.continuous_high, dtype=torch.float32, device=device).expand(batch_size, -1),
            "binary_allow_zero": torch.as_tensor(self.binary_allow_zero, dtype=torch.bool, device=device).expand(batch_size, -1),
            "binary_allow_one": torch.as_tensor(self.binary_allow_one, dtype=torch.bool, device=device).expand(batch_size, -1),
        }


class SupportAwareActionAdapter:
    """Translate frozen local support and weather caps into policy masks/bounds."""

    def __init__(self, env: Any):
        self.env = env

    def feasible_spec(self) -> FeasibleActionSpec:
        state = self.env._current_scaled.loc[list(self.env.observation_columns)].to_frame().T
        local = self.env.support.local_action_support(state)[0]
        fallback = self.env._safe_fallback_action()
        low = []
        high = []
        for column in MDP_V1_CONTINUOUS_ACTION_COLUMNS:
            lower, upper = local["continuous"][column]
            low.append(float(lower))
            high.append(float(upper))
        rain = float(self.env._current_physical.get("obs_rain_flag", 0.0)) > 0.0
        wind = float(self.env._current_physical.get("obs_outdoor_wind_speed", 0.0))
        vent_cap = 1.0
        if rain:
            vent_cap = min(vent_cap, self.env.mdp_config.rain_vent_open_cap)
        if np.isfinite(wind) and wind > self.env.mdp_config.wind_cap_threshold:
            vent_cap = min(vent_cap, self.env.mdp_config.wind_vent_open_cap)
        high[0] = min(high[0], vent_cap)
        low[0] = min(low[0], high[0])
        allow_zero = []
        allow_one = []
        for column in MDP_V1_BINARY_ACTION_COLUMNS:
            allowed = set(int(value) for value in local["binary"][column])
            if not allowed:
                allowed = {int(float(fallback[column]) > 0.5)}
            allow_zero.append(0 in allowed)
            allow_one.append(1 in allowed)
        return FeasibleActionSpec(
            continuous_low=np.asarray(low, dtype=np.float32),
            continuous_high=np.asarray(high, dtype=np.float32),
            binary_allow_zero=np.asarray(allow_zero, dtype=bool),
            binary_allow_one=np.asarray(allow_one, dtype=bool),
        )


def _activation(name: str) -> type[nn.Module]:
    return nn.Tanh if name == "tanh" else nn.ReLU


class HybridActorCritic(nn.Module):
    """Shared encoder with Beta, Bernoulli, and scalar value heads."""

    def __init__(self, config: HybridPPOConfig):
        super().__init__()
        self.config = config
        activation = _activation(config.activation)
        layers: list[nn.Module] = []
        previous = config.observation_dim
        for size in config.hidden_sizes:
            layers.extend((nn.Linear(previous, size), activation()))
            previous = size
        self.encoder = nn.Sequential(*layers)
        self.continuous_alpha_head = nn.Linear(previous, config.continuous_action_dim)
        self.continuous_beta_head = nn.Linear(previous, config.continuous_action_dim)
        self.binary_logits_head = nn.Linear(previous, config.binary_action_dim)
        self.value_head = nn.Linear(previous, 1)
        self.softplus = nn.Softplus()

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.encoder(observations)
        floor = self.config.beta_concentration_floor
        alpha = self.softplus(self.continuous_alpha_head(features)) + floor
        beta = self.softplus(self.continuous_beta_head(features)) + floor
        logits = self.binary_logits_head(features)
        value = self.value_head(features).squeeze(-1)
        return alpha, beta, logits, value

    @staticmethod
    def _masked_binary_probability(
        logits: torch.Tensor,
        allow_zero: torch.Tensor,
        allow_one: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(logits)
        fixed_zero = allow_zero & ~allow_one
        fixed_one = ~allow_zero & allow_one
        free = allow_zero & allow_one
        probability = torch.where(fixed_zero, torch.zeros_like(probability), probability)
        probability = torch.where(fixed_one, torch.ones_like(probability), probability)
        probability = torch.where(free, probability, probability)
        return probability, free

    def act(
        self,
        observations: torch.Tensor,
        feasible: Mapping[str, torch.Tensor],
        *,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        alpha, beta, logits, value = self(observations)
        continuous_distribution = Beta(alpha, beta)
        unit_action = alpha / (alpha + beta) if deterministic else continuous_distribution.sample()
        low = feasible["continuous_low"]
        high = feasible["continuous_high"]
        scale = high - low
        continuous_action = low + scale * unit_action
        probability, binary_free = self._masked_binary_probability(
            logits, feasible["binary_allow_zero"], feasible["binary_allow_one"]
        )
        binary_action = (probability >= 0.5).float() if deterministic else Bernoulli(probs=probability).sample()
        action = torch.cat((continuous_action, binary_action), dim=-1)
        log_prob, entropy = self._log_prob_entropy(
            continuous_distribution, unit_action, scale, probability, binary_free, binary_action
        )
        return {"action": action, "log_prob": log_prob, "entropy": entropy, "value": value}

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        feasible: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        alpha, beta, logits, value = self(observations)
        distribution = Beta(alpha, beta)
        low = feasible["continuous_low"]
        scale = feasible["continuous_high"] - low
        variable = scale > torch.finfo(scale.dtype).eps
        unit = torch.where(variable, (actions[:, :3] - low) / torch.clamp(scale, min=1e-8), torch.full_like(scale, 0.5))
        unit = torch.clamp(unit, 1e-6, 1.0 - 1e-6)
        probability, binary_free = self._masked_binary_probability(
            logits, feasible["binary_allow_zero"], feasible["binary_allow_one"]
        )
        log_prob, entropy = self._log_prob_entropy(
            distribution, unit, scale, probability, binary_free, actions[:, 3:]
        )
        return {"log_prob": log_prob, "entropy": entropy, "value": value}

    @staticmethod
    def _log_prob_entropy(
        continuous_distribution: Beta,
        unit_action: torch.Tensor,
        scale: torch.Tensor,
        binary_probability: torch.Tensor,
        binary_free: torch.Tensor,
        binary_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        variable = scale > torch.finfo(scale.dtype).eps
        continuous_log_prob = torch.where(
            variable,
            continuous_distribution.log_prob(torch.clamp(unit_action, 1e-6, 1.0 - 1e-6))
            - torch.log(torch.clamp(scale, min=1e-8)),
            torch.zeros_like(scale),
        ).sum(dim=-1)
        continuous_entropy = torch.where(
            variable,
            continuous_distribution.entropy() + torch.log(torch.clamp(scale, min=1e-8)),
            torch.zeros_like(scale),
        ).sum(dim=-1)
        safe_probability = torch.clamp(binary_probability, 1e-6, 1.0 - 1e-6)
        binary_distribution = Bernoulli(probs=safe_probability)
        binary_log_prob = torch.where(
            binary_free, binary_distribution.log_prob(binary_action), torch.zeros_like(binary_action)
        ).sum(dim=-1)
        binary_entropy = torch.where(
            binary_free, binary_distribution.entropy(), torch.zeros_like(binary_probability)
        ).sum(dim=-1)
        return continuous_log_prob + binary_log_prob, continuous_entropy + binary_entropy


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE; truncations bootstrap while true terminations do not."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    if not (rewards.shape == values.shape == next_values.shape == terminated.shape):
        raise ValueError("GAE arrays must have identical shapes.")
    advantages = np.zeros_like(rewards)
    accumulator = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if terminated[index] else 1.0
        delta = rewards[index] + gamma * next_values[index] * nonterminal - values[index]
        accumulator = delta + gamma * gae_lambda * nonterminal * accumulator
        advantages[index] = accumulator
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    policy_actions: list[np.ndarray] = field(default_factory=list)
    executed_actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    next_values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    feasible_specs: list[FeasibleActionSpec] = field(default_factory=list)
    warning_ood_penalties: list[float] = field(default_factory=list)
    projected: list[bool] = field(default_factory=list)
    fallback: list[bool] = field(default_factory=list)
    state_ood_scores: list[float] = field(default_factory=list)
    joint_ood_scores: list[float] = field(default_factory=list)
    joint_ood_levels: list[str] = field(default_factory=list)
    termination_reasons: list[str | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rewards)

    def add(self, **transition: Any) -> None:
        optional = {
            "state_ood_scores": float("nan"), "joint_ood_scores": float("nan"),
            "joint_ood_levels": "unknown", "termination_reasons": None,
        }
        for key, default in optional.items():
            transition.setdefault(key, default)
        for key, value in transition.items():
            getattr(self, key).append(value)
        lengths = {len(getattr(self, field_name)) for field_name in self.__dataclass_fields__}
        if len(lengths) != 1:
            raise ValueError("RolloutBuffer transition fields are incomplete.")

    def arrays(self, config: HybridPPOConfig) -> dict[str, np.ndarray]:
        advantages, returns = compute_gae(
            np.asarray(self.rewards), np.asarray(self.values), np.asarray(self.next_values),
            np.asarray(self.terminated), gamma=config.gamma, gae_lambda=config.gae_lambda,
        )
        if config.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "observations": np.asarray(self.observations, dtype=np.float32),
            "actions": np.asarray(self.policy_actions, dtype=np.float32),
            "old_log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "old_values": np.asarray(self.values, dtype=np.float32),
            "advantages": advantages.astype(np.float32),
            "returns": returns.astype(np.float32),
            "continuous_low": np.asarray([spec.continuous_low for spec in self.feasible_specs], dtype=np.float32),
            "continuous_high": np.asarray([spec.continuous_high for spec in self.feasible_specs], dtype=np.float32),
            "binary_allow_zero": np.asarray([spec.binary_allow_zero for spec in self.feasible_specs], dtype=bool),
            "binary_allow_one": np.asarray([spec.binary_allow_one for spec in self.feasible_specs], dtype=bool),
        }

    def intervention_metrics(self) -> dict[str, float]:
        count = max(len(self), 1)
        known_levels = [level for level in self.joint_ood_levels if level != "unknown"]
        return {
            "projection_rate": float(sum(self.projected) / count),
            "fallback_rate": float(sum(self.fallback) / count),
            "mean_warning_ood_penalty": float(np.mean(self.warning_ood_penalties)) if self.warning_ood_penalties else 0.0,
            "policy_executed_action_l1": float(np.mean(np.abs(
                np.asarray(self.policy_actions) - np.asarray(self.executed_actions)
            ))) if self.policy_actions else 0.0,
            "in_support_rate": float(sum(level == "in_support" for level in known_levels) / len(known_levels)) if known_levels else 0.0,
            "warning_ood_rate": float(sum(level == "warning" for level in known_levels) / len(known_levels)) if known_levels else 0.0,
            "severe_ood_rate": float(sum(level == "severe" for level in known_levels) / len(known_levels)) if known_levels else 0.0,
            "severe_state_termination_rate": float(sum(reason == "severe_state_ood" for reason in self.termination_reasons) / count),
        }


class HybridPPOTrainer:
    def __init__(self, policy: HybridActorCritic, config: HybridPPOConfig, *, device: str = "cpu"):
        self.policy = policy.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.scheduler: Any | None = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda _: 1.0
        )
        self.update_count = 0

    def update(self, buffer: RolloutBuffer) -> dict[str, float | int | bool]:
        if len(buffer) == 0:
            raise ValueError("Cannot update PPO from an empty rollout buffer.")
        arrays = buffer.arrays(self.config)
        parameters_before = torch.cat([
            parameter.detach().flatten().cpu() for parameter in self.policy.parameters()
        ])
        tensors = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in arrays.items()
        }
        batch_size = len(buffer)
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + self.update_count)
        metrics = {key: [] for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "gradient_norm", "ratio_mean")}
        early_stopped = False
        epochs_completed = 0
        for epoch in range(self.config.update_epochs):
            permutation = torch.randperm(batch_size, generator=generator)
            for start in range(0, batch_size, self.config.minibatch_size):
                indices = permutation[start:start + self.config.minibatch_size].to(self.device)
                feasible = {key: tensors[key][indices] for key in (
                    "continuous_low", "continuous_high", "binary_allow_zero", "binary_allow_one"
                )}
                evaluated = self.policy.evaluate_actions(
                    tensors["observations"][indices], tensors["actions"][indices], feasible
                )
                log_ratio = evaluated["log_prob"] - tensors["old_log_probs"][indices]
                ratio = torch.exp(log_ratio)
                advantages = tensors["advantages"][indices]
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
                ) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value = evaluated["value"]
                old_value = tensors["old_values"][indices]
                returns = tensors["returns"][indices]
                clipped_value = old_value + torch.clamp(
                    value - old_value,
                    -self.config.value_clip_range,
                    self.config.value_clip_range,
                )
                value_loss = 0.5 * torch.maximum(
                    (value - returns).square(), (clipped_value - returns).square()
                ).mean()
                entropy = evaluated["entropy"].mean()
                loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss is NaN or Inf.")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_gradient_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.config.clip_range).float().mean()
                for key, value_metric in (
                    ("policy_loss", policy_loss), ("value_loss", value_loss),
                    ("entropy", entropy), ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction), ("gradient_norm", gradient_norm),
                    ("ratio_mean", ratio.mean()),
                ):
                    metrics[key].append(float(value_metric.detach().cpu()))
                if self.config.target_kl is not None and float(approx_kl) > self.config.target_kl:
                    early_stopped = True
                    break
            epochs_completed = epoch + 1
            if early_stopped:
                break
        if self.scheduler is not None:
            self.scheduler.step()
        self.update_count += 1
        result: dict[str, float | int | bool] = {
            key: float(np.mean(values)) for key, values in metrics.items()
        }
        with torch.no_grad():
            feasible_all = {key: tensors[key] for key in (
                "continuous_low", "continuous_high", "binary_allow_zero", "binary_allow_one"
            )}
            predicted_values = self.policy.evaluate_actions(
                tensors["observations"], tensors["actions"], feasible_all
            )["value"].detach().cpu().numpy()
        returns = arrays["returns"]
        return_variance = float(np.var(returns))
        result["explained_variance"] = (
            float(1.0 - np.var(returns - predicted_values) / return_variance)
            if return_variance > 1e-12 else 0.0
        )
        parameters_after = torch.cat([
            parameter.detach().flatten().cpu() for parameter in self.policy.parameters()
        ])
        result["parameter_update_l2"] = float(torch.linalg.vector_norm(parameters_after - parameters_before))
        result.update({"epochs_completed": epochs_completed, "target_kl_early_stop": early_stopped})
        result.update(buffer.intervention_metrics())
        return result


def collect_rollout(
    env: Any,
    trainer: HybridPPOTrainer,
    *,
    steps: int,
    seed: int,
) -> RolloutBuffer:
    """Collect support-aware transitions; intended for Step 19+ callers."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    observation, _ = env.reset(seed=seed)
    adapter = SupportAwareActionAdapter(env)
    buffer = RolloutBuffer()
    for _ in range(steps):
        spec = adapter.feasible_spec()
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=trainer.device).unsqueeze(0)
        feasible = spec.batched(1, device=trainer.device)
        with torch.no_grad():
            output = trainer.policy.act(obs_tensor, feasible, deterministic=False)
        policy_action = output["action"][0].cpu().numpy().astype(np.float32)
        next_observation, reward, terminated, truncated, info = env.step(policy_action)
        next_tensor = torch.as_tensor(next_observation, dtype=torch.float32, device=trainer.device).unsqueeze(0)
        with torch.no_grad():
            next_value = float(trainer.policy(next_tensor)[-1][0].cpu())
        executed = info.get("executed_action")
        executed_array = np.asarray(
            [executed[column] for column in MDP_V1_ACTION_COLUMNS], dtype=np.float32
        ) if executed is not None else policy_action.copy()
        buffer.add(
            observations=observation.copy(), policy_actions=policy_action,
            executed_actions=executed_array, log_probs=float(output["log_prob"][0].cpu()),
            values=float(output["value"][0].cpu()), next_values=next_value,
            rewards=float(reward), terminated=bool(terminated), truncated=bool(truncated),
            feasible_specs=spec,
            warning_ood_penalties=float(info.get("warning_ood_penalty", 0.0)),
            projected=bool(info.get("support_projection", {}).get("changes", {})),
            fallback=bool(info.get("safety_fallback_applied", False)),
            state_ood_scores=float(info.get("state_ood_score", float("nan"))),
            joint_ood_scores=float(info.get("joint_ood_score", float("nan"))),
            joint_ood_levels=str(info.get("joint_ood_level", "unknown")),
            termination_reasons=info.get("termination_reason"),
        )
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
            adapter = SupportAwareActionAdapter(env)
    return buffer


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_ppo_checkpoint(
    path: str | Path,
    trainer: HybridPPOTrainer,
    *,
    artifact_hashes: Mapping[str, str],
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    required = {"observation_scaler", "environment", "transition_model", "support"}
    missing = sorted(required - set(artifact_hashes))
    if missing:
        raise ValueError(f"Checkpoint is missing artifact hashes: {missing}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    payload = {
        "schema_version": STEP18_PPO_VERSION,
        "config": asdict(trainer.config),
        "policy_state": trainer.policy.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "scheduler_state": None if trainer.scheduler is None else trainer.scheduler.state_dict(),
        "update_count": trainer.update_count,
        "rng_state": _rng_state(),
        "artifact_hashes": dict(artifact_hashes),
        "extra_state": dict(extra_state or {}),
    }
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def load_ppo_checkpoint(
    path: str | Path,
    *,
    expected_artifact_hashes: Mapping[str, str] | None = None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> tuple[HybridPPOTrainer, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != STEP18_PPO_VERSION:
        raise ValueError("Unsupported PPO checkpoint schema.")
    if expected_artifact_hashes is not None and dict(expected_artifact_hashes) != payload["artifact_hashes"]:
        raise ValueError("Checkpoint artifact hashes differ from the requested environment.")
    config_payload = dict(payload["config"])
    config_payload["hidden_sizes"] = tuple(config_payload["hidden_sizes"])
    config = HybridPPOConfig(**config_payload)
    policy = HybridActorCritic(config)
    trainer = HybridPPOTrainer(policy, config, device=device)
    trainer.policy.load_state_dict(payload["policy_state"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.update_count = int(payload["update_count"])
    if trainer.scheduler is not None and payload["scheduler_state"] is not None:
        trainer.scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng:
        state = payload["rng_state"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    return trainer, payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "FeasibleActionSpec", "HybridActorCritic", "HybridPPOConfig", "HybridPPOTrainer",
    "RolloutBuffer", "STEP18_PPO_VERSION", "SupportAwareActionAdapter", "collect_rollout",
    "compute_gae", "load_ppo_checkpoint", "save_ppo_checkpoint", "seed_everything",
]
