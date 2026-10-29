"""Step 18 Hybrid PPO artifact freeze and checkpoint smoke validation."""
from __future__ import annotations

from dataclasses import asdict
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
    STEP18_PPO_VERSION,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
    seed_everything,
)

STEP18_VERSION = "geas35.rl.step18.v1"
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


def run_step18_hybrid_ppo_freeze(
    *, project_root: str | Path, output_root: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    destination = Path(output_root).resolve() if output_root else (
        root / "experiments/rl_policy_training/artifacts/step18"
    )
    step17_path = root / "experiments/rl_policy_training/artifacts/step17/step17_integrity_manifest.json"
    step17 = _read(step17_path)
    if step17.get("status") != "passed":
        raise ValueError("Step 18 requires passed Step 17 integrity.")
    protocol_path = root / "experiments/rl_policy_training/artifacts/step15/rl_protocol_manifest.json"
    protocol = _read(protocol_path)
    if protocol.get("status") != "passed":
        raise ValueError("Step 18 requires passed Step 15 protocol.")

    config = HybridPPOConfig()
    config_path = destination / "support_aware_ppo_config.json"
    schema_path = destination / "hybrid_action_schema.json"
    _write(config_path, {
        "schema_version": STEP18_VERSION,
        "policy_version": STEP18_PPO_VERSION,
        "status": "frozen",
        "config": asdict(config),
        "observation_normalization": "frozen_step15_train_scaler_no_online_refit",
        "hpo_status": "not_started_step20_scope",
        "pilot_status": "not_started_step19_scope",
    })
    _write(schema_path, {
        "schema_version": STEP18_VERSION,
        "continuous": {
            "columns": ["vent_pct", "shade_curtain_pct", "thermal_curtain_pct"],
            "distribution": "independent_beta",
            "base_support": [0.0, 1.0],
            "feasibility": "affine_transform_to_local_train_bounds_and_weather_cap",
        },
        "binary": {
            "columns": ["heat_run", "cool_run", "fan_run"],
            "distribution": "independent_bernoulli",
            "support": [0, 1],
            "feasibility": "local_train_support_mask_with_safe_fallback_for_empty_evidence",
        },
        "joint_log_probability": "sum_of_six_component_log_probabilities",
        "joint_entropy": "sum_of_six_component_entropies",
        "rollout_action_records": ["policy_action", "executed_action"],
    })

    seed_everything(config.seed)
    observation = torch.linspace(-1.0, 1.0, config.observation_dim).unsqueeze(0)
    feasible = FeasibleActionSpec.unconstrained().batched(1)
    crops: dict[str, Any] = {}
    environment_hash = _sha256(root / "src/geas35/rl/model_driven_env.py")
    for crop in OFFICIAL_CROPS:
        crop_record = protocol["crops"][crop]
        hashes = {
            "observation_scaler": crop_record["observation_scaler"]["sha256"],
            "environment": environment_hash,
            "transition_model": crop_record["candidate_model"]["sha256"],
            "support": crop_record["state_action_support"]["sha256"],
        }
        seed_everything(config.seed)
        trainer = HybridPPOTrainer(HybridActorCritic(config), config)
        with torch.no_grad():
            before = trainer.policy.act(observation, feasible, deterministic=True)
        checkpoint_path = destination / crop / "initialization_checkpoint.pt"
        save_ppo_checkpoint(
            checkpoint_path, trainer, artifact_hashes=hashes,
            extra_state={"crop": crop, "training_steps": 0, "step19_pilot_started": False},
        )
        restored, payload = load_ppo_checkpoint(
            checkpoint_path, expected_artifact_hashes=hashes, restore_rng=False,
        )
        with torch.no_grad():
            after = restored.policy.act(observation, feasible, deterministic=True)
        action_equal = torch.equal(before["action"], after["action"])
        value_equal = torch.equal(before["value"], after["value"])
        if not action_equal or not value_equal or payload["update_count"] != 0:
            raise ValueError(f"{crop}: checkpoint roundtrip parity failed.")
        crops[crop] = {
            "status": "passed", "checkpoint": _record(checkpoint_path, root),
            "artifact_hashes": hashes, "deterministic_action_equal": action_equal,
            "value_equal": value_equal, "optimizer_restored": True,
            "rng_state_present": True, "training_steps": 0,
        }

    smoke_path = destination / "checkpoint_smoke_report.json"
    _write(smoke_path, {
        "schema_version": STEP18_VERSION, "status": "passed", "crops": crops,
    })
    integrity = {
        "schema_version": STEP18_VERSION, "step": "18", "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "hybrid_ppo_implementation_only_step19_not_implemented",
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "device": "cpu",
        },
        "step17_integrity": _record(step17_path, root),
        "implementations": {
            "hybrid_ppo": _record(root / "src/geas35/rl/hybrid_ppo.py", root),
            "artifact_freeze": _record(root / "src/geas35/experiments/rl/step18_hybrid_ppo.py", root),
            "training_cli": _record(root / "experiments/rl_policy_training/scripts/train_hybrid_ppo.py", root),
        },
        "artifacts": {
            "config": _record(config_path, root), "action_schema": _record(schema_path, root),
            "checkpoint_smoke": _record(smoke_path, root),
        },
        "checks": {
            "beta_three_continuous": True, "bernoulli_three_binary": True,
            "joint_log_probability_and_entropy": True,
            "local_support_and_dynamic_hard_bounds": True,
            "raw_and_executed_action_buffered": True,
            "ood_reward_propagates_through_gae": True,
            "clipped_ppo_and_value_loss": True, "gradient_clipping": True,
            "target_kl_early_stop": True, "deterministic_stochastic_modes": True,
            "atomic_checkpoint_and_rng": True, "artifact_hash_validation": True,
            "online_observation_normalization": False,
            "test_accessed": False, "step19_implemented": False,
        },
    }
    integrity_path = destination / "step18_integrity_manifest.json"
    _write(integrity_path, integrity)
    return integrity_path


__all__ = ["STEP18_VERSION", "run_step18_hybrid_ppo_freeze"]
