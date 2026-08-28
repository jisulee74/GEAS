#!/usr/bin/env python3
"""Generic Step 18 Hybrid PPO training CLI; Step 19 protocol is separate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.rl.hybrid_ppo import (
    HybridActorCritic, HybridPPOConfig, HybridPPOTrainer, collect_rollout,
    load_ppo_checkpoint, save_ppo_checkpoint, seed_everything,
)
from geas35.rl.model_driven_env import load_model_driven_env_from_step15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(crop: str) -> dict[str, str]:
    protocol = json.loads((
        PROJECT_ROOT / "experiments/rl_policy_training/artifacts/step15/rl_protocol_manifest.json"
    ).read_text())
    record = protocol["crops"][crop]
    return {
        "observation_scaler": record["observation_scaler"]["sha256"],
        "environment": _sha256(PROJECT_ROOT / "src/geas35/rl/model_driven_env.py"),
        "transition_model": record["candidate_model"]["sha256"],
        "support": record["state_action_support"]["sha256"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the GEAS support-aware Hybrid PPO policy.")
    parser.add_argument("--crop", required=True, choices=("strawberry", "melon", "cucumber"))
    parser.add_argument("--total-steps", required=True, type=int)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--metrics-output")
    args = parser.parse_args(argv)
    if args.total_steps <= 0 or args.rollout_steps <= 0:
        parser.error("--total-steps and --rollout-steps must be positive")
    hashes = _artifact_hashes(args.crop)
    seed_everything(args.seed)
    env = load_model_driven_env_from_step15(
        project_root=PROJECT_ROOT, crop=args.crop, split="train"
    )
    if args.resume:
        trainer, payload = load_ppo_checkpoint(
            args.resume, expected_artifact_hashes=hashes, restore_rng=True,
        )
        completed = int(payload.get("extra_state", {}).get("training_steps", 0))
    else:
        config = HybridPPOConfig(observation_dim=env.observation_shape[0], seed=args.seed)
        trainer = HybridPPOTrainer(HybridActorCritic(config), config)
        completed = 0
    history = []
    while completed < args.total_steps:
        count = min(args.rollout_steps, args.total_steps - completed)
        buffer = collect_rollout(env, trainer, steps=count, seed=args.seed + trainer.update_count)
        metrics = trainer.update(buffer)
        completed += count
        history.append({"training_steps": completed, **metrics})
    checkpoint = save_ppo_checkpoint(
        args.checkpoint, trainer, artifact_hashes=hashes,
        extra_state={"crop": args.crop, "training_steps": completed},
    )
    metrics_path = Path(args.metrics_output) if args.metrics_output else checkpoint.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({
        "schema_version": "geas35.rl.hybrid_ppo.training_metrics.v1",
        "crop": args.crop, "training_steps": completed, "history": history,
        "test_accessed": False,
    }, indent=2) + "\n")
    print(f"Checkpoint: {checkpoint}")
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
