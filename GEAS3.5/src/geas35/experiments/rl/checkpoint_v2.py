"""Device-portable PPO checkpoint loading for the v2 RL protocol."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from geas35.rl.hybrid_ppo import HybridPPOTrainer, load_ppo_checkpoint


def load_ppo_checkpoint_v2(
    path: str | Path,
    *,
    expected_artifact_hashes: Mapping[str, str] | None = None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> tuple[HybridPPOTrainer, dict[str, Any]]:
    """Load on the requested device while restoring RNG from CPU ByteTensors."""
    trainer, payload = load_ppo_checkpoint(
        path,
        expected_artifact_hashes=expected_artifact_hashes,
        device=device,
        restore_rng=False,
    )
    if restore_rng:
        state = payload["rng_state"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].detach().to(device="cpu", dtype=torch.uint8))
        if torch.cuda.is_available() and state["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(
                [item.detach().to(device="cpu", dtype=torch.uint8) for item in state["torch_cuda"]]
            )
    return trainer, payload


__all__ = ["load_ppo_checkpoint_v2"]
