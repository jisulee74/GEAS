"""Composition root for a validated deployment bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from geas35.realtime.artifacts import load_deployment_bundle
from geas35.realtime.engine import OnlineInferenceEngine
from geas35.realtime.policy import PPOCheckpointPolicy
from geas35.realtime.preprocessing import (
    ArtifactQualityProcessor, CausalPreprocessor, load_observation_scaler,
)
from geas35.rl import MDP_V1_ACTION_COLUMNS, MdpV1Config
from geas35.rl.support_v1 import StateActionSupportModel


def _checkpoint_hash_contract(bundle) -> dict[str, str]:
    raw = bundle.metadata.get("checkpoint_artifact_hashes")
    if not isinstance(raw, dict):
        raise ValueError("Deployment bundle metadata requires checkpoint_artifact_hashes")
    expected = {
        "observation_scaler": bundle.artifact_hashes["observation_scaler"],
        "transition_model": bundle.artifact_hashes["transition_model"],
        "support": bundle.artifact_hashes["state_action_support"],
    }
    for key, value in expected.items():
        if str(raw.get(key, "")) != value:
            raise ValueError(f"Checkpoint provenance differs from deployment bundle: {key}")
    if not raw.get("environment"):
        raise ValueError("Checkpoint provenance requires an environment hash")
    return {str(key): str(value) for key, value in raw.items()}


def build_online_inference_engine(
    manifest_path: str | Path, *, mode: str, sink,
    crop_cycles: str | Path | None = None, device: str = "cpu",
    live_enabled: bool = False, safe_fallback: Mapping[str, float] | None = None,
    stale_after_seconds: float = 600.0, max_gap_minutes: float = 10.0,
    mdp_config: MdpV1Config | None = None,
) -> OnlineInferenceEngine:
    bundle = load_deployment_bundle(manifest_path, require_policy=True)
    checkpoint_hashes = _checkpoint_hash_contract(bundle)
    fallback = dict(safe_fallback or bundle.metadata.get("safe_fallback") or {})
    if set(fallback) != set(MDP_V1_ACTION_COLUMNS):
        raise ValueError("A complete six-action safe_fallback is required")
    quality = ArtifactQualityProcessor(
        bundle.path("quality_model"),
        required_columns=("in_temp_representative", "in_hum_representative", "in_co2_representative"),
    )
    preprocessor = CausalPreprocessor(
        quality, crop_cycles=crop_cycles, mdp_config=mdp_config,
    )
    scaler = load_observation_scaler(bundle.path("observation_scaler"))
    support = StateActionSupportModel(bundle.path("state_action_support"))
    policy = PPOCheckpointPolicy(
        bundle.path("ppo_policy"), expected_artifact_hashes=checkpoint_hashes, device=device,
    )
    return OnlineInferenceEngine(
        mode=mode, preprocessor=preprocessor, scaler=scaler, policy=policy,
        support=support, sink=sink, artifact_hashes=bundle.artifact_hashes,
        safe_fallback=fallback, mdp_config=mdp_config,
        stale_after_seconds=stale_after_seconds, max_gap_minutes=max_gap_minutes,
        live_enabled=live_enabled,
    )


__all__ = ["build_online_inference_engine"]
