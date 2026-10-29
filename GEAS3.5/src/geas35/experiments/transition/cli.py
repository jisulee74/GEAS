"""Command-line entry point for transition model experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from geas35.experiments.transition.config import (
    LoadedTransitionExperimentConfig,
    load_transition_experiment_config,
    read_configured_transition_frames,
)
from geas35.experiments.transition.runner import (
    TransitionExperimentResult,
    run_transition_model_experiment,
)
from geas35.models.transition.causal_state import load_frozen_observation_scaler
from geas35.models.transition.feature_variants import (
    feature_variant_contract, observation_columns_for_variant,
    prepare_frame_for_feature_variant, project_frozen_observation_scaler,
)
from geas35.rl import MDP_V1_ACTION_COLUMNS
from geas35.io_utils import write_json
import hashlib
import json
from geas35.models.transition.registry import TransitionModelFactory


@dataclass(frozen=True)
class TransitionEndToEndExperimentResult:
    """Step 10 transition experiment result loaded from YAML."""

    loaded_config: LoadedTransitionExperimentConfig
    experiment_result: TransitionExperimentResult

    @property
    def output_root(self) -> str:
        return self.experiment_result.output_root

    @property
    def experiment_summary_path(self) -> str:
        return self.experiment_result.experiment_summary_path


def run_from_config(
    config_path: str | Path,
    *,
    registry: Mapping[str, TransitionModelFactory] | None = None,
) -> TransitionEndToEndExperimentResult:
    """Run a transition experiment from a YAML config file."""

    loaded = load_transition_experiment_config(config_path)
    train_df, validation_df, test_df, rollout_df = read_configured_transition_frames(
        loaded, include_test=False
    )
    variant = loaded.experiment_config.feature_variant
    if variant is None:
        scaler, scaler_hash = load_frozen_observation_scaler(loaded.scaler_path)
    else:
        train_df = prepare_frame_for_feature_variant(train_df, variant)
        validation_df = prepare_frame_for_feature_variant(validation_df, variant)
        rollout_df = None if rollout_df is None else prepare_frame_for_feature_variant(rollout_df, variant)
        source_scaler, _ = load_frozen_observation_scaler(loaded.scaler_path)
        scaler = project_frozen_observation_scaler(
            source_scaler, observation_columns_for_variant(variant)
        )
        scaler_payload = {
            "stage": "transition_observation_scaler", "fit_split": "train",
            "feature_contract": feature_variant_contract(variant, action_columns=MDP_V1_ACTION_COLUMNS),
            "columns": list(scaler.columns), "mean": dict(scaler.means), "scale": dict(scaler.scales),
        }
        scaler_bytes = json.dumps(scaler_payload, sort_keys=True, separators=(",", ":")).encode()
        scaler_hash = hashlib.sha256(scaler_bytes).hexdigest()
        scaler_path = Path(loaded.experiment_config.output_root) / "observation_scaler.json"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(scaler_path, scaler_payload, convert=True)
    experiment_result = run_transition_model_experiment(
        config=loaded.experiment_config,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        rollout_df=rollout_df,
        registry=registry,
        observation_scaler=scaler,
        observation_scaler_hash=scaler_hash,
    )
    return TransitionEndToEndExperimentResult(
        loaded_config=loaded,
        experiment_result=experiment_result,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the GEAS transition model experiment from YAML."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML experiment config.",
    )
    args = parser.parse_args(argv)
    result = run_from_config(args.config)
    print(f"Experiment output: {result.output_root}")
    print(f"Experiment summary: {result.experiment_summary_path}")
    return 0


__all__ = [
    "TransitionEndToEndExperimentResult",
    "main",
    "run_from_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
