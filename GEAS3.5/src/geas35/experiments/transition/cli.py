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
    def selected_manifest_path(self) -> str:
        return self.experiment_result.selected_manifest_path

    @property
    def experiment_summary_path(self) -> str:
        return self.experiment_result.experiment_summary_path

    @property
    def selected_test_metrics_path(self) -> str | None:
        return self.experiment_result.selected_test_metrics_path


def run_from_config(
    config_path: str | Path,
    *,
    registry: Mapping[str, TransitionModelFactory] | None = None,
) -> TransitionEndToEndExperimentResult:
    """Run a transition experiment from a YAML config file."""

    loaded = load_transition_experiment_config(config_path)
    train_df, validation_df, test_df, rollout_df = read_configured_transition_frames(
        loaded
    )
    experiment_result = run_transition_model_experiment(
        config=loaded.experiment_config,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        rollout_df=rollout_df,
        registry=registry,
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
    print(f"Selected transition model: {result.selected_manifest_path}")
    print(f"Experiment summary: {result.experiment_summary_path}")
    print(f"Selected model test metrics: {result.selected_test_metrics_path}")
    return 0


__all__ = [
    "TransitionEndToEndExperimentResult",
    "main",
    "run_from_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
