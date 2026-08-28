"""Experiment runner for GEAS transition model training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from geas35.models.transition.datasets import build_transition_dataset_from_rl_frame
from geas35.models.transition.comparison import (
    write_transition_candidate_comparison_outputs,
    write_transition_candidate_test_outputs,
)
from geas35.models.transition.registry import (
    TransitionModelFactory,
    build_transition_candidate_specs,
)
from geas35.models.transition.selector import (
    RankingMetricStrategy,
    build_ranking_strategy,
)
from geas35.models.transition.hyperparameters import TransitionHPOConfig
from geas35.models.transition.trainer import (
    DEFAULT_ROLLOUT_HORIZON_STEPS,
    DEFAULT_ROLLOUT_STEP_MINUTES,
    TransitionTrainingResult,
    train_transition_candidates,
)
from geas35.io_utils import write_json


@dataclass(frozen=True)
class TransitionModelExperimentSpec:
    """Config-level model candidate spec for transition experiments."""

    model_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionExperimentConfig:
    """End-to-end transition experiment config excluding dataset payloads."""

    crop: str
    output_root: Path
    models: tuple[TransitionModelExperimentSpec, ...]
    ranking_strategy_name: str = "mean_rmse"
    ranking_strategy_params: Mapping[str, Any] = field(default_factory=dict)
    hpo_config: TransitionHPOConfig = field(default_factory=TransitionHPOConfig)
    rollout_enabled: bool = False
    rollout_horizon_steps: tuple[int, ...] = DEFAULT_ROLLOUT_HORIZON_STEPS
    rollout_step_minutes: int = DEFAULT_ROLLOUT_STEP_MINUTES
    random_seed: int = 0


@dataclass(frozen=True)
class TransitionExperimentResult:
    """Artifacts produced by one transition experiment run."""

    output_root: str
    crop: str
    training_result: TransitionTrainingResult
    config_json_path: str
    experiment_summary_path: str
    comparison_output_paths: Mapping[str, str] = field(default_factory=dict)
    test_output_paths: Mapping[str, str] = field(default_factory=dict)


def run_transition_model_experiment(
    *,
    config: TransitionExperimentConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    rollout_df: pd.DataFrame | None = None,
    registry: Mapping[str, TransitionModelFactory] | None = None,
    ranking_strategy: RankingMetricStrategy | None = None,
) -> TransitionExperimentResult:
    """Run transition model training from in-memory RL-ready frames."""

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config.json", _config_payload(config), convert=True)

    train_build = build_transition_dataset_from_rl_frame(train_df)
    validation_build = build_transition_dataset_from_rl_frame(validation_df)
    test_build = (
        None
        if test_df is None
        else build_transition_dataset_from_rl_frame(test_df)
    )
    candidate_specs = build_transition_candidate_specs(
        config.models,
        registry=registry,
    )
    strategy = ranking_strategy or build_ranking_strategy(
        config.ranking_strategy_name,
        **dict(config.ranking_strategy_params),
    )
    rollout_source = None
    if config.rollout_enabled:
        rollout_source = rollout_df if rollout_df is not None else validation_df

    training_result = train_transition_candidates(
        train_build.dataset,
        validation_build.dataset,
        candidate_specs,
        crop=config.crop,
        models_root=output_root,
        ranking_strategy=strategy,
        rollout_frame=rollout_source,
        rollout_horizon_steps=config.rollout_horizon_steps,
        rollout_step_minutes=config.rollout_step_minutes,
        hpo_config=config.hpo_config,
        artifact_metadata={
            "experiment": "transition",
            "random_seed": int(config.random_seed),
        },
    )
    summary_path = output_root / "experiment_summary.json"
    write_json(
        summary_path,
        _summary_payload(
            config=config,
            training_result=training_result,
            train_rows=len(train_df.index),
            validation_rows=len(validation_df.index),
            test_rows=None if test_df is None else len(test_df.index),
            train_dataset_rows=len(train_build.dataset.x.index),
            validation_dataset_rows=len(validation_build.dataset.x.index),
            test_dataset_rows=(
                None if test_build is None else len(test_build.dataset.x.index)
            ),
            rollout_rows=None if rollout_source is None else len(rollout_source.index),
        ),
        convert=True,
    )
    config_payload = _config_payload(config)
    dataset_payload = _dataset_payload(
        train_rows=len(train_df.index),
        validation_rows=len(validation_df.index),
        test_rows=None if test_df is None else len(test_df.index),
        train_dataset_rows=len(train_build.dataset.x.index),
        validation_dataset_rows=len(validation_build.dataset.x.index),
        test_dataset_rows=(
            None if test_build is None else len(test_build.dataset.x.index)
        ),
        rollout_rows=None if rollout_source is None else len(rollout_source.index),
    )
    comparison_output_paths = write_transition_candidate_comparison_outputs(
        output_root,
        training_result,
        config_payload=config_payload,
        dataset_payload=dataset_payload,
    )
    test_output_paths = {}
    if test_build is not None and test_df is not None:
        test_output_paths = write_transition_candidate_test_outputs(
            output_root,
            training_result,
            test_dataset=test_build.dataset,
            test_frame=test_df,
            rollout_horizon_steps=config.rollout_horizon_steps,
            rollout_step_minutes=config.rollout_step_minutes,
        )
    return TransitionExperimentResult(
        output_root=str(output_root),
        crop=config.crop,
        training_result=training_result,
        config_json_path=str(output_root / "config.json"),
        experiment_summary_path=str(summary_path),
        comparison_output_paths=comparison_output_paths,
        test_output_paths=test_output_paths,
    )


def _config_payload(config: TransitionExperimentConfig) -> dict[str, Any]:
    return {
        "stage": "transition_experiment_config",
        "crop": config.crop,
        "output_root": str(config.output_root),
        "models": [asdict(model) for model in config.models],
        "ranking": {
            "strategy": config.ranking_strategy_name,
            "params": dict(config.ranking_strategy_params),
            "automatic_model_selection": False,
            "test_used_for_validation_ranking": False,
        },
        "hpo": config.hpo_config.to_artifact(),
        "rollout": {
            "enabled": bool(config.rollout_enabled),
            "horizon_steps": list(config.rollout_horizon_steps),
            "step_minutes": int(config.rollout_step_minutes),
        },
        "random_seed": int(config.random_seed),
    }


def _summary_payload(
    *,
    config: TransitionExperimentConfig,
    training_result: TransitionTrainingResult,
    train_rows: int,
    validation_rows: int,
    test_rows: int | None,
    train_dataset_rows: int,
    validation_dataset_rows: int,
    test_dataset_rows: int | None,
    rollout_rows: int | None,
) -> dict[str, Any]:
    return {
        "stage": "transition_experiment_summary",
        "crop": config.crop,
        "automatic_model_selection": False,
        "validation_ranking": training_result.validation_ranking.to_artifact(),
        "hpo": config.hpo_config.to_artifact(),
        "candidate_artifacts": {
            result.model_name: str(result.artifact.artifact_dir)
            for result in training_result.candidate_results
        },
        **_dataset_payload(
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            train_dataset_rows=train_dataset_rows,
            validation_dataset_rows=validation_dataset_rows,
            test_dataset_rows=test_dataset_rows,
            rollout_rows=rollout_rows,
        ),
        "test_used_for_hpo": False,
        "test_used_for_validation_ranking": False,
        "test_used_for_selection": False,
    }


def _dataset_payload(
    *,
    train_rows: int,
    validation_rows: int,
    test_rows: int | None,
    train_dataset_rows: int,
    validation_dataset_rows: int,
    test_dataset_rows: int | None,
    rollout_rows: int | None,
) -> dict[str, Any]:
    return {
        "source_rows": {
            "train": int(train_rows),
            "validation": int(validation_rows),
            "test": None if test_rows is None else int(test_rows),
            "rollout": None if rollout_rows is None else int(rollout_rows),
        },
        "transition_dataset_rows": {
            "train": int(train_dataset_rows),
            "validation": int(validation_dataset_rows),
            "test": None if test_dataset_rows is None else int(test_dataset_rows),
        },
    }


__all__ = [
    "TransitionExperimentConfig",
    "TransitionExperimentResult",
    "TransitionModelExperimentSpec",
    "run_transition_model_experiment",
]
