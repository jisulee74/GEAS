"""Experiment runner for GEAS transition model training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from geas35.models.transition.datasets import build_transition_dataset_from_rl_frame
from geas35.models.transition.registry import (
    TransitionModelFactory,
    build_transition_candidate_specs,
)
from geas35.models.transition.selector import (
    SelectionMetricStrategy,
    build_selection_strategy,
)
from geas35.models.transition.evaluator import (
    TransitionOneStepEvaluationReport,
    evaluate_transition_model_one_step,
)
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
    selection_strategy_name: str = "mean_rmse"
    selection_strategy_params: Mapping[str, Any] = field(default_factory=dict)
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
    selected_test_metrics_path: str | None = None

    @property
    def selected_manifest_path(self) -> str:
        return str(self.training_result.selected_manifest_path)

    @property
    def selected_model_name(self) -> str:
        return self.training_result.selection_result.selected_model_name


def run_transition_model_experiment(
    *,
    config: TransitionExperimentConfig,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    rollout_df: pd.DataFrame | None = None,
    registry: Mapping[str, TransitionModelFactory] | None = None,
    selection_strategy: SelectionMetricStrategy | None = None,
) -> TransitionExperimentResult:
    """Run transition model training from in-memory RL-ready frames."""

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config.json", _config_payload(config), convert=True)

    train_build = build_transition_dataset_from_rl_frame(train_df)
    validation_build = build_transition_dataset_from_rl_frame(validation_df)
    test_build = (
        None if test_df is None else build_transition_dataset_from_rl_frame(test_df)
    )
    candidate_specs = build_transition_candidate_specs(
        config.models,
        registry=registry,
    )
    strategy = selection_strategy or build_selection_strategy(
        config.selection_strategy_name,
        **dict(config.selection_strategy_params),
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
        selection_strategy=strategy,
        rollout_frame=rollout_source,
        rollout_horizon_steps=config.rollout_horizon_steps,
        rollout_step_minutes=config.rollout_step_minutes,
        artifact_metadata={
            "experiment": "transition",
            "random_seed": int(config.random_seed),
        },
    )
    selected_test_report = None
    selected_test_metrics_path = None
    if test_build is not None:
        selected_test_report = evaluate_transition_model_one_step(
            training_result.selected_candidate.model,
            test_build.dataset,
        )
        selected_test_metrics_path = (
            output_root / training_result.crop / "selected_transition_model_test_metrics.json"
        )
        write_json(
            selected_test_metrics_path,
            _selected_test_metrics_payload(
                training_result=training_result,
                report=selected_test_report,
            ),
            convert=True,
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
            test_dataset_rows=None
            if test_build is None
            else len(test_build.dataset.x.index),
            rollout_rows=None if rollout_source is None else len(rollout_source.index),
            selected_test_metrics_path=selected_test_metrics_path,
            selected_test_report=selected_test_report,
        ),
        convert=True,
    )
    return TransitionExperimentResult(
        output_root=str(output_root),
        crop=config.crop,
        training_result=training_result,
        config_json_path=str(output_root / "config.json"),
        experiment_summary_path=str(summary_path),
        selected_test_metrics_path=None
        if selected_test_metrics_path is None
        else str(selected_test_metrics_path),
    )


def _config_payload(config: TransitionExperimentConfig) -> dict[str, Any]:
    return {
        "stage": "transition_experiment_config",
        "crop": config.crop,
        "output_root": str(config.output_root),
        "models": [asdict(model) for model in config.models],
        "selection": {
            "strategy": config.selection_strategy_name,
            "params": dict(config.selection_strategy_params),
        },
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
    selected_test_metrics_path: Path | None,
    selected_test_report: TransitionOneStepEvaluationReport | None,
) -> dict[str, Any]:
    return {
        "stage": "transition_experiment_summary",
        "crop": config.crop,
        "selected_model_name": training_result.selection_result.selected_model_name,
        "selected_manifest_path": str(training_result.selected_manifest_path),
        "selection_result": training_result.selection_result.to_artifact(),
        "candidate_artifacts": {
            result.model_name: str(result.artifact.artifact_dir)
            for result in training_result.candidate_results
        },
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
        "selected_test_evaluation": {
            "selected_model_name": training_result.selection_result.selected_model_name,
            "selection_split": "validation",
            "evaluation_split": "test",
            "test_used_for_selection": False,
            "metrics_path": None
            if selected_test_metrics_path is None
            else str(selected_test_metrics_path),
            "metrics": None
            if selected_test_report is None
            else selected_test_report.to_artifact(),
        },
    }


def _selected_test_metrics_payload(
    *,
    training_result: TransitionTrainingResult,
    report: TransitionOneStepEvaluationReport,
) -> dict[str, Any]:
    return {
        "stage": "selected_transition_model_test_evaluation",
        "crop": training_result.crop,
        "selected_model_name": training_result.selection_result.selected_model_name,
        "selected_manifest_path": str(training_result.selected_manifest_path),
        "selection_split": "validation",
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "selection_result": training_result.selection_result.to_artifact(),
        "test_report": report.to_artifact(),
    }


__all__ = [
    "TransitionExperimentConfig",
    "TransitionExperimentResult",
    "TransitionModelExperimentSpec",
    "run_transition_model_experiment",
]
