"""Experiment runner for GEAS transition model training."""

from geas35.experiments.transition.config import (
    LoadedTransitionExperimentConfig,
    load_transition_experiment_config,
    read_configured_transition_frames,
)
from geas35.experiments.transition.runner import (
    TransitionExperimentConfig,
    TransitionExperimentResult,
    TransitionModelExperimentSpec,
    run_transition_model_experiment,
)


def run_from_config(*args, **kwargs):
    """Run a transition experiment from YAML without eager CLI imports."""

    from geas35.experiments.transition.cli import run_from_config as _run_from_config

    return _run_from_config(*args, **kwargs)


__all__ = [
    "LoadedTransitionExperimentConfig",
    "TransitionExperimentConfig",
    "TransitionExperimentResult",
    "TransitionModelExperimentSpec",
    "load_transition_experiment_config",
    "read_configured_transition_frames",
    "run_from_config",
    "run_transition_model_experiment",
]
