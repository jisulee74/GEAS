"""Experiment runner for GEAS transition model training."""

from geas35.experiments.transition.integrity_handoff import (
    INTEGRITY_MANIFEST_NAME,
    INTEGRITY_VERSION,
    STEP12_HANDOFF_NAME,
    Step116IntegrityError,
    finalize_step11_integrity,
)
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
from geas35.experiments.transition.reproducibility_review import (
    STEP13_MANIFEST_NAME,
    STEP13_VERSION,
    Step13ReviewError,
    run_step13_reproducibility_review,
)
from geas35.experiments.transition.deployment_handoff import (
    RESEARCHER_DECISION_VERSION,
    STEP14_HANDOFF_NAME,
    STEP14_MANIFEST_NAME,
    STEP14_VERSION,
    Step14HandoffError,
    create_step14_deployment_handoff,
)


def run_from_config(*args, **kwargs):
    """Run a transition experiment from YAML without eager CLI imports."""

    from geas35.experiments.transition.cli import run_from_config as _run_from_config

    return _run_from_config(*args, **kwargs)


__all__ = [
    "finalize_step11_integrity",
    "Step116IntegrityError",
    "STEP12_HANDOFF_NAME",
    "INTEGRITY_VERSION",
    "INTEGRITY_MANIFEST_NAME",
    "LoadedTransitionExperimentConfig",
    "TransitionExperimentConfig",
    "TransitionExperimentResult",
    "TransitionModelExperimentSpec",
    "load_transition_experiment_config",
    "read_configured_transition_frames",
    "run_from_config",
    "run_transition_model_experiment",
    "STEP13_MANIFEST_NAME",
    "STEP13_VERSION",
    "Step13ReviewError",
    "run_step13_reproducibility_review",
    "RESEARCHER_DECISION_VERSION",
    "STEP14_HANDOFF_NAME",
    "STEP14_MANIFEST_NAME",
    "STEP14_VERSION",
    "Step14HandoffError",
    "create_step14_deployment_handoff",
]
