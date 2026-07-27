"""Model registry helpers for transition experiment configs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from geas35.models.transition.base import BaseTransitionModel
from geas35.models.transition.boosting_models import (
    CatBoostTransitionModel,
    LightGBMTransitionModel,
    XGBoostTransitionModel,
)
from geas35.models.transition.deep import TCNTransitionModel
from geas35.models.transition.sklearn_models import (
    KNNTransitionModel,
    LinearRegressionTransitionModel,
    LinearSVRTransitionModel,
    MLPTransitionModel,
)
from geas35.models.transition.trainer import TransitionCandidateSpec

TransitionModelFactory = Callable[..., BaseTransitionModel]


def default_transition_model_registry() -> dict[str, TransitionModelFactory]:
    """Return built-in transition model factories available to configs."""

    return {
        "linear_regression": LinearRegressionTransitionModel,
        "linear_svr": LinearSVRTransitionModel,
        "knn": KNNTransitionModel,
        "mlp": MLPTransitionModel,
        "lightgbm": LightGBMTransitionModel,
        "catboost": CatBoostTransitionModel,
        "xgboost": XGBoostTransitionModel,
        "tcn": TCNTransitionModel,
    }


def build_transition_model(
    model_name: str,
    *,
    params: Mapping[str, Any] | None = None,
    registry: Mapping[str, TransitionModelFactory] | None = None,
) -> BaseTransitionModel:
    """Build one transition model by registry name."""

    name = str(model_name).strip()
    if not name:
        raise ValueError("model_name is required.")
    model_registry = default_transition_model_registry() if registry is None else dict(registry)
    if name not in model_registry:
        raise ValueError(f"Unknown transition model: {name}")
    return model_registry[name](**dict(params or {}))


def build_transition_candidate_specs(
    model_specs: Sequence[Any],
    *,
    registry: Mapping[str, TransitionModelFactory] | None = None,
) -> tuple[TransitionCandidateSpec, ...]:
    """Convert config-level model specs into trainer candidate specs."""

    if not model_specs:
        raise ValueError("At least one transition model spec is required.")
    candidates = []
    seen_names: set[str] = set()
    for spec in model_specs:
        name = _model_spec_name(spec)
        if name in seen_names:
            raise ValueError(f"Duplicate transition model spec: {name}")
        seen_names.add(name)
        params = _model_spec_params(spec)
        metadata = {
            "registry_name": name,
            "params": dict(params),
            **_model_spec_metadata(spec),
        }
        candidates.append(
            TransitionCandidateSpec(
                model_name=name,
                build_model=lambda name=name, params=params: build_transition_model(
                    name,
                    params=params,
                    registry=registry,
                ),
                metadata=metadata,
            )
        )
    return tuple(candidates)


def _model_spec_name(spec: Any) -> str:
    if isinstance(spec, Mapping):
        value = spec.get("name", spec.get("model_name"))
    else:
        value = getattr(spec, "model_name", getattr(spec, "name", None))
    if value is None:
        raise ValueError("Transition model spec must define name or model_name.")
    return str(value).strip()


def _model_spec_params(spec: Any) -> Mapping[str, Any]:
    if isinstance(spec, Mapping):
        value = spec.get("params", spec.get("estimator_kwargs", {}))
    else:
        value = getattr(spec, "params", getattr(spec, "estimator_kwargs", {}))
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Transition model params must be a mapping.")
    return value


def _model_spec_metadata(spec: Any) -> Mapping[str, Any]:
    if isinstance(spec, Mapping):
        value = spec.get("metadata", {})
    else:
        value = getattr(spec, "metadata", {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Transition model metadata must be a mapping.")
    return value


__all__ = [
    "TransitionModelFactory",
    "build_transition_candidate_specs",
    "build_transition_model",
    "default_transition_model_registry",
]
