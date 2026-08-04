"""Validation HPO helpers for GEAS transition candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from geas35.models.transition.evaluator import TransitionOneStepEvaluationReport


PERSISTENCE_CANDIDATE_NAME = "persistence"
TRAINABLE_TRANSITION_CANDIDATES = (
    "linear_regression",
    "linear_svr",
    "knn",
    "extra_trees",
    "lightgbm",
    "xgboost",
    "mlp",
)
DEFAULT_HPO_BUDGET = 20
DEFAULT_HPO_RANDOM_SEED = 42
DEFAULT_VALIDATION_ROLLOUT_OBJECTIVE_WEIGHTS = {
    "one_step": 0.10,
    "15min": 0.10,
    "30min": 0.25,
    "60min": 0.45,
    "drift": 0.10,
}


@dataclass(frozen=True)
class TransitionHPOConfig:
    """Random-search configuration for transition candidate HPO."""

    enabled: bool = True
    budget: int = DEFAULT_HPO_BUDGET
    random_seed: int = DEFAULT_HPO_RANDOM_SEED
    search_spaces: Mapping[str, Any] = field(default_factory=dict)
    objective_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_VALIDATION_ROLLOUT_OBJECTIVE_WEIGHTS)
    )

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("TransitionHPOConfig.budget must be positive.")

    def to_artifact(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "budget": int(self.budget),
            "random_seed": int(self.random_seed),
            "search_spaces": _jsonable(self.search_spaces),
            "objective_weights": {
                str(key): float(value)
                for key, value in self.objective_weights.items()
            },
            "test_used_for_hpo": False,
            "test_used_for_validation_ranking": False,
            "test_used_for_selection": False,
        }


@dataclass(frozen=True)
class TransitionHPOTrialResult:
    """Validation result for one transition HPO trial."""

    trial_index: int
    model_name: str
    params: Mapping[str, Any]
    validation_one_step_metrics: Mapping[str, Any]
    validation_rollout_metrics: Mapping[str, Any]
    objective_components: Mapping[str, float]
    validation_rollout_weighted_score: float

    def to_artifact(self) -> dict[str, Any]:
        return {
            "trial_index": int(self.trial_index),
            "model_name": self.model_name,
            "params": _jsonable(self.params),
            "validation_one_step_metrics": _jsonable(self.validation_one_step_metrics),
            "validation_rollout_metrics": _jsonable(self.validation_rollout_metrics),
            "objective_components": _jsonable(self.objective_components),
            "validation_rollout_weighted_score": _jsonable_float(
                self.validation_rollout_weighted_score
            ),
        }


@dataclass(frozen=True)
class TransitionHPOResult:
    """Best validation configuration and all random-search trials."""

    model_name: str
    status: str
    best_trial_index: int | None
    best_params: Mapping[str, Any]
    best_score: float | None
    trials: tuple[TransitionHPOTrialResult, ...]
    config: TransitionHPOConfig
    reason: str | None = None

    @property
    def applied(self) -> bool:
        return self.status == "completed"

    def to_artifact(self) -> dict[str, Any]:
        return {
            "stage": "transition_hyperparameter_optimization",
            "model_name": self.model_name,
            "status": self.status,
            "reason": self.reason,
            "hpo_applied": bool(self.applied),
            "best_trial_index": self.best_trial_index,
            "best_config": _jsonable(self.best_params),
            "best_validation_rollout_weighted_score": _jsonable_float(
                self.best_score
            ),
            "objective": {
                "name": "validation_rollout_weighted_score",
                "mode": "min",
                "weights": {
                    str(key): float(value)
                    for key, value in self.config.objective_weights.items()
                },
                "test_used_for_hpo": False,
                "test_used_for_validation_ranking": False,
                "test_used_for_selection": False,
            },
            "config": self.config.to_artifact(),
            "trials": [trial.to_artifact() for trial in self.trials],
        }

    def best_config_artifact(self) -> dict[str, Any]:
        return {
            "stage": "transition_best_candidate_config",
            "model_name": self.model_name,
            "status": self.status,
            "reason": self.reason,
            "hpo_applied": bool(self.applied),
            "best_trial_index": self.best_trial_index,
            "best_config": _jsonable(self.best_params),
            "validation_rollout_weighted_score": _jsonable_float(self.best_score),
            "test_used_for_hpo": False,
            "test_used_for_validation_ranking": False,
            "test_used_for_selection": False,
        }

    @classmethod
    def not_applicable(
        cls,
        *,
        model_name: str,
        config: TransitionHPOConfig,
        params: Mapping[str, Any],
        reason: str,
    ) -> "TransitionHPOResult":
        return cls(
            model_name=model_name,
            status="not_applicable",
            best_trial_index=None,
            best_params=dict(params),
            best_score=None,
            trials=(),
            config=config,
            reason=reason,
        )


def transition_target_normalization_scales(
    y: pd.DataFrame,
    target_columns: Sequence[str],
) -> dict[str, float]:
    """Return validation target scales used by one-step and rollout NRMSE."""

    return {str(column): _target_scale(y[str(column)]) for column in target_columns}


def enrich_rollout_metrics_with_normalized_errors(
    rollout_metrics: Mapping[str, float],
    errors: pd.DataFrame,
    normalization_scales: Mapping[str, float],
) -> dict[str, float]:
    """Add aggregate normalized rollout metrics for Step 5 HPO scoring."""

    out = {str(key): float(value) for key, value in rollout_metrics.items()}
    trajectory_nrmse = []
    final_step_nrmse = []
    drift_nmae = []
    for column in errors.columns:
        scale = float(normalization_scales.get(str(column), 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        values = errors[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        trajectory_nrmse.append(float(np.sqrt(np.mean(finite**2))) / scale)
        final = finite[-1]
        final_step_nrmse.append(float(abs(final)) / scale)
        step_abs = np.abs(values) / scale
        drift_nmae.append(_drift_slope(step_abs))
    out["trajectory_nrmse"] = _finite_mean(trajectory_nrmse)
    out["final_step_nrmse"] = _finite_mean(final_step_nrmse)
    out["drift_slope_nmae"] = _finite_mean(drift_nmae)
    return out


def validation_rollout_weighted_score(
    one_step_report: TransitionOneStepEvaluationReport,
    rollout_metrics: Mapping[str, Mapping[str, float]],
    *,
    objective_weights: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return the Step 5 Validation Rollout Weighted Score."""

    weights = dict(objective_weights or DEFAULT_VALIDATION_ROLLOUT_OBJECTIVE_WEIGHTS)
    one_step = _metric(
        one_step_report.aggregate_metrics,
        "normalized_mean_rmse",
        fallback_key="mean_rmse",
    )
    components = {
        "one_step": one_step,
        "15min": _horizon_metric(rollout_metrics, "15min", fallback=one_step),
        "30min": _horizon_metric(rollout_metrics, "30min", fallback=one_step),
        "60min": _horizon_metric(rollout_metrics, "60min", fallback=one_step),
        "drift": _drift_metric(rollout_metrics, fallback=0.0),
    }
    score = 0.0
    for key, weight in weights.items():
        value = float(components.get(key, np.nan))
        if not np.isfinite(value):
            value = one_step if np.isfinite(one_step) else float("inf")
        score += float(weight) * value
    return float(score), components


def sample_transition_hpo_params(
    model_name: str,
    base_params: Mapping[str, Any],
    config: TransitionHPOConfig,
) -> tuple[dict[str, Any], ...]:
    """Sample random-search parameter dictionaries for one candidate."""

    rng = np.random.default_rng(int(config.random_seed) + _stable_name_offset(model_name))
    space = {
        **_default_search_space(model_name, seed=int(config.random_seed)),
        **_configured_search_space(model_name, config.search_spaces),
    }
    trials = []
    for _ in range(int(config.budget)):
        params = dict(base_params)
        for key, spec in space.items():
            params[str(key)] = _sample_value(spec, rng)
        trials.append(params)
    return tuple(trials)


def _configured_search_space(
    model_name: str,
    search_spaces: Mapping[str, Any],
) -> dict[str, Any]:
    if not search_spaces:
        return {}
    model_specific = search_spaces.get("model_specific", {})
    if isinstance(model_specific, Mapping) and model_name in model_specific:
        value = model_specific[model_name]
        if not isinstance(value, Mapping):
            raise ValueError(f"hpo.search_spaces.model_specific.{model_name} must be a mapping.")
        return dict(value)
    if model_name in search_spaces:
        value = search_spaces[model_name]
        if not isinstance(value, Mapping):
            raise ValueError(f"hpo.search_spaces.{model_name} must be a mapping.")
        return dict(value)
    return {}


def _default_search_space(model_name: str, *, seed: int) -> dict[str, Any]:
    if model_name == "linear_regression":
        return {"fit_intercept": {"values": [True, False]}}
    if model_name == "linear_svr":
        return {
            "C": {"values": [0.1, 1.0, 10.0]},
            "epsilon": {"values": [0.0, 0.01, 0.1]},
            "max_iter": {"values": [1000, 3000, 5000]},
            "random_state": {"value": seed},
        }
    if model_name == "knn":
        return {
            "n_neighbors": {"values": [3, 5, 7, 11]},
            "weights": {"values": ["uniform", "distance"]},
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": {"values": [50, 100, 200]},
            "max_depth": {"values": [None, 8, 16]},
            "min_samples_leaf": {"values": [1, 2, 4]},
            "random_state": {"value": seed},
            "n_jobs": {"value": -1},
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": {"values": [50, 100, 200]},
            "learning_rate": {"values": [0.03, 0.05, 0.1]},
            "num_leaves": {"values": [15, 31, 63]},
            "random_state": {"value": seed},
            "verbosity": {"value": -1},
        }
    if model_name == "xgboost":
        return {
            "n_estimators": {"values": [50, 100, 200]},
            "learning_rate": {"values": [0.03, 0.05, 0.1]},
            "max_depth": {"values": [3, 5, 7]},
            "random_state": {"value": seed},
            "verbosity": {"value": 0},
        }
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": {"values": [(32,), (64,), (64, 32)]},
            "alpha": {"values": [0.0001, 0.001, 0.01]},
            "learning_rate_init": {"values": [0.0005, 0.001, 0.005]},
            "max_iter": {"value": 300},
            "random_state": {"value": seed},
        }
    return {}


def _sample_value(spec: Any, rng: np.random.Generator) -> Any:
    if isinstance(spec, Mapping):
        if "value" in spec:
            return spec["value"]
        if "values" in spec:
            values = list(spec["values"])
            if not values:
                raise ValueError("HPO values search space must not be empty.")
            return values[int(rng.integers(0, len(values)))]
        if "low" in spec and "high" in spec:
            low = float(spec["low"])
            high = float(spec["high"])
            if spec.get("type") in {"int", "integer", "randint"}:
                return int(rng.integers(int(low), int(high) + 1))
            if spec.get("scale") == "log":
                return float(np.exp(rng.uniform(np.log(low), np.log(high))))
            return float(rng.uniform(low, high))
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("HPO list search space must not be empty.")
        return spec[int(rng.integers(0, len(spec)))]
    return spec


def _horizon_metric(
    rollout_metrics: Mapping[str, Mapping[str, float]],
    horizon: str,
    *,
    fallback: float,
) -> float:
    metrics = rollout_metrics.get(horizon, {})
    if not isinstance(metrics, Mapping):
        return fallback
    return _metric(metrics, "trajectory_nrmse", fallback_key="trajectory_rmse", default=fallback)


def _drift_metric(
    rollout_metrics: Mapping[str, Mapping[str, float]],
    *,
    fallback: float,
) -> float:
    values = []
    for metrics in rollout_metrics.values():
        if not isinstance(metrics, Mapping):
            continue
        values.append(
            _metric(metrics, "drift_slope_nmae", fallback_key="drift_slope_mae")
        )
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float(fallback)
    return float(np.mean(finite))


def _metric(
    metrics: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    default: float = float("nan"),
) -> float:
    value = pd.to_numeric(metrics.get(key), errors="coerce")
    if pd.isna(value) and fallback_key is not None:
        value = pd.to_numeric(metrics.get(fallback_key), errors="coerce")
    if pd.isna(value):
        return float(default)
    return float(value)


def _target_scale(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return 1.0
    std = float(np.std(finite, ddof=0))
    if np.isfinite(std) and std > 0.0:
        return std
    value_range = float(np.max(finite) - np.min(finite))
    if np.isfinite(value_range) and value_range > 0.0:
        return value_range
    return 1.0


def _drift_slope(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        return 0.0
    steps = np.arange(len(finite), dtype=float)
    slope, _ = np.polyfit(steps, finite, deg=1)
    return float(slope)


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def _stable_name_offset(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return _jsonable_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _jsonable_float(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


__all__ = [
    "DEFAULT_HPO_BUDGET",
    "DEFAULT_HPO_RANDOM_SEED",
    "DEFAULT_VALIDATION_ROLLOUT_OBJECTIVE_WEIGHTS",
    "PERSISTENCE_CANDIDATE_NAME",
    "TRAINABLE_TRANSITION_CANDIDATES",
    "TransitionHPOConfig",
    "TransitionHPOResult",
    "TransitionHPOTrialResult",
    "enrich_rollout_metrics_with_normalized_errors",
    "sample_transition_hpo_params",
    "transition_target_normalization_scales",
    "validation_rollout_weighted_score",
]
