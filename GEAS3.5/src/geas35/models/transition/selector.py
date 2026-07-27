"""Model-selection strategies for GEAS transition models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from geas35.models.transition.evaluator import TransitionOneStepEvaluationReport

DEFAULT_HORIZON_WEIGHTS = {
    "15min": 0.2,
    "30min": 0.3,
    "60min": 0.5,
}


@dataclass(frozen=True)
class TransitionModelSelectionCandidate:
    """One evaluated candidate available for model selection."""

    model_name: str
    one_step_report: TransitionOneStepEvaluationReport | Mapping[str, Any] | None = None
    rollout_metrics: Mapping[str, Mapping[str, float] | float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionModelCandidateScore:
    """Score assigned to one candidate by a selection strategy."""

    model_name: str
    score: float
    rank: int
    selected: bool = False


@dataclass(frozen=True)
class TransitionModelSelectionResult:
    """Best-model selection result and full candidate ranking."""

    selected_model_name: str
    selected_score: float
    strategy_name: str
    higher_is_better: bool
    candidate_scores: tuple[TransitionModelCandidateScore, ...]

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-safe model selection payload."""

        return {
            "stage": "transition_model_selection",
            "selected_model_name": self.selected_model_name,
            "selected_score": _jsonable_float(self.selected_score),
            "strategy_name": self.strategy_name,
            "higher_is_better": bool(self.higher_is_better),
            "candidate_scores": [
                {
                    "model_name": score.model_name,
                    "score": _jsonable_float(score.score),
                    "rank": int(score.rank),
                    "selected": bool(score.selected),
                }
                for score in self.candidate_scores
            ],
        }


class SelectionMetricStrategy(ABC):
    """Pluggable scoring strategy for transition model selection."""

    name = "base"
    higher_is_better = False

    @abstractmethod
    def score(self, candidate: TransitionModelSelectionCandidate) -> float:
        """Return the candidate score."""


class MeanRmseStrategy(SelectionMetricStrategy):
    """Select by one-step aggregate mean RMSE."""

    name = "mean_rmse"
    higher_is_better = False

    def score(self, candidate: TransitionModelSelectionCandidate) -> float:
        return _aggregate_metric(candidate, "mean_rmse")


class WeightedRmseStrategy(SelectionMetricStrategy):
    """Select by weighted one-step target RMSE."""

    name = "weighted_rmse"
    higher_is_better = False

    def __init__(self, target_weights: Mapping[str, float] | None = None) -> None:
        self.target_weights = dict(target_weights or {})

    def score(self, candidate: TransitionModelSelectionCandidate) -> float:
        target_metrics = _target_metrics(candidate)
        if not target_metrics:
            return _aggregate_metric(candidate, "mean_rmse")
        values = []
        weights = []
        for target, metrics in target_metrics.items():
            if "rmse" not in metrics:
                continue
            value = float(metrics["rmse"])
            weight = float(self.target_weights.get(target, 1.0))
            if np.isfinite(value) and np.isfinite(weight) and weight > 0.0:
                values.append(value)
                weights.append(weight)
        if not values:
            return float("inf")
        return float(np.average(values, weights=weights))


class TemperaturePriorityStrategy(WeightedRmseStrategy):
    """Select by RMSE with indoor temperature prioritized."""

    name = "temperature_priority"

    def __init__(self, target_weights: Mapping[str, float] | None = None) -> None:
        weights = {"obs_indoor_temp_c": 3.0, "obs_indoor_humidity_pct": 1.0}
        weights.update(dict(target_weights or {}))
        super().__init__(weights)


class HumidityPriorityStrategy(WeightedRmseStrategy):
    """Select by RMSE with indoor humidity prioritized."""

    name = "humidity_priority"

    def __init__(self, target_weights: Mapping[str, float] | None = None) -> None:
        weights = {"obs_indoor_temp_c": 1.0, "obs_indoor_humidity_pct": 3.0}
        weights.update(dict(target_weights or {}))
        super().__init__(weights)


class RolloutWeightedRmseStrategy(SelectionMetricStrategy):
    """Select by weighted rollout RMSE over configured horizons."""

    name = "rollout_weighted_rmse"
    higher_is_better = False

    def __init__(
        self,
        horizon_weights: Mapping[str, float] | None = None,
        *,
        metric_key: str = "trajectory_rmse",
    ) -> None:
        self.horizon_weights = dict(horizon_weights or DEFAULT_HORIZON_WEIGHTS)
        self.metric_key = metric_key

    def score(self, candidate: TransitionModelSelectionCandidate) -> float:
        values = []
        weights = []
        for horizon, weight in self.horizon_weights.items():
            value = _rollout_metric(candidate, horizon, self.metric_key)
            if np.isfinite(value) and np.isfinite(weight) and float(weight) > 0.0:
                values.append(value)
                weights.append(float(weight))
        if not values:
            return float("inf")
        return float(np.average(values, weights=weights))


class CustomScoreStrategy(SelectionMetricStrategy):
    """Select by a custom score stored in candidate metadata."""

    name = "custom_score"

    def __init__(
        self,
        score_key: str = "custom_score",
        *,
        higher_is_better: bool = False,
    ) -> None:
        self.score_key = score_key
        self.higher_is_better = higher_is_better

    def score(self, candidate: TransitionModelSelectionCandidate) -> float:
        if self.score_key not in candidate.metadata:
            raise KeyError(f"Missing custom selection score: {self.score_key}")
        return float(candidate.metadata[self.score_key])


StrategyFactory = Callable[..., SelectionMetricStrategy]


def default_selection_strategy_registry() -> dict[str, StrategyFactory]:
    """Return the built-in transition model selection strategy registry."""

    return {
        MeanRmseStrategy.name: MeanRmseStrategy,
        WeightedRmseStrategy.name: WeightedRmseStrategy,
        TemperaturePriorityStrategy.name: TemperaturePriorityStrategy,
        HumidityPriorityStrategy.name: HumidityPriorityStrategy,
        RolloutWeightedRmseStrategy.name: RolloutWeightedRmseStrategy,
        CustomScoreStrategy.name: CustomScoreStrategy,
    }


def build_selection_strategy(name: str, **kwargs: Any) -> SelectionMetricStrategy:
    """Build a selection strategy by registry name."""

    registry = default_selection_strategy_registry()
    if name not in registry:
        raise ValueError(f"Unknown transition selection strategy: {name}")
    return registry[name](**kwargs)


def select_transition_model(
    candidates: Sequence[TransitionModelSelectionCandidate],
    *,
    strategy: SelectionMetricStrategy | None = None,
) -> TransitionModelSelectionResult:
    """Select and rank transition model candidates."""

    if not candidates:
        raise ValueError("At least one transition model candidate is required.")
    strategy = strategy or RolloutWeightedRmseStrategy()
    raw_scores = [
        (candidate, float(strategy.score(candidate)))
        for candidate in candidates
    ]
    higher_is_better = bool(strategy.higher_is_better)
    raw_scores.sort(
        key=lambda item: _sort_score(item[1], higher_is_better=higher_is_better),
        reverse=higher_is_better,
    )
    candidate_scores = tuple(
        TransitionModelCandidateScore(
            model_name=candidate.model_name,
            score=score,
            rank=index + 1,
            selected=index == 0,
        )
        for index, (candidate, score) in enumerate(raw_scores)
    )
    selected = candidate_scores[0]
    return TransitionModelSelectionResult(
        selected_model_name=selected.model_name,
        selected_score=selected.score,
        strategy_name=strategy.name,
        higher_is_better=higher_is_better,
        candidate_scores=candidate_scores,
    )


def _sort_score(score: float, *, higher_is_better: bool) -> float:
    if np.isfinite(score):
        return float(score)
    return float("-inf") if higher_is_better else float("inf")


def _aggregate_metric(
    candidate: TransitionModelSelectionCandidate,
    key: str,
) -> float:
    report = candidate.one_step_report
    if isinstance(report, TransitionOneStepEvaluationReport):
        metrics = report.aggregate_metrics
    elif isinstance(report, Mapping):
        metrics = _nested_mapping(report, "aggregate_metrics")
    else:
        metrics = {}
    if key not in metrics:
        return float("inf")
    return float(metrics[key])


def _target_metrics(
    candidate: TransitionModelSelectionCandidate,
) -> Mapping[str, Mapping[str, float]]:
    report = candidate.one_step_report
    if isinstance(report, TransitionOneStepEvaluationReport):
        return report.target_metrics
    if isinstance(report, Mapping):
        return _nested_mapping(report, "target_metrics")
    return {}


def _rollout_metric(
    candidate: TransitionModelSelectionCandidate,
    horizon: str,
    metric_key: str,
) -> float:
    rollout = candidate.rollout_metrics
    horizon_metrics = rollout.get(horizon)
    if isinstance(horizon_metrics, Mapping) and metric_key in horizon_metrics:
        return float(horizon_metrics[metric_key])
    flat_key = f"{horizon}_{metric_key}"
    if flat_key in rollout:
        return float(rollout[flat_key])
    return float("nan")


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _jsonable_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


__all__ = [
    "DEFAULT_HORIZON_WEIGHTS",
    "CustomScoreStrategy",
    "HumidityPriorityStrategy",
    "MeanRmseStrategy",
    "RolloutWeightedRmseStrategy",
    "SelectionMetricStrategy",
    "TemperaturePriorityStrategy",
    "TransitionModelCandidateScore",
    "TransitionModelSelectionCandidate",
    "TransitionModelSelectionResult",
    "WeightedRmseStrategy",
    "build_selection_strategy",
    "default_selection_strategy_registry",
    "select_transition_model",
]
