"""Validation-ranking strategies for GEAS transition models."""

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
class TransitionModelRankingCandidate:
    """One evaluated candidate available for validation ranking."""

    model_name: str
    one_step_report: TransitionOneStepEvaluationReport | Mapping[str, Any] | None = None
    rollout_metrics: Mapping[str, Mapping[str, float] | float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionModelCandidateScore:
    """Score assigned to one candidate by a ranking strategy."""

    model_name: str
    score: float
    rank: int


@dataclass(frozen=True)
class TransitionModelRankingResult:
    """Validation ranking result without automatic model choice."""

    strategy_name: str
    higher_is_better: bool
    candidate_scores: tuple[TransitionModelCandidateScore, ...]

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-safe model ranking payload."""

        return {
            "stage": "transition_candidate_validation_ranking",
            "strategy_name": self.strategy_name,
            "higher_is_better": bool(self.higher_is_better),
            "automatic_model_selection": False,
            "test_used_for_validation_ranking": False,
            "test_used_for_selection": False,
            "candidate_scores": [
                {
                    "model_name": score.model_name,
                    "score": _jsonable_float(score.score),
                    "rank": int(score.rank),
                }
                for score in self.candidate_scores
            ],
        }


class RankingMetricStrategy(ABC):
    """Pluggable scoring strategy for transition candidate ranking."""

    name = "base"
    higher_is_better = False

    @abstractmethod
    def score(self, candidate: TransitionModelRankingCandidate) -> float:
        """Return the candidate score."""


class MeanRmseStrategy(RankingMetricStrategy):
    """Select by one-step aggregate mean RMSE."""

    name = "mean_rmse"
    higher_is_better = False

    def score(self, candidate: TransitionModelRankingCandidate) -> float:
        return _aggregate_metric(candidate, "mean_rmse")


class WeightedRmseStrategy(RankingMetricStrategy):
    """Select by weighted one-step target RMSE."""

    name = "weighted_rmse"
    higher_is_better = False

    def __init__(self, target_weights: Mapping[str, float] | None = None) -> None:
        self.target_weights = dict(target_weights or {})

    def score(self, candidate: TransitionModelRankingCandidate) -> float:
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


class RolloutWeightedRmseStrategy(RankingMetricStrategy):
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

    def score(self, candidate: TransitionModelRankingCandidate) -> float:
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


class CustomScoreStrategy(RankingMetricStrategy):
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

    def score(self, candidate: TransitionModelRankingCandidate) -> float:
        if self.score_key not in candidate.metadata:
            raise KeyError(f"Missing custom ranking score: {self.score_key}")
        return float(candidate.metadata[self.score_key])


StrategyFactory = Callable[..., RankingMetricStrategy]


def default_ranking_strategy_registry() -> dict[str, StrategyFactory]:
    """Return the built-in transition model ranking strategy registry."""

    return {
        MeanRmseStrategy.name: MeanRmseStrategy,
        WeightedRmseStrategy.name: WeightedRmseStrategy,
        TemperaturePriorityStrategy.name: TemperaturePriorityStrategy,
        HumidityPriorityStrategy.name: HumidityPriorityStrategy,
        RolloutWeightedRmseStrategy.name: RolloutWeightedRmseStrategy,
        CustomScoreStrategy.name: CustomScoreStrategy,
    }


def build_ranking_strategy(name: str, **kwargs: Any) -> RankingMetricStrategy:
    """Build a ranking strategy by registry name."""

    registry = default_ranking_strategy_registry()
    if name not in registry:
        raise ValueError(f"Unknown transition ranking strategy: {name}")
    return registry[name](**kwargs)


def rank_transition_models(
    candidates: Sequence[TransitionModelRankingCandidate],
    *,
    strategy: RankingMetricStrategy | None = None,
) -> TransitionModelRankingResult:
    """Rank transition candidates without selecting a final model."""

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
        )
        for index, (candidate, score) in enumerate(raw_scores)
    )
    return TransitionModelRankingResult(
        strategy_name=strategy.name,
        higher_is_better=higher_is_better,
        candidate_scores=candidate_scores,
    )


def _sort_score(score: float, *, higher_is_better: bool) -> float:
    if np.isfinite(score):
        return float(score)
    return float("-inf") if higher_is_better else float("inf")


def _aggregate_metric(
    candidate: TransitionModelRankingCandidate,
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
    candidate: TransitionModelRankingCandidate,
) -> Mapping[str, Mapping[str, float]]:
    report = candidate.one_step_report
    if isinstance(report, TransitionOneStepEvaluationReport):
        return report.target_metrics
    if isinstance(report, Mapping):
        return _nested_mapping(report, "target_metrics")
    return {}


def _rollout_metric(
    candidate: TransitionModelRankingCandidate,
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
    "RankingMetricStrategy",
    "TemperaturePriorityStrategy",
    "TransitionModelCandidateScore",
    "TransitionModelRankingCandidate",
    "TransitionModelRankingResult",
    "WeightedRmseStrategy",
    "build_ranking_strategy",
    "default_ranking_strategy_registry",
    "rank_transition_models",
]
