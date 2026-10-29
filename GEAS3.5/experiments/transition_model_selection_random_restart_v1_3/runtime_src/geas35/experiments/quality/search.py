"""Random-search candidate generation for quality-model experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from geas35.models.quality import HyperparameterCandidate


@dataclass(frozen=True)
class SearchSpaceParameter:
    """One configurable random-search parameter."""

    spec: Mapping[str, Any]

    def sample(self, rng: np.random.Generator) -> Any:
        if "value" in self.spec:
            return self.spec["value"]
        if "values" in self.spec:
            values = list(self.spec["values"])
            if not values:
                raise ValueError("Search parameter 'values' must not be empty.")
            return values[int(rng.integers(0, len(values)))]

        kind = str(self.spec.get("type", "float")).lower()
        low = self.spec.get("low")
        high = self.spec.get("high")
        if low is None or high is None:
            raise ValueError("Search parameter must define low/high, value, or values.")

        if kind == "int":
            return int(rng.integers(int(low), int(high) + 1))
        if kind == "float":
            low_f = float(low)
            high_f = float(high)
            if bool(self.spec.get("log", False)):
                if low_f <= 0.0 or high_f <= 0.0:
                    raise ValueError("Log-uniform float parameters require positive bounds.")
                return float(np.exp(rng.uniform(np.log(low_f), np.log(high_f))))
            return float(rng.uniform(low_f, high_f))
        if kind == "categorical":
            values = list(self.spec.get("choices", ()))
            if not values:
                raise ValueError("Categorical search parameters require choices.")
            return values[int(rng.integers(0, len(values)))]

        raise ValueError(f"Unsupported search parameter type: {kind}")


@dataclass(frozen=True)
class SearchSpace:
    """Model search space split into common and model-specific parameters."""

    common: Mapping[str, SearchSpaceParameter]
    model_specific: Mapping[str, SearchSpaceParameter]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SearchSpace":
        common = {
            str(name): SearchSpaceParameter(spec)
            for name, spec in dict(config.get("common", {})).items()
        }
        model_specific = {
            str(name): SearchSpaceParameter(spec)
            for name, spec in dict(config.get("model_specific", {})).items()
        }
        return cls(common=common, model_specific=model_specific)

    def sample(self, rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
        common = {name: param.sample(rng) for name, param in self.common.items()}
        model_specific = {
            name: param.sample(rng) for name, param in self.model_specific.items()
        }
        return common, model_specific


@dataclass(frozen=True)
class RandomSearchCandidateGenerator:
    """Generate HyperparameterCandidate objects from an experiment config space."""

    model_name: str
    search_space: SearchSpace
    budget: int
    random_seed: int = 0

    def generate(self) -> list[HyperparameterCandidate]:
        if self.budget < 1:
            raise ValueError("Random search budget must be >= 1.")
        rng = np.random.default_rng(self.random_seed)
        candidates: list[HyperparameterCandidate] = []
        for idx in range(self.budget):
            common, model_specific = self.search_space.sample(rng)
            candidates.append(
                HyperparameterCandidate(
                    name=f"{self.model_name}_seed{self.random_seed}_candidate{idx:03d}",
                    common=common,
                    model_specific=model_specific,
                )
            )
        return candidates


__all__ = [
    "RandomSearchCandidateGenerator",
    "SearchSpace",
    "SearchSpaceParameter",
]
