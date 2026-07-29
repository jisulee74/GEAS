"""Optional gradient-boosting transition model wrappers."""

from __future__ import annotations

import importlib
from typing import Any

from geas35.models.transition.base import TransitionModelCapabilities
from geas35.models.transition.sklearn_models import (
    IndependentTargetTransitionModel,
    optional_dependency_unavailable,
)


class LightGBMTransitionModel(IndependentTargetTransitionModel):
    """Independent-target wrapper around lightgbm.LGBMRegressor."""

    model_name = "lightgbm"
    capabilities = TransitionModelCapabilities(
        multi_output_strategy="independent",
        optional_dependency="lightgbm",
    )

    def __init__(self, **estimator_kwargs: Any) -> None:
        self.estimator_kwargs = dict(estimator_kwargs)
        super().__init__(
            self._build_estimator,
            model_name=self.model_name,
            capabilities=self.capabilities,
        )

    def _build_estimator(self) -> Any:
        try:
            lightgbm = importlib.import_module("lightgbm")
        except ModuleNotFoundError as exc:
            raise optional_dependency_unavailable(
                "LightGBMTransitionModel",
                "lightgbm",
            ) from exc
        return lightgbm.LGBMRegressor(**self.estimator_kwargs)


class XGBoostTransitionModel(IndependentTargetTransitionModel):
    """Independent-target wrapper around xgboost.XGBRegressor."""

    model_name = "xgboost"
    capabilities = TransitionModelCapabilities(
        multi_output_strategy="independent",
        optional_dependency="xgboost",
    )

    def __init__(self, **estimator_kwargs: Any) -> None:
        self.estimator_kwargs = dict(estimator_kwargs)
        super().__init__(
            self._build_estimator,
            model_name=self.model_name,
            capabilities=self.capabilities,
        )

    def _build_estimator(self) -> Any:
        try:
            xgboost = importlib.import_module("xgboost")
        except ModuleNotFoundError as exc:
            raise optional_dependency_unavailable(
                "XGBoostTransitionModel",
                "xgboost",
            ) from exc
        return xgboost.XGBRegressor(**self.estimator_kwargs)


__all__ = [
    "LightGBMTransitionModel",
    "XGBoostTransitionModel",
]
