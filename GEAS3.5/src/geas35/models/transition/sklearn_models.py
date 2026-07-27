"""Sklearn-backed baseline transition model wrappers."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import Any

import numpy as np
import pandas as pd

from geas35.models.transition.base import (
    BaseTransitionModel,
    TransitionDataset,
    TransitionModelCapabilities,
    TransitionPrediction,
)


class OptionalDependencyError(ImportError):
    """Raised when a transition model optional dependency is unavailable."""


EstimatorFactory = Callable[[], Any]


class IndependentTargetTransitionModel(BaseTransitionModel):
    """Fit one independent estimator per transition target column."""

    model_name = "independent_target_transition"
    capabilities = TransitionModelCapabilities(multi_output_strategy="independent")

    def __init__(
        self,
        estimator_factory: EstimatorFactory,
        *,
        model_name: str | None = None,
        capabilities: TransitionModelCapabilities | None = None,
    ) -> None:
        self.estimator_factory = estimator_factory
        if model_name is not None:
            self.model_name = model_name
        if capabilities is not None:
            self.capabilities = capabilities
        self.estimators_: dict[str, Any] | None = None

    def fit(
        self,
        dataset: TransitionDataset,
        validation_dataset: TransitionDataset | None = None,
    ) -> "IndependentTargetTransitionModel":
        del validation_dataset
        x = dataset.x.loc[:, list(dataset.input_columns)]
        estimators: dict[str, Any] = {}
        for target in dataset.target_columns:
            estimator = self.estimator_factory()
            estimator.fit(x, dataset.y[target])
            estimators[target] = estimator
        self.estimators_ = estimators
        self.input_columns_ = dataset.input_columns
        self.target_columns_ = dataset.target_columns
        return self

    def predict(
        self,
        dataset_or_frame: TransitionDataset | pd.DataFrame,
    ) -> TransitionPrediction:
        input_columns, target_columns = self._require_fitted()
        if self.estimators_ is None:
            raise ValueError("Transition model is not fitted.")
        frame = (
            dataset_or_frame.x
            if isinstance(dataset_or_frame, TransitionDataset)
            else dataset_or_frame
        )
        x = frame.loc[:, list(input_columns)]
        predictions = {
            target: _prediction_1d(self.estimators_[target].predict(x))
            for target in target_columns
        }
        return TransitionPrediction(
            next_observation=pd.DataFrame(predictions, index=frame.index),
            target_columns=target_columns,
        )


class LinearRegressionTransitionModel(IndependentTargetTransitionModel):
    """Independent-target wrapper around sklearn LinearRegression."""

    model_name = "linear_regression"
    capabilities = TransitionModelCapabilities(
        supports_native_multi_output=True,
        multi_output_strategy="independent",
        optional_dependency="scikit-learn",
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
            from sklearn.linear_model import LinearRegression
        except ModuleNotFoundError as exc:
            raise _sklearn_unavailable("LinearRegressionTransitionModel") from exc
        return LinearRegression(**self.estimator_kwargs)


class LinearSVRTransitionModel(IndependentTargetTransitionModel):
    """Independent-target wrapper around sklearn LinearSVR."""

    model_name = "linear_svr"
    capabilities = TransitionModelCapabilities(
        multi_output_strategy="independent",
        optional_dependency="scikit-learn",
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
            from sklearn.svm import LinearSVR
        except ModuleNotFoundError as exc:
            raise _sklearn_unavailable("LinearSVRTransitionModel") from exc
        return LinearSVR(**self.estimator_kwargs)


class KNNTransitionModel(IndependentTargetTransitionModel):
    """Independent-target wrapper around sklearn KNeighborsRegressor."""

    model_name = "knn"
    capabilities = TransitionModelCapabilities(
        supports_native_multi_output=True,
        multi_output_strategy="independent",
        optional_dependency="scikit-learn",
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
            sklearn_neighbors = importlib.import_module("sklearn.neighbors")
        except ModuleNotFoundError as exc:
            raise _sklearn_unavailable("KNNTransitionModel") from exc
        return sklearn_neighbors.KNeighborsRegressor(**self.estimator_kwargs)


class MLPTransitionModel(BaseTransitionModel):
    """Sklearn MLP transition wrapper with selectable multi-output strategy."""

    model_name = "mlp"
    capabilities = TransitionModelCapabilities(
        supports_native_multi_output=True,
        multi_output_strategy="independent",
        optional_dependency="scikit-learn",
    )

    def __init__(
        self,
        *,
        multi_output_strategy: str = "independent",
        **estimator_kwargs: Any,
    ) -> None:
        if multi_output_strategy not in {"independent", "native"}:
            raise ValueError("multi_output_strategy must be 'independent' or 'native'.")
        self.multi_output_strategy = multi_output_strategy
        self.estimator_kwargs = dict(estimator_kwargs)
        self.estimator_: Any | None = None
        self.independent_model_: IndependentTargetTransitionModel | None = None
        self.capabilities = TransitionModelCapabilities(
            supports_native_multi_output=True,
            multi_output_strategy=multi_output_strategy,
            optional_dependency="scikit-learn",
        )

    def fit(
        self,
        dataset: TransitionDataset,
        validation_dataset: TransitionDataset | None = None,
    ) -> "MLPTransitionModel":
        del validation_dataset
        if self.multi_output_strategy == "independent":
            self.independent_model_ = IndependentTargetTransitionModel(
                self._build_estimator,
                model_name=self.model_name,
                capabilities=self.capabilities,
            ).fit(dataset)
            self.input_columns_ = self.independent_model_.input_columns_
            self.target_columns_ = self.independent_model_.target_columns_
            return self

        estimator = self._build_estimator()
        estimator.fit(
            dataset.x.loc[:, list(dataset.input_columns)],
            dataset.y.loc[:, list(dataset.target_columns)],
        )
        self.estimator_ = estimator
        self.input_columns_ = dataset.input_columns
        self.target_columns_ = dataset.target_columns
        return self

    def predict(
        self,
        dataset_or_frame: TransitionDataset | pd.DataFrame,
    ) -> TransitionPrediction:
        input_columns, target_columns = self._require_fitted()
        frame = (
            dataset_or_frame.x
            if isinstance(dataset_or_frame, TransitionDataset)
            else dataset_or_frame
        )
        if self.multi_output_strategy == "independent":
            if self.independent_model_ is None:
                raise ValueError("Transition model is not fitted.")
            return self.independent_model_.predict(frame)
        if self.estimator_ is None:
            raise ValueError("Transition model is not fitted.")
        values = np.asarray(
            self.estimator_.predict(frame.loc[:, list(input_columns)]),
            dtype=float,
        )
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return TransitionPrediction(
            next_observation=pd.DataFrame(
                values[:, : len(target_columns)],
                columns=target_columns,
                index=frame.index,
            ),
            target_columns=target_columns,
        )

    def _build_estimator(self) -> Any:
        try:
            sklearn_neural_network = importlib.import_module("sklearn.neural_network")
        except ModuleNotFoundError as exc:
            raise _sklearn_unavailable("MLPTransitionModel") from exc
        return sklearn_neural_network.MLPRegressor(**self.estimator_kwargs)


def _prediction_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr


def _sklearn_unavailable(model_label: str) -> OptionalDependencyError:
    return optional_dependency_unavailable(model_label, "scikit-learn")


def optional_dependency_unavailable(
    model_label: str,
    dependency: str,
) -> OptionalDependencyError:
    return OptionalDependencyError(
        f"{model_label} requires {dependency}. Install {dependency} before "
        f"using this transition model wrapper."
    )


__all__ = [
    "IndependentTargetTransitionModel",
    "KNNTransitionModel",
    "LinearRegressionTransitionModel",
    "LinearSVRTransitionModel",
    "MLPTransitionModel",
    "OptionalDependencyError",
    "optional_dependency_unavailable",
]
