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
    ResidualTransitionAdapter,
    TransitionModelCapabilities,
    TransitionPrediction,
    TransitionTargetFormulation,
)

from geas35.rl import MdpV1ObservationScaler

class OptionalDependencyError(ImportError):
    """Raised when a transition model optional dependency is unavailable."""


EstimatorFactory = Callable[[], Any]


class PersistenceTransitionModel(BaseTransitionModel):
    """No-training baseline that carries each dynamic observation forward."""

    model_name = "persistence"
    capabilities = TransitionModelCapabilities(multi_output_strategy="deterministic")

    def __init__(self, *, observation_scaler: MdpV1ObservationScaler | None = None) -> None:
        self.observation_scaler = observation_scaler
        self.target_formulation = TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value
        self.deployment_eligible = False

    def fit(
        self,
        dataset: TransitionDataset,
        validation_dataset: TransitionDataset | None = None,
    ) -> "PersistenceTransitionModel":
        del validation_dataset
        missing_targets = [
            column for column in dataset.target_columns if column not in dataset.input_columns
        ]
        if missing_targets:
            raise ValueError(
                "PersistenceTransitionModel requires every target in the current "
                f"observation inputs: {missing_targets[0]}"
            )
        self.input_columns_ = dataset.input_columns
        self.target_columns_ = dataset.target_columns
        return self

    def predict(
        self,
        dataset_or_frame: TransitionDataset | pd.DataFrame,
    ) -> TransitionPrediction:
        _, target_columns = self._require_fitted()
        frame = (
            dataset_or_frame.x
            if isinstance(dataset_or_frame, TransitionDataset)
            else dataset_or_frame
        )
        missing = [column for column in target_columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing persistence input column: {missing[0]}")
        return TransitionPrediction(
            next_observation=self._physical_targets(frame, target_columns),
            target_columns=target_columns,
        )

    def _physical_targets(self, frame, target_columns):
        if self.observation_scaler is None:
            raise ValueError("Persistence requires the frozen Train observation scaler.")
        return self.observation_scaler.inverse_transform_frame(frame).loc[:, list(target_columns)].copy()


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
        target_estimator_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.target_estimator_kwargs = {
            str(target): dict(params)
            for target, params in (target_estimator_kwargs or {}).items()
        }
        if model_name is not None:
            self.model_name = model_name
        if capabilities is not None:
            self.capabilities = capabilities
        self.estimators_: dict[str, Any] | None = None
        self.target_formulation = TransitionTargetFormulation.ABSOLUTE_STATE.value
        self.observation_scaler: MdpV1ObservationScaler | None = None

    def configure_target_formulation(
        self,
        formulation: TransitionTargetFormulation | str,
        *,
        observation_scaler: MdpV1ObservationScaler | None = None,
    ) -> "IndependentTargetTransitionModel":
        value = TransitionTargetFormulation(formulation).value
        if (
            value == TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value
            and observation_scaler is None
        ):
            raise ValueError("Residual transition models require the frozen Train scaler.")
        self.target_formulation = value
        self.observation_scaler = observation_scaler
        return self

    def fit(
        self,
        dataset: TransitionDataset,
        validation_dataset: TransitionDataset | None = None,
    ) -> "IndependentTargetTransitionModel":
        del validation_dataset
        x = dataset.x.loc[:, list(dataset.input_columns)]
        residual_targets: pd.DataFrame | None = None
        if self.target_formulation == TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value:
            if self.observation_scaler is None:
                raise ValueError("Residual transition models require the frozen Train scaler.")
            current_physical = self.observation_scaler.inverse_transform_frame(x)
            residual_targets = dataset.y.loc[:, list(dataset.target_columns)].subtract(
                current_physical.loc[:, list(dataset.target_columns)]
            )
        estimators: dict[str, Any] = {}
        for target in dataset.target_columns:
            estimator = self.estimator_factory()
            target_kwargs = self.target_estimator_kwargs.get(str(target), {})
            if target_kwargs:
                set_params = getattr(estimator, "set_params", None)
                if not callable(set_params):
                    raise TypeError(
                        f"Estimator for {target} does not support target-specific parameters."
                    )
                estimator.set_params(**target_kwargs)
            fit_target = residual_targets[target] if residual_targets is not None else dataset.y[target]
            estimator.fit(x, fit_target)
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
        raw_prediction = pd.DataFrame(predictions, index=frame.index)
        next_observation = raw_prediction
        if self.target_formulation == TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value:
            if self.observation_scaler is None:
                raise ValueError("Residual transition models require the frozen Train scaler.")
            next_observation = ResidualTransitionAdapter(
                self.observation_scaler
            ).to_physical_next_state(x, raw_prediction, target_columns)
        return TransitionPrediction(
            next_observation=next_observation,
            target_columns=target_columns,
            raw_prediction=(
                raw_prediction
                if self.target_formulation == TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value
                else None
            ),
        )


class LinearRegressionTransitionModel(IndependentTargetTransitionModel):
    """Independent-target wrapper around sklearn LinearRegression."""

    model_name = "linear_regression"
    capabilities = TransitionModelCapabilities(
        supports_native_multi_output=True,
        multi_output_strategy="independent",
        optional_dependency="scikit-learn",
    )

    def __init__(
        self,
        *,
        target_estimator_kwargs: dict[str, dict[str, Any]] | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        self.estimator_kwargs = dict(estimator_kwargs)
        super().__init__(
            self._build_estimator,
            model_name=self.model_name,
            capabilities=self.capabilities,
            target_estimator_kwargs=target_estimator_kwargs,
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

    def __init__(
        self,
        *,
        target_estimator_kwargs: dict[str, dict[str, Any]] | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        self.estimator_kwargs = dict(estimator_kwargs)
        super().__init__(
            self._build_estimator,
            model_name=self.model_name,
            capabilities=self.capabilities,
            target_estimator_kwargs=target_estimator_kwargs,
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

    def __init__(
        self,
        *,
        target_estimator_kwargs: dict[str, dict[str, Any]] | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        self.estimator_kwargs = dict(estimator_kwargs)
        super().__init__(
            self._build_estimator,
            model_name=self.model_name,
            capabilities=self.capabilities,
            target_estimator_kwargs=target_estimator_kwargs,
        )

    def _build_estimator(self) -> Any:
        try:
            sklearn_neighbors = importlib.import_module("sklearn.neighbors")
        except ModuleNotFoundError as exc:
            raise _sklearn_unavailable("KNNTransitionModel") from exc
        return sklearn_neighbors.KNeighborsRegressor(**self.estimator_kwargs)


class ExtraTreesTransitionModel(IndependentTargetTransitionModel):
    """Independent-target wrapper around sklearn ExtraTreesRegressor."""

    model_name = "extra_trees"
    capabilities = TransitionModelCapabilities(
        supports_native_multi_output=True,
        multi_output_strategy="independent",
        optional_dependency="scikit-learn",
    )

    def __init__(
        self,
        *,
        target_estimator_kwargs: dict[str, dict[str, Any]] | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        self.estimator_kwargs = dict(estimator_kwargs)
        super().__init__(
            self._build_estimator,
            model_name=self.model_name,
            capabilities=self.capabilities,
            target_estimator_kwargs=target_estimator_kwargs,
        )

    def _build_estimator(self) -> Any:
        try:
            sklearn_ensemble = importlib.import_module("sklearn.ensemble")
        except ModuleNotFoundError as exc:
            raise _sklearn_unavailable("ExtraTreesTransitionModel") from exc
        return sklearn_ensemble.ExtraTreesRegressor(**self.estimator_kwargs)


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
        target_estimator_kwargs: dict[str, dict[str, Any]] | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        if multi_output_strategy not in {"independent", "native"}:
            raise ValueError("multi_output_strategy must be 'independent' or 'native'.")
        self.multi_output_strategy = multi_output_strategy
        self.estimator_kwargs = dict(estimator_kwargs)
        self.target_estimator_kwargs = {
            str(target): dict(params)
            for target, params in (target_estimator_kwargs or {}).items()
        }
        self.estimator_: Any | None = None
        self.independent_model_: IndependentTargetTransitionModel | None = None
        self.target_formulation = TransitionTargetFormulation.ABSOLUTE_STATE.value
        self.observation_scaler: MdpV1ObservationScaler | None = None

    def configure_target_formulation(
        self,
        formulation: TransitionTargetFormulation | str,
        *,
        observation_scaler: MdpV1ObservationScaler | None = None,
    ) -> "MLPTransitionModel":
        value = TransitionTargetFormulation(formulation).value
        if value == TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value and observation_scaler is None:
            raise ValueError("Residual transition models require the frozen Train scaler.")
        self.target_formulation = value
        self.observation_scaler = observation_scaler
        return self
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
                target_estimator_kwargs=self.target_estimator_kwargs,
            ).configure_target_formulation(
                self.target_formulation, observation_scaler=self.observation_scaler
            ).fit(dataset)
            self.input_columns_ = self.independent_model_.input_columns_
            self.target_columns_ = self.independent_model_.target_columns_
            return self

        estimator = self._build_estimator()
        x = dataset.x.loc[:, list(dataset.input_columns)]
        fit_target = dataset.y.loc[:, list(dataset.target_columns)]
        if self.target_formulation == TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value:
            if self.observation_scaler is None:
                raise ValueError("Residual transition models require the frozen Train scaler.")
            current = self.observation_scaler.inverse_transform_frame(x).loc[:, list(dataset.target_columns)]
            fit_target = fit_target.subtract(current)
        estimator.fit(x, fit_target)
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
        raw_prediction = pd.DataFrame(
            values[:, : len(target_columns)], columns=target_columns, index=frame.index
        )
        next_observation = raw_prediction
        if self.target_formulation == TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value:
            next_observation = ResidualTransitionAdapter(
                self.observation_scaler
            ).to_physical_next_state(frame.loc[:, list(input_columns)], raw_prediction, target_columns)
        return TransitionPrediction(
            next_observation=next_observation,
            target_columns=target_columns,
            raw_prediction=(raw_prediction if self.target_formulation == TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value else None),
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
    "ExtraTreesTransitionModel",
    "IndependentTargetTransitionModel",
    "KNNTransitionModel",
    "LinearRegressionTransitionModel",
    "LinearSVRTransitionModel",
    "MLPTransitionModel",
    "OptionalDependencyError",
    "PersistenceTransitionModel",
    "optional_dependency_unavailable",
]
