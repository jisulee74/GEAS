import unittest

import numpy as np
import pandas as pd

from geas35.models.transition import (
    IndependentTargetTransitionModel,
    KNNTransitionModel,
    LinearRegressionTransitionModel,
    LinearSVRTransitionModel,
    OptionalDependencyError,
    TransitionDataset,
)


class MeanEstimator:
    def fit(self, x, y):
        self.row_count_ = len(x)
        self.mean_ = float(pd.Series(y).mean())
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


class LinearEstimator:
    def fit(self, x, y):
        x_values = np.asarray(x, dtype=float)
        y_values = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(len(x_values)), x_values])
        self.coef_, *_ = np.linalg.lstsq(design, y_values, rcond=None)
        return self

    def predict(self, x):
        x_values = np.asarray(x, dtype=float)
        design = np.column_stack([np.ones(len(x_values)), x_values])
        return design @ self.coef_


def _dataset():
    x = pd.DataFrame(
        {
            "obs_indoor_temp_c": [20.0, 21.0, 22.0, 23.0],
            "vent_pct": [0.0, 0.1, 0.2, 0.3],
        }
    )
    y = pd.DataFrame(
        {
            "obs_indoor_temp_c": [20.5, 21.5, 22.5, 23.5],
            "obs_indoor_humidity_pct": [80.0, 79.0, 78.0, 77.0],
        }
    )
    return TransitionDataset(
        x=x,
        y=y,
        input_columns=("obs_indoor_temp_c", "vent_pct"),
        target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        observation_columns=("obs_indoor_temp_c",),
        action_columns=("vent_pct",),
    )


class TransitionSklearnModelsTest(unittest.TestCase):
    def test_independent_target_model_fits_one_estimator_per_target(self):
        created = []

        def factory():
            estimator = MeanEstimator()
            created.append(estimator)
            return estimator

        dataset = _dataset()
        model = IndependentTargetTransitionModel(factory).fit(dataset)

        self.assertEqual(len(created), 2)
        self.assertEqual(set(model.estimators_), set(dataset.target_columns))
        self.assertEqual(model.input_columns_, dataset.input_columns)
        self.assertEqual(model.target_columns_, dataset.target_columns)

    def test_independent_target_model_predicts_dataframe_with_target_schema(self):
        dataset = _dataset()
        model = IndependentTargetTransitionModel(MeanEstimator).fit(dataset)

        prediction = model.predict(dataset.x.iloc[:2])

        self.assertEqual(prediction.target_columns, dataset.target_columns)
        self.assertEqual(list(prediction.next_observation.columns), list(dataset.target_columns))
        self.assertEqual(list(prediction.next_observation.index), [0, 1])
        self.assertAlmostEqual(
            prediction.next_observation["obs_indoor_temp_c"].iloc[0],
            22.0,
        )

    def test_independent_target_model_accepts_transition_dataset_for_prediction(self):
        dataset = _dataset()
        model = IndependentTargetTransitionModel(LinearEstimator).fit(dataset)

        prediction = model.predict(dataset)

        self.assertEqual(len(prediction.next_observation), len(dataset.x))
        self.assertAlmostEqual(
            prediction.next_observation["obs_indoor_temp_c"].iloc[0],
            20.5,
        )

    def test_independent_target_model_rejects_prediction_before_fit(self):
        with self.assertRaises(ValueError):
            IndependentTargetTransitionModel(MeanEstimator).predict(_dataset().x)

    def test_sklearn_wrappers_are_lazy_and_report_capabilities(self):
        self.assertEqual(LinearRegressionTransitionModel.model_name, "linear_regression")
        self.assertEqual(LinearSVRTransitionModel.model_name, "linear_svr")
        self.assertEqual(KNNTransitionModel.model_name, "knn")
        self.assertEqual(
            LinearRegressionTransitionModel.capabilities.optional_dependency,
            "scikit-learn",
        )
        self.assertEqual(
            LinearSVRTransitionModel.capabilities.multi_output_strategy,
            "independent",
        )

    def test_sklearn_wrappers_raise_clear_error_when_dependency_missing(self):
        try:
            import sklearn  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaises(OptionalDependencyError):
                LinearRegressionTransitionModel().fit(_dataset())
        else:
            prediction = LinearRegressionTransitionModel().fit(_dataset()).predict(_dataset())
            self.assertEqual(prediction.target_columns, _dataset().target_columns)


if __name__ == "__main__":
    unittest.main()
