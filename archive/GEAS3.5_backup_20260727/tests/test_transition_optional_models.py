import importlib.util
import unittest
import warnings
from unittest.mock import patch

import pandas as pd

from geas35.models.transition import (
    CatBoostTransitionModel,
    LightGBMTransitionModel,
    MLPTransitionModel,
    OptionalDependencyError,
    TCNTransitionModel,
    TransitionDataset,
    XGBoostTransitionModel,
)


def _dataset():
    x = pd.DataFrame(
        {
            "obs_indoor_temp_c": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            "obs_indoor_humidity_pct": [70.0, 71.0, 72.0, 73.0, 74.0, 75.0],
            "vent_pct": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    y = pd.DataFrame(
        {
            "obs_indoor_temp_c": [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
            "obs_indoor_humidity_pct": [72.0, 73.0, 74.0, 75.0, 76.0, 77.0],
        }
    )
    return TransitionDataset(
        x=x,
        y=y,
        input_columns=(
            "obs_indoor_temp_c",
            "obs_indoor_humidity_pct",
            "vent_pct",
        ),
        target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        observation_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        action_columns=("vent_pct",),
    )


def _block_import(module_name):
    real_import = __import__("importlib").import_module

    def fake_import(name, package=None):
        if name == module_name:
            raise ModuleNotFoundError(name)
        return real_import(name, package=package)

    return fake_import


class TransitionOptionalModelsTest(unittest.TestCase):
    def test_optional_model_capabilities_are_declared(self):
        self.assertEqual(LightGBMTransitionModel.model_name, "lightgbm")
        self.assertEqual(CatBoostTransitionModel.model_name, "catboost")
        self.assertEqual(XGBoostTransitionModel.model_name, "xgboost")
        self.assertEqual(MLPTransitionModel.model_name, "mlp")
        self.assertEqual(TCNTransitionModel.model_name, "tcn")
        self.assertEqual(
            LightGBMTransitionModel.capabilities.multi_output_strategy,
            "independent",
        )
        self.assertEqual(
            TCNTransitionModel.capabilities.multi_output_strategy,
            "native",
        )

    def test_boosting_wrappers_raise_clear_error_when_dependency_missing(self):
        cases = [
            (LightGBMTransitionModel, "lightgbm"),
            (CatBoostTransitionModel, "catboost"),
            (XGBoostTransitionModel, "xgboost"),
        ]
        for model_cls, module_name in cases:
            with self.subTest(model=model_cls.__name__):
                with patch(
                    "importlib.import_module",
                    side_effect=_block_import(module_name),
                ):
                    with self.assertRaises(OptionalDependencyError):
                        model_cls().fit(_dataset())

    def test_mlp_independent_and_native_smoke_when_sklearn_is_available(self):
        if importlib.util.find_spec("sklearn") is None:
            with self.assertRaises(OptionalDependencyError):
                MLPTransitionModel().fit(_dataset())
            return

        dataset = _dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for strategy in ("independent", "native"):
                model = MLPTransitionModel(
                    multi_output_strategy=strategy,
                    hidden_layer_sizes=(4,),
                    max_iter=100,
                    random_state=7,
                ).fit(dataset)
                prediction = model.predict(dataset.x.iloc[:2])
                self.assertEqual(prediction.target_columns, dataset.target_columns)
                self.assertEqual(prediction.next_observation.shape, (2, 2))

    def test_boosting_smoke_only_when_dependencies_are_installed(self):
        cases = [
            (
                LightGBMTransitionModel,
                "lightgbm",
                {"n_estimators": 2, "random_state": 7, "verbose": -1},
            ),
            (
                CatBoostTransitionModel,
                "catboost",
                {"iterations": 2, "random_seed": 7},
            ),
            (
                XGBoostTransitionModel,
                "xgboost",
                {"n_estimators": 2, "random_state": 7, "verbosity": 0},
            ),
        ]
        dataset = _dataset()
        for model_cls, module_name, kwargs in cases:
            with self.subTest(model=model_cls.__name__):
                if importlib.util.find_spec(module_name) is None:
                    self.skipTest(f"{module_name} is not installed")
                model = model_cls(**kwargs).fit(dataset)
                prediction = model.predict(dataset.x.iloc[:2])
                self.assertEqual(prediction.target_columns, dataset.target_columns)
                self.assertEqual(prediction.next_observation.shape, (2, 2))

    def test_tcn_raises_clear_error_when_torch_is_missing(self):
        with patch("importlib.import_module", side_effect=_block_import("torch")):
            with self.assertRaises(OptionalDependencyError):
                TCNTransitionModel(epochs=1).fit(_dataset())

    def test_tcn_smoke_only_when_torch_is_installed(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")
        dataset = _dataset()
        model = TCNTransitionModel(
            epochs=1,
            hidden_channels=4,
            batch_size=3,
            random_state=7,
        ).fit(dataset)
        prediction = model.predict(dataset.x.iloc[:2])
        self.assertEqual(prediction.target_columns, dataset.target_columns)
        self.assertEqual(prediction.next_observation.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
