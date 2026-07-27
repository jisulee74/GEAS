import unittest

import pandas as pd

from geas35.models.transition import (
    BaseTransitionModel,
    TransitionDataset,
    TransitionModelCapabilities,
    TransitionPrediction,
)


class DummyTransitionModel(BaseTransitionModel):
    model_name = "dummy_transition"
    capabilities = TransitionModelCapabilities(supports_native_multi_output=True)

    def fit(self, dataset, validation_dataset=None):
        self.input_columns_ = dataset.input_columns
        self.target_columns_ = dataset.target_columns
        return self

    def predict(self, dataset_or_frame):
        _, target_columns = self._require_fitted()
        row_count = len(dataset_or_frame.x) if isinstance(dataset_or_frame, TransitionDataset) else len(dataset_or_frame)
        return TransitionPrediction(
            next_observation=pd.DataFrame(
                {col: [0.0] * row_count for col in target_columns}
            ),
            target_columns=target_columns,
        )


class TransitionBaseTest(unittest.TestCase):
    def test_transition_dataset_preserves_columns(self):
        dataset = TransitionDataset(
            x=pd.DataFrame({"obs_temp": [1.0, 2.0], "vent_pct": [0.0, 0.5]}),
            y=pd.DataFrame({"obs_next_temp": [1.5, 2.5]}),
            metadata=pd.DataFrame({"reg_date": ["a", "b"]}),
            input_columns=("obs_temp", "vent_pct"),
            target_columns=("obs_next_temp",),
            observation_columns=("obs_temp",),
            action_columns=("vent_pct",),
        )

        self.assertEqual(dataset.input_columns, ("obs_temp", "vent_pct"))
        self.assertEqual(dataset.target_columns, ("obs_next_temp",))
        self.assertEqual(dataset.observation_columns, ("obs_temp",))
        self.assertEqual(dataset.action_columns, ("vent_pct",))

    def test_transition_dataset_rejects_row_mismatch(self):
        with self.assertRaises(ValueError):
            TransitionDataset(
                x=pd.DataFrame({"x": [1.0, 2.0]}),
                y=pd.DataFrame({"y": [1.0]}),
                input_columns=("x",),
                target_columns=("y",),
            )

    def test_transition_dataset_rejects_missing_input_column(self):
        with self.assertRaises(KeyError):
            TransitionDataset(
                x=pd.DataFrame({"x": [1.0]}),
                y=pd.DataFrame({"y": [1.0]}),
                input_columns=("missing_x",),
                target_columns=("y",),
            )

    def test_transition_dataset_rejects_missing_target_column(self):
        with self.assertRaises(KeyError):
            TransitionDataset(
                x=pd.DataFrame({"x": [1.0]}),
                y=pd.DataFrame({"y": [1.0]}),
                input_columns=("x",),
                target_columns=("missing_y",),
            )

    def test_transition_prediction_rejects_missing_target_column(self):
        with self.assertRaises(KeyError):
            TransitionPrediction(
                next_observation=pd.DataFrame({"obs_temp": [1.0]}),
                target_columns=("obs_humidity",),
            )

    def test_base_transition_model_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseTransitionModel()

    def test_dummy_model_requires_fit_before_predict(self):
        model = DummyTransitionModel()
        with self.assertRaises(ValueError):
            model.predict(pd.DataFrame({"obs_temp": [1.0]}))

    def test_dummy_model_fit_and_predict_uses_common_interface(self):
        dataset = TransitionDataset(
            x=pd.DataFrame({"obs_temp": [1.0, 2.0], "vent_pct": [0.0, 0.5]}),
            y=pd.DataFrame({"obs_next_temp": [1.5, 2.5]}),
            input_columns=("obs_temp", "vent_pct"),
            target_columns=("obs_next_temp",),
        )

        model = DummyTransitionModel().fit(dataset)
        prediction = model.predict(dataset)

        self.assertEqual(model.input_columns_, ("obs_temp", "vent_pct"))
        self.assertEqual(model.target_columns_, ("obs_next_temp",))
        self.assertEqual(prediction.target_columns, ("obs_next_temp",))
        self.assertEqual(len(prediction.next_observation), 2)


if __name__ == "__main__":
    unittest.main()
