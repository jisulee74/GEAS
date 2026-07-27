import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.models.transition import (
    BaseTransitionModel,
    TransitionModelCapabilities,
    TransitionPrediction,
    build_predicted_next_row,
    load_selected_transition_model,
    predict_next_observation,
    predict_next_observation_reward,
    save_selected_transition_model,
    save_transition_model_artifact,
)
from geas35.models.transition.selector import TransitionModelSelectionResult


class AdditiveInferenceModel(BaseTransitionModel):
    model_name = "additive_inference"
    capabilities = TransitionModelCapabilities()

    def fit(self, dataset, validation_dataset=None):
        self.input_columns_ = dataset.input_columns
        self.target_columns_ = dataset.target_columns
        return self

    def predict(self, dataset_or_frame):
        frame = dataset_or_frame.x if hasattr(dataset_or_frame, "x") else dataset_or_frame
        return TransitionPrediction(
            next_observation=pd.DataFrame(
                {
                    "obs_indoor_temp_c": frame["obs_indoor_temp_c"].astype(float)
                    + 1.0
                    + frame["vent_pct"].astype(float),
                    "obs_indoor_humidity_pct": frame[
                        "obs_indoor_humidity_pct"
                    ].astype(float)
                    + 2.0,
                },
                index=frame.index,
            ),
            target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        )


def _fitted_model():
    model = AdditiveInferenceModel()
    model.input_columns_ = (
        "obs_indoor_temp_c",
        "obs_indoor_humidity_pct",
        "vent_pct",
        "heat_run",
    )
    model.target_columns_ = ("obs_indoor_temp_c", "obs_indoor_humidity_pct")
    return model


def _current_row():
    return pd.Series(
        {
            "obs_indoor_temp_c": 20.0,
            "obs_indoor_humidity_pct": 70.0,
            "obs_current_target_temp_min_c": 20.0,
            "obs_current_target_temp_max_c": 22.0,
            "obs_current_target_temp_c": 21.0,
            "obs_current_vpd_kpa": 1.0,
            "obs_current_condensation_margin_c": 4.0,
            "obs_outdoor_temp_c": 18.0,
            "obs_outdoor_wind_speed": 0.0,
        }
    )


def _action():
    return {
        "vent_pct": 0.5,
        "shade_curtain_pct": 0.0,
        "thermal_curtain_pct": 0.0,
        "heat_run": 0.0,
        "cool_run": 0.0,
        "fan_run": 0.0,
    }


class TransitionInferenceTest(unittest.TestCase):
    def test_load_selected_transition_model_loads_selected_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = save_transition_model_artifact(
                _fitted_model(),
                tmp,
                "tomato",
                model_name="additive_inference",
                training_summary={"train_rows": 3},
            )
            save_selected_transition_model(
                tmp,
                "tomato",
                TransitionModelSelectionResult(
                    selected_model_name="additive_inference",
                    selected_score=0.0,
                    strategy_name="mean_rmse",
                    higher_is_better=False,
                    candidate_scores=(),
                ),
                metadata={
                    "candidate_artifacts": {
                        "additive_inference": str(artifact.artifact_dir)
                    }
                },
            )

            loaded = load_selected_transition_model(tmp, "tomato")

            self.assertEqual(loaded.crop, "tomato")
            self.assertEqual(loaded.model_name, "additive_inference")
            self.assertEqual(loaded.model_artifact_dir, artifact.artifact_dir)
            self.assertIsNotNone(loaded.model_manifest)
            prediction = predict_next_observation(
                loaded,
                _current_row(),
                _action(),
            )
            self.assertAlmostEqual(
                prediction.next_observation["obs_indoor_temp_c"].iloc[0],
                21.5,
            )

    def test_predict_next_observation_accepts_direct_model_and_action_mapping(self):
        prediction = predict_next_observation(
            _fitted_model(),
            _current_row(),
            _action(),
        )

        self.assertEqual(
            prediction.target_columns,
            ("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        )
        self.assertAlmostEqual(
            prediction.next_observation["obs_indoor_humidity_pct"].iloc[0],
            72.0,
        )

    def test_build_predicted_next_row_overlays_dynamic_targets_only(self):
        current = _current_row()
        prediction = predict_next_observation(_fitted_model(), current, _action())

        next_row = build_predicted_next_row(current, prediction)

        self.assertAlmostEqual(next_row["obs_indoor_temp_c"], 21.5)
        self.assertAlmostEqual(next_row["obs_indoor_humidity_pct"], 72.0)
        self.assertAlmostEqual(next_row["obs_current_target_temp_c"], 21.0)

    def test_predict_next_observation_reward_uses_existing_mdp_v1_reward(self):
        result = predict_next_observation_reward(
            _fitted_model(),
            _current_row(),
            _action(),
        )

        self.assertAlmostEqual(result.next_row["obs_indoor_temp_c"], 21.5)
        self.assertAlmostEqual(result.next_row["obs_indoor_humidity_pct"], 72.0)
        self.assertLessEqual(result.reward, 0.0)
        self.assertIn("raw_temp_penalty", result.reward_terms)
        self.assertIn("raw_rh_penalty", result.reward_terms)
        self.assertEqual(result.reward_terms["raw_temp_penalty"], 0.0)

    def test_sequence_action_maps_to_mdp_v1_action_columns(self):
        prediction = predict_next_observation(
            _fitted_model(),
            _current_row(),
            [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

        self.assertAlmostEqual(
            prediction.next_observation["obs_indoor_temp_c"].iloc[0],
            21.25,
        )


if __name__ == "__main__":
    unittest.main()
