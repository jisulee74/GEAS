import unittest

import pandas as pd

from geas35.models.transition import (
    ActionProvider,
    BaseTransitionModel,
    LoggedActionProvider,
    PolicyActionProvider,
    TransitionModelCapabilities,
    TransitionPrediction,
    TransitionRolloutSimulator,
)


class PerfectIncrementModel(BaseTransitionModel):
    model_name = "perfect_increment"
    capabilities = TransitionModelCapabilities()

    def __init__(self):
        self.seen_inputs = []

    def fit(self, dataset, validation_dataset=None):
        self.input_columns_ = dataset.input_columns
        self.target_columns_ = dataset.target_columns
        return self

    def predict(self, dataset_or_frame):
        frame = dataset_or_frame.x if hasattr(dataset_or_frame, "x") else dataset_or_frame
        self.seen_inputs.append(frame.copy())
        return TransitionPrediction(
            next_observation=pd.DataFrame(
                {
                    "obs_indoor_temp_c": frame["obs_indoor_temp_c"].astype(float) + 1.0,
                    "obs_indoor_humidity_pct": frame["obs_indoor_humidity_pct"].astype(float) + 2.0,
                },
                index=frame.index,
            ),
            target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        )


def _source_frame(rows=12):
    data = []
    for step in range(rows):
        temp = 20.0 + step
        humidity = 70.0 + 2.0 * step
        data.append(
            {
                "obs_indoor_temp_c": temp,
                "obs_indoor_humidity_pct": humidity,
                "obs_prev_vent_pct": max((step - 1) / 10.0, 0.0),
                "vent_pct": step / 10.0,
                "heat_run": float(step % 2),
                "next_obs_indoor_temp_c": temp + 1.0,
                "next_obs_indoor_humidity_pct": humidity + 2.0,
            }
        )
    return pd.DataFrame(data)


class TransitionRolloutTest(unittest.TestCase):
    def _fitted_model(self):
        model = PerfectIncrementModel()
        model.input_columns_ = (
            "obs_indoor_temp_c",
            "obs_indoor_humidity_pct",
            "obs_prev_vent_pct",
            "vent_pct",
            "heat_run",
        )
        model.target_columns_ = ("obs_indoor_temp_c", "obs_indoor_humidity_pct")
        return model

    def test_logged_action_provider_replays_logged_actions(self):
        provider = LoggedActionProvider(action_columns=("vent_pct", "heat_run"))
        frame = _source_frame(rows=2)
        context = type(
            "Context",
            (),
            {
                "frame": frame,
                "action_columns": ("vent_pct", "heat_run"),
                "target_columns": ("obs_indoor_temp_c",),
            },
        )()
        provider.reset(context)
        step_context = type(
            "StepContext",
            (),
            {
                "frame": frame,
                "source_position": 1,
                "action_columns": ("vent_pct", "heat_run"),
            },
        )()

        self.assertEqual(provider.action(step_context), {"vent_pct": 0.1, "heat_run": 1.0})

    def test_policy_action_provider_is_interface_only(self):
        with self.assertRaises(TypeError):
            PolicyActionProvider()
        with self.assertRaises(TypeError):
            ActionProvider()

    def test_rollout_with_perfect_predictor_has_zero_drift_for_3_6_12_steps(self):
        frame = _source_frame(rows=12)
        for horizon in (3, 6, 12):
            result = TransitionRolloutSimulator(self._fitted_model()).simulate(
                frame,
                start_index=0,
                horizon_steps=horizon,
            )

            self.assertEqual(len(result.predictions), horizon)
            self.assertAlmostEqual(result.metrics["trajectory_mae"], 0.0)
            self.assertAlmostEqual(result.metrics["trajectory_rmse"], 0.0)
            self.assertAlmostEqual(result.metrics["final_step_mae"], 0.0)
            self.assertAlmostEqual(result.metrics["drift_slope_mae"], 0.0)

    def test_rollout_updates_next_step_dynamic_state_with_prediction(self):
        frame = _source_frame(rows=3)
        frame.loc[1, "obs_indoor_temp_c"] = 999.0
        model = self._fitted_model()

        TransitionRolloutSimulator(model).simulate(
            frame,
            start_index=0,
            horizon_steps=2,
        )

        second_input = model.seen_inputs[1]
        self.assertAlmostEqual(second_input["obs_indoor_temp_c"].iloc[0], 21.0)
        self.assertAlmostEqual(second_input["obs_prev_vent_pct"].iloc[0], 0.0)
        self.assertAlmostEqual(second_input["vent_pct"].iloc[0], 0.1)

    def test_rollout_raises_when_truth_column_is_missing(self):
        frame = _source_frame(rows=3).drop(columns=["next_obs_indoor_humidity_pct"])

        with self.assertRaises(KeyError):
            TransitionRolloutSimulator(self._fitted_model()).simulate(
                frame,
                start_index=0,
                horizon_steps=2,
            )

    def test_rollout_raises_when_horizon_exceeds_source_rows(self):
        with self.assertRaises(ValueError):
            TransitionRolloutSimulator(self._fitted_model()).simulate(
                _source_frame(rows=2),
                start_index=0,
                horizon_steps=3,
            )


if __name__ == "__main__":
    unittest.main()
