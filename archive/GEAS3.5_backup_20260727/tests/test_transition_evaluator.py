import unittest

import pandas as pd

from geas35.models.transition import (
    IndependentTargetTransitionModel,
    TransitionDataset,
    TransitionPrediction,
    evaluate_transition_model_one_step,
    evaluate_transition_predictions,
)


class MemorizedMeanEstimator:
    def fit(self, x, y):
        self.value_ = float(pd.Series(y).mean())
        return self

    def predict(self, x):
        return [self.value_] * len(x)


def _true_frame():
    return pd.DataFrame(
        {
            "obs_indoor_temp_c": [1.0, 2.0, 3.0, 4.0],
            "obs_indoor_humidity_pct": [80.0, 81.0, 82.0, 83.0],
            "obs_outdoor_temp_c": [10.0, 11.0, 12.0, 13.0],
        }
    )


class TransitionEvaluatorTest(unittest.TestCase):
    def test_evaluate_transition_predictions_reports_target_and_aggregate_metrics(self):
        y_true = _true_frame()
        y_pred = y_true.copy()
        y_pred.loc[3, "obs_indoor_temp_c"] = 6.0

        report = evaluate_transition_predictions(y_true, y_pred)
        temp_metrics = report.target_metrics["obs_indoor_temp_c"]

        self.assertEqual(report.row_count, 4)
        self.assertEqual(
            report.target_columns,
            (
                "obs_indoor_temp_c",
                "obs_indoor_humidity_pct",
                "obs_outdoor_temp_c",
            ),
        )
        self.assertIn("r2", temp_metrics)
        self.assertIn("mae", temp_metrics)
        self.assertIn("rmse", temp_metrics)
        self.assertIn("q90", temp_metrics)
        self.assertIn("cvar90", temp_metrics)
        self.assertAlmostEqual(temp_metrics["mae"], 0.5)
        self.assertAlmostEqual(report.aggregate_metrics["indoor_temp_rmse"], 1.0)
        self.assertIn("normalized_mean_rmse", report.aggregate_metrics)

    def test_evaluate_transition_predictions_accepts_transition_prediction(self):
        y_true = _true_frame()
        prediction = TransitionPrediction(
            next_observation=y_true.copy(),
            target_columns=tuple(y_true.columns),
        )

        report = evaluate_transition_predictions(y_true, prediction)

        self.assertAlmostEqual(report.aggregate_metrics["mean_rmse"], 0.0)

    def test_evaluate_transition_predictions_supports_target_groups(self):
        y_true = _true_frame()
        y_pred = y_true.copy()
        y_pred["obs_indoor_humidity_pct"] = y_pred["obs_indoor_humidity_pct"] + 1.0

        report = evaluate_transition_predictions(
            y_true,
            y_pred,
            target_groups={"custom_indoor": ("obs_indoor_temp_c", "obs_indoor_humidity_pct")},
        )

        self.assertIn("custom_indoor", report.target_group_metrics)
        self.assertIn("mean_rmse", report.target_group_metrics["custom_indoor"])

    def test_evaluate_transition_predictions_uses_explicit_normalization_scales(self):
        y_true = _true_frame()[["obs_indoor_temp_c"]]
        y_pred = pd.DataFrame({"obs_indoor_temp_c": [2.0, 3.0, 4.0, 5.0]})

        report = evaluate_transition_predictions(
            y_true,
            y_pred,
            normalization_scales={"obs_indoor_temp_c": 2.0},
        )

        self.assertAlmostEqual(report.aggregate_metrics["mean_rmse"], 1.0)
        self.assertAlmostEqual(report.aggregate_metrics["normalized_mean_rmse"], 0.5)

    def test_evaluate_transition_predictions_rejects_missing_columns(self):
        with self.assertRaises(KeyError):
            evaluate_transition_predictions(
                _true_frame(),
                pd.DataFrame({"obs_indoor_temp_c": [1.0, 2.0, 3.0, 4.0]}),
                target_columns=("obs_indoor_humidity_pct",),
            )

    def test_evaluate_transition_model_one_step_runs_model_prediction(self):
        y_true = _true_frame()[["obs_indoor_temp_c", "obs_indoor_humidity_pct"]]
        dataset = TransitionDataset(
            x=pd.DataFrame({"obs_indoor_temp_c": [1.0, 2.0, 3.0, 4.0]}),
            y=y_true,
            input_columns=("obs_indoor_temp_c",),
            target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        )
        model = IndependentTargetTransitionModel(MemorizedMeanEstimator).fit(dataset)

        report = evaluate_transition_model_one_step(model, dataset)

        self.assertEqual(report.target_columns, dataset.target_columns)
        self.assertIn("mean_rmse", report.aggregate_metrics)

    def test_report_to_artifact_converts_nan_to_none(self):
        report = evaluate_transition_predictions(
            pd.DataFrame({"obs_indoor_temp_c": [1.0, 1.0]}),
            pd.DataFrame({"obs_indoor_temp_c": [1.0, 1.0]}),
        )

        artifact = report.to_artifact()

        self.assertEqual(artifact["stage"], "transition_one_step_evaluation")
        self.assertIsNone(artifact["target_metrics"]["obs_indoor_temp_c"]["r2"])
        self.assertEqual(artifact["aggregate_metrics"]["mean_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
