import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.models.quality import (
    CandidateTrainingResult,
    EarlyStoppingConfig,
    EarlyStoppingTracker,
    GridSearchHyperparameterOptimizer,
    HyperparameterCandidate,
    HyperparameterObjective,
    MetricGoal,
    write_hyperparameter_artifact,
)


class HyperparameterOptimizationTest(unittest.TestCase):
    def test_candidate_rejects_threshold_keys(self):
        with self.assertRaisesRegex(ValueError, "Threshold calibration is separate"):
            HyperparameterCandidate(
                name="bad",
                common={"lookback": 12, "threshold": 0.5},
            )

        with self.assertRaisesRegex(ValueError, "Threshold calibration is separate"):
            HyperparameterCandidate(
                name="bad_nested",
                model_specific={"block": {"score_threshold": 2.0}},
            )

    def test_grid_search_selects_best_validation_config_without_threshold(self):
        train = pd.DataFrame({"in_temp": [20.0, 21.0, 22.0]})
        validation = pd.DataFrame({"in_temp": [23.0, 24.0]})
        candidates = [
            HyperparameterCandidate(name="a", common={"lookback": 6}),
            HyperparameterCandidate(name="b", common={"lookback": 12}),
            HyperparameterCandidate(name="c", common={"lookback": 24}),
        ]
        metrics_by_name = {
            "a": {
                "validation_synthetic_masking_rmse": 2.0,
                "validation_synthetic_masking_mae": 1.0,
                "validation_anomaly_pr_auc": 0.9,
            },
            "b": {
                "validation_synthetic_masking_rmse": 1.0,
                "validation_synthetic_masking_mae": 0.9,
                "validation_anomaly_pr_auc": 0.4,
            },
            "c": {
                "validation_synthetic_masking_rmse": 1.0,
                "validation_synthetic_masking_mae": 0.7,
                "validation_anomaly_pr_auc": 0.2,
            },
        }
        seen = []

        def train_candidate(candidate, train_df, validation_df, columns, early_stopping):
            seen.append((candidate.name, columns, early_stopping.patience))
            return CandidateTrainingResult(
                candidate=candidate,
                validation_metrics=metrics_by_name[candidate.name],
                training_history=[
                    {
                        "epoch": 1,
                        "validation_reconstruction_loss": metrics_by_name[
                            candidate.name
                        ]["validation_synthetic_masking_rmse"],
                    }
                ],
            )

        optimizer = GridSearchHyperparameterOptimizer(
            candidates,
            train_candidate,
            early_stopping=EarlyStoppingConfig(patience=3),
        )

        result = optimizer.optimize(train, validation, ["in_temp"])

        self.assertEqual(result.best_candidate.name, "c")
        self.assertEqual([item[0] for item in seen], ["a", "b", "c"])
        self.assertEqual(seen[0][1], ("in_temp",))
        self.assertEqual(seen[0][2], 3)
        artifact = result.to_artifact()
        self.assertFalse(artifact["threshold_calibration_included"])
        self.assertFalse(artifact["threshold_fields_allowed_in_candidates"])
        self.assertNotIn("threshold", artifact["best_candidate"]["common"])

    def test_custom_objective_can_maximize_validation_metric(self):
        train = pd.DataFrame({"in_temp": [20.0]})
        validation = pd.DataFrame({"in_temp": [21.0]})
        candidates = [
            HyperparameterCandidate(name="small"),
            HyperparameterCandidate(name="large"),
        ]

        def train_candidate(candidate, train_df, validation_df, columns, early_stopping):
            score = 0.1 if candidate.name == "small" else 0.9
            return CandidateTrainingResult(
                candidate=candidate,
                validation_metrics={"validation_anomaly_pr_auc": score},
            )

        optimizer = GridSearchHyperparameterOptimizer(
            candidates,
            train_candidate,
            objective=HyperparameterObjective(
                primary=MetricGoal("validation_anomaly_pr_auc", "max"),
                secondary=(),
            ),
        )

        result = optimizer.optimize(train, validation, ["in_temp"])

        self.assertEqual(result.best_candidate.name, "large")

    def test_early_stopping_tracker_stops_after_patience(self):
        tracker = EarlyStoppingTracker(
            EarlyStoppingConfig(
                monitor="validation_reconstruction_loss",
                patience=1,
                min_delta=0.0,
            )
        )

        self.assertFalse(tracker.update(1, {"validation_reconstruction_loss": 1.0}))
        self.assertFalse(tracker.update(2, {"validation_reconstruction_loss": 1.1}))
        self.assertTrue(tracker.update(3, {"validation_reconstruction_loss": 1.2}))
        self.assertEqual(tracker.best_epoch, 1)
        self.assertEqual(tracker.stopped_epoch, 3)

    def test_write_hyperparameter_artifact_schema(self):
        candidate = HyperparameterCandidate(
            name="modern_tcn_small",
            common={"lookback": 12, "masking_ratio": 0.15},
            model_specific={"dropout": 0.1},
        )
        result = CandidateTrainingResult(
            candidate=candidate,
            validation_metrics={
                "validation_synthetic_masking_rmse": 1.5,
                "validation_synthetic_masking_mae": 1.2,
            },
            training_history=[
                {"epoch": 1, "validation_reconstruction_loss": 1.5}
            ],
        )
        hpo_result = GridSearchHyperparameterOptimizer(
            [candidate],
            lambda candidate, train_df, validation_df, columns, early_stopping: result,
        ).optimize(
            pd.DataFrame({"in_temp": [20.0]}),
            pd.DataFrame({"in_temp": [21.0]}),
            ["in_temp"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hpo.json"
            write_hyperparameter_artifact(path, hpo_result)
            text = path.read_text(encoding="utf-8")

        self.assertIn('"stage": "hyperparameter_optimization"', text)
        self.assertIn('"threshold_calibration_included": false', text)
        self.assertIn('"best_candidate"', text)


if __name__ == "__main__":
    unittest.main()
