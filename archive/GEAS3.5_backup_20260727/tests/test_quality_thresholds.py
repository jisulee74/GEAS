import unittest

import numpy as np
import pandas as pd

from geas35.models.quality import (
    GridSearchThresholdOptimizer,
    MedianQualityModel,
    RuleOnlyQualityModel,
    evaluate_synthetic_anomaly_detection,
    make_synthetic_anomaly_injection,
)


class SyntheticAnomalyDetectionEvaluationTest(unittest.TestCase):
    def test_synthetic_anomaly_injection_marks_ground_truth_cells(self):
        df = pd.DataFrame(
            {
                "in_temp": [20.0, 21.0, np.nan, 22.0],
                "in_temp_missing_flag": [0, 0, 1, 0],
            }
        )

        injection = make_synthetic_anomaly_injection(
            df,
            ["in_temp"],
            anomaly_fraction=1.0,
            anomaly_scale=5.0,
            random_state=1,
        )

        self.assertEqual(injection.injected_cells, 3)
        self.assertEqual(injection.anomaly_mask["in_temp"].tolist(), [True, True, False, True])
        self.assertTrue(pd.isna(injection.frame.loc[2, "in_temp"]))
        self.assertNotEqual(injection.frame.loc[0, "in_temp"], df.loc[0, "in_temp"])

    def test_anomaly_detection_metrics_include_confusion_auc_and_pr_auc(self):
        train = pd.DataFrame({"in_temp": [20.0, 21.0, 22.0]})
        validation = pd.DataFrame(
            {
                "in_temp": [20.0, 21.0, 22.0, 23.0],
                "in_temp_missing_flag": [0, 0, 0, 0],
                "in_temp_rule_outlier_flag": [0, 0, 0, 0],
            }
        )
        model = MedianQualityModel().fit(train, ["in_temp"])

        result = evaluate_synthetic_anomaly_detection(
            model,
            validation,
            ["in_temp"],
            threshold=2.0,
            anomaly_fraction=0.5,
            anomaly_scale=10.0,
            random_state=0,
        )

        cm = result.metrics.confusion_matrix
        self.assertEqual(result.injected_cells, 2)
        self.assertEqual(cm.true_positive, 2)
        self.assertGreaterEqual(result.metrics.precision, 0.0)
        self.assertGreaterEqual(result.metrics.recall, 0.0)
        self.assertGreaterEqual(result.metrics.f1_score, 0.0)
        self.assertTrue(np.isfinite(result.metrics.roc_auc))
        self.assertTrue(np.isfinite(result.metrics.pr_auc))

    def test_rule_only_model_can_be_evaluated_with_same_interface(self):
        validation = pd.DataFrame(
            {
                "in_temp": [20.0, 21.0, 22.0],
                "in_temp_missing_flag": [0, 0, 0],
                "in_temp_rule_outlier_flag": [0, 0, 0],
            }
        )
        model = RuleOnlyQualityModel().fit(validation, ["in_temp"])

        result = evaluate_synthetic_anomaly_detection(
            model,
            validation,
            ["in_temp"],
            threshold=0.5,
            anomaly_fraction=1.0,
            random_state=0,
        )

        self.assertEqual(result.injected_cells, 3)
        self.assertEqual(result.metrics.recall, 0.0)
        self.assertEqual(result.metrics.confusion_matrix.false_negative, 3)


class GridSearchThresholdOptimizerTest(unittest.TestCase):
    def test_grid_search_selects_best_validation_threshold_and_reports_metrics(self):
        train = pd.DataFrame({"in_temp": [20.0, 21.0, 22.0]})
        validation = pd.DataFrame(
            {
                "in_temp": [20.0, 21.0, 22.0, 23.0],
                "in_temp_missing_flag": [0, 0, 0, 0],
                "in_temp_rule_outlier_flag": [0, 0, 0, 0],
            }
        )
        model = MedianQualityModel().fit(train, ["in_temp"])
        optimizer = GridSearchThresholdOptimizer(
            [0.1, 2.0, 100.0],
            objective_metric="f1_score",
            anomaly_fraction=0.5,
            anomaly_scale=10.0,
            mask_fraction=0.5,
            random_state=0,
        )

        result = optimizer.optimize(model, validation, ["in_temp"])

        self.assertIn(result.best_threshold, [0.1, 2.0, 100.0])
        self.assertEqual(result.objective_metric, "f1_score")
        self.assertEqual(len(result.detection_results), 3)
        self.assertIsNotNone(result.masking_result)
        self.assertGreaterEqual(result.best_detection_result.metrics.f1_score, 0.0)

    def test_grid_search_rejects_empty_candidates(self):
        with self.assertRaisesRegex(ValueError, "candidate_thresholds"):
            GridSearchThresholdOptimizer([])


if __name__ == "__main__":
    unittest.main()
