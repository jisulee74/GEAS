import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.models.quality import (
    CandidateTrainingResult,
    EarlyStoppingConfig,
    HyperparameterCandidate,
    HyperparameterObjective,
    HyperparameterOptimizationResult,
    MedianQualityModel,
    QualityModelSelectionCandidate,
    build_quality_model_comparison_report,
    evaluate_quality_model_for_report,
    measure_online_inference_efficiency,
    write_quality_model_comparison_report,
)


def _frame(values):
    return pd.DataFrame(
        {
            "in_temp": values,
            "in_temp_missing_flag": [0] * len(values),
            "in_temp_rule_outlier_flag": [0] * len(values),
        }
    )


def _hpo_result(name: str, rmse: float) -> HyperparameterOptimizationResult:
    candidate = HyperparameterCandidate(
        name=f"{name}_best",
        common={"lookback": 4, "mask_fraction": 0.2},
        model_specific={"width": 8},
    )
    training_result = CandidateTrainingResult(
        candidate=candidate,
        validation_metrics={
            "validation_synthetic_masking_rmse": rmse,
            "validation_synthetic_masking_mae": rmse / 2.0,
        },
        training_history=[
            {"epoch": 1, "validation_reconstruction_loss": rmse}
        ],
    )
    return HyperparameterOptimizationResult(
        best_result=training_result,
        candidate_results=[training_result],
        objective=HyperparameterObjective(),
        early_stopping=EarlyStoppingConfig(patience=2),
    )


class QualitySelectionReportTest(unittest.TestCase):
    def test_evaluation_report_includes_required_metrics_and_efficiency(self):
        validation = _frame([20.0, 21.0, 22.0, 23.0])
        model = MedianQualityModel().fit(validation, ["in_temp"])

        report = evaluate_quality_model_for_report(
            model,
            validation,
            ["in_temp"],
            split="validation",
            calibrated_threshold=2.0,
            mask_fraction=0.5,
            anomaly_fraction=0.5,
            random_state=0,
            efficiency_repeats=1,
        )

        self.assertEqual(report.split, "validation")
        self.assertGreater(report.reconstruction.masked_cells, 0)
        self.assertIsInstance(report.reconstruction.mae, float)
        self.assertIsInstance(report.reconstruction.rmse, float)
        self.assertGreaterEqual(report.anomaly_detection.precision, 0.0)
        self.assertGreaterEqual(report.anomaly_detection.recall, 0.0)
        self.assertGreaterEqual(report.anomaly_detection.f1_score, 0.0)
        self.assertIn("in_temp", report.anomaly_detection.per_column)
        self.assertIsNotNone(report.online_efficiency)
        self.assertGreaterEqual(
            report.online_efficiency.inference_latency_ms_per_row,
            0.0,
        )
        self.assertGreaterEqual(report.online_efficiency.peak_memory_bytes, 0)
        self.assertGreater(report.online_efficiency.model_size_bytes, 0)

    def test_comparison_report_selects_by_validation_and_keeps_test_separate(self):
        validation = _frame([20.0, 21.0, 22.0, 23.0])
        test = _frame([100.0, 101.0, 102.0, 103.0])
        validation_good = MedianQualityModel().fit(validation, ["in_temp"])
        validation_bad = MedianQualityModel().fit(_frame([100.0, 101.0, 102.0]), ["in_temp"])
        candidates = [
            QualityModelSelectionCandidate(
                model_name="validation_good",
                model=validation_good,
                calibrated_threshold=2.0,
                hyperparameter_result=_hpo_result("validation_good", rmse=1.0),
            ),
            QualityModelSelectionCandidate(
                model_name="validation_bad",
                model=validation_bad,
                calibrated_threshold=2.0,
                hyperparameter_result=_hpo_result("validation_bad", rmse=10.0),
            ),
        ]

        report = build_quality_model_comparison_report(
            candidates,
            validation,
            ["in_temp"],
            test_df=test,
            mask_fraction=0.5,
            anomaly_fraction=0.5,
            random_state=0,
            efficiency_repeats=1,
        )

        self.assertEqual(report.stage, "quality_model_final_selection")
        self.assertEqual(report.selection_split, "validation")
        self.assertEqual(report.test_split, "test")
        self.assertFalse(report.test_used_for_selection)
        self.assertEqual(report.selected_model, "validation_good")
        self.assertEqual(report.selected_threshold, 2.0)
        selected_entries = [entry for entry in report.entries if entry.selected]
        self.assertEqual(len(selected_entries), 1)
        self.assertEqual(selected_entries[0].model_name, "validation_good")
        for entry in report.entries:
            self.assertIsNotNone(entry.test)
            self.assertIsNotNone(entry.hyperparameter_selection)
            self.assertEqual(entry.calibrated_threshold, 2.0)
            self.assertFalse(
                entry.hyperparameter_selection["threshold_calibration_included"]
            )
            self.assertIn("best_candidate", entry.hyperparameter_selection)

    def test_report_artifact_json_schema(self):
        validation = _frame([20.0, 21.0, 22.0, 23.0])
        model = MedianQualityModel().fit(validation, ["in_temp"])
        report = build_quality_model_comparison_report(
            [
                QualityModelSelectionCandidate(
                    model_name="median",
                    model=model,
                    calibrated_threshold=2.0,
                    hyperparameter_result=_hpo_result("median", rmse=1.0),
                )
            ],
            validation,
            ["in_temp"],
            mask_fraction=0.5,
            anomaly_fraction=0.5,
            random_state=0,
            efficiency_repeats=1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality_model_selection.json"
            write_quality_model_comparison_report(path, report)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["stage"], "quality_model_final_selection")
        self.assertEqual(payload["selected_model"], "median")
        self.assertFalse(payload["test_used_for_selection"])
        entry = payload["entries"][0]
        self.assertIn("calibrated_threshold", entry)
        self.assertIn("hyperparameter_selection", entry)
        self.assertIn("reconstruction", entry["validation"])
        self.assertIn("anomaly_detection", entry["validation"])
        self.assertIn("online_efficiency", entry["validation"])
        anomaly = entry["validation"]["anomaly_detection"]
        self.assertIn("precision", anomaly)
        self.assertIn("recall", anomaly)
        self.assertIn("f1_score", anomaly)
        self.assertIn("roc_auc", anomaly)
        self.assertIn("pr_auc", anomaly)
        efficiency = entry["validation"]["online_efficiency"]
        self.assertIn("inference_latency_ms_per_row", efficiency)
        self.assertIn("model_size_bytes", efficiency)

    def test_efficiency_measure_rejects_invalid_repeats(self):
        model = MedianQualityModel().fit(_frame([20.0, 21.0]), ["in_temp"])

        with self.assertRaisesRegex(ValueError, "repeats"):
            measure_online_inference_efficiency(
                model,
                _frame([20.0, 21.0]),
                ["in_temp"],
                threshold=2.0,
                repeats=0,
            )


if __name__ == "__main__":
    unittest.main()
