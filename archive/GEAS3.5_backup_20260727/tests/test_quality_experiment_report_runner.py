import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.experiments.quality import (
    ModelExperimentSpec,
    ModelImplementation,
    QualityEvaluationConfig,
    QualityExperimentConfig,
    QualityHPOConfig,
    QualityOnlineBenchmarkConfig,
    QualityReportConfig,
    QualityThresholdCalibrationConfig,
    quality_report_config_from_experiment_config,
    run_quality_hpo,
    run_quality_online_benchmark,
    run_quality_report_generation,
    run_quality_threshold_calibration,
    run_quality_validation_test_evaluation,
)
from geas35.models.quality import (
    CandidateTrainingResult,
    EarlyStoppingConfig,
    MedianQualityModel,
)


def _frame(values):
    return pd.DataFrame(
        {
            "in_temp": values,
            "in_temp_missing_flag": [0] * len(values),
            "in_temp_rule_outlier_flag": [0] * len(values),
        }
    )


def _search_space():
    return {
        "common": {
            "lookback": {"values": [2, 3]},
            "mask_fraction": {"value": 0.2},
        },
        "model_specific": {
            "width": {"values": [4, 8]},
        },
    }


def _fake_train_candidate(candidate, train_df, validation_df, columns, early_stopping):
    lookback = float(candidate.common["lookback"])
    return CandidateTrainingResult(
        candidate=candidate,
        validation_metrics={
            "validation_synthetic_masking_rmse": lookback,
            "validation_synthetic_masking_mae": lookback / 2.0,
        },
        training_history=[
            {
                "epoch": 1,
                "validation_reconstruction_loss": lookback,
                "patience": early_stopping.patience,
            }
        ],
    )


def _fake_build_model(candidate):
    return MedianQualityModel()


class QualityExperimentReportRunnerTest(unittest.TestCase):
    def test_report_generation_writes_csv_json_and_markdown_without_selection_or_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "experiment"
            registry = {
                "fake_modern_tcn": ModelImplementation(
                    train_candidate=_fake_train_candidate,
                    build_model=_fake_build_model,
                ),
                "fake_timesnet": ModelImplementation(
                    train_candidate=_fake_train_candidate,
                    build_model=_fake_build_model,
                ),
            }
            train_df = _frame([20.0, 21.0, 22.0, 23.0])
            validation_df = _frame([20.0, 21.0, 22.0, 23.0, 24.0])
            test_df = _frame([25.0, 26.0, 27.0, 28.0, 29.0])
            hpo_result = run_quality_hpo(
                config=QualityHPOConfig(
                    output_root=output_root,
                    models=(
                        ModelExperimentSpec(
                            model_name="fake_modern_tcn",
                            search_space=_search_space(),
                            hpo_budget=2,
                            random_seed=1,
                        ),
                        ModelExperimentSpec(
                            model_name="fake_timesnet",
                            search_space=_search_space(),
                            hpo_budget=2,
                            random_seed=2,
                        ),
                    ),
                    early_stopping=EarlyStoppingConfig(patience=2),
                ),
                train_df=train_df,
                validation_df=validation_df,
                observation_columns=["in_temp"],
                registry=registry,
            )
            calibration_result = run_quality_threshold_calibration(
                config=QualityThresholdCalibrationConfig(
                    output_root=output_root,
                    threshold_candidates=(0.1, 1.0, 2.0),
                    early_stopping=EarlyStoppingConfig(patience=2),
                    mask_fraction=0.5,
                    anomaly_fraction=0.5,
                    random_state=0,
                ),
                hpo_result=hpo_result,
                train_df=train_df,
                validation_df=validation_df,
                observation_columns=["in_temp"],
                registry=registry,
            )
            evaluation_result = run_quality_validation_test_evaluation(
                config=QualityEvaluationConfig(
                    output_root=output_root,
                    mask_fraction=0.5,
                    anomaly_fraction=0.5,
                    random_state=0,
                ),
                calibration_result=calibration_result,
                validation_df=validation_df,
                test_df=test_df,
                observation_columns=["in_temp"],
            )
            benchmark_result = run_quality_online_benchmark(
                config=QualityOnlineBenchmarkConfig(
                    output_root=output_root,
                    repeats=1,
                ),
                calibration_result=calibration_result,
                validation_df=validation_df,
                test_df=test_df,
                observation_columns=["in_temp"],
            )

            result = run_quality_report_generation(
                config=QualityReportConfig(output_root=output_root),
                evaluation_result=evaluation_result,
                online_benchmark_result=benchmark_result,
            )

            for path in (
                result.model_comparison_json_path,
                result.model_comparison_csv_path,
                result.markdown_summary_path,
                result.reconstruction_metric_table_path,
                result.anomaly_detection_metric_table_path,
                result.online_benchmark_table_path,
            ):
                self.assertTrue(Path(path).exists())
            self.assertFalse((output_root / "figures").exists())

            comparison = json.loads((output_root / "model_comparison.json").read_text())
            self.assertEqual(comparison["stage"], "quality_model_experiment_report")
            self.assertFalse(comparison["automatic_best_model_selection"])
            self.assertFalse(comparison["test_used_for_selection"])
            self.assertTrue(comparison["online_benchmark_included"])
            self.assertNotIn("selected_model", comparison)
            self.assertEqual(len(comparison["models"]), 2)
            self.assertIn("validation_rmse", comparison["models"][0])
            self.assertIn("validation_latency_ms_per_row", comparison["models"][0])

            csv_text = (output_root / "model_comparison.csv").read_text()
            self.assertIn("validation_pr_auc", csv_text)
            self.assertIn("test_model_size_bytes", csv_text)
            markdown = (output_root / "markdown_summary.md").read_text()
            self.assertIn("does not automatically select", markdown)
            self.assertIn("Online benchmark included: True", markdown)

    def test_report_generation_can_run_without_online_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "experiment"
            registry = {
                "fake_modern_tcn": ModelImplementation(
                    train_candidate=_fake_train_candidate,
                    build_model=_fake_build_model,
                ),
            }
            train_df = _frame([20.0, 21.0, 22.0, 23.0])
            validation_df = _frame([20.0, 21.0, 22.0, 23.0, 24.0])
            test_df = _frame([25.0, 26.0, 27.0, 28.0, 29.0])
            hpo_result = run_quality_hpo(
                config=QualityHPOConfig(
                    output_root=output_root,
                    models=(
                        ModelExperimentSpec(
                            model_name="fake_modern_tcn",
                            search_space=_search_space(),
                            hpo_budget=1,
                            random_seed=1,
                        ),
                    ),
                    early_stopping=EarlyStoppingConfig(patience=2),
                ),
                train_df=train_df,
                validation_df=validation_df,
                observation_columns=["in_temp"],
                registry=registry,
            )
            calibration_result = run_quality_threshold_calibration(
                config=QualityThresholdCalibrationConfig(
                    output_root=output_root,
                    threshold_candidates=(0.1, 1.0),
                    early_stopping=EarlyStoppingConfig(patience=2),
                    mask_fraction=0.5,
                    anomaly_fraction=0.5,
                    random_state=0,
                ),
                hpo_result=hpo_result,
                train_df=train_df,
                validation_df=validation_df,
                observation_columns=["in_temp"],
                registry=registry,
            )
            evaluation_result = run_quality_validation_test_evaluation(
                config=QualityEvaluationConfig(
                    output_root=output_root,
                    mask_fraction=0.5,
                    anomaly_fraction=0.5,
                    random_state=0,
                ),
                calibration_result=calibration_result,
                validation_df=validation_df,
                test_df=test_df,
                observation_columns=["in_temp"],
            )

            run_quality_report_generation(
                config=QualityReportConfig(output_root=output_root),
                evaluation_result=evaluation_result,
            )

            comparison = json.loads((output_root / "model_comparison.json").read_text())
            self.assertFalse(comparison["online_benchmark_included"])
            self.assertIsNone(comparison["models"][0]["validation_latency_ms_per_row"])
            online_table = (output_root / "online_benchmark_table.csv").read_text()
            self.assertEqual(online_table.strip(), "")

    def test_report_config_can_be_derived_from_full_experiment_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment_config = QualityExperimentConfig(
                output_root=Path(tmp) / "experiment",
                models=(
                    ModelExperimentSpec(
                        model_name="fake_modern_tcn",
                        search_space=_search_space(),
                        hpo_budget=1,
                        random_seed=7,
                    ),
                ),
                threshold_candidates=(0.25, 0.5),
            )

            config = quality_report_config_from_experiment_config(experiment_config)

            self.assertEqual(config.output_root, experiment_config.output_root)


if __name__ == "__main__":
    unittest.main()
