import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.experiments.quality import (
    FIGURE_STEMS,
    ModelExperimentSpec,
    ModelImplementation,
    QualityEvaluationConfig,
    QualityExperimentConfig,
    QualityHPOConfig,
    QualityOnlineBenchmarkConfig,
    QualityReportConfig,
    QualityThresholdCalibrationConfig,
    QualityVisualizationConfig,
    quality_visualization_config_from_experiment_config,
    run_quality_hpo,
    run_quality_online_benchmark,
    run_quality_report_generation,
    run_quality_threshold_calibration,
    run_quality_validation_test_evaluation,
    run_quality_visualization_generation,
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
                "train_reconstruction_loss": lookback + 1.0,
                "validation_reconstruction_loss": lookback,
                "patience": early_stopping.patience,
            },
            {
                "epoch": 2,
                "train_reconstruction_loss": lookback,
                "validation_reconstruction_loss": lookback / 2.0,
                "patience": early_stopping.patience,
            },
        ],
    )


def _fake_build_model(candidate):
    return MedianQualityModel()


class QualityExperimentVisualizationRunnerTest(unittest.TestCase):
    def test_visualization_generation_writes_png_pdf_and_manifest(self):
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
            run_quality_report_generation(
                config=QualityReportConfig(output_root=output_root),
                evaluation_result=evaluation_result,
                online_benchmark_result=benchmark_result,
            )

            result = run_quality_visualization_generation(
                config=QualityVisualizationConfig(output_root=output_root)
            )

            self.assertTrue((output_root / "figures").exists())
            self.assertTrue((output_root / "visualization_manifest.json").exists())
            self.assertEqual(len(result.figure_paths), len(FIGURE_STEMS) * 2)
            manifest = json.loads(
                (output_root / "visualization_manifest.json").read_text()
            )
            self.assertEqual(manifest["stage"], "quality_model_visualization")
            self.assertFalse(manifest["automatic_best_model_selection"])
            self.assertFalse(manifest["test_used_for_selection"])
            self.assertEqual(manifest["figure_count"], len(FIGURE_STEMS) * 2)
            for stem in FIGURE_STEMS:
                self.assertTrue((output_root / "figures" / f"{stem}.png").exists())
                self.assertTrue((output_root / "figures" / f"{stem}.pdf").exists())
            self.assertTrue((output_root / "figures" / "f1_comparison.png").exists())
            self.assertTrue(
                (output_root / "figures" / "online_benchmark_summary.png").exists()
            )
            self.assertTrue(
                (output_root / "figures" / "reconstruction_error_heatmap.png").exists()
            )
            self.assertFalse((output_root / "artifact_integrity.json").exists())

    def test_visualization_config_can_be_derived_from_full_experiment_config(self):
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

            config = quality_visualization_config_from_experiment_config(
                experiment_config
            )

            self.assertEqual(config.output_root, experiment_config.output_root)


if __name__ == "__main__":
    unittest.main()
