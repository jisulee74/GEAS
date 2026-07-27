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
    QualityThresholdCalibrationConfig,
    quality_evaluation_config_from_experiment_config,
    run_quality_hpo,
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


class QualityExperimentEvaluationRunnerTest(unittest.TestCase):
    def test_validation_test_evaluation_uses_calibrated_threshold_without_online_benchmark(self):
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

            result = run_quality_validation_test_evaluation(
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

            self.assertEqual(len(result.model_results), 2)
            self.assertTrue((output_root / "evaluation_config.json").exists())
            self.assertTrue((output_root / "evaluation.json").exists())
            self.assertFalse((output_root / "model_comparison.json").exists())
            self.assertFalse((output_root / "online_benchmark.json").exists())
            payload = json.loads((output_root / "evaluation.json").read_text())
            self.assertEqual(payload["stage"], "quality_model_validation_test_evaluation")
            self.assertFalse(payload["automatic_best_model_selection"])
            self.assertFalse(payload["test_used_for_selection"])
            self.assertFalse(payload["online_benchmark_included"])

            for model_name in ("fake_modern_tcn", "fake_timesnet"):
                model_dir = output_root / model_name
                evaluation_path = model_dir / "evaluation.json"
                self.assertTrue(evaluation_path.exists())
                model_payload = json.loads(evaluation_path.read_text())
                self.assertEqual(model_payload["validation"]["split"], "validation")
                self.assertEqual(model_payload["test"]["split"], "test")
                self.assertIsNone(model_payload["validation"]["online_efficiency"])
                self.assertIsNone(model_payload["test"]["online_efficiency"])
                self.assertIn("rmse", model_payload["validation"]["reconstruction"])
                self.assertIn("f1_score", model_payload["validation"]["anomaly_detection"])

    def test_evaluation_config_can_be_derived_from_full_experiment_config(self):
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
                early_stopping=EarlyStoppingConfig(patience=3),
                mask_fraction=0.4,
                anomaly_fraction=0.3,
                anomaly_scale=5.0,
                evaluation_random_seed=11,
            )

            config = quality_evaluation_config_from_experiment_config(experiment_config)

            self.assertEqual(config.output_root, experiment_config.output_root)
            self.assertEqual(config.mask_fraction, 0.4)
            self.assertEqual(config.anomaly_fraction, 0.3)
            self.assertEqual(config.anomaly_scale, 5.0)
            self.assertEqual(config.random_state, 11)
            self.assertFalse(config.include_efficiency)


if __name__ == "__main__":
    unittest.main()
