import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.experiments.quality import (
    ModelExperimentSpec,
    ModelImplementation,
    QualityExperimentConfig,
    QualityHPOConfig,
    QualityThresholdCalibrationConfig,
    quality_threshold_config_from_experiment_config,
    run_quality_hpo,
    run_quality_threshold_calibration,
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


class QualityExperimentCalibrationRunnerTest(unittest.TestCase):
    def test_threshold_calibration_consumes_hpo_result_and_writes_calibration_only(self):
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
            hpo_config = QualityHPOConfig(
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
            )
            train_df = _frame([20.0, 21.0, 22.0, 23.0])
            validation_df = _frame([20.0, 21.0, 22.0, 23.0, 24.0])
            hpo_result = run_quality_hpo(
                config=hpo_config,
                train_df=train_df,
                validation_df=validation_df,
                observation_columns=["in_temp"],
                registry=registry,
            )
            calibration_config = QualityThresholdCalibrationConfig(
                output_root=output_root,
                threshold_candidates=(0.1, 1.0, 2.0),
                early_stopping=EarlyStoppingConfig(patience=2),
                mask_fraction=0.5,
                anomaly_fraction=0.5,
                random_state=0,
            )

            result = run_quality_threshold_calibration(
                config=calibration_config,
                hpo_result=hpo_result,
                train_df=train_df,
                validation_df=validation_df,
                observation_columns=["in_temp"],
                registry=registry,
            )

            self.assertEqual(len(result.model_results), 2)
            self.assertTrue((output_root / "threshold_config.json").exists())
            self.assertTrue((output_root / "threshold_calibration.json").exists())
            self.assertFalse((output_root / "evaluation.json").exists())
            self.assertFalse((output_root / "model_comparison.json").exists())
            payload = json.loads((output_root / "threshold_calibration.json").read_text())
            self.assertEqual(payload["stage"], "threshold_calibration_summary")
            self.assertTrue(payload["threshold_calibration_after_hpo"])
            self.assertFalse(payload["threshold_is_hpo_parameter"])

            for model_name in ("fake_modern_tcn", "fake_timesnet"):
                model_dir = output_root / model_name
                threshold_path = model_dir / "threshold_calibration.json"
                self.assertTrue(threshold_path.exists())
                self.assertFalse((model_dir / "evaluation.json").exists())
                model_payload = json.loads(threshold_path.read_text())
                self.assertIn("best_threshold", model_payload)
                self.assertIn(model_payload["best_threshold"], [0.1, 1.0, 2.0])
                self.assertEqual(len(model_payload["candidate_results"]), 3)
                self.assertTrue(model_payload["threshold_calibration_after_hpo"])
                self.assertFalse(model_payload["threshold_is_hpo_parameter"])

    def test_threshold_config_can_be_derived_from_full_experiment_config(self):
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

            config = quality_threshold_config_from_experiment_config(experiment_config)

            self.assertEqual(config.output_root, experiment_config.output_root)
            self.assertEqual(config.threshold_candidates, (0.25, 0.5))
            self.assertEqual(config.early_stopping.patience, 3)
            self.assertEqual(config.mask_fraction, 0.4)
            self.assertEqual(config.anomaly_fraction, 0.3)
            self.assertEqual(config.anomaly_scale, 5.0)
            self.assertEqual(config.random_state, 11)


if __name__ == "__main__":
    unittest.main()
