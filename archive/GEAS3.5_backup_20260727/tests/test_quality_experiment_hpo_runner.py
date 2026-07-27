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
    quality_hpo_config_from_experiment_config,
    run_quality_hpo,
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
            "lookback": {"values": [2, 3, 4]},
            "mask_fraction": {"value": 0.2},
        },
        "model_specific": {
            "width": {"values": [4, 8]},
        },
    }


def _fake_train_candidate(candidate, train_df, validation_df, columns, early_stopping):
    lookback = float(candidate.common["lookback"])
    width = float(candidate.model_specific["width"])
    score = lookback + width / 100.0
    return CandidateTrainingResult(
        candidate=candidate,
        validation_metrics={
            "validation_synthetic_masking_rmse": score,
            "validation_synthetic_masking_mae": score / 2.0,
        },
        training_history=[
            {
                "epoch": 1,
                "validation_reconstruction_loss": score,
                "patience": early_stopping.patience,
            }
        ],
        model_artifact={"columns": list(columns)},
    )


def _fake_build_model(candidate):
    return MedianQualityModel()


class QualityExperimentHPORunnerTest(unittest.TestCase):
    def test_hpo_runner_writes_model_and_root_hpo_artifacts_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "hpo"
            config = QualityHPOConfig(
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
                        hpo_budget=3,
                        random_seed=2,
                    ),
                ),
                early_stopping=EarlyStoppingConfig(patience=2),
            )
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

            result = run_quality_hpo(
                config=config,
                train_df=_frame([20.0, 21.0, 22.0]),
                validation_df=_frame([20.0, 21.0, 22.0, 23.0]),
                observation_columns=["in_temp"],
                registry=registry,
            )

            self.assertEqual(len(result.model_results), 2)
            self.assertTrue((output_root / "hpo_config.json").exists())
            self.assertTrue((output_root / "hpo_results.json").exists())
            self.assertTrue((output_root / "best_config.json").exists())
            self.assertTrue((output_root / "training_history.csv").exists())
            self.assertFalse((output_root / "threshold_calibration.json").exists())
            self.assertFalse((output_root / "evaluation.json").exists())
            self.assertFalse((output_root / "model_comparison.json").exists())

            hpo_payload = json.loads((output_root / "hpo_results.json").read_text())
            self.assertEqual(hpo_payload["stage"], "hyperparameter_optimization_summary")
            self.assertFalse(hpo_payload["threshold_calibration_included"])
            self.assertFalse(hpo_payload["threshold_is_hpo_parameter"])
            self.assertEqual(
                len(hpo_payload["models"]["fake_modern_tcn"]["candidate_results"]),
                2,
            )
            self.assertEqual(
                len(hpo_payload["models"]["fake_timesnet"]["candidate_results"]),
                3,
            )

            for model_name in ("fake_modern_tcn", "fake_timesnet"):
                model_dir = output_root / model_name
                self.assertTrue((model_dir / "hpo_results.json").exists())
                self.assertTrue((model_dir / "best_config.json").exists())
                self.assertTrue((model_dir / "training_history.csv").exists())
                self.assertFalse((model_dir / "threshold_calibration.json").exists())
                self.assertFalse((model_dir / "evaluation.json").exists())

    def test_hpo_config_can_be_derived_from_full_experiment_config(self):
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
                threshold_candidates=(0.1, 1.0),
                early_stopping=EarlyStoppingConfig(patience=3),
            )

            hpo_config = quality_hpo_config_from_experiment_config(experiment_config)

            self.assertEqual(hpo_config.output_root, experiment_config.output_root)
            self.assertEqual(hpo_config.models, experiment_config.models)
            self.assertEqual(hpo_config.early_stopping.patience, 3)


if __name__ == "__main__":
    unittest.main()
