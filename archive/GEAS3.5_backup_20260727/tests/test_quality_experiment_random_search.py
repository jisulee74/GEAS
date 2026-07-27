import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.experiments.quality import (
    ModelExperimentSpec,
    ModelImplementation,
    QualityExperimentConfig,
    RandomSearchCandidateGenerator,
    SearchSpace,
    run_quality_model_experiment,
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
            "mask_fraction": {"type": "float", "low": 0.1, "high": 0.2},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
        },
        "model_specific": {
            "width": {"type": "int", "low": 4, "high": 8},
            "dropout": {"value": 0.0},
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
        model_artifact={"model_name": "fake_median"},
    )


def _fake_build_model(candidate):
    return MedianQualityModel()


class QualityExperimentRandomSearchTest(unittest.TestCase):
    def test_random_search_generates_reproducible_candidates_from_config_space(self):
        space = SearchSpace.from_config(_search_space())
        first = RandomSearchCandidateGenerator(
            "modern_tcn",
            space,
            budget=3,
            random_seed=7,
        ).generate()
        second = RandomSearchCandidateGenerator(
            "modern_tcn",
            space,
            budget=3,
            random_seed=7,
        ).generate()

        self.assertEqual([c.to_artifact() for c in first], [c.to_artifact() for c in second])
        self.assertEqual(len(first), 3)
        self.assertTrue(first[0].name.startswith("modern_tcn_seed7_candidate"))
        self.assertIn(first[0].common["lookback"], [2, 3, 4])
        self.assertEqual(first[0].model_specific["dropout"], 0.0)

    def test_threshold_keys_are_rejected_by_generated_candidates(self):
        space = SearchSpace.from_config(
            {
                "common": {"threshold": {"values": [1.0]}},
                "model_specific": {},
            }
        )

        with self.assertRaisesRegex(ValueError, "Threshold calibration is separate"):
            RandomSearchCandidateGenerator(
                "modern_tcn",
                space,
                budget=1,
            ).generate()

    def test_experiment_runner_writes_hpo_calibration_evaluation_and_no_selection_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "experiment"
            config = QualityExperimentConfig(
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
                threshold_candidates=(0.1, 2.0),
                early_stopping=EarlyStoppingConfig(patience=2),
                mask_fraction=0.5,
                anomaly_fraction=0.5,
                efficiency_repeats=1,
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

            result = run_quality_model_experiment(
                config=config,
                train_df=_frame([20.0, 21.0, 22.0]),
                validation_df=_frame([20.0, 21.0, 22.0, 23.0]),
                test_df=_frame([24.0, 25.0, 26.0, 27.0]),
                observation_columns=["in_temp"],
                registry=registry,
            )

            self.assertEqual(len(result.model_results), 2)
            for model_name in ("fake_modern_tcn", "fake_timesnet"):
                model_dir = output_root / model_name
                self.assertTrue((model_dir / "hpo_results.json").exists())
                self.assertTrue((model_dir / "best_config.json").exists())
                self.assertTrue((model_dir / "training_history.csv").exists())
                self.assertTrue((model_dir / "threshold_calibration.json").exists())
                self.assertTrue((model_dir / "evaluation.json").exists())

                hpo_payload = json.loads((model_dir / "hpo_results.json").read_text())
                threshold_payload = json.loads(
                    (model_dir / "threshold_calibration.json").read_text()
                )
                self.assertFalse(hpo_payload["threshold_calibration_included"])
                self.assertTrue(threshold_payload["threshold_calibration_after_hpo"])
                self.assertFalse(threshold_payload["threshold_is_hpo_parameter"])

            comparison = json.loads((output_root / "model_comparison.json").read_text())
            self.assertEqual(comparison["stage"], "quality_model_experiment_comparison")
            self.assertFalse(comparison["automatic_best_model_selection"])
            self.assertFalse(comparison["test_used_for_selection"])
            self.assertNotIn("selected_model", comparison)
            self.assertTrue((output_root / "model_comparison.csv").exists())
            self.assertTrue((output_root / "markdown_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
