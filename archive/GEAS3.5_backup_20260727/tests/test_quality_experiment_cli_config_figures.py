import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.experiments.quality.config import (
    load_quality_experiment_config,
    read_configured_datasets,
)
from geas35.experiments.quality.figures import (
    FIGURE_STEMS,
    generate_quality_experiment_figures,
)
from geas35.experiments.quality.runner import (
    ModelExperimentSpec,
    ModelImplementation,
    QualityExperimentConfig,
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
            "lookback": {"values": [2, 3]},
            "mask_fraction": {"value": 0.2},
        },
        "model_specific": {"width": {"values": [4, 8]}},
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
            },
            {
                "epoch": 2,
                "train_reconstruction_loss": lookback,
                "validation_reconstruction_loss": lookback / 2.0,
            },
        ],
    )


def _fake_build_model(candidate):
    return MedianQualityModel()


class QualityExperimentCliConfigFiguresTest(unittest.TestCase):
    def test_yaml_config_loader_reads_datasets_and_search_space_without_pyyaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            validation = root / "validation.csv"
            test = root / "test.csv"
            _frame([20.0, 21.0, 22.0]).to_csv(train, index=False)
            _frame([20.0, 21.0, 22.0, 23.0]).to_csv(validation, index=False)
            _frame([24.0, 25.0, 26.0, 27.0]).to_csv(test, index=False)
            config_path = root / "quality.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        f"  train: {train}",
                        f"  validation: {validation}",
                        f"  test: {test}",
                        "  observation_columns: [in_temp]",
                        "experiment:",
                        f"  output_dir: {root / 'output'}",
                        "  random_seed: 3",
                        "hpo:",
                        "  budget: 2",
                        "  models: [fake_modern_tcn]",
                        "  early_stopping:",
                        "    patience: 2",
                        "  search_spaces:",
                        "    common:",
                        "      lookback:",
                        "        values: [2, 3]",
                        "    model_specific:",
                        "      fake_modern_tcn:",
                        "        width:",
                        "          values: [4, 8]",
                        "threshold:",
                        "  enabled: true",
                        "  candidates: [0.1, 1.0]",
                        "report:",
                        "  generate_figures: true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = load_quality_experiment_config(config_path)
            train_df, validation_df, test_df, columns = read_configured_datasets(loaded)

            self.assertEqual(columns, ("in_temp",))
            self.assertEqual(len(train_df), 3)
            self.assertEqual(len(validation_df), 4)
            self.assertEqual(len(test_df), 4)
            self.assertEqual(loaded.experiment_config.models[0].hpo_budget, 2)
            self.assertTrue(loaded.generate_figures)

    def test_runner_writes_root_artifacts_and_figure_generator_writes_png_and_pdf(self):
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
            outputs = generate_quality_experiment_figures(result.output_root)

            for name in (
                "hpo_results.json",
                "best_config.json",
                "threshold_calibration.json",
                "evaluation.json",
                "training_history.csv",
            ):
                self.assertTrue((output_root / name).exists())
            comparison = json.loads((output_root / "model_comparison.json").read_text())
            self.assertFalse(comparison["automatic_best_model_selection"])
            self.assertNotIn("selected_model", comparison)
            self.assertEqual(len(outputs), len(FIGURE_STEMS) * 2)
            for stem in FIGURE_STEMS:
                self.assertTrue((output_root / "figures" / f"{stem}.png").exists())
                self.assertTrue((output_root / "figures" / f"{stem}.pdf").exists())


if __name__ == "__main__":
    unittest.main()
