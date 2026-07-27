import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.experiments.quality import (
    FIGURE_STEMS,
    ModelImplementation,
    QualityEndToEndExperimentResult,
    run_from_config,
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


def _write_config(root: Path, *, generate_figures: bool = True) -> Path:
    train = root / "train.csv"
    validation = root / "validation.csv"
    test = root / "test.csv"
    output = root / "output"
    _frame([20.0, 21.0, 22.0, 23.0]).to_csv(train, index=False)
    _frame([20.0, 21.0, 22.0, 23.0, 24.0]).to_csv(validation, index=False)
    _frame([25.0, 26.0, 27.0, 28.0, 29.0]).to_csv(test, index=False)
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
                f"  output_dir: {output}",
                "  random_seed: 3",
                "hpo:",
                "  budget: 2",
                "  models: [fake_modern_tcn, fake_timesnet]",
                "  early_stopping:",
                "    patience: 2",
                "  search_spaces:",
                "    common:",
                "      lookback:",
                "        values: [2, 3]",
                "      mask_fraction:",
                "        value: 0.2",
                "    model_specific:",
                "      fake_modern_tcn:",
                "        width:",
                "          values: [4, 8]",
                "      fake_timesnet:",
                "        width:",
                "          values: [4, 8]",
                "threshold:",
                "  enabled: true",
                "  candidates: [0.1, 1.0, 2.0]",
                "evaluation:",
                "  mask_fraction: 0.5",
                "  anomaly_fraction: 0.5",
                "  random_seed: 0",
                "  efficiency_repeats: 1",
                "report:",
                f"  generate_figures: {'true' if generate_figures else 'false'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


class QualityExperimentEndToEndCliTest(unittest.TestCase):
    def test_run_from_config_executes_step_1_to_8_and_writes_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root, generate_figures=True)
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

            result = run_from_config(config_path, registry=registry)

            self.assertIsInstance(result, QualityEndToEndExperimentResult)
            output = root / "output"
            self.assertEqual(Path(result.output_root), output)
            for name in (
                "hpo_results.json",
                "best_config.json",
                "training_history.csv",
                "threshold_calibration.json",
                "evaluation.json",
                "online_benchmark.json",
                "model_comparison.json",
                "model_comparison.csv",
                "markdown_summary.md",
                "reconstruction_metric_table.csv",
                "anomaly_detection_metric_table.csv",
                "online_benchmark_table.csv",
                "visualization_manifest.json",
                "artifact_integrity.json",
            ):
                self.assertTrue((output / name).exists(), name)
            self.assertTrue(result.integrity_result.passed)
            self.assertIsNotNone(result.visualization_result)
            self.assertEqual(
                len(result.visualization_result.figure_paths),
                len(FIGURE_STEMS) * 2,
            )
            comparison = json.loads((output / "model_comparison.json").read_text())
            self.assertNotIn("selected_model", comparison)
            self.assertFalse(comparison["automatic_best_model_selection"])
            integrity = json.loads((output / "artifact_integrity.json").read_text())
            self.assertTrue(integrity["passed"])
            self.assertFalse(integrity["automatic_best_model_selection"])

    def test_run_from_config_can_skip_figures_and_integrity_figure_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root, generate_figures=False)
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

            result = run_from_config(config_path, registry=registry)

            output = root / "output"
            self.assertIsNone(result.visualization_result)
            self.assertFalse((output / "figures").exists())
            self.assertFalse((output / "visualization_manifest.json").exists())
            self.assertTrue((output / "artifact_integrity.json").exists())
            self.assertTrue(result.integrity_result.passed)


if __name__ == "__main__":
    unittest.main()
