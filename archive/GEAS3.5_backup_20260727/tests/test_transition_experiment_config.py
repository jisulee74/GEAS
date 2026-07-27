import tempfile
import unittest
from pathlib import Path

from geas35.experiments.transition.config import load_transition_experiment_config
from geas35.models.transition import (
    LinearRegressionTransitionModel,
    build_selection_strategy,
    build_transition_candidate_specs,
    build_transition_model,
)


class TransitionExperimentConfigTest(unittest.TestCase):
    def test_load_transition_experiment_config_parses_paths_models_and_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train.parquet", "validation.parquet", "test.parquet"):
                (root / name).touch()
            config_path = root / "transition.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  crop: tomato",
                        "  train: train.parquet",
                        "  validation: validation.parquet",
                        "  test: test.parquet",
                        "experiment:",
                        "  output_dir: out",
                        "  random_seed: 7",
                        "models:",
                        "  candidates:",
                        "    - name: linear_regression",
                        "      params: {}",
                        "    - name: knn",
                        "      params:",
                        "        n_neighbors: 1",
                        "selection:",
                        "  strategy: weighted_rmse",
                        "  params:",
                        "    target_weights:",
                        "      obs_indoor_temp_c: 2.0",
                        "rollout:",
                        "  enabled: true",
                        "  horizon_steps: [3, 6]",
                        "  step_minutes: 5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = load_transition_experiment_config(config_path)

            self.assertEqual(loaded.experiment_config.crop, "tomato")
            self.assertEqual(loaded.train_path, root / "train.parquet")
            self.assertEqual(loaded.validation_path, root / "validation.parquet")
            self.assertEqual(loaded.test_path, root / "test.parquet")
            self.assertEqual(loaded.experiment_config.output_root, Path.cwd() / "out")
            self.assertEqual(loaded.experiment_config.random_seed, 7)
            self.assertEqual(
                [model.model_name for model in loaded.experiment_config.models],
                ["linear_regression", "knn"],
            )
            self.assertEqual(
                loaded.experiment_config.selection_strategy_name,
                "weighted_rmse",
            )
            self.assertEqual(
                loaded.experiment_config.selection_strategy_params[
                    "target_weights"
                ]["obs_indoor_temp_c"],
                2.0,
            )
            self.assertTrue(loaded.experiment_config.rollout_enabled)
            self.assertEqual(loaded.experiment_config.rollout_horizon_steps, (3, 6))

    def test_registry_builds_configured_candidate_specs(self):
        model = build_transition_model("linear_regression")
        self.assertIsInstance(model, LinearRegressionTransitionModel)
        self.assertEqual(build_selection_strategy("mean_rmse").name, "mean_rmse")

        specs = build_transition_candidate_specs(
            [
                {
                    "name": "linear_regression",
                    "params": {},
                    "metadata": {"tag": "baseline"},
                }
            ]
        )

        self.assertEqual(specs[0].model_name, "linear_regression")
        self.assertEqual(specs[0].metadata["tag"], "baseline")
        self.assertIsInstance(specs[0].build_model(), LinearRegressionTransitionModel)

    def test_registry_rejects_unknown_model_name(self):
        with self.assertRaises(ValueError):
            build_transition_model("unknown_model")


if __name__ == "__main__":
    unittest.main()
