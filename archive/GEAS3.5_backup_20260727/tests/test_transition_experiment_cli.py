import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.experiments.transition.cli import main, run_from_config


def _rl_frame(rows=8):
    data = []
    for step in range(rows):
        temp = 20.0 + step
        humidity = 70.0 + 2.0 * step
        data.append(
            {
                "crop": "tomato",
                "obs_indoor_temp_c": temp,
                "obs_indoor_humidity_pct": humidity,
                "obs_prev_vent_pct": max((step - 1) / 10.0, 0.0),
                "vent_pct": step / 10.0,
                "next_obs_indoor_temp_c": temp + 1.0,
                "next_obs_indoor_humidity_pct": humidity + 2.0,
                "rl_valid_transition": 1,
            }
        )
    return pd.DataFrame(data)


class TransitionExperimentCliTest(unittest.TestCase):
    def test_run_from_config_trains_and_writes_transition_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.parquet"
            validation_path = root / "validation.parquet"
            test_path = root / "test.parquet"
            rollout_path = root / "rollout.parquet"
            output_dir = root / "out"
            frame = _rl_frame()
            frame.to_parquet(train_path)
            frame.to_parquet(validation_path)
            frame.to_parquet(test_path)
            frame.to_parquet(rollout_path)
            config_path = _write_config(
                root,
                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                rollout_path=rollout_path,
                output_dir=output_dir,
            )

            result = run_from_config(config_path)

            selected_path = Path(result.selected_manifest_path)
            summary_path = Path(result.experiment_summary_path)
            self.assertEqual(result.experiment_result.selected_model_name, "linear_regression")
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(selected_path.exists())
            self.assertEqual(selected_path.name, "selected_transition_model.json")
            self.assertEqual(selected_path.parent.name, "tomato")

            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(selected["selected_model_name"], "linear_regression")
            self.assertEqual(summary["selected_model_name"], "linear_regression")
            self.assertEqual(summary["source_rows"]["test"], len(frame.index))
            self.assertIn("linear_regression", summary["candidate_artifacts"])

    def test_main_cli_smoke_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.parquet"
            validation_path = root / "validation.parquet"
            test_path = root / "test.parquet"
            output_dir = root / "out"
            frame = _rl_frame()
            frame.to_parquet(train_path)
            frame.to_parquet(validation_path)
            frame.to_parquet(test_path)
            config_path = _write_config(
                root,
                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                rollout_path=None,
                output_dir=output_dir,
                rollout_enabled=False,
            )

            self.assertEqual(main(["--config", str(config_path)]), 0)
            self.assertTrue((output_dir / "tomato" / "selected_transition_model.json").exists())


def _write_config(
    root,
    *,
    train_path,
    validation_path,
    test_path,
    rollout_path,
    output_dir,
    rollout_enabled=True,
):
    config_path = root / "transition.yaml"
    rollout_line = (
        f"  rollout: {rollout_path.as_posix()}\n"
        if rollout_path is not None
        else ""
    )
    config_path.write_text(
        (
            "dataset:\n"
            "  crop: tomato\n"
            f"  train: {train_path.as_posix()}\n"
            f"  validation: {validation_path.as_posix()}\n"
            f"  test: {test_path.as_posix()}\n"
            f"{rollout_line}"
            "experiment:\n"
            f"  output_dir: {output_dir.as_posix()}\n"
            "  random_seed: 11\n"
            "models:\n"
            "  candidates:\n"
            "    - name: linear_regression\n"
            "      params: {}\n"
            "selection:\n"
            "  strategy: mean_rmse\n"
            "  params: {}\n"
            "rollout:\n"
            f"  enabled: {str(rollout_enabled).lower()}\n"
            "  horizon_steps: [3]\n"
            "  step_minutes: 5\n"
        ),
        encoding="utf-8",
    )
    return config_path


if __name__ == "__main__":
    unittest.main()
