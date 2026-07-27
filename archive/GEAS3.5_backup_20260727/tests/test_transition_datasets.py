import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.models.transition import (
    RL_VALID_TRANSITION_COLUMN,
    TransitionDatasetBuilder,
    build_transition_dataset_from_rl_frame,
    build_transition_dataset_from_rl_path,
)
from geas35.offline.prepare_rl_dataset_splits import prepare_rl_dataset_frame
from geas35.preprocessing import ACTION_COLUMNS, prepare_geas_input_schema


def _rl_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reg_date": pd.date_range("2026-07-01 08:00:00", periods=3, freq="5min"),
            "crop": ["cucumber", "cucumber", "cucumber"],
            "obs_indoor_temp_c": [23.0, 24.0, 25.0],
            "obs_indoor_humidity_pct": [80.0, 82.0, 84.0],
            "obs_hour_sin": [0.1, 0.2, 0.3],
            "obs_growth_stage_3": [1.0, 1.0, 1.0],
            "obs_quality_outlier_flag": [0.0, 0.0, 0.0],
            "obs_prev_vent_pct": [0.0, 0.1, 0.2],
            "vent_pct": [0.1, 0.2, 0.3],
            "heat_run": [0.0, 1.0, 0.0],
            "next_obs_indoor_temp_c": [24.0, 25.0, 26.0],
            "next_obs_indoor_humidity_pct": [82.0, 84.0, 86.0],
            "next_obs_hour_sin": [0.2, 0.3, 0.4],
            "target_next_indoor_temp_c": [24.0, 25.0, 26.0],
            "reward": [-0.1, -0.2, -0.3],
            "done": [0, 0, 1],
            RL_VALID_TRANSITION_COLUMN: [1, 0, 1],
        }
    )


def _clean_quality_frame() -> pd.DataFrame:
    rows = 4
    df = prepare_geas_input_schema(
        pd.DataFrame(
            {
                "reg_date": pd.date_range(
                    "2025-03-01 08:00:00",
                    periods=rows,
                    freq="5min",
                ),
                "in_temp": [23.0, 24.0, 25.0, 26.0],
                "in_hum": [80.0, 82.0, 84.0, 86.0],
                "out_temp": [20.0, 20.5, 21.0, 21.5],
                "out_hum": [70.0, 71.0, 72.0, 73.0],
                "out_light": [300.0, 350.0, 400.0, 450.0],
                "out_windsp": [1.0, 1.1, 1.2, 1.3],
                "out_rain": [0.0, 0.0, 0.0, 0.0],
                "growth_stage_order": [3, 3, 3, 3],
                "growth_stage_dat": [31, 32, 33, 34],
                "target_day_temp_c": [24.0, 24.0, 24.0, 24.0],
                "target_day_temp_min_c": [23.0, 23.0, 23.0, 23.0],
                "target_day_temp_max_c": [26.0, 26.0, 26.0, 26.0],
                "target_night_temp_c": [12.0, 12.0, 12.0, 12.0],
                "target_night_temp_min_c": [10.0, 10.0, 10.0, 10.0],
                "target_night_temp_max_c": [14.0, 14.0, 14.0, 14.0],
                "missing_imputed_flag": [0, 0, 0, 0],
                "outlier_flag": [0, 0, 0, 0],
            }
        ),
        keep_extra_columns=True,
    )
    for col in ACTION_COLUMNS:
        df[col] = 0.0
    df["cont_skyl_vol"] = [0.1, None, 0.3, 0.4]
    df["cont_skyr_vol"] = [0.1, 0.2, 0.3, 0.4]
    df["cont_cur_vol"] = [0.0, 0.0, 0.1, 0.1]
    df["cont_kwcur_vol"] = [0.0, 0.0, 0.0, 0.0]
    df["cont_heater_run"] = [0, 1, 0, 0]
    df["cont_cooler_run"] = [0, 0, 1, 0]
    df["cont_fan_run"] = [0, 1, 1, 0]
    return df


class TransitionDatasetBuilderTest(unittest.TestCase):
    def test_build_transition_dataset_filters_invalid_rows_and_renames_targets(self):
        result = build_transition_dataset_from_rl_frame(_rl_frame())
        dataset = result.dataset

        self.assertEqual(result.summary.input_rows, 3)
        self.assertEqual(result.summary.output_rows, 2)
        self.assertEqual(result.summary.excluded_invalid_transition_rows, 1)
        self.assertEqual(
            dataset.target_columns,
            ("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        )
        self.assertEqual(
            result.summary.next_target_columns,
            ("next_obs_indoor_temp_c", "next_obs_indoor_humidity_pct"),
        )
        self.assertEqual(dataset.y["obs_indoor_temp_c"].tolist(), [24.0, 26.0])
        self.assertEqual(dataset.y["obs_indoor_humidity_pct"].tolist(), [82.0, 86.0])

    def test_build_transition_dataset_preserves_inputs_and_metadata(self):
        result = TransitionDatasetBuilder().from_rl_dataset_frame(_rl_frame())
        dataset = result.dataset

        self.assertIn("obs_hour_sin", dataset.input_columns)
        self.assertIn("obs_quality_outlier_flag", dataset.input_columns)
        self.assertIn("vent_pct", dataset.input_columns)
        self.assertIn("heat_run", dataset.input_columns)
        self.assertNotIn("reward", dataset.input_columns)
        self.assertEqual(dataset.action_columns, ("vent_pct", "heat_run"))
        self.assertIn("reg_date", result.summary.metadata_columns)
        self.assertIn("crop", result.summary.metadata_columns)
        self.assertIn(RL_VALID_TRANSITION_COLUMN, result.summary.metadata_columns)
        self.assertEqual(dataset.metadata["crop"].tolist(), ["cucumber", "cucumber"])

    def test_build_transition_dataset_from_parquet_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rl.parquet"
            _rl_frame().to_parquet(path, index=False)

            result = build_transition_dataset_from_rl_path(path)

        self.assertEqual(result.summary.output_rows, 2)
        self.assertEqual(len(result.dataset.x), 2)

    def test_build_transition_dataset_rejects_empty_frame(self):
        with self.assertRaises(ValueError):
            build_transition_dataset_from_rl_frame(pd.DataFrame())

    def test_build_transition_dataset_rejects_frame_without_targets(self):
        frame = _rl_frame().drop(
            columns=["next_obs_indoor_temp_c", "next_obs_indoor_humidity_pct"]
        )

        with self.assertRaises(ValueError):
            build_transition_dataset_from_rl_frame(frame)

    def test_builder_integrates_with_existing_rl_dataset_preparation(self):
        rl_frame, _ = prepare_rl_dataset_frame(_clean_quality_frame())

        result = build_transition_dataset_from_rl_frame(rl_frame)

        self.assertEqual(result.summary.input_rows, len(rl_frame))
        self.assertEqual(result.summary.output_rows, len(rl_frame))
        self.assertIn("obs_indoor_temp_c", result.dataset.target_columns)
        self.assertIn("obs_indoor_humidity_pct", result.dataset.target_columns)
        self.assertIn("vent_pct", result.dataset.input_columns)
        self.assertTrue(result.dataset.y.notna().all().all())


if __name__ == "__main__":
    unittest.main()
