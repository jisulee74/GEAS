import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.offline.prepare_rl_dataset_splits import (
    NEXT_OBSERVATION_PREFIX,
    RL_DONE_COLUMN,
    RL_REWARD_COLUMN,
    RL_VALID_TRANSITION_COLUMN,
    prepare_rl_dataset_frame,
    prepare_rl_dataset_splits,
)
from geas35.preprocessing import ACTION_COLUMNS, prepare_geas_input_schema
from geas35.rl import MDP_V1_ACTION_COLUMNS, MDP_V1_TRANSITION_TARGET_COLUMNS


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


class RlDatasetPreparationTest(unittest.TestCase):
    def test_prepare_rl_dataset_frame_excludes_action_nan_transition_starts(self):
        out, summary = prepare_rl_dataset_frame(_clean_quality_frame())

        self.assertEqual(summary["input_rows"], 4)
        self.assertEqual(summary["output_rows"], 2)
        self.assertEqual(summary["excluded_action_nan_rows"], 1)
        self.assertEqual(summary["excluded_invalid_transition_rows"], 1)
        self.assertEqual(out["reg_date"].dt.strftime("%H:%M:%S").tolist(), ["08:00:00", "08:10:00"])
        self.assertNotIn("08:05:00", out["reg_date"].dt.strftime("%H:%M:%S").tolist())

        for col in MDP_V1_ACTION_COLUMNS:
            self.assertIn(col, out.columns)
            self.assertFalse(out[col].isna().any())
        for col in MDP_V1_TRANSITION_TARGET_COLUMNS:
            self.assertIn(col, out.columns)
        self.assertIn(f"{NEXT_OBSERVATION_PREFIX}obs_indoor_temp_c", out.columns)
        self.assertIn(RL_REWARD_COLUMN, out.columns)
        self.assertIn(RL_DONE_COLUMN, out.columns)
        self.assertIn(RL_VALID_TRANSITION_COLUMN, out.columns)
        self.assertEqual(out[RL_VALID_TRANSITION_COLUMN].tolist(), [1, 1])
        self.assertEqual(out[RL_DONE_COLUMN].tolist(), [1, 1])
        self.assertTrue(out[RL_REWARD_COLUMN].notna().all())

    def test_prepare_rl_dataset_splits_writes_stage_outputs_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            input_dir = (
                dataset_root
                / "4_preprocessed"
                / "3_missing_outliers_handled"
                / "strawberry"
            )
            input_dir.mkdir(parents=True)
            _clean_quality_frame().to_parquet(input_dir / "train.parquet", index=False)

            summaries = prepare_rl_dataset_splits(
                dataset_root=dataset_root,
                crops=["strawberry"],
                splits=["train"],
            )

            output_path = dataset_root / "5_rl_dataset" / "strawberry" / "train.parquet"
            manifest_path = dataset_root / "5_rl_dataset" / "rl_dataset_manifest.json"

            self.assertEqual(len(summaries), 1)
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())

            result = pd.read_parquet(output_path)
            self.assertEqual(len(result), 2)
            self.assertIn("vent_pct", result.columns)
            self.assertIn("next_obs_indoor_temp_c", result.columns)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "rl_dataset")
            self.assertFalse(manifest["policy"]["ai_action_imputation"])
            self.assertEqual(
                manifest["policy"]["action_nan_policy"],
                "exclude_transition_start",
            )
            self.assertEqual(
                manifest["required_source_action_columns"],
                list(ACTION_COLUMNS),
            )
            self.assertEqual(manifest["splits"][0]["excluded_action_nan_rows"], 1)
            self.assertEqual(manifest["splits"][0]["output_rows"], 2)


if __name__ == "__main__":
    unittest.main()
