import unittest

import numpy as np
import pandas as pd

from geas35.preprocessing import (
    ACTION_COLUMNS,
    FEATURE_COLUMNS,
    STATE_COLUMNS,
    action_restored_flag_column,
    missing_flag_column,
    prepare_geas_input_schema,
    prepare_missing_features,
)


def schema_ready(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_geas_input_schema(df, keep_extra_columns=True)


class MissingFeaturePreparationTest(unittest.TestCase):
    def test_feature_contract_has_39_columns(self):
        self.assertEqual(len(FEATURE_COLUMNS), 39)
        self.assertEqual(len(STATE_COLUMNS), 26)
        self.assertEqual(len(ACTION_COLUMNS), 12)

    def test_raw_input_must_pass_input_schema_first(self):
        df = pd.DataFrame(
            {
                "reg_date": ["2025-03-01 00:05:00"],
                "in_temp": [20.0],
            }
        )

        with self.assertRaises(ValueError):
            prepare_missing_features(df, keep_extra_columns=False)

    def test_state_missing_values_remain_nan(self):
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-01 00:00:00"],
                    "in_temp": [np.nan],
                }
            )
        )

        out = prepare_missing_features(df, keep_extra_columns=False)

        self.assertTrue(pd.isna(out.loc[0, "in_temp"]))
        self.assertTrue(pd.isna(out.loc[0, "out_temp"]))

    def test_missing_flags_are_feature_level_only(self):
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-01 00:00:00"],
                    "in_temp": [np.nan],
                    "cont_skyl_vol": [10.0],
                }
            )
        )

        out = prepare_missing_features(df)

        self.assertEqual(out.loc[0, missing_flag_column("in_temp")], 1)
        self.assertEqual(out.loc[0, missing_flag_column("cont_skyl_vol")], 0)
        self.assertNotIn("missing_flag", out.columns)

    def test_missing_action_values_restore_from_same_timestamp_control_log(self):
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-01 00:00:00", "2025-03-01 00:05:00"],
                    "cont_skyl_vol": [np.nan, 30.0],
                    "cont_heater_run": [np.nan, np.nan],
                }
            )
        )
        control_log = pd.DataFrame(
            {
                "reg_date": ["2025-03-01 00:00:00", "2025-03-01 00:05:00"],
                "pred_ltw": [12.0, 99.0],
                "pred_heater": [1.0, 0.0],
            }
        )

        out = prepare_missing_features(df, control_log, keep_extra_columns=False)

        self.assertEqual(out.loc[0, "cont_skyl_vol"], 12.0)
        self.assertEqual(out.loc[0, "cont_heater_run"], 1.0)
        self.assertEqual(out.loc[1, "cont_skyl_vol"], 30.0)
        self.assertEqual(out.loc[1, "cont_heater_run"], 0.0)

    def test_control_log_restore_marks_action_restored_flags(self):
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": [
                        "2025-03-01 00:00:00",
                        "2025-03-01 00:05:00",
                        "2025-03-01 00:10:00",
                    ],
                    "cont_skyl_vol": [np.nan, 30.0, np.nan],
                }
            )
        )
        control_log = pd.DataFrame(
            {
                "reg_date": [
                    "2025-03-01 00:00:00",
                    "2025-03-01 00:05:00",
                    "2025-03-01 00:10:00",
                ],
                "pred_ltw": [12.0, 99.0, np.nan],
            }
        )

        out = prepare_missing_features(df, control_log)

        self.assertEqual(out.loc[0, "cont_skyl_vol"], 12.0)
        self.assertEqual(out.loc[1, "cont_skyl_vol"], 30.0)
        self.assertTrue(pd.isna(out.loc[2, "cont_skyl_vol"]))
        restored_col = action_restored_flag_column("cont_skyl_vol")
        self.assertEqual(out[restored_col].tolist(), [1, 0, 0])
        self.assertNotIn("cont_skyl_vol_restore_source", out.columns)

    def test_missing_action_values_stay_nan_without_control_log(self):
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-01 00:00:00"],
                    "cont_skyl_vol": [np.nan],
                }
            )
        )

        out = prepare_missing_features(df, keep_extra_columns=False)

        self.assertTrue(pd.isna(out.loc[0, "cont_skyl_vol"]))

    def test_locf_is_not_applied_to_state_or_action_values(self):
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": [
                        "2025-03-01 00:00:00",
                        "2025-03-01 00:05:00",
                    ],
                    "segment_id": ["seg01", "seg01"],
                    "is_resampled_row": [False, True],
                    "in_temp": [20.0, np.nan],
                    "cont_skyl_vol": [30.0, np.nan],
                }
            )
        )

        out = prepare_missing_features(df)

        self.assertTrue(pd.isna(out.loc[1, "in_temp"]))
        self.assertTrue(pd.isna(out.loc[1, "cont_skyl_vol"]))
        self.assertTrue(pd.isna(out.loc[1, "in_temp"]))
        self.assertEqual(out.loc[1, missing_flag_column("in_temp")], 1)
        self.assertEqual(out.loc[1, missing_flag_column("cont_skyl_vol")], 1)


if __name__ == "__main__":
    unittest.main()
