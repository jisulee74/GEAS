import unittest

import numpy as np
import pandas as pd

from geas35.models.quality.deep import (
    FeatureMaskingConfig,
    apply_feature_mask,
    build_sliding_windows,
    build_valid_observation_mask,
    confidence_from_scores,
    current_timestep_loss_mask,
    make_current_timestep_feature_mask,
    score_iqr_scale,
)


class DeepQualityCommonTest(unittest.TestCase):
    def test_sliding_windows_do_not_cross_groups_or_time_gaps(self):
        df = pd.DataFrame(
            {
                "series_id": ["a", "a", "a", "a", "a", "b", "b"],
                "reg_date": pd.to_datetime(
                    [
                        "2024-01-01 00:00",
                        "2024-01-01 00:05",
                        "2024-01-01 00:10",
                        "2024-01-01 00:20",
                        "2024-01-01 00:25",
                        "2024-01-01 00:00",
                        "2024-01-01 00:05",
                    ]
                ),
            }
        )

        windows = build_sliding_windows(
            df,
            lookback=2,
            group_columns=("series_id",),
            expected_frequency="5min",
        )

        self.assertEqual(
            [window.positions for window in windows],
            [(0, 1), (1, 2), (3, 4), (5, 6)],
        )
        self.assertEqual([window.current_position for window in windows], [1, 2, 4, 6])

    def test_valid_observation_mask_excludes_missing_rule_ai_and_invalid_flags(self):
        df = pd.DataFrame(
            {
                "in_temp": [20.0, np.nan, 22.0, 23.0, 24.0],
                "in_hum": [50.0, 51.0, 52.0, 53.0, 54.0],
                "in_temp_missing_flag": [0, 1, 0, 0, 0],
                "in_temp_rule_outlier_flag": [0, 0, 1, 0, 0],
                "in_temp_ai_outlier_flag": [0, 0, 0, 1, 0],
                "in_temp_invalid_flag": [0, 0, 0, 0, 1],
            }
        )

        mask = build_valid_observation_mask(df, ["in_temp", "in_hum"])

        self.assertEqual(mask["in_temp"].tolist(), [True, False, False, False, False])
        self.assertEqual(mask["in_hum"].tolist(), [True, True, True, True, True])

    def test_action_columns_are_rejected_by_mask_builders(self):
        df = pd.DataFrame({"in_temp": [20.0], "cont_skyl_vol": [0.5]})

        with self.assertRaisesRegex(ValueError, "Action columns are not allowed"):
            build_valid_observation_mask(df, ["in_temp", "cont_skyl_vol"])

        with self.assertRaisesRegex(ValueError, "Action columns are not allowed"):
            make_current_timestep_feature_mask(
                df,
                ["in_temp", "cont_skyl_vol"],
                windows=[],
            )

    def test_current_timestep_feature_mask_does_not_mask_past_rows_or_whole_timestamps(self):
        df = pd.DataFrame(
            {
                "reg_date": pd.date_range("2024-01-01", periods=4, freq="5min"),
                "in_temp": [20.0, 21.0, 22.0, 23.0],
                "in_hum": [50.0, 51.0, 52.0, 53.0],
            }
        )
        windows = build_sliding_windows(df, lookback=3, expected_frequency="5min")

        mask = make_current_timestep_feature_mask(
            df,
            ["in_temp", "in_hum"],
            windows=windows,
            config=FeatureMaskingConfig(mask_fraction=1.0, random_state=3),
        )

        current_positions = {window.current_position for window in windows}
        for row_pos in range(len(df)):
            row_masked = int(mask.iloc[row_pos].sum())
            if row_pos in current_positions:
                self.assertEqual(row_masked, 1)
            else:
                self.assertEqual(row_masked, 0)

    def test_apply_feature_mask_preserves_action_columns(self):
        df = pd.DataFrame(
            {
                "in_temp": [20.0, 21.0],
                "in_hum": [50.0, 51.0],
                "cont_skyl_vol": [0.1, 0.2],
            }
        )
        feature_mask = pd.DataFrame(
            {"in_temp": [False, True], "in_hum": [False, False]},
            index=df.index,
        )

        masked = apply_feature_mask(df, feature_mask, ["in_temp", "in_hum"])

        self.assertEqual(masked["in_temp"].isna().tolist(), [False, True])
        self.assertEqual(masked["in_hum"].tolist(), [50.0, 51.0])
        self.assertEqual(masked["cont_skyl_vol"].tolist(), [0.1, 0.2])

    def test_loss_mask_is_selected_masked_valid_cells_only(self):
        feature_mask = pd.DataFrame(
            {"in_temp": [True, True], "in_hum": [False, True]}
        )
        valid_mask = pd.DataFrame(
            {"in_temp": [True, False], "in_hum": [True, True]}
        )

        loss_mask = current_timestep_loss_mask(
            feature_mask,
            valid_mask,
            ["in_temp", "in_hum"],
        )

        self.assertEqual(loss_mask["in_temp"].tolist(), [True, False])
        self.assertEqual(loss_mask["in_hum"].tolist(), [False, True])

    def test_confidence_from_scores_uses_score_threshold_margin(self):
        scores = pd.DataFrame(
            {
                "in_temp": [0.0, 1.0, 2.0, np.nan],
                "in_hum": [1.0, 2.0, 3.0, 4.0],
            }
        )

        confidence = confidence_from_scores(
            scores,
            thresholds={"in_temp": 1.0, "in_hum": 2.0},
            scales={"in_temp": 1.0, "in_hum": 2.0},
        )

        self.assertLess(confidence.loc[0, "in_temp"], 0.5)
        self.assertEqual(confidence.loc[1, "in_temp"], 0.5)
        self.assertGreater(confidence.loc[2, "in_temp"], 0.5)
        self.assertTrue(np.isnan(confidence.loc[3, "in_temp"]))
        self.assertEqual(confidence.loc[1, "in_hum"], 0.5)

    def test_score_iqr_scale_uses_epsilon_floor(self):
        scores = pd.DataFrame({"in_temp": [1.0, 1.0, 1.0], "in_hum": [1.0, 2.0, 3.0]})

        scale = score_iqr_scale(scores, epsilon=0.25)

        self.assertEqual(scale["in_temp"], 0.25)
        self.assertEqual(scale["in_hum"], 1.0)


if __name__ == "__main__":
    unittest.main()
