import unittest
from datetime import datetime

from geas35.core import (
    DEFAULT_GROWTH_STAGE_RULE_PATH,
    calculate_dat,
    classify_solar_period_from_bounds,
    classify_sunshine_ratio,
    find_growth_stage,
    load_growth_stage_rules,
)
from geas35.data import crop_info_record_from_row


class GrowthStageTest(unittest.TestCase):
    def test_default_rule_file_lives_inside_geas35(self):
        self.assertTrue(DEFAULT_GROWTH_STAGE_RULE_PATH.exists())
        self.assertIn("GEAS3.5", str(DEFAULT_GROWTH_STAGE_RULE_PATH))

    def test_calculate_dat_is_one_based(self):
        transplant_date = datetime(2026, 6, 16, 0, 0)

        self.assertEqual(calculate_dat(datetime(2026, 6, 16, 23, 55), transplant_date), 1)
        self.assertEqual(calculate_dat(datetime(2026, 6, 17, 0, 0), transplant_date), 2)

    def test_cucumber_late_harvest_uses_lowered_midpoints(self):
        stage = find_growth_stage(
            target_ts=datetime(2026, 8, 24, 12, 0),
            transplant_date=datetime(2026, 6, 16, 0, 0),
            subj_cd="04",
        )

        self.assertEqual(stage.crop, "cucumber")
        self.assertEqual(stage.dat, 70)
        self.assertEqual(stage.stage_name, "late_harvest")
        self.assertEqual(stage.target_day_temp_c, 24.5)
        self.assertEqual(stage.target_night_temp_c, 13.5)
        self.assertEqual(stage.night_temp.as_tuple(), (11.0, 16.0))
        self.assertEqual(stage.night_early_temp.as_tuple(), (13.0, 16.0))
        self.assertEqual(stage.night_late_temp.as_tuple(), (11.0, 13.0))

    def test_controller_stage_config_includes_split_night_ranges(self):
        stage = find_growth_stage(
            target_ts=datetime(2026, 8, 24, 12, 0),
            transplant_date=datetime(2026, 6, 16, 0, 0),
            subj_cd="04",
        )

        config = stage.to_controller_stage_config()

        self.assertEqual(config["temp"], {"day": 24.5, "night": 13.5})
        self.assertEqual(config["night_early_temp_range_c"], (13.0, 16.0))
        self.assertEqual(config["night_late_temp_range_c"], (11.0, 13.0))

    def test_solar_period_bounds_split_early_and_late_night(self):
        self.assertEqual(
            classify_solar_period_from_bounds(
                now=datetime(2026, 7, 14, 12, 0),
                sunrise=datetime(2026, 7, 14, 7, 0),
                sunset=datetime(2026, 7, 14, 18, 0),
            ),
            "day",
        )
        self.assertEqual(
            classify_solar_period_from_bounds(
                now=datetime(2026, 7, 14, 23, 0),
                sunrise=datetime(2026, 7, 14, 7, 0),
                sunset=datetime(2026, 7, 14, 18, 0),
                next_sunrise=datetime(2026, 7, 15, 7, 0),
            ),
            "early_night",
        )
        self.assertEqual(
            classify_solar_period_from_bounds(
                now=datetime(2026, 7, 15, 1, 0),
                sunrise=datetime(2026, 7, 15, 7, 0),
                sunset=datetime(2026, 7, 15, 18, 0),
                previous_sunset=datetime(2026, 7, 14, 18, 0),
            ),
            "late_night",
        )

    def test_cucumber_main_harvest_uses_period_specific_night_targets(self):
        stage = find_growth_stage(
            target_ts=datetime(2026, 7, 14, 12, 0),
            transplant_date=datetime(2026, 6, 16, 0, 0),
            subj_cd="04",
        )

        early = stage.effective_temperature_target_for_period("early_night")
        late = stage.effective_temperature_target_for_period("late_night")

        self.assertEqual(stage.stage_name, "main_harvest")
        self.assertEqual(early["target_temp_c"], 16.5)
        self.assertEqual(early["temp_range_c"], (15.0, 18.0))
        self.assertEqual(late["target_temp_c"], 14.0)
        self.assertEqual(late["temp_range_c"], (13.0, 15.0))

    def test_sunshine_ratio_classification_uses_wmo_thresholds(self):
        self.assertEqual(
            classify_sunshine_ratio(
                actual_sunshine_minutes=700,
                possible_sunshine_minutes_value=1000,
            ),
            "sunny",
        )
        self.assertEqual(
            classify_sunshine_ratio(
                actual_sunshine_minutes=301,
                possible_sunshine_minutes_value=1000,
            ),
            "partly_cloudy",
        )
        self.assertEqual(
            classify_sunshine_ratio(
                actual_sunshine_minutes=300,
                possible_sunshine_minutes_value=1000,
            ),
            "cloudy",
        )
        self.assertEqual(
            classify_sunshine_ratio(
                actual_sunshine_minutes=None,
                possible_sunshine_minutes_value=1000,
            ),
            "unknown",
        )

    def test_cucumber_main_harvest_cloudy_and_partly_cloudy_use_cloudy_day_range(self):
        stage = find_growth_stage(
            target_ts=datetime(2026, 7, 14, 12, 0),
            transplant_date=datetime(2026, 6, 16, 0, 0),
            subj_cd="04",
        )

        cloudy = stage.effective_temperature_target_for_period(
            "day",
            daylight_condition="cloudy",
        )
        partly_cloudy = stage.effective_temperature_target_for_period(
            "day",
            daylight_condition="partly_cloudy",
        )
        unknown = stage.effective_temperature_target_for_period(
            "day",
            daylight_condition="unknown",
        )

        self.assertEqual(stage.day_temp.as_tuple(), (25.0, 28.0))
        self.assertEqual(stage.day_cloudy_temp.as_tuple(), (20.0, 25.0))
        self.assertEqual(cloudy["target_temp_c"], 22.5)
        self.assertEqual(cloudy["temp_range_c"], (20.0, 25.0))
        self.assertEqual(partly_cloudy["target_temp_c"], 22.5)
        self.assertEqual(partly_cloudy["temp_range_c"], (20.0, 25.0))
        self.assertEqual(unknown["target_temp_c"], 26.5)
        self.assertEqual(unknown["temp_range_c"], (25.0, 28.0))

    def test_controller_stage_config_can_use_cloudy_day_condition(self):
        stage = find_growth_stage(
            target_ts=datetime(2026, 7, 14, 12, 0),
            transplant_date=datetime(2026, 6, 16, 0, 0),
            subj_cd="04",
        )

        config = stage.to_controller_stage_config(
            target_ts=datetime(2026, 7, 14, 12, 0),
            lat=36.46,
            lon=128.22,
            daylight_condition="partly_cloudy",
        )

        self.assertEqual(config["solar_period"], "day")
        self.assertEqual(config["daylight_condition"], "partly_cloudy")
        self.assertEqual(config["temp"], {"day": 22.5, "night": 15.5})
        self.assertEqual(config["current_temp_range_c"], (20.0, 25.0))

    def test_cucumber_late_harvest_uses_period_specific_night_targets(self):
        stage = find_growth_stage(
            target_ts=datetime(2026, 8, 24, 12, 0),
            transplant_date=datetime(2026, 6, 16, 0, 0),
            subj_cd="04",
        )

        early = stage.effective_temperature_target_for_period("early_night")
        late = stage.effective_temperature_target_for_period("late_night")

        self.assertEqual(stage.stage_name, "late_harvest")
        self.assertEqual(early["target_temp_c"], 14.5)
        self.assertEqual(early["temp_range_c"], (13.0, 16.0))
        self.assertEqual(late["target_temp_c"], 12.0)
        self.assertEqual(late["temp_range_c"], (11.0, 13.0))

    def test_controller_stage_config_can_use_current_solar_period(self):
        stage = find_growth_stage(
            target_ts=datetime(2026, 8, 24, 12, 0),
            transplant_date=datetime(2026, 6, 16, 0, 0),
            subj_cd="04",
        )

        config = stage.to_controller_stage_config(
            target_ts=datetime(2026, 8, 24, 23, 0),
            lat=36.46,
            lon=128.22,
        )

        self.assertEqual(config["solar_period"], "early_night")
        self.assertEqual(config["temp"], {"day": 24.5, "night": 14.5})
        self.assertEqual(config["current_temp_range_c"], (13.0, 16.0))

    def test_melon_after_day_75_remains_in_final_stage(self):
        stage = find_growth_stage(
            target_ts=datetime(2025, 8, 1, 12, 0),
            transplant_date=datetime(2025, 5, 8, 0, 0),
            crop="melon",
        )

        self.assertEqual(stage.dat, 86)
        self.assertEqual(stage.stage_name, "maturity_harvest")
        self.assertEqual(stage.target_day_temp_c, 27.0)
        self.assertEqual(stage.target_night_temp_c, 15.0)

    def test_strawberry_numeric_range_midpoint_is_target(self):
        stage = find_growth_stage(
            target_ts=datetime(2025, 12, 1, 12, 0),
            transplant_date=datetime(2025, 10, 17, 0, 0),
            crop="strawberry",
        )

        self.assertEqual(stage.dat, 46)
        self.assertEqual(stage.stage_name, "flowering")
        self.assertEqual(stage.target_day_temp_c, 24.0)
        self.assertEqual(stage.target_night_temp_c, 6.5)

    def test_initial_strawberry_uses_establishment_night_target(self):
        stage = find_growth_stage(
            target_ts=datetime(2024, 9, 11, 12, 0),
            transplant_date=datetime(2024, 9, 11, 0, 0),
            subj_cd="01",
        )

        self.assertEqual(stage.stage_name, "establishment")
        self.assertEqual(stage.target_day_temp_c, 25.0)
        self.assertEqual(stage.target_night_temp_c, 11.5)
        self.assertEqual(stage.night_temp.as_tuple(), (10.0, 13.0))

    def test_crop_info_record_calculates_growth_stage(self):
        record = crop_info_record_from_row(
            {
                "idx": 71,
                "iot_data_idx": 97,
                "crop_nm": "2026 6 cucumber",
                "subj_cd": "04",
                "kind_cd": "0401",
                "trans_crop_date": datetime(2026, 6, 16, 0, 0),
                "crop_end_date": None,
                "end_yn": "N",
                "del_yn": "N",
            }
        )

        stage = record.growth_stage(datetime(2026, 7, 14, 12, 0))

        self.assertEqual(record.crop, "cucumber")
        self.assertEqual(stage.dat, 29)
        self.assertEqual(stage.stage_name, "main_harvest")

    def test_load_rules_contains_expected_crops(self):
        crops = {rule.crop for rule in load_growth_stage_rules()}

        self.assertEqual(crops, {"cucumber", "melon", "strawberry"})


if __name__ == "__main__":
    unittest.main()
