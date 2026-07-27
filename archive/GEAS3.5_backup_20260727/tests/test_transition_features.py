import unittest

import pandas as pd

from geas35.models.transition import (
    NEXT_OBSERVATION_PREFIX,
    TransitionTargetPolicy,
    infer_action_columns,
    infer_observation_columns,
    is_denied_observation_column,
    resolve_transition_feature_schema,
)


class TransitionFeaturesTest(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            {
                "obs_indoor_temp_c": [23.0],
                "obs_indoor_humidity_pct": [80.0],
                "obs_outdoor_temp_c": [20.0],
                "obs_rain_flag": [0.0],
                "obs_current_vpd_kpa": [0.6],
                "obs_derived_light_sum": [100.0],
                "obs_derived_condensation_risk_10m": [0.0],
                "obs_hour_sin": [0.1],
                "obs_growth_stage_3": [1.0],
                "obs_solar_period_day": [1.0],
                "obs_daylight_condition_unknown": [0.0],
                "obs_quality_outlier_flag": [0.0],
                "obs_prev_vent_pct": [0.2],
                "vent_pct": [0.3],
                "heat_run": [1.0],
                "target_next_indoor_temp_c": [23.5],
                "reward": [-0.1],
                f"{NEXT_OBSERVATION_PREFIX}obs_indoor_temp_c": [23.5],
                f"{NEXT_OBSERVATION_PREFIX}obs_indoor_humidity_pct": [82.0],
                f"{NEXT_OBSERVATION_PREFIX}obs_rain_flag": [0.0],
                f"{NEXT_OBSERVATION_PREFIX}obs_current_vpd_kpa": [0.55],
                f"{NEXT_OBSERVATION_PREFIX}obs_derived_light_sum": [120.0],
            }
        )

    def test_infer_current_observation_and_action_columns(self):
        frame = self._frame()

        observations = infer_observation_columns(frame)
        actions = infer_action_columns(frame)

        self.assertIn("obs_indoor_temp_c", observations)
        self.assertIn("obs_quality_outlier_flag", observations)
        self.assertNotIn("next_obs_indoor_temp_c", observations)
        self.assertEqual(actions, ("vent_pct", "heat_run"))

    def test_resolve_schema_selects_only_dynamic_targets_with_next_columns(self):
        schema = resolve_transition_feature_schema(self._frame())

        self.assertEqual(
            schema.dynamic_target_columns,
            (
                "obs_indoor_temp_c",
                "obs_indoor_humidity_pct",
                "obs_rain_flag",
                "obs_current_vpd_kpa",
                "obs_derived_light_sum",
            ),
        )
        self.assertEqual(
            schema.next_target_columns,
            (
                "next_obs_indoor_temp_c",
                "next_obs_indoor_humidity_pct",
                "next_obs_rain_flag",
                "next_obs_current_vpd_kpa",
                "next_obs_derived_light_sum",
            ),
        )
        self.assertIn("obs_outdoor_temp_c", schema.missing_dynamic_target_columns)

    def test_resolve_schema_excludes_metadata_flags_onehot_and_postprocessed_columns(self):
        schema = resolve_transition_feature_schema(self._frame())

        for column in (
            "obs_hour_sin",
            "obs_growth_stage_3",
            "obs_solar_period_day",
            "obs_daylight_condition_unknown",
            "obs_quality_outlier_flag",
            "obs_prev_vent_pct",
            "obs_derived_condensation_risk_10m",
        ):
            self.assertIn(column, schema.excluded_observation_columns)

        self.assertEqual(schema.deterministic_observation_columns, (
            "obs_hour_sin",
            "obs_growth_stage_3",
            "obs_solar_period_day",
            "obs_daylight_condition_unknown",
        ))
        self.assertEqual(schema.postprocessed_columns, ("obs_prev_vent_pct",))

    def test_resolve_schema_keeps_current_observations_and_actions_as_inputs(self):
        schema = resolve_transition_feature_schema(self._frame())

        self.assertIn("obs_hour_sin", schema.input_columns)
        self.assertIn("obs_quality_outlier_flag", schema.input_columns)
        self.assertIn("vent_pct", schema.input_columns)
        self.assertIn("heat_run", schema.input_columns)
        self.assertNotIn("reward", schema.input_columns)

    def test_target_policy_rejects_denied_allowlist_columns(self):
        with self.assertRaises(ValueError):
            TransitionTargetPolicy(dynamic_target_allowlist=("obs_hour_sin",))

    def test_denied_observation_column_policy_covers_flags_and_management_columns(self):
        self.assertTrue(is_denied_observation_column("obs_quality_missing_imputed_flag"))
        self.assertTrue(is_denied_observation_column("obs_temp_missing_flag"))
        self.assertTrue(is_denied_observation_column("obs_temp_outlier_flag"))
        self.assertTrue(is_denied_observation_column("next_obs_indoor_temp_c"))
        self.assertTrue(is_denied_observation_column("target_next_indoor_temp_c"))
        self.assertTrue(is_denied_observation_column("reward_term_temp_penalty"))
        self.assertFalse(is_denied_observation_column("obs_indoor_temp_c"))


if __name__ == "__main__":
    unittest.main()
