import unittest

import pandas as pd

from geas35.rl import (
    MDP_V1_DAYLIGHT_CONDITION_ONEHOT_COLUMNS,
    MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS,
    MDP_V1_OBSERVATION_COLUMNS,
    MDP_V1_QUALITY_FLAG_FALLBACKS,
    MDP_V1_ROLLOUT_ID_COLUMN,
    MDP_V1_SOLAR_PERIOD_ONEHOT_COLUMNS,
    MDP_V1_VALID_TRANSITION_COLUMN,
    MdpV1Config,
    MdpV1Env,
    MdpV1RewardNormalizer,
    compute_mdp_v1_reward,
    evaluate_mdp_v1_transition_predictions,
    fit_mdp_v1_observation_scaler,
    fit_mdp_v1_reward_normalizer,
    logged_mdp_v1_action,
    mdp_v1_observation_columns,
    prepare_mdp_v1_frame,
)


class MdpV1EnvTest(unittest.TestCase):
    def _sample_frame(self):
        return pd.DataFrame(
            {
                "reg_date": [
                    "2026-07-15 08:00:00",
                    "2026-07-15 08:05:00",
                    "2026-07-15 08:10:00",
                ],
                "in_temp": [23.0, 24.5, 25.5],
                "in_hum": [80.0, 85.0, 91.0],
                "in_co2": [650.0, 700.0, 720.0],
                "out_temp": [20.0, 21.0, 22.0],
                "out_hum": [70.0, 72.0, 75.0],
                "out_light": [300.0, 400.0, 500.0],
                "out_light_sum": [10.0, 20.0, 30.0],
                "out_windsp": [1.0, 1.2, 1.4],
                "out_rain": [0.0, 0.0, 0.0],
                "cont_skyl_vol": [0.1, 0.2, 0.3],
                "cont_skyr_vol": [0.0, 0.0, 0.0],
                "cont_cur_vol": [0.0, 0.0, 0.0],
                "cont_kwcur_vol": [0.0, 0.0, 0.0],
                "cont_3way1_vol": [0.1, 0.2, 0.1],
                "cont_3way2_vol": [0.2, 0.3, 0.2],
                "cont_heater_run": [0, 1, 0],
                "cont_cooler_run": [0, 0, 1],
                "cont_co2_run": [0, 1, 1],
                "cont_pump1_run": [0, 1, 0],
                "cont_pump2_run": [0, 0, 1],
                "cont_fan_run": [0, 1, 1],
                "missing_imputed_flag": [0, 1, 0],
                "in_temp_rule_outlier_flag": [0, 0, 1],
                "outlier_flag": [0, 0, 1],
                "growth_stage_order": [3, 3, 3],
                "growth_stage_dat": [31, 32, 33],
                "target_day_temp_c": [24.0, 24.0, 24.0],
                "target_day_temp_min_c": [23.0, 23.0, 23.0],
                "target_day_temp_max_c": [26.0, 26.0, 26.0],
                "target_night_temp_c": [12.0, 12.0, 12.0],
                "target_night_temp_min_c": [10.0, 10.0, 10.0],
                "target_night_temp_max_c": [14.0, 14.0, 14.0],
            }
        )

    def test_prepare_mdp_v1_frame_adds_observations_and_targets(self):
        result = prepare_mdp_v1_frame(self._sample_frame())

        for col in MDP_V1_OBSERVATION_COLUMNS:
            self.assertIn(col, result.columns)
        self.assertIn("obs_growth_stage_order", result.columns)
        self.assertNotIn("obs_growth_stage_order", MDP_V1_OBSERVATION_COLUMNS)
        self.assertIn("obs_growth_stage_3", MDP_V1_OBSERVATION_COLUMNS)
        self.assertIn("obs_growth_stage_dat", MDP_V1_OBSERVATION_COLUMNS)
        self.assertEqual(result.loc[0, "obs_growth_stage_3"], 1.0)
        self.assertEqual(result.loc[0, "obs_growth_stage_1"], 0.0)
        self.assertEqual(result.loc[0, "obs_growth_stage_dat"], 31)
        self.assertEqual(len(MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS), 6)
        self.assertIn("obs_solar_period_day", MDP_V1_SOLAR_PERIOD_ONEHOT_COLUMNS)
        self.assertIn("obs_daylight_condition_unknown", MDP_V1_DAYLIGHT_CONDITION_ONEHOT_COLUMNS)
        self.assertEqual(result.loc[0, "obs_solar_period_day"], 1.0)
        self.assertEqual(result.loc[0, "obs_daylight_condition_unknown"], 1.0)
        self.assertEqual(result.loc[0, "obs_current_target_temp_min_c"], 23.0)
        self.assertEqual(result.loc[0, "obs_current_target_temp_max_c"], 26.0)
        self.assertEqual(result.loc[0, "obs_current_target_temp_c"], 24.0)
        self.assertIn("obs_derived_rh90_minutes_1h", MDP_V1_OBSERVATION_COLUMNS)
        self.assertIn("obs_derived_light_sum_ratio", MDP_V1_OBSERVATION_COLUMNS)
        self.assertIn("obs_derived_ramp_limit_pct", MDP_V1_OBSERVATION_COLUMNS)
        self.assertGreaterEqual(result.loc[2, "obs_derived_low_vpd_minutes_1h"], 0.0)
        self.assertEqual(result.loc[0, "obs_derived_light_sum"], 10.0)
        self.assertEqual(result.loc[0, "obs_prev_vent_pct"], 0.05)
        self.assertNotIn("obs_prev_rail_3way_pct", result.columns)
        self.assertNotIn("obs_prev_fcu_3way_pct", result.columns)
        self.assertEqual(result.loc[0, "target_next_indoor_temp_c"], 24.5)
        self.assertGreater(result.loc[0, "obs_current_vpd_kpa"], 0.0)
        self.assertIn("obs_quality_missing_imputed_flag", result.columns)
        self.assertIn("obs_quality_outlier_flag", result.columns)
        self.assertNotIn("obs_quality_rule_outlier_flag", result.columns)
        self.assertNotIn("obs_quality_in_temp_rule_outlier_flag", result.columns)
        self.assertEqual(result.loc[0, MDP_V1_VALID_TRANSITION_COLUMN], 1)

    def test_prepare_mdp_v1_frame_accepts_new_quality_flag_fallbacks(self):
        data = self._sample_frame().drop(
            columns=["missing_imputed_flag", "outlier_flag"]
        )
        data["imputed_flag"] = [0, 1, 0]
        data["invalid_flag"] = [0, 0, 1]

        result = prepare_mdp_v1_frame(data)

        self.assertIn("missing_imputed_flag", MDP_V1_QUALITY_FLAG_FALLBACKS)
        self.assertIn("outlier_flag", MDP_V1_QUALITY_FLAG_FALLBACKS)
        self.assertEqual(result["obs_quality_missing_imputed_flag"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(result["obs_quality_outlier_flag"].tolist(), [0.0, 0.0, 1.0])

    def test_logged_mdp_v1_action_maps_geas_action_columns(self):
        row = prepare_mdp_v1_frame(self._sample_frame()).iloc[1]

        action = logged_mdp_v1_action(row)

        self.assertEqual(action["vent_pct"], 0.1)
        self.assertEqual(action["heat_run"], 1.0)
        self.assertEqual(action["cool_run"], 0.0)
        self.assertEqual(action["fan_run"], 1.0)
        self.assertNotIn("rail_3way_pct", action)
        self.assertNotIn("fcu_3way_pct", action)
        self.assertNotIn("co2_run", action)
        self.assertNotIn("rail_pump_run", action)
        self.assertNotIn("fcu_pump_run", action)

    def test_mdp_v1_steps_over_replay_trajectory(self):
        env = MdpV1Env(self._sample_frame())

        obs = env.reset()
        next_obs, reward, terminated, truncated, info = env.step(
            {
                "vent_pct": 0.2,
                "shade_curtain_pct": 0.0,
                "thermal_curtain_pct": 0.0,
                "heat_run": 0.0,
                "cool_run": 0.0,
                "fan_run": 0.0,
            }
        )

        expected_columns = mdp_v1_observation_columns(env.frame)
        self.assertGreater(len(expected_columns), len(MDP_V1_OBSERVATION_COLUMNS))
        self.assertEqual(len(obs), len(expected_columns))
        self.assertEqual(len(next_obs), len(expected_columns))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("reward_terms", info)
        self.assertIn("transition_target", info)
        self.assertIn("valid_transition", info)
        self.assertIn("rollout_id", info)
        self.assertIn("constraint_info", info)

    def test_mdp_v1_projects_vent_action_for_rain_and_high_wind(self):
        rain_data = self._sample_frame()
        rain_data.loc[0, "out_rain"] = 1.0
        rain_env = MdpV1Env(rain_data)
        rain_env.reset()
        _, _, _, _, rain_info = rain_env.step(
            {
                "vent_pct": 0.8,
                "shade_curtain_pct": 0.0,
                "thermal_curtain_pct": 0.0,
                "heat_run": 0.0,
                "cool_run": 0.0,
                "fan_run": 0.0,
            }
        )

        self.assertEqual(rain_info["raw_action"]["vent_pct"], 0.8)
        self.assertEqual(rain_info["action"]["vent_pct"], 0.0)
        self.assertTrue(rain_info["constraint_info"]["rain_vent_constraint_applied"])

        wind_data = self._sample_frame()
        wind_data.loc[0, "out_windsp"] = 6.0
        wind_env = MdpV1Env(wind_data)
        wind_env.reset()
        _, _, _, _, wind_info = wind_env.step(
            {
                "vent_pct": 0.8,
                "shade_curtain_pct": 0.0,
                "thermal_curtain_pct": 0.0,
                "heat_run": 0.0,
                "cool_run": 0.0,
                "fan_run": 0.0,
            }
        )

        self.assertEqual(wind_info["raw_action"]["vent_pct"], 0.8)
        self.assertEqual(wind_info["action"]["vent_pct"], 0.2)
        self.assertTrue(wind_info["constraint_info"]["wind_vent_constraint_applied"])

    def test_reward_uses_target_temperature_bounds(self):
        frame = prepare_mdp_v1_frame(self._sample_frame())
        reward, terms = compute_mdp_v1_reward(
            frame.iloc[2],
            {
                "vent_pct": 0.0,
                "shade_curtain_pct": 0.0,
                "thermal_curtain_pct": 0.0,
                "heat_run": 0.0,
                "cool_run": 0.0,
                "fan_run": 0.0,
            },
        )

        self.assertEqual(terms["target_temp_min_c"], 23.0)
        self.assertEqual(terms["target_temp_max_c"], 26.0)
        self.assertEqual(terms["raw_temp_penalty"], 0.0)
        self.assertEqual(terms["normalized_temp_penalty"], 0.0)
        self.assertEqual(terms["temp_penalty"], 0.0)
        self.assertLessEqual(reward, 0.0)

    def test_reward_penalizes_only_low_vpd(self):
        frame = prepare_mdp_v1_frame(self._sample_frame())
        row = frame.iloc[2].copy()
        action = {
            "vent_pct": 0.0,
            "shade_curtain_pct": 0.0,
            "thermal_curtain_pct": 0.0,
            "heat_run": 0.0,
            "cool_run": 0.0,
            "fan_run": 0.0,
        }

        row["obs_current_vpd_kpa"] = 2.0
        _, terms = compute_mdp_v1_reward(row, action)
        self.assertEqual(terms["raw_vpd_penalty"], 0.0)

        row["obs_current_vpd_kpa"] = 0.2
        _, terms = compute_mdp_v1_reward(row, action)
        self.assertAlmostEqual(terms["raw_vpd_penalty"], 1.0 / 3.0)

    def test_reward_uses_physics_hv_penalty(self):
        frame = prepare_mdp_v1_frame(self._sample_frame())
        row = frame.iloc[2].copy()
        row["obs_indoor_temp_c"] = 35.0
        row["obs_outdoor_temp_c"] = 5.0
        row["obs_outdoor_wind_speed"] = 0.0
        config = MdpV1Config(
            physics_k_heat=100.0,
            physics_rho_cp=10.0,
            physics_greenhouse_volume_m3=360.0,
            physics_ach_base=0.0,
            physics_ach_vent_coef=1.0,
            physics_ach_wind_coef=0.0,
            heat_vent_alpha=0.25,
        )

        _, terms = compute_mdp_v1_reward(
            row,
            {
                "vent_pct": 1.0,
                "shade_curtain_pct": 0.0,
                "thermal_curtain_pct": 0.0,
                "heat_run": 1.0,
                "cool_run": 0.0,
                "fan_run": 0.0,
            },
            config=config,
        )

        self.assertAlmostEqual(terms["heat_vent_q_heat"], 100.0)
        self.assertAlmostEqual(terms["heat_vent_q_ventloss"], 30.0)
        self.assertAlmostEqual(terms["heat_vent_threshold"], 25.0)
        self.assertAlmostEqual(terms["heat_vent_excess_ratio"], 0.2)
        self.assertAlmostEqual(terms["heat_vent_excess_heat_fraction"], 0.05)
        self.assertAlmostEqual(terms["raw_hv_penalty"], 0.1)
        self.assertNotIn("raw_energy_penalty", terms)
        self.assertNotIn("raw_co2_penalty", terms)
        self.assertNotIn("raw_fan_penalty", terms)

        q_max_config = MdpV1Config(
            physics_k_heat=5.0,
            physics_rho_cp=10.0,
            physics_greenhouse_volume_m3=360.0,
            physics_ach_base=0.0,
            physics_ach_vent_coef=1.0,
            physics_ach_wind_coef=0.0,
            heat_vent_alpha=0.25,
            hv_q_max=5.0,
        )
        _, q_max_terms = compute_mdp_v1_reward(
            row,
            {
                "vent_pct": 1.0,
                "shade_curtain_pct": 0.0,
                "thermal_curtain_pct": 0.0,
                "heat_run": 1.0,
                "cool_run": 0.0,
                "fan_run": 0.0,
            },
            config=q_max_config,
        )
        self.assertEqual(q_max_terms["heat_vent_q_heat"], q_max_terms["heat_vent_q_max"])
        self.assertEqual(q_max_terms["raw_hv_penalty"], 0.0)

    def test_reward_act_uses_direction_reversal_and_direct_policy_actions(self):
        frame = prepare_mdp_v1_frame(self._sample_frame())
        row = frame.iloc[2]
        prev_prev_action = {
            "vent_pct": 0.2,
            "shade_curtain_pct": 0.0,
            "thermal_curtain_pct": 0.0,
            "heat_run": 0.0,
            "cool_run": 1.0,
            "fan_run": 0.0,
        }
        prev_action = {
            "vent_pct": 0.5,
            "shade_curtain_pct": 0.0,
            "thermal_curtain_pct": 0.0,
            "heat_run": 1.0,
            "cool_run": 0.0,
            "fan_run": 1.0,
        }
        action = {
            "vent_pct": 0.4,
            "shade_curtain_pct": 0.0,
            "thermal_curtain_pct": 0.0,
            "heat_run": 0.0,
            "cool_run": 1.0,
            "fan_run": 0.0,
        }

        _, terms = compute_mdp_v1_reward(
            row,
            action,
            prev_action=prev_action,
            prev_prev_action=prev_prev_action,
        )

        continuous_weighted_reversal = (0.3 + 0.1) / 2.0
        binary_weighted_reversal = 3.0 * 2.0
        total_weight = 3.0 * 1.0 + 3.0 * 2.0
        expected_penalty = (
            continuous_weighted_reversal + binary_weighted_reversal
        ) / total_weight
        self.assertAlmostEqual(
            terms["act_unbounded"],
            expected_penalty,
        )
        self.assertAlmostEqual(terms["raw_act_penalty"], expected_penalty)

    def test_reward_normalizer_is_domain_bounded_compatibility_noop(self):
        frame = prepare_mdp_v1_frame(self._sample_frame())
        normalizer = MdpV1RewardNormalizer(scales={"rh": 4.0})

        _, terms = compute_mdp_v1_reward(
            frame.iloc[2],
            {
                "vent_pct": 0.0,
                "shade_curtain_pct": 0.0,
                "thermal_curtain_pct": 0.0,
                "heat_run": 0.0,
                "cool_run": 0.0,
                "fan_run": 0.0,
            },
            reward_normalizer=normalizer,
        )

        self.assertAlmostEqual(
            terms["normalized_rh_penalty"],
            terms["raw_rh_penalty"],
        )
        self.assertAlmostEqual(
            terms["rh_penalty"],
            terms["normalized_rh_penalty"] / 6.0,
        )

    def test_fit_reward_normalizer_from_logged_transitions(self):
        normalizer = fit_mdp_v1_reward_normalizer(
            self._sample_frame(),
            quantile=0.95,
        )

        self.assertIn("rh", normalizer.scales)
        self.assertEqual(normalizer.scales["rh"], 1.0)
        self.assertEqual(normalizer.method, "domain_bounded")

    def test_transition_prediction_metrics_include_q90_and_cvar90(self):
        y_true = pd.DataFrame({"target_next_indoor_temp_c": [1.0, 2.0, 3.0, 4.0]})
        y_pred = pd.DataFrame({"target_next_indoor_temp_c": [1.0, 2.0, 3.0, 6.0]})

        metrics = evaluate_mdp_v1_transition_predictions(y_true, y_pred)
        temp_metrics = metrics["target_next_indoor_temp_c"]

        self.assertIn("r2", temp_metrics)
        self.assertIn("mae", temp_metrics)
        self.assertIn("rmse", temp_metrics)
        self.assertIn("q90", temp_metrics)
        self.assertIn("cvar90", temp_metrics)
        self.assertAlmostEqual(temp_metrics["mae"], 0.5)
        self.assertGreaterEqual(temp_metrics["cvar90"], temp_metrics["q90"])

    def test_mdp_v1_cuts_rollout_on_nan_and_time_gap(self):
        data = self._sample_frame()
        data.loc[1, "reg_date"] = "2026-07-15 08:10:00"
        data.loc[2, "reg_date"] = "2026-07-15 08:15:00"
        data.loc[1, "in_temp"] = pd.NA

        result = prepare_mdp_v1_frame(data)

        self.assertEqual(result.loc[0, MDP_V1_VALID_TRANSITION_COLUMN], 0)
        self.assertEqual(result.loc[1, MDP_V1_VALID_TRANSITION_COLUMN], 0)
        self.assertGreaterEqual(result[MDP_V1_ROLLOUT_ID_COLUMN].nunique(), 2)

    def test_mdp_v1_observation_scaler_scales_only_continuous_columns(self):
        frame = prepare_mdp_v1_frame(self._sample_frame())
        scaler = fit_mdp_v1_observation_scaler(frame)
        scaled = scaler.transform_frame(frame)

        self.assertIn("obs_indoor_temp_c", scaler.columns)
        self.assertIn("obs_growth_stage_dat", scaler.columns)
        self.assertNotIn("obs_growth_stage_3", scaler.columns)
        self.assertNotIn("obs_rain_flag", scaler.columns)
        self.assertAlmostEqual(scaled.loc[0, "obs_indoor_temp_c"], -1.0)
        self.assertAlmostEqual(scaled.loc[1, "obs_indoor_temp_c"], 1.0)
        self.assertEqual(scaled.loc[0, "obs_growth_stage_3"], 1.0)

    def test_mdp_v1_env_can_return_scaled_observations(self):
        frame = prepare_mdp_v1_frame(self._sample_frame())
        scaler = fit_mdp_v1_observation_scaler(frame)
        env = MdpV1Env(self._sample_frame(), observation_scaler=scaler)

        obs = env.reset()
        temp_index = env.observation_columns.index("obs_indoor_temp_c")
        stage_index = env.observation_columns.index("obs_growth_stage_3")

        self.assertAlmostEqual(obs[temp_index], -1.0)
        self.assertEqual(obs[stage_index], 1.0)


if __name__ == "__main__":
    unittest.main()
