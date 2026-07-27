import unittest

import pandas as pd

from geas35.preprocessing import (
    DEFAULT_AGGREGATE_FLAG,
    DEFAULT_AI_OUTLIER_FLAG_COLUMN,
    DEFAULT_AI_WARNING_FLAG_COLUMN,
    DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN,
    DEFAULT_IMPUTED_FLAG_COLUMN,
    DEFAULT_RULE_AGGREGATE_FLAG,
    DEFAULT_TCN_FLAG_COLUMN,
    DEFAULT_TCN_PRED_VALUE_COLUMN,
    DEFAULT_TCN_SCORE_COLUMN,
    DomainRangeRule,
    TCNOutlierConfig,
    apply_outlier_flags,
    apply_rule_based_outlier_flags,
    apply_tcn_outlier_flags,
    controller_value_column,
    raw_value_column,
    rule_outlier_flag_column,
    tcn_pred_value_column,
    tcn_score_column,
)


class OutlierHandlingTest(unittest.TestCase):
    def test_rule_based_flags_do_not_modify_feature_values(self):
        df = pd.DataFrame({"in_temp": [2.0, 1.9, 50.1, None]})
        rules = [
            DomainRangeRule(
                column="in_temp",
                lower=2.0,
                upper=50.0,
                source_code="FG-EI-TI",
                source_name="internal_temperature",
                unit="celsius",
            )
        ]

        out = apply_rule_based_outlier_flags(df, rules)

        self.assertEqual(
            out[rule_outlier_flag_column("in_temp")].tolist(),
            [0, 1, 1, 0],
        )
        self.assertEqual(out[DEFAULT_RULE_AGGREGATE_FLAG].tolist(), [0, 1, 1, 0])
        self.assertEqual(out["in_temp"].tolist()[:3], [2.0, 1.9, 50.1])

    def test_rule_based_flags_ignore_action_columns(self):
        df = pd.DataFrame({"cont_skyl_vol": [0.5, 2.0]})
        rules = [
            DomainRangeRule(
                column="cont_skyl_vol",
                lower=0.0,
                upper=1.0,
                source_code="ACTION",
                source_name="left_window",
                unit="ratio",
            )
        ]

        out = apply_rule_based_outlier_flags(df, rules)

        self.assertNotIn("cont_skyl_vol_rule_outlier_flag", out.columns)
        self.assertEqual(out[DEFAULT_RULE_AGGREGATE_FLAG].tolist(), [0, 0])

    def test_tcn_placeholder_adds_stable_columns_when_model_is_absent(self):
        df = pd.DataFrame({"in_temp": [20.0, 21.0]})

        out = apply_tcn_outlier_flags(df)

        self.assertTrue(out[DEFAULT_TCN_SCORE_COLUMN].isna().all())
        self.assertTrue(out[DEFAULT_TCN_PRED_VALUE_COLUMN].isna().all())
        self.assertEqual(out[DEFAULT_TCN_FLAG_COLUMN].tolist(), [0, 0])

    def test_tcn_model_scores_are_thresholded(self):
        df = pd.DataFrame({"in_temp": [20.0, 21.0]})

        out = apply_tcn_outlier_flags(
            df,
            model=lambda frame: [0.1, 0.8],
            config=TCNOutlierConfig(score_threshold=0.5),
        )

        self.assertEqual(out[DEFAULT_TCN_SCORE_COLUMN].tolist(), [0.1, 0.8])
        self.assertEqual(out[DEFAULT_TCN_FLAG_COLUMN].tolist(), [0, 1])

    def test_outlier_pipeline_aggregates_rule_and_tcn_flags(self):
        df = pd.DataFrame({"in_temp": [1.9, 20.0]})
        rules = [
            DomainRangeRule(
                column="in_temp",
                lower=2.0,
                upper=50.0,
                source_code="FG-EI-TI",
                source_name="internal_temperature",
                unit="celsius",
            )
        ]

        out = apply_outlier_flags(
            df,
            rules=rules,
            tcn_model=lambda frame: pd.DataFrame(
                {
                    tcn_score_column("in_temp"): [0.1, 0.8],
                    tcn_pred_value_column("in_temp"): [2.5, 19.5],
                }
            ),
            tcn_config=TCNOutlierConfig(
                warning_threshold=0.5,
                outlier_threshold=0.7,
            ),
        )

        self.assertEqual(out[DEFAULT_RULE_AGGREGATE_FLAG].tolist(), [1, 0])
        self.assertEqual(out["in_temp_tcn_outlier_flag"].tolist(), [0, 1])
        self.assertEqual(out[DEFAULT_AGGREGATE_FLAG].tolist(), [1, 1])
        self.assertEqual(out[raw_value_column("in_temp")].tolist(), [1.9, 20.0])
        self.assertEqual(
            out[controller_value_column("in_temp")].tolist(),
            [2.5, 19.5],
        )
        self.assertEqual(out["in_temp_outlier_imputed_flag"].tolist(), [1, 1])

    def test_policy_matrix_keeps_raw_on_warning_and_replaces_on_outlier(self):
        df = pd.DataFrame({"in_temp": [20.0, 20.0, 80.0, 80.0]})
        rules = [
            DomainRangeRule(
                column="in_temp",
                lower=2.0,
                upper=50.0,
                source_code="FG-EI-TI",
                source_name="internal_temperature",
                unit="celsius",
            )
        ]

        out = apply_outlier_flags(
            df,
            rules=rules,
            tcn_model=lambda frame: pd.DataFrame(
                {
                    tcn_score_column("in_temp"): [0.1, 0.6, 0.1, 0.9],
                    tcn_pred_value_column("in_temp"): [21.0, 22.0, 45.0, 46.0],
                }
            ),
            tcn_config=TCNOutlierConfig(
                warning_threshold=0.5,
                outlier_threshold=0.8,
            ),
        )

        self.assertEqual(out["in_temp_ai_warning_flag"].tolist(), [0, 1, 0, 0])
        self.assertEqual(out["in_temp_ai_outlier_flag"].tolist(), [0, 0, 0, 1])
        self.assertEqual(out["in_temp_high_confidence_outlier_flag"].tolist(), [0, 0, 0, 1])
        self.assertEqual(out["in_temp_outlier_imputed_flag"].tolist(), [0, 0, 1, 1])
        self.assertEqual(
            out[controller_value_column("in_temp")].tolist(),
            [20.0, 20.0, 45.0, 46.0],
        )
        self.assertEqual(out[DEFAULT_AI_WARNING_FLAG_COLUMN].tolist(), [0, 1, 0, 0])
        self.assertEqual(out[DEFAULT_AI_OUTLIER_FLAG_COLUMN].tolist(), [0, 0, 0, 1])
        self.assertEqual(out[DEFAULT_IMPUTED_FLAG_COLUMN].tolist(), [0, 0, 1, 1])
        self.assertEqual(out[DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN].tolist(), [0, 0, 0, 1])

    def test_no_tcn_model_does_not_replace_rule_outlier_without_prediction(self):
        df = pd.DataFrame({"in_temp": [80.0]})
        rules = [
            DomainRangeRule(
                column="in_temp",
                lower=2.0,
                upper=50.0,
                source_code="FG-EI-TI",
                source_name="internal_temperature",
                unit="celsius",
            )
        ]

        out = apply_outlier_flags(df, rules=rules, tcn_model=None)

        self.assertEqual(out[controller_value_column("in_temp")].tolist(), [80.0])
        self.assertEqual(out["in_temp_outlier_imputed_flag"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
