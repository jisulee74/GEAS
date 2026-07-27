import unittest
from importlib import import_module

import pandas as pd

from geas35.preprocessing import (
    DomainRangeRule,
    apply_domain_range_flags,
    canonicalize_feature_units,
    normalize_missing_input,
    prepare_geas_input_schema,
    prepare_missing_features,
    prepare_quality_features,
    prepare_missing_outliers_handled_features,
    rule_outlier_flag_column,
)


class PreprocessingCompatibilityTest(unittest.TestCase):
    def test_legacy_unit_canonicalization_module_wraps_new_stage(self):
        legacy = import_module("geas35.preprocessing.3_unit_canonicalization")
        df = pd.DataFrame({"cont_skyl_vol": [50.0]})

        current = canonicalize_feature_units(df)
        wrapped = legacy.canonicalize_feature_units(df)

        self.assertEqual(current.loc[0, "cont_skyl_vol"], 0.5)
        self.assertEqual(wrapped.loc[0, "cont_skyl_vol"], 0.5)

    def test_legacy_missing_alias_still_points_to_missing_stage(self):
        self.assertIs(normalize_missing_input, prepare_missing_features)

    def test_quality_stage_readable_alias_still_points_to_combined_stage(self):
        self.assertIs(prepare_quality_features, prepare_missing_outliers_handled_features)

    def test_legacy_domain_range_alias_still_flags_observation_rules(self):
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

        out = apply_domain_range_flags(df, rules)

        self.assertEqual(out.loc[0, rule_outlier_flag_column("in_temp")], 1)

    def test_combined_stage_does_not_restore_old_aggregate_missing_flag(self):
        df = prepare_geas_input_schema(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-01 00:00:00"],
                    "in_temp": [None],
                }
            ),
            keep_extra_columns=True,
        )

        out = prepare_quality_features(df, observation_columns=["in_temp"])

        self.assertIn("in_temp_missing_flag", out.columns)
        self.assertNotIn("missing_flag", out.columns)


if __name__ == "__main__":
    unittest.main()
