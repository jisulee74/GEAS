import unittest

import pandas as pd

from geas35.preprocessing import (
    FEATURE_COLUMNS,
    prepare_geas_input_schema,
)


class InputSchemaPreparationTest(unittest.TestCase):
    def test_prepare_geas_input_schema_parses_time_and_drops_invalid_rows(self):
        df = pd.DataFrame(
            {
                "reg_date": ["2025-03-01 00:05:00", "bad", None],
                "in_temp": ["20.5", "21.0", "22.0"],
            }
        )

        result = prepare_geas_input_schema(df, keep_extra_columns=False)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "reg_date"], pd.Timestamp("2025-03-01 00:05:00"))

    def test_prepare_geas_input_schema_ensures_39_feature_columns(self):
        df = pd.DataFrame({"reg_date": ["2025-03-01 00:00:00"]})

        result = prepare_geas_input_schema(df, keep_extra_columns=False)

        self.assertEqual(result.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(len(result.columns), 39)
        self.assertTrue(pd.isna(result.loc[0, "in_temp"]))
        self.assertTrue(pd.isna(result.loc[0, "cont_skyl_vol"]))

    def test_prepare_geas_input_schema_coerces_state_and_action_to_numeric(self):
        df = pd.DataFrame(
            {
                "reg_date": ["2025-03-01 00:00:00"],
                "in_temp": ["23.5"],
                "in_hum": ["error"],
                "cont_heater_run": ["1"],
            }
        )

        result = prepare_geas_input_schema(df, keep_extra_columns=False)

        self.assertEqual(result.loc[0, "in_temp"], 23.5)
        self.assertTrue(pd.isna(result.loc[0, "in_hum"]))
        self.assertEqual(result.loc[0, "cont_heater_run"], 1)


if __name__ == "__main__":
    unittest.main()
