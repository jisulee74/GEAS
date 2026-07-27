import unittest

from importlib import import_module

import numpy as np
import pandas as pd

from geas35.preprocessing import (
    canonicalize_feature_units,
    canonicalize_for_derived,
)
from geas35.realtime import sanitize_for_controller


class NormalizationTest(unittest.TestCase):
    def test_legacy_unit_canonicalization_module_wraps_stage_2(self):
        legacy = import_module("geas35.preprocessing.3_unit_canonicalization")
        current = import_module("geas35.preprocessing.2_unit_canonicalization")

        self.assertIs(
            legacy.canonicalize_feature_units,
            current.canonicalize_feature_units,
        )

    def test_canonicalize_feature_units_converts_units_without_clipping(self):
        df = pd.DataFrame(
            {
                "reg_date": pd.to_datetime(
                    [
                        "2024-01-02 00:05:00",
                        "2024-01-02 00:10:00",
                        "2024-01-02 00:00:00",
                    ]
                ),
                "in_hum": [-5.0, 50.0, 150.0],
                "etc_blackout": [-1.0, 0.5, 2.0],
                "cont_skyl_vol": [0.5, 0.7, 1.2],
                "cont_heater_run": [-0.5, 0.5, 2.0],
                "out_winddirec": [-10.0, 90.0, 400.0],
                "out_windsp": [-1.0, 2.0, 3.0],
                "out_rain": [-1.0, 0.5, 2.0],
            }
        )

        result = canonicalize_feature_units(df)

        self.assertEqual(len(result), 3)
        self.assertEqual(result["reg_date"].tolist(), df["reg_date"].tolist())
        self.assertEqual(result["in_hum"].tolist(), [-5.0, 50.0, 150.0])
        self.assertEqual(result["etc_blackout"].tolist(), [-1.0, 0.5, 2.0])
        self.assertEqual(result["cont_skyl_vol"].tolist(), [0.5, 0.7, 1.2])
        self.assertEqual(result["cont_heater_run"].tolist(), [-0.5, 0.5, 2.0])
        self.assertEqual(result["out_winddirec"].tolist(), [-10.0, 90.0, 400.0])
        self.assertEqual(result["out_windsp"].tolist(), [-1.0, 2.0, 3.0])
        self.assertEqual(result["out_rain"].tolist(), [-1.0, 0.5, 2.0])

    def test_canonicalize_feature_units_converts_percent_scale_to_0_1(self):
        df = pd.DataFrame(
            {
                "reg_date": ["2024-01-02 00:00:00", "2024-01-02 00:05:00"],
                "cont_skyl_vol": [25.0, 150.0],
            }
        )

        result = canonicalize_feature_units(df)

        self.assertEqual(result["cont_skyl_vol"].tolist(), [0.25, 1.5])

    def test_canonicalize_feature_units_requires_schema_numeric_input(self):
        df = pd.DataFrame({"cont_skyl_vol": ["0.5"]})

        with self.assertRaisesRegex(TypeError, "1_input_schema_preparation"):
            canonicalize_feature_units(df)

    def test_canonicalize_for_derived_creates_expected_aliases_without_clipping(self):
        df = pd.DataFrame(
            {
                "reg_date": ["2024-01-02 00:05:00", "2024-01-02 00:00:00"],
                "in_hum": [120.0, 80.0],
                "out_windsp": [-2.0, 3.5],
                "cont_skyl_vol": [0.25, 0.5],
                "cont_heater_run": [2.0, np.nan],
            }
        )

        result = canonicalize_for_derived(df)

        self.assertEqual(result["in_rh"].tolist(), [120.0, 80.0])
        self.assertEqual(result["wind_speed"].tolist(), [-2.0, 3.5])
        self.assertEqual(result["window_pct"].tolist(), [0.25, 0.5])
        self.assertEqual(result["heater_duty"].iloc[0], 2.0)
        self.assertTrue(np.isnan(result["heater_duty"].iloc[1]))

    def test_sanitize_for_controller_clips_after_qc(self):
        df = pd.DataFrame(
            {
                "in_hum": [-5.0, 150.0],
                "etc_blackout": [-1.0, 2.0],
                "cont_skyl_vol": [0.5, 1.2],
                "cont_heater_run": [-0.5, 2.0],
                "out_winddirec": [-10.0, 400.0],
                "out_windsp": [-1.0, 3.0],
                "out_rain": [-1.0, 2.0],
            }
        )

        result = sanitize_for_controller(df)

        self.assertEqual(result["in_hum"].tolist(), [0.0, 100.0])
        self.assertEqual(result["etc_blackout"].tolist(), [0.0, 1.0])
        self.assertEqual(result["cont_skyl_vol"].tolist(), [0.5, 1.0])
        self.assertEqual(result["cont_heater_run"].tolist(), [0.0, 1.0])
        self.assertEqual(result["out_winddirec"].tolist(), [0.0, 360.0])
        self.assertEqual(result["out_windsp"].tolist(), [0.0, 3.0])
        self.assertEqual(result["out_rain"].tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
