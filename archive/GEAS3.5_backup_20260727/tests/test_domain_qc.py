import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.qc import (
    DEFAULT_DOMAIN_RANGE_PATH,
    apply_default_domain_flags,
    apply_domain_range_flags,
    load_domain_range_rules,
)


class DomainQCTest(unittest.TestCase):
    def test_default_rule_file_lives_inside_geas35(self):
        self.assertTrue(DEFAULT_DOMAIN_RANGE_PATH.exists())
        self.assertIn("GEAS3.5", str(DEFAULT_DOMAIN_RANGE_PATH))

    def test_load_rules_uses_canonical_ascii_schema(self):
        csv_text = (
            "column,lower,upper,source_code,source_name,unit\n"
            "in_temp,2,50,FG-EI-TI,internal_temperature,celsius\n"
            "out_windsp,0,30,FG-EO-WS,outdoor_wind_speed,m_s\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ranges.csv"
            path.write_text(csv_text, encoding="utf-8")

            rules = load_domain_range_rules(path)

        by_col = {rule.column: rule for rule in rules}
        self.assertEqual(by_col["in_temp"].lower, 2.0)
        self.assertEqual(by_col["in_temp"].upper, 50.0)
        self.assertTrue(by_col["in_temp"].upper_inclusive)
        self.assertEqual(by_col["out_windsp"].upper, 30.0)

    def test_apply_domain_range_flags_marks_values_outside_configured_bounds(self):
        rules = load_domain_range_rules()
        df = pd.DataFrame(
            {
                "in_temp": [2.0, 1.9, 49.9, 50.0, None],
                "out_windsp": [0.0, 31.0, 30.0, None, 2.0],
                "out_rain": [0.0, 0.5, 1.0, 2.0, None],
            }
        )

        out = apply_domain_range_flags(df, rules)

        self.assertEqual(out["in_temp_rule_outlier_flag"].tolist(), [0, 1, 0, 1, 0])
        self.assertEqual(out["out_windsp_rule_outlier_flag"].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(out["out_rain_rule_outlier_flag"].tolist(), [0, 1, 0, 1, 0])
        self.assertEqual(out["rule_outlier_flag"].tolist(), [0, 1, 0, 1, 0])
        self.assertEqual(out["in_temp"].tolist()[:4], [2.0, 1.9, 49.9, 50.0])

    def test_apply_default_domain_flags_loads_internal_config(self):
        df = pd.DataFrame({"out_light": [0.0, 1400.0, 1400.1]})

        out = apply_default_domain_flags(df)

        self.assertEqual(out["out_light_rule_outlier_flag"].tolist(), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
