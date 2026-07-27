import unittest

import pandas as pd

from geas35.preprocessing import (
    GROWTH_STAGE_DAT_COLUMN,
    GROWTH_STAGE_NAME_COLUMN,
    GROWTH_STAGE_ORDER_COLUMN,
    add_growth_stage_columns,
)


class GrowthStageFeatureTest(unittest.TestCase):
    def test_add_growth_stage_columns_uses_series_transplant_dates(self):
        df = pd.DataFrame(
            {
                "reg_date": ["2026-03-05 00:00:00", "2026-03-30 12:00:00"],
                "series_id": ["cucumber_s1", "cucumber_s1"],
                "crop": ["cucumber", "cucumber"],
            }
        )

        result = add_growth_stage_columns(
            df,
            transplant_date_by_series={"cucumber_s1": "2026-03-05"},
        )

        self.assertEqual(result[GROWTH_STAGE_DAT_COLUMN].tolist(), [1, 26])
        self.assertEqual(result[GROWTH_STAGE_ORDER_COLUMN].tolist(), [1, 3])
        self.assertNotIn(GROWTH_STAGE_NAME_COLUMN, result.columns)

    def test_add_growth_stage_columns_can_include_stage_name_metadata(self):
        df = pd.DataFrame(
            {
                "reg_date": ["2025-05-15"],
                "crop": ["melon"],
            }
        )

        result = add_growth_stage_columns(
            df,
            transplant_date="2025-05-08",
            include_stage_name=True,
        )

        self.assertEqual(result.loc[0, GROWTH_STAGE_DAT_COLUMN], 8)
        self.assertEqual(result.loc[0, GROWTH_STAGE_ORDER_COLUMN], 2)
        self.assertEqual(result.loc[0, GROWTH_STAGE_NAME_COLUMN], "vine_extension")


if __name__ == "__main__":
    unittest.main()
