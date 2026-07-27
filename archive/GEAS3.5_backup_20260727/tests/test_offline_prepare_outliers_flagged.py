import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.offline.prepare_outliers_flagged_splits import (
    prepare_outliers_flagged_splits,
)


class OfflinePrepareOutliersFlaggedTest(unittest.TestCase):
    def test_prepare_outliers_flagged_splits_writes_parquet_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            input_dir = (
                dataset_root
                / "4_preprocessed"
                / "3_unit_canonicalized"
                / "strawberry"
            )
            input_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "reg_date": [
                        "2025-03-01 00:00:00",
                        "2025-03-01 00:05:00",
                    ],
                    "in_temp": [20.0, 80.0],
                    "cont_skyl_vol": [0.5, 2.0],
                }
            ).to_parquet(input_dir / "train.parquet", index=False)

            domain_csv = dataset_root / "ranges.csv"
            domain_csv.write_text(
                "column,lower,upper,source_code,source_name,unit\n"
                "in_temp,2,50,FG-EI-TI,internal_temperature,celsius\n"
                "cont_skyl_vol,0,1,ACTION,left_window,ratio\n",
                encoding="utf-8",
            )

            summaries = prepare_outliers_flagged_splits(
                dataset_root=dataset_root,
                crops=["strawberry"],
                splits=["train"],
                domain_csv_path=domain_csv,
            )

            output_path = (
                dataset_root
                / "4_preprocessed"
                / "4_outliers_flagged"
                / "strawberry"
                / "train.parquet"
            )
            manifest_path = (
                dataset_root
                / "4_preprocessed"
                / "4_outliers_flagged"
                / "outliers_flagged_manifest.json"
            )

            self.assertEqual(len(summaries), 1)
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())

            result = pd.read_parquet(output_path)
            self.assertEqual(result["in_temp_rule_outlier_flag"].tolist(), [0, 1])
            self.assertNotIn("cont_skyl_vol_rule_outlier_flag", result.columns)
            self.assertEqual(result["rule_outlier_flag"].tolist(), [0, 1])
            self.assertEqual(result["tcn_outlier_flag"].tolist(), [0, 0])
            self.assertEqual(result["outlier_flag"].tolist(), [0, 1])
            self.assertEqual(result["in_temp_raw_value"].tolist(), [20.0, 80.0])
            self.assertEqual(result["in_temp_controller_value"].tolist(), [20.0, 80.0])
            self.assertEqual(result["in_temp_outlier_imputed_flag"].tolist(), [0, 0])

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "outliers_flagged")
            self.assertEqual(manifest["splits"][0]["input_rows"], 2)
            self.assertEqual(manifest["splits"][0]["outlier_rows"], 1)
            self.assertEqual(manifest["splits"][0]["rule_outlier_cells"], 1)
            self.assertEqual(manifest["splits"][0]["imputed_cells"], 0)


if __name__ == "__main__":
    unittest.main()
