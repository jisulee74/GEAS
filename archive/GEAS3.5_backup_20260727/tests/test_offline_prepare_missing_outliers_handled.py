import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.offline.prepare_missing_outliers_handled_splits import (
    prepare_missing_outliers_handled_splits,
)
from geas35.preprocessing import (
    QUALITY_INVALID_FLAG_COLUMN,
    STATE_COLUMNS,
    invalid_flag_column,
    prepare_geas_input_schema,
    rule_outlier_flag_column,
)


class OfflinePrepareMissingOutliersHandledTest(unittest.TestCase):
    def test_prepare_missing_outliers_handled_splits_writes_parquet_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            input_dir = (
                dataset_root
                / "4_preprocessed"
                / "2_unit_canonicalized"
                / "strawberry"
            )
            input_dir.mkdir(parents=True)
            df = prepare_geas_input_schema(
                pd.DataFrame(
                    {
                        "reg_date": [
                            "2025-03-01 00:00:00",
                            "2025-03-01 00:05:00",
                        ],
                        "in_temp": [20.0, 80.0],
                        "cont_skyl_vol": [0.2, None],
                    }
                ),
                keep_extra_columns=True,
            )
            for col in STATE_COLUMNS:
                if col != "in_temp":
                    df[col] = 0.0
            df.to_parquet(input_dir / "train.parquet", index=False)

            domain_path = dataset_root / "domain_ranges.csv"
            pd.DataFrame(
                {
                    "column": ["in_temp", "cont_skyl_vol"],
                    "lower": [2.0, 0.0],
                    "upper": [50.0, 1.0],
                    "source_code": ["FG-EI-TI", "ACTION"],
                    "source_name": ["internal_temperature", "left_window"],
                    "unit": ["celsius", "ratio"],
                }
            ).to_csv(domain_path, index=False)

            summaries = prepare_missing_outliers_handled_splits(
                dataset_root=dataset_root,
                crops=["strawberry"],
                splits=["train"],
                domain_csv_path=domain_path,
            )

            output_path = (
                dataset_root
                / "4_preprocessed"
                / "3_missing_outliers_handled"
                / "strawberry"
                / "train.parquet"
            )
            manifest_path = (
                dataset_root
                / "4_preprocessed"
                / "3_missing_outliers_handled"
                / "missing_outliers_handled_manifest.json"
            )

            self.assertEqual(len(summaries), 1)
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())

            result = pd.read_parquet(output_path)
            self.assertEqual(result[rule_outlier_flag_column("in_temp")].tolist(), [0, 1])
            self.assertEqual(result[invalid_flag_column("in_temp")].tolist(), [0, 1])
            self.assertEqual(result[QUALITY_INVALID_FLAG_COLUMN].tolist(), [0, 1])
            self.assertNotIn(rule_outlier_flag_column("cont_skyl_vol"), result.columns)
            self.assertNotIn("missing_flag", result.columns)
            self.assertNotIn("cont_skyl_vol_restore_source", result.columns)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "missing_outliers_handled")
            self.assertEqual(manifest["quality_model"], "rule_only")
            self.assertEqual(manifest["splits"][0]["input_rows"], 2)
            self.assertEqual(manifest["splits"][0]["output_rows"], 2)
            self.assertEqual(manifest["splits"][0]["rule_outlier_rows"], 1)
            self.assertEqual(manifest["splits"][0]["invalid_rows"], 1)


if __name__ == "__main__":
    unittest.main()
