import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.offline.prepare_input_schema_prepared_splits import (
    prepare_input_schema_prepared_splits,
)
from geas35.preprocessing import (
    FEATURE_COLUMNS,
    GROWTH_STAGE_DAT_COLUMN,
    GROWTH_STAGE_ORDER_COLUMN,
)


class OfflinePrepareInputSchemaTest(unittest.TestCase):
    def test_prepare_input_schema_prepared_splits_writes_parquet_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            raw_dir = dataset_root / "1_raw"
            raw_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "series_id": ["strawberry_test"],
                    "crop": ["strawberry"],
                    "actual_start": ["2025-03-01 00:00:00"],
                }
            ).to_csv(raw_dir / "manifest.csv", index=False)

            input_dir = dataset_root / "3_splits" / "strawberry"
            input_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-01 00:05:00", "bad"],
                    "in_temp": ["23.5", "24.0"],
                    "series_id": ["strawberry_test", "strawberry_test"],
                    "segment_id": ["seg01", "seg01"],
                    "split": ["train", "train"],
                }
            ).to_parquet(input_dir / "train.parquet", index=False)

            summaries = prepare_input_schema_prepared_splits(
                dataset_root=dataset_root,
                crops=["strawberry"],
                splits=["train"],
            )

            output_path = (
                dataset_root
                / "4_preprocessed"
                / "1_input_schema_prepared"
                / "strawberry"
                / "train.parquet"
            )
            manifest_path = (
                dataset_root
                / "4_preprocessed"
                / "1_input_schema_prepared"
                / "input_schema_manifest.json"
            )

            self.assertEqual(len(summaries), 1)
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())

            result = pd.read_parquet(output_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "in_temp"], 23.5)
            self.assertIn("segment_id", result.columns)
            self.assertEqual(result.loc[0, GROWTH_STAGE_DAT_COLUMN], 1)
            self.assertEqual(result.loc[0, GROWTH_STAGE_ORDER_COLUMN], 1)
            for col in FEATURE_COLUMNS:
                self.assertIn(col, result.columns)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "input_schema")
            self.assertEqual(
                manifest["growth_stage_columns"],
                [GROWTH_STAGE_ORDER_COLUMN, GROWTH_STAGE_DAT_COLUMN],
            )
            self.assertIn("growth_stage_rules_sha256", manifest)
            self.assertEqual(manifest["splits"][0]["input_rows"], 2)
            self.assertEqual(manifest["splits"][0]["output_rows"], 1)
            self.assertEqual(manifest["splits"][0]["dropped_invalid_time_rows"], 1)
            self.assertEqual(manifest["splits"][0]["growth_stage_dat_missing_rows"], 0)
            self.assertEqual(manifest["splits"][0]["growth_stage_order_missing_rows"], 0)


if __name__ == "__main__":
    unittest.main()
