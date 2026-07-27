import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.offline.prepare_unit_canonicalized_splits import (
    prepare_unit_canonicalized_splits,
)
from geas35.preprocessing import prepare_geas_input_schema


class OfflinePrepareUnitCanonicalizedTest(unittest.TestCase):
    def test_prepare_unit_canonicalized_splits_writes_parquet_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            input_dir = (
                dataset_root
                / "4_preprocessed"
                / "1_input_schema_prepared"
                / "strawberry"
            )
            input_dir.mkdir(parents=True)
            df = prepare_geas_input_schema(
                pd.DataFrame(
                    {
                        "reg_date": ["2025-03-01 00:00:00"],
                        "cont_skyl_vol": [50.0],
                        "in_hum": [120.0],
                    }
                ),
                keep_extra_columns=True,
            )
            df.to_parquet(input_dir / "train.parquet", index=False)

            summaries = prepare_unit_canonicalized_splits(
                dataset_root=dataset_root,
                crops=["strawberry"],
                splits=["train"],
            )

            output_path = (
                dataset_root
                / "4_preprocessed"
                / "2_unit_canonicalized"
                / "strawberry"
                / "train.parquet"
            )
            manifest_path = (
                dataset_root
                / "4_preprocessed"
                / "2_unit_canonicalized"
                / "unit_canonicalized_manifest.json"
            )

            self.assertEqual(len(summaries), 1)
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())

            result = pd.read_parquet(output_path)
            self.assertEqual(result.loc[0, "cont_skyl_vol"], 0.5)
            self.assertEqual(result.loc[0, "in_hum"], 120.0)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "unit_canonicalized")
            self.assertEqual(manifest["splits"][0]["input_rows"], 1)
            self.assertEqual(manifest["splits"][0]["output_rows"], 1)
            self.assertEqual(manifest["splits"][0]["changed_percent_action_cells"], 1)


if __name__ == "__main__":
    unittest.main()
