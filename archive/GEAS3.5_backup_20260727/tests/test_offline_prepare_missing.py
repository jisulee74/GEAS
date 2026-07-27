import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from geas35.offline.prepare_missing_values_handled_splits import (
    prepare_missing_values_handled_splits,
)
from geas35.preprocessing import (
    missing_flag_column,
    prepare_geas_input_schema,
)


class OfflinePrepareMissingTest(unittest.TestCase):
    def test_prepare_missing_values_handled_splits_writes_parquet_and_manifest(self):
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
                        "reg_date": [
                            "2025-03-01 00:00:00",
                            "2025-03-01 00:05:00",
                        ],
                        "segment_id": ["seg01", "seg01"],
                        "is_resampled_row": [False, True],
                        "in_temp": [20.0, np.nan],
                    }
                ),
                keep_extra_columns=True,
            )
            df.to_parquet(input_dir / "train.parquet", index=False)

            summaries = prepare_missing_values_handled_splits(
                dataset_root=dataset_root,
                crops=["strawberry"],
                splits=["train"],
            )

            output_path = (
                dataset_root
                / "4_preprocessed"
                / "2_missing_values_handled"
                / "strawberry"
                / "train.parquet"
            )
            manifest_path = (
                dataset_root
                / "4_preprocessed"
                / "2_missing_values_handled"
                / "missing_manifest.json"
            )

            self.assertEqual(len(summaries), 1)
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())

            result = pd.read_parquet(output_path)
            self.assertTrue(pd.isna(result.loc[1, "in_temp"]))
            self.assertEqual(result.loc[1, missing_flag_column("in_temp")], 1)
            self.assertNotIn("missing_flag", result.columns)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "missing")
            self.assertEqual(manifest["splits"][0]["input_rows"], 2)
            self.assertEqual(manifest["splits"][0]["output_rows"], 2)
            self.assertEqual(manifest["splits"][0]["missing_cells"], 75)
            self.assertEqual(manifest["splits"][0]["restored_action_cells"], 0)


if __name__ == "__main__":
    unittest.main()
