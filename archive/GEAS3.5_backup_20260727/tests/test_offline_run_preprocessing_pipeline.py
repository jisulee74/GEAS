import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.offline.run_preprocessing_pipeline import run_preprocessing_pipeline
from geas35.preprocessing import (
    QUALITY_INVALID_FLAG_COLUMN,
    STATE_COLUMNS,
    ai_anomaly_score_column,
    prepare_geas_input_schema,
    quality_ai_outlier_flag_column,
)


def _input_frame(values: list[float]) -> pd.DataFrame:
    df = prepare_geas_input_schema(
        pd.DataFrame(
            {
                "reg_date": pd.date_range(
                    "2025-03-01 00:00:00",
                    periods=len(values),
                    freq="5min",
                ),
                "in_temp": values,
                "cont_skyl_vol": [0.2] * len(values),
            }
        ),
        keep_extra_columns=True,
    )
    for col in STATE_COLUMNS:
        if col != "in_temp":
            df[col] = 0.0
    return df


class OfflineRunPreprocessingPipelineTest(unittest.TestCase):
    def test_pipeline_fits_selects_transforms_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            input_dir = (
                dataset_root
                / "4_preprocessed"
                / "2_unit_canonicalized"
                / "strawberry"
            )
            input_dir.mkdir(parents=True)
            _input_frame([20.0, 21.0, 22.0, 23.0]).to_parquet(
                input_dir / "train.parquet",
                index=False,
            )
            _input_frame([20.0, 21.0, 22.0, 23.0]).to_parquet(
                input_dir / "validation.parquet",
                index=False,
            )
            _input_frame([20.0, 40.0]).to_parquet(
                input_dir / "test.parquet",
                index=False,
            )

            domain_path = dataset_root / "domain_ranges.csv"
            pd.DataFrame(
                {
                    "column": ["in_temp"],
                    "lower": [0.0],
                    "upper": [100.0],
                    "source_code": ["FG-EI-TI"],
                    "source_name": ["internal_temperature"],
                    "unit": ["celsius"],
                }
            ).to_csv(domain_path, index=False)

            summaries = run_preprocessing_pipeline(
                dataset_root=dataset_root,
                crops=["strawberry"],
                candidate_model_names=["median"],
                candidate_thresholds=[0.1, 2.0],
                observation_columns=["in_temp"],
                domain_csv_path=domain_path,
            )

            output_root = dataset_root / "4_preprocessed" / "3_missing_outliers_handled"
            model_artifact_path = (
                output_root
                / "_artifacts"
                / "strawberry"
                / "selected_quality_model.json"
            )
            validation_report_path = (
                output_root
                / "_artifacts"
                / "strawberry"
                / "validation_selection_report.json"
            )
            manifest_path = output_root / "preprocessing_pipeline_manifest.json"

            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].selected_model, "median")
            self.assertTrue(model_artifact_path.exists())
            self.assertTrue(validation_report_path.exists())
            self.assertTrue(manifest_path.exists())

            for split in ("train", "validation", "test"):
                self.assertTrue((output_root / "strawberry" / f"{split}.parquet").exists())

            test_out = pd.read_parquet(output_root / "strawberry" / "test.parquet")
            self.assertIn(ai_anomaly_score_column("in_temp"), test_out.columns)
            self.assertIn(quality_ai_outlier_flag_column("in_temp"), test_out.columns)
            self.assertIn(QUALITY_INVALID_FLAG_COLUMN, test_out.columns)

            model_artifact = json.loads(
                model_artifact_path.read_text(encoding="utf-8")
            )
            self.assertEqual(model_artifact["model_name"], "median")
            self.assertEqual(model_artifact["observation_columns"], ["in_temp"])
            self.assertIn("threshold", model_artifact)
            self.assertIn("medians", model_artifact)

            validation_report = json.loads(
                validation_report_path.read_text(encoding="utf-8")
            )
            self.assertEqual(validation_report["selection_split"], "validation")
            self.assertFalse(validation_report["test_used_for_selection"])
            self.assertEqual(validation_report["selected_model"], "median")
            self.assertIn("threshold_results", validation_report)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "preprocessing_pipeline")
            self.assertEqual(manifest["scope"], "quality_model_only")
            self.assertFalse(manifest["selection_policy"]["test_used_for_selection"])
            self.assertFalse((dataset_root / "5_rl_dataset").exists())


if __name__ == "__main__":
    unittest.main()
