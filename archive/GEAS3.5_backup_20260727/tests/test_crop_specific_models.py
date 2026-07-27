import tempfile
import unittest
from pathlib import Path

import pandas as pd

from geas35.models import (
    discover_dataset_crops,
    load_crop_model,
    predict_with_crop_model,
    save_crop_model,
    train_models_by_crop,
)


class ConstantPredictor:
    def __init__(self, value):
        self.value = value

    def predict(self, inputs):
        return [self.value for _ in range(len(inputs))]


class CropSpecificModelsTest(unittest.TestCase):
    def _write_split(self, root: Path, crop: str, split: str, values: list[int]) -> None:
        crop_dir = root / crop
        crop_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": values}).to_parquet(crop_dir / f"{split}.parquet", index=False)

    def test_train_models_by_crop_saves_independent_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            models_root = tmp_path / "models"
            self._write_split(dataset_root, "cucumber", "train", [1, 2])
            self._write_split(dataset_root, "cucumber", "validation", [3])
            self._write_split(dataset_root, "melon", "train", [10])
            self._write_split(dataset_root, "melon", "validation", [20, 30])

            seen = []

            def trainer(crop, train_df, validation_df):
                seen.append((crop, len(train_df), len(validation_df)))
                return ConstantPredictor(crop)

            results = train_models_by_crop(
                dataset_root,
                models_root,
                trainer,
                model_name="mdp_v1_policy",
            )

            self.assertEqual([result.crop for result in results], ["cucumber", "melon"])
            self.assertEqual(seen, [("cucumber", 2, 1), ("melon", 1, 2)])
            self.assertEqual(
                predict_with_crop_model(
                    models_root,
                    "cucumber",
                    pd.DataFrame({"x": [0, 1]}),
                    model_name="mdp_v1_policy",
                ),
                ["cucumber", "cucumber"],
            )
            self.assertEqual(
                predict_with_crop_model(
                    models_root,
                    "melon",
                    pd.DataFrame({"x": [0]}),
                    model_name="mdp_v1_policy",
                ),
                ["melon"],
            )

    def test_subject_code_loads_matching_crop_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_root = Path(tmp)
            save_crop_model(
                ConstantPredictor("cucumber-model"),
                models_root,
                "cucumber",
                model_name="transition",
            )

            model = load_crop_model(models_root, "04", model_name="transition")

            self.assertEqual(model.predict([1]), ["cucumber-model"])

    def test_discover_dataset_crops_ignores_missing_train_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            self._write_split(dataset_root, "strawberry", "train", [1])
            self._write_split(dataset_root, "melon", "validation", [2])

            self.assertEqual(discover_dataset_crops(dataset_root), ["strawberry"])


if __name__ == "__main__":
    unittest.main()
