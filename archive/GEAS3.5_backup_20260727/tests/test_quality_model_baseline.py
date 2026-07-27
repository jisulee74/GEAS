import unittest

import numpy as np
import pandas as pd

from geas35.models.quality import (
    MedianQualityModel,
    evaluate_synthetic_masking,
    make_synthetic_mask,
)


class MedianQualityModelTest(unittest.TestCase):
    def test_fit_rejects_action_columns(self):
        df = pd.DataFrame({"in_temp": [20.0], "cont_skyl_vol": [0.5]})

        with self.assertRaisesRegex(ValueError, "Action columns are not allowed"):
            MedianQualityModel().fit(df, ["in_temp", "cont_skyl_vol"])

    def test_median_model_imputes_invalid_observation_cells_only(self):
        train = pd.DataFrame(
            {
                "in_temp": [20.0, 22.0, 24.0],
                "in_hum": [60.0, 70.0, 80.0],
            }
        )
        df = pd.DataFrame(
            {
                "in_temp": [19.0, np.nan, 80.0],
                "in_hum": [55.0, 65.0, 75.0],
                "cont_skyl_vol": [0.1, np.nan, 0.3],
                "in_temp_missing_flag": [0, 1, 0],
                "in_temp_rule_outlier_flag": [0, 0, 1],
            }
        )
        model = MedianQualityModel().fit(train, ["in_temp", "in_hum"])
        output = model.predict_outlier(df)
        prediction = model.reconstruct(df)

        result = model.impute(df, output.invalid_mask, prediction)

        self.assertEqual(result["in_temp"].tolist(), [19.0, 22.0, 22.0])
        self.assertEqual(result["in_hum"].tolist(), [55.0, 65.0, 75.0])
        self.assertTrue(pd.isna(result.loc[1, "cont_skyl_vol"]))

    def test_anomaly_score_is_absolute_residual_from_median(self):
        train = pd.DataFrame({"in_temp": [20.0, 22.0, 24.0]})
        df = pd.DataFrame({"in_temp": [21.0, 25.0]})
        model = MedianQualityModel().fit(train, ["in_temp"])

        prediction = model.reconstruct(df)
        scores = model.anomaly_score(df, prediction)

        self.assertEqual(prediction["in_temp"].tolist(), [22.0, 22.0])
        self.assertEqual(scores["in_temp"].tolist(), [1.0, 3.0])

    def test_predict_outlier_applies_selected_threshold_to_scores(self):
        train = pd.DataFrame({"in_temp": [20.0, 22.0, 24.0]})
        df = pd.DataFrame({"in_temp": [21.0, 30.0]})
        model = MedianQualityModel().fit(train, ["in_temp"])

        output = model.predict_outlier(df, thresholds=2.0)

        self.assertEqual(output.outlier_flags["in_temp"].tolist(), [0, 1])
        self.assertEqual(output.invalid_mask["in_temp"].tolist(), [False, True])


class SyntheticMaskingEvaluationTest(unittest.TestCase):
    def test_make_synthetic_mask_uses_only_normal_observed_cells(self):
        df = pd.DataFrame(
            {
                "in_temp": [20.0, np.nan, 80.0, 21.0],
                "in_temp_missing_flag": [0, 1, 0, 0],
                "in_temp_rule_outlier_flag": [0, 0, 1, 0],
            }
        )

        mask = make_synthetic_mask(
            df,
            ["in_temp"],
            mask_fraction=1.0,
            random_state=7,
        )

        self.assertEqual(mask["in_temp"].tolist(), [True, False, False, True])

    def test_evaluate_synthetic_masking_returns_mae_and_rmse(self):
        train = pd.DataFrame({"in_temp": [20.0, 22.0, 24.0]})
        validation = pd.DataFrame(
            {
                "in_temp": [20.0, 24.0],
                "in_temp_missing_flag": [0, 0],
                "in_temp_rule_outlier_flag": [0, 0],
            }
        )
        model = MedianQualityModel().fit(train, ["in_temp"])

        result = evaluate_synthetic_masking(
            model,
            validation,
            ["in_temp"],
            mask_fraction=1.0,
            random_state=1,
        )

        self.assertEqual(result.masked_cells, 2)
        self.assertEqual(result.mae, 2.0)
        self.assertEqual(result.rmse, 2.0)
        self.assertEqual(result.per_column["in_temp"]["masked_cells"], 2.0)


if __name__ == "__main__":
    unittest.main()
