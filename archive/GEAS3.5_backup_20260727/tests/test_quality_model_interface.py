import unittest

import numpy as np
import pandas as pd

from geas35.models.quality import (
    QualityModelOutput,
    RuleOnlyQualityModel,
    validate_observation_columns,
)


class QualityModelInterfaceTest(unittest.TestCase):
    def test_validate_observation_columns_rejects_action_columns(self):
        with self.assertRaisesRegex(ValueError, "Action columns are not allowed"):
            validate_observation_columns(["in_temp", "cont_skyl_vol"])

    def test_rule_only_fit_rejects_action_columns(self):
        model = RuleOnlyQualityModel()
        df = pd.DataFrame({"in_temp": [20.0], "cont_skyl_vol": [0.5]})

        with self.assertRaisesRegex(ValueError, "Action columns are not allowed"):
            model.fit(df, ["in_temp", "cont_skyl_vol"])

    def test_rule_only_invalid_mask_uses_missing_and_rule_flags(self):
        df = pd.DataFrame(
            {
                "in_temp": [20.0, np.nan, 80.0],
                "in_hum": [50.0, 60.0, 70.0],
                "in_temp_missing_flag": [0, 1, 0],
                "in_temp_rule_outlier_flag": [0, 0, 1],
                "in_hum_missing_flag": [0, 0, 0],
            }
        )
        model = RuleOnlyQualityModel().fit(df, ["in_temp", "in_hum"])

        output = model.predict_outlier(df)

        self.assertEqual(output.observation_columns, ("in_temp", "in_hum"))
        self.assertEqual(output.invalid_mask["in_temp"].tolist(), [False, True, True])
        self.assertEqual(output.invalid_mask["in_hum"].tolist(), [False, False, False])
        self.assertEqual(output.outlier_flags["in_temp"].tolist(), [0, 0, 0])
        self.assertEqual(output.anomaly_scores["in_temp"].tolist(), [0.0, 0.0, 0.0])
        self.assertIsNone(output.confidence_scores)

    def test_quality_model_output_accepts_optional_confidence_scores(self):
        df = pd.DataFrame({"in_temp": [20.0, 21.0]})
        scores = pd.DataFrame({"in_temp": [0.1, 0.9]})
        flags = pd.DataFrame({"in_temp": [0, 1]})
        invalid = pd.DataFrame({"in_temp": [False, True]})
        confidence = pd.DataFrame({"in_temp": [0.2, 0.8]})

        output = QualityModelOutput(
            frame=df,
            observation_columns=("in_temp",),
            anomaly_scores=scores,
            outlier_flags=flags,
            invalid_mask=invalid,
            confidence_scores=confidence,
        )

        self.assertEqual(output.confidence_scores["in_temp"].tolist(), [0.2, 0.8])

    def test_rule_only_impute_is_noop(self):
        df = pd.DataFrame({"in_temp": [20.0, np.nan]})
        model = RuleOnlyQualityModel().fit(df, ["in_temp"])
        prediction = model.reconstruct(df)
        invalid_mask = pd.DataFrame({"in_temp": [False, True]})

        result = model.impute(df, invalid_mask, prediction)

        self.assertTrue(result.equals(df))


if __name__ == "__main__":
    unittest.main()
