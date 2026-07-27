import unittest

import numpy as np
import pandas as pd

from geas35.models.quality import MedianQualityModel, QualityModelOutput
from geas35.preprocessing import (
    QUALITY_AI_OUTLIER_FLAG_COLUMN,
    DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
    QUALITY_IMPUTED_FLAG_COLUMN,
    QUALITY_INVALID_FLAG_COLUMN,
    DomainRangeRule,
    action_restored_flag_column,
    ai_anomaly_score_column,
    ai_confidence_column,
    invalid_flag_column,
    missing_flag_column,
    prepare_geas_input_schema,
    prepare_missing_outliers_handled_features,
    quality_ai_outlier_flag_column,
    quality_imputed_flag_column,
    raw_value_column,
    rule_outlier_flag_column,
)


def schema_ready(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_geas_input_schema(df, keep_extra_columns=True)


class ConfidenceMedianQualityModel(MedianQualityModel):
    def predict_outlier(
        self,
        df: pd.DataFrame,
        thresholds: object | None = None,
        observation_columns=None,
    ) -> QualityModelOutput:
        output = super().predict_outlier(df, thresholds, observation_columns)
        confidence = pd.DataFrame(
            {
                col: [0.25 + 0.1 * row for row in range(len(df))]
                for col in output.observation_columns
            },
            index=df.index,
        )
        return QualityModelOutput(
            frame=output.frame,
            observation_columns=output.observation_columns,
            anomaly_scores=output.anomaly_scores,
            outlier_flags=output.outlier_flags,
            invalid_mask=output.invalid_mask,
            confidence_scores=confidence,
        )


class ConfidenceGatedQualityModel(MedianQualityModel):
    def reconstruct(
        self,
        df: pd.DataFrame,
        observation_columns=None,
        *,
        time_column: str = "reg_date",
        group_columns=(),
    ) -> pd.DataFrame:
        columns = self._resolved_columns(observation_columns)
        return pd.DataFrame({col: 999.0 for col in columns}, index=df.index)

    def predict_outlier(
        self,
        df: pd.DataFrame,
        thresholds: object | None = None,
        observation_columns=None,
    ) -> QualityModelOutput:
        columns = self._resolved_columns(observation_columns)
        scores = pd.DataFrame({col: [9.0] * len(df) for col in columns}, index=df.index)
        flags = pd.DataFrame({col: [1] * len(df) for col in columns}, index=df.index)
        invalid = flags.astype(bool)
        confidence = pd.DataFrame(
            {col: [0.1, 0.1, 0.4, 0.9] for col in columns},
            index=df.index,
        )
        return QualityModelOutput(
            frame=df.copy(),
            observation_columns=columns,
            anomaly_scores=scores,
            outlier_flags=flags,
            invalid_mask=invalid,
            confidence_scores=confidence,
        )


class MissingOutliersHandlingTest(unittest.TestCase):
    def test_combined_stage_restores_actions_and_imputes_observations_only(self):
        train = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": [
                        "2025-03-01 00:00:00",
                        "2025-03-01 00:05:00",
                        "2025-03-01 00:10:00",
                    ],
                    "in_temp": [20.0, 22.0, 24.0],
                    "in_hum": [60.0, 70.0, 80.0],
                }
            )
        )
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": [
                        "2025-03-02 00:00:00",
                        "2025-03-02 00:05:00",
                        "2025-03-02 00:10:00",
                    ],
                    "in_temp": [19.0, np.nan, 80.0],
                    "in_hum": [55.0, 65.0, 75.0],
                    "cont_skyl_vol": [np.nan, 0.3, np.nan],
                }
            )
        )
        control_log = pd.DataFrame(
            {
                "reg_date": [
                    "2025-03-02 00:00:00",
                    "2025-03-02 00:05:00",
                    "2025-03-02 00:10:00",
                ],
                "pred_ltw": [0.1, 0.9, np.nan],
            }
        )
        rules = [
            DomainRangeRule(
                column="in_temp",
                lower=2.0,
                upper=50.0,
                source_code="FG-EI-TI",
                source_name="internal_temperature",
                unit="celsius",
            ),
            DomainRangeRule(
                column="cont_skyl_vol",
                lower=0.0,
                upper=1.0,
                source_code="ACTION",
                source_name="left_window",
                unit="ratio",
            ),
        ]
        model = MedianQualityModel().fit(train, ["in_temp", "in_hum"])

        out = prepare_missing_outliers_handled_features(
            df,
            df_control_log=control_log,
            quality_model=model,
            rules=rules,
            observation_columns=["in_temp", "in_hum"],
        )

        self.assertEqual(out.loc[0, "cont_skyl_vol"], 0.1)
        self.assertEqual(out.loc[1, "cont_skyl_vol"], 0.3)
        self.assertTrue(pd.isna(out.loc[2, "cont_skyl_vol"]))
        self.assertEqual(
            out[action_restored_flag_column("cont_skyl_vol")].tolist(),
            [1, 0, 0],
        )
        self.assertNotIn("missing_flag", out.columns)
        self.assertNotIn("cont_skyl_vol_restore_source", out.columns)
        self.assertNotIn(rule_outlier_flag_column("cont_skyl_vol"), out.columns)

        self.assertEqual(out[missing_flag_column("in_temp")].tolist(), [0, 1, 0])
        self.assertEqual(out[rule_outlier_flag_column("in_temp")].tolist(), [0, 0, 1])
        self.assertEqual(out[invalid_flag_column("in_temp")].tolist(), [0, 1, 1])
        self.assertEqual(out[quality_imputed_flag_column("in_temp")].tolist(), [0, 1, 1])
        self.assertEqual(out["in_temp"].tolist(), [19.0, 22.0, 22.0])

        self.assertIn(raw_value_column("in_temp"), out.columns)
        self.assertIn(raw_value_column("cont_skyl_vol"), out.columns)
        self.assertTrue(pd.isna(out.loc[1, raw_value_column("in_temp")]))
        self.assertTrue(pd.isna(out.loc[0, raw_value_column("cont_skyl_vol")]))

        self.assertIn(ai_anomaly_score_column("in_temp"), out.columns)
        self.assertNotIn(ai_confidence_column("in_temp"), out.columns)
        self.assertIn(quality_ai_outlier_flag_column("in_temp"), out.columns)
        self.assertEqual(out[quality_ai_outlier_flag_column("in_temp")].tolist(), [0, 0, 0])
        self.assertEqual(out[QUALITY_AI_OUTLIER_FLAG_COLUMN].tolist(), [0, 0, 0])
        self.assertEqual(out[QUALITY_INVALID_FLAG_COLUMN].tolist(), [0, 1, 1])
        self.assertEqual(out[QUALITY_IMPUTED_FLAG_COLUMN].tolist(), [0, 1, 1])

    def test_quality_stage_writes_feature_level_ai_confidence_when_available(self):
        train = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-01 00:00:00", "2025-03-01 00:05:00"],
                    "in_temp": [20.0, 22.0],
                    "in_hum": [60.0, 70.0],
                }
            )
        )
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-02 00:00:00", "2025-03-02 00:05:00"],
                    "in_temp": [21.0, 23.0],
                    "in_hum": [65.0, 75.0],
                }
            )
        )
        model = ConfidenceMedianQualityModel().fit(train, ["in_temp", "in_hum"])

        out = prepare_missing_outliers_handled_features(
            df,
            quality_model=model,
            rules=[],
            observation_columns=["in_temp", "in_hum"],
        )

        self.assertEqual(out[ai_confidence_column("in_temp")].tolist(), [0.25, 0.35])
        self.assertEqual(out[ai_confidence_column("in_hum")].tolist(), [0.25, 0.35])
        self.assertNotIn("ai_confidence", out.columns)

    def test_confidence_gate_controls_ai_only_imputation_without_blocking_missing_or_rule(self):
        train = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-01 00:00:00", "2025-03-01 00:05:00"],
                    "in_temp": [20.0, 22.0],
                }
            )
        )
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": [
                        "2025-03-02 00:00:00",
                        "2025-03-02 00:05:00",
                        "2025-03-02 00:10:00",
                        "2025-03-02 00:15:00",
                    ],
                    "in_temp": [np.nan, 80.0, 40.0, 41.0],
                }
            )
        )
        rules = [
            DomainRangeRule(
                column="in_temp",
                lower=2.0,
                upper=50.0,
                source_code="FG-EI-TI",
                source_name="internal_temperature",
                unit="celsius",
            )
        ]
        model = ConfidenceGatedQualityModel().fit(train, ["in_temp"])

        out = prepare_missing_outliers_handled_features(
            df,
            quality_model=model,
            rules=rules,
            observation_columns=["in_temp"],
            imputation_confidence_threshold=DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
        )

        self.assertEqual(out[ai_confidence_column("in_temp")].tolist(), [0.1, 0.1, 0.4, 0.9])
        self.assertEqual(out[quality_ai_outlier_flag_column("in_temp")].tolist(), [1, 1, 1, 1])
        self.assertEqual(out[invalid_flag_column("in_temp")].tolist(), [1, 1, 1, 1])
        self.assertEqual(out["in_temp"].tolist(), [999.0, 999.0, 40.0, 999.0])
        self.assertEqual(out[quality_imputed_flag_column("in_temp")].tolist(), [1, 1, 0, 1])
        self.assertEqual(out[QUALITY_INVALID_FLAG_COLUMN].tolist(), [1, 1, 1, 1])
        self.assertEqual(out[QUALITY_IMPUTED_FLAG_COLUMN].tolist(), [1, 1, 0, 1])

    def test_confidence_threshold_must_be_probability(self):
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-02 00:00:00"],
                    "in_temp": [20.0],
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "imputation_confidence_threshold"):
            prepare_missing_outliers_handled_features(
                df,
                rules=[],
                observation_columns=["in_temp"],
                imputation_confidence_threshold=1.5,
            )

    def test_default_rule_only_model_does_not_impute_invalid_observations(self):
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-02 00:00:00"],
                    "in_temp": [80.0],
                }
            )
        )
        rules = [
            DomainRangeRule(
                column="in_temp",
                lower=2.0,
                upper=50.0,
                source_code="FG-EI-TI",
                source_name="internal_temperature",
                unit="celsius",
            )
        ]

        out = prepare_missing_outliers_handled_features(
            df,
            rules=rules,
            observation_columns=["in_temp"],
        )

        self.assertEqual(out.loc[0, "in_temp"], 80.0)
        self.assertEqual(out.loc[0, invalid_flag_column("in_temp")], 1)
        self.assertEqual(out.loc[0, quality_imputed_flag_column("in_temp")], 0)
        self.assertEqual(out.loc[0, QUALITY_IMPUTED_FLAG_COLUMN], 0)

    def test_quality_stage_rejects_action_observation_columns(self):
        df = schema_ready(
            pd.DataFrame(
                {
                    "reg_date": ["2025-03-02 00:00:00"],
                    "cont_skyl_vol": [0.2],
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "Action columns are not allowed"):
            prepare_missing_outliers_handled_features(
                df,
                observation_columns=["cont_skyl_vol"],
            )


if __name__ == "__main__":
    unittest.main()
