import unittest
from unittest.mock import patch

import pandas as pd

from geas35.models.quality import (
    ModernTCNConfig,
    ModernTCNQualityModel,
)
from geas35.models.quality.deep import TorchUnavailableError
from geas35.offline.run_preprocessing_pipeline import (
    DEFAULT_CANDIDATE_MODELS,
    _make_quality_model,
)


class ConstantModernTCNQualityModel(ModernTCNQualityModel):
    def reconstruct(
        self,
        df: pd.DataFrame,
        observation_columns=None,
        *,
        time_column: str = "reg_date",
        group_columns=(),
    ) -> pd.DataFrame:
        columns = self._resolved_columns(observation_columns)
        return pd.DataFrame({col: 20.0 for col in columns}, index=df.index)


class ModernTCNQualityModelTest(unittest.TestCase):
    def test_fit_rejects_action_columns_before_torch_is_required(self):
        df = pd.DataFrame({"in_temp": [20.0], "cont_skyl_vol": [0.5]})

        with self.assertRaisesRegex(ValueError, "Action columns are not allowed"):
            ModernTCNQualityModel().fit(df, ["in_temp", "cont_skyl_vol"])

    def test_fit_reports_torch_optional_dependency_when_unavailable(self):
        df = pd.DataFrame({"in_temp": [20.0, 21.0, 22.0]})

        with patch(
            "geas35.models.quality.modern_tcn.require_torch",
            side_effect=TorchUnavailableError("torch unavailable"),
        ), self.assertRaises(TorchUnavailableError):
            ModernTCNQualityModel().fit(df, ["in_temp"])

    def test_predict_outlier_returns_scores_flags_and_confidence(self):
        df = pd.DataFrame(
            {
                "in_temp": [20.0, 23.0, 25.0],
                "cont_skyl_vol": [0.1, None, 0.3],
            }
        )
        model = ConstantModernTCNQualityModel()
        model.observation_columns_ = ("in_temp",)
        model.confidence_scales_ = pd.Series({"in_temp": 1.0})
        model.rule_model_.fit(df, ["in_temp"])

        output = model.predict_outlier(df, thresholds=2.0)

        self.assertEqual(output.anomaly_scores["in_temp"].tolist(), [0.0, 3.0, 5.0])
        self.assertEqual(output.outlier_flags["in_temp"].tolist(), [0, 1, 1])
        self.assertEqual(output.invalid_mask["in_temp"].tolist(), [False, True, True])
        self.assertIsNotNone(output.confidence_scores)
        self.assertLess(output.confidence_scores.loc[0, "in_temp"], 0.5)
        self.assertGreater(output.confidence_scores.loc[1, "in_temp"], 0.5)

    def test_impute_preserves_action_columns(self):
        df = pd.DataFrame(
            {
                "in_temp": [20.0, 25.0],
                "cont_skyl_vol": [0.1, None],
            }
        )
        prediction = pd.DataFrame({"in_temp": [20.0, 21.0]})
        invalid = pd.DataFrame({"in_temp": [False, True]})
        model = ConstantModernTCNQualityModel()
        model.observation_columns_ = ("in_temp",)

        result = model.impute(df, invalid, prediction, ["in_temp"])

        self.assertEqual(result["in_temp"].tolist(), [20.0, 21.0])
        self.assertTrue(pd.isna(result.loc[1, "cont_skyl_vol"]))

    def test_pipeline_factory_supports_modern_tcn_without_defaulting_to_it(self):
        model = _make_quality_model("modern_tcn")

        self.assertIsInstance(model, ModernTCNQualityModel)
        self.assertNotIn("modern_tcn", DEFAULT_CANDIDATE_MODELS)

    def test_config_from_hyperparameter_candidate_uses_common_and_model_specific(self):
        from geas35.models.quality import HyperparameterCandidate

        candidate = HyperparameterCandidate(
            name="small",
            common={"lookback": 3, "mask_fraction": 0.5},
            model_specific={"channel_width": 4, "depth": 1},
        )

        config = ModernTCNConfig.from_candidate(candidate)

        self.assertEqual(config.lookback, 3)
        self.assertEqual(config.mask_fraction, 0.5)
        self.assertEqual(config.channel_width, 4)
        self.assertEqual(config.depth, 1)

    def test_small_synthetic_series_fit_and_reconstruct_when_torch_is_available(self):
        df = pd.DataFrame(
            {
                "reg_date": pd.date_range("2025-01-01", periods=5, freq="5min"),
                "in_temp": [20.0, 21.0, 22.0, 23.0, 24.0],
            }
        )
        model = ModernTCNQualityModel(
            ModernTCNConfig(
                lookback=2,
                mask_fraction=1.0,
                batch_size=2,
                epochs=2,
                channel_width=4,
                depth=1,
                random_state=1,
            )
        ).fit(df, ["in_temp"], validation_df=df)

        prediction = model.reconstruct(df, ["in_temp"])

        self.assertEqual(prediction.shape, (5, 1))
        self.assertEqual(prediction.columns.tolist(), ["in_temp"])
        self.assertGreaterEqual(len(model.training_history_), 1)


if __name__ == "__main__":
    unittest.main()
