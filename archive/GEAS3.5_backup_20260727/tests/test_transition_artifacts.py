import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from geas35.models.transition import (
    IndependentTargetTransitionModel,
    TransitionDataset,
    TransitionOneStepEvaluationReport,
    load_selected_transition_model_manifest,
    load_transition_model_artifact,
    save_selected_transition_model,
    save_transition_model_artifact,
)
from geas35.models.transition.selector import TransitionModelSelectionResult


class MeanEstimator:
    def fit(self, x, y):
        self.mean_ = float(pd.Series(y).mean())
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


def _dataset():
    x = pd.DataFrame(
        {
            "obs_indoor_temp_c": [20.0, 21.0, 22.0],
            "vent_pct": [0.0, 0.2, 0.4],
        }
    )
    y = pd.DataFrame(
        {
            "obs_indoor_temp_c": [21.0, 22.0, 23.0],
            "obs_indoor_humidity_pct": [72.0, 73.0, 74.0],
        }
    )
    return TransitionDataset(
        x=x,
        y=y,
        input_columns=("obs_indoor_temp_c", "vent_pct"),
        target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        observation_columns=("obs_indoor_temp_c",),
        action_columns=("vent_pct",),
    )


def _report():
    return TransitionOneStepEvaluationReport(
        row_count=3,
        target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        target_metrics={
            "obs_indoor_temp_c": {
                "r2": 1.0,
                "mae": 0.0,
                "rmse": 0.0,
                "q90": 0.0,
                "cvar90": 0.0,
            },
            "obs_indoor_humidity_pct": {
                "r2": 1.0,
                "mae": 0.0,
                "rmse": 0.0,
                "q90": 0.0,
                "cvar90": 0.0,
            },
        },
        aggregate_metrics={"mean_rmse": 0.0},
        target_group_metrics={},
    )


class TransitionArtifactsTest(unittest.TestCase):
    def test_save_and_load_transition_model_artifact(self):
        dataset = _dataset()
        model = IndependentTargetTransitionModel(MeanEstimator).fit(dataset)

        with tempfile.TemporaryDirectory() as tmp:
            artifact = save_transition_model_artifact(
                model,
                tmp,
                "tomato",
                model_name="mean_baseline",
                feature_schema={
                    "input_columns": dataset.input_columns,
                    "target_columns": dataset.target_columns,
                },
                one_step_report=_report(),
                rollout_metrics={"15min": {"trajectory_rmse": 0.1}},
                training_summary={"train_rows": len(dataset.x)},
                metadata={"step": 8},
            )

            self.assertTrue(artifact.model_path.exists())
            self.assertTrue(artifact.manifest_path.exists())
            self.assertTrue(artifact.feature_schema_path.exists())
            self.assertTrue(artifact.one_step_metrics_path.exists())
            self.assertTrue(artifact.rollout_metrics_path.exists())
            self.assertTrue(artifact.training_summary_path.exists())

            manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["crop"], "tomato")
            self.assertEqual(manifest["model_name"], "mean_baseline")
            self.assertEqual(manifest["model_filename"], "model.pkl")
            self.assertEqual(manifest["metadata"]["step"], 8)

            loaded = load_transition_model_artifact(
                tmp,
                "tomato",
                model_name="mean_baseline",
            )
            prediction = loaded.predict(dataset.x.iloc[:1])
            self.assertEqual(
                tuple(prediction.next_observation.columns),
                dataset.target_columns,
            )

    def test_selected_transition_model_manifest_round_trip(self):
        selection_result = TransitionModelSelectionResult(
            selected_model_name="linear_regression",
            selected_score=0.25,
            strategy_name="mean_rmse",
            higher_is_better=False,
            candidate_scores=(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = save_selected_transition_model(
                Path(tmp),
                "tomato",
                selection_result,
                metadata={"candidate_artifacts": {"linear_regression": "artifact-dir"}},
            )
            loaded = load_selected_transition_model_manifest(tmp, "tomato")

            self.assertEqual(path.name, "selected_transition_model.json")
            self.assertEqual(loaded["selected_model_name"], "linear_regression")
            self.assertEqual(loaded["selected_score"], 0.25)
            self.assertEqual(
                loaded["metadata"]["candidate_artifacts"]["linear_regression"],
                "artifact-dir",
            )


if __name__ == "__main__":
    unittest.main()
