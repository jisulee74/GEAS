import tempfile
import unittest

import pandas as pd

from geas35.models.transition import (
    BaseTransitionModel,
    MeanRmseStrategy,
    TransitionCandidateSpec,
    TransitionDataset,
    TransitionModelCapabilities,
    TransitionPrediction,
    load_selected_transition_model_manifest,
    train_transition_candidates,
)


class OffsetTransitionModel(BaseTransitionModel):
    model_name = "offset_transition"
    capabilities = TransitionModelCapabilities()

    def __init__(self, offset=0.0):
        self.offset = float(offset)

    def fit(self, dataset, validation_dataset=None):
        self.input_columns_ = dataset.input_columns
        self.target_columns_ = dataset.target_columns
        return self

    def predict(self, dataset_or_frame):
        frame = dataset_or_frame.x if hasattr(dataset_or_frame, "x") else dataset_or_frame
        return TransitionPrediction(
            next_observation=pd.DataFrame(
                {
                    "obs_indoor_temp_c": frame["obs_indoor_temp_c"].astype(float)
                    + 1.0
                    + self.offset,
                    "obs_indoor_humidity_pct": frame[
                        "obs_indoor_humidity_pct"
                    ].astype(float)
                    + 2.0
                    + self.offset,
                },
                index=frame.index,
            ),
            target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        )


def _dataset(rows=8):
    x_rows = []
    y_rows = []
    for step in range(rows):
        temp = 20.0 + step
        humidity = 70.0 + 2.0 * step
        x_rows.append(
            {
                "obs_indoor_temp_c": temp,
                "obs_indoor_humidity_pct": humidity,
                "obs_prev_vent_pct": max((step - 1) / 10.0, 0.0),
                "vent_pct": step / 10.0,
            }
        )
        y_rows.append(
            {
                "obs_indoor_temp_c": temp + 1.0,
                "obs_indoor_humidity_pct": humidity + 2.0,
            }
        )
    return TransitionDataset(
        x=pd.DataFrame(x_rows),
        y=pd.DataFrame(y_rows),
        input_columns=(
            "obs_indoor_temp_c",
            "obs_indoor_humidity_pct",
            "obs_prev_vent_pct",
            "vent_pct",
        ),
        target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        observation_columns=(
            "obs_indoor_temp_c",
            "obs_indoor_humidity_pct",
            "obs_prev_vent_pct",
        ),
        action_columns=("vent_pct",),
    )


def _rollout_frame(rows=8):
    frame = _dataset(rows).x.copy()
    frame["next_obs_indoor_temp_c"] = frame["obs_indoor_temp_c"] + 1.0
    frame["next_obs_indoor_humidity_pct"] = frame["obs_indoor_humidity_pct"] + 2.0
    return frame


class TransitionTrainerTest(unittest.TestCase):
    def test_train_candidates_writes_artifacts_and_selected_manifest(self):
        train_dataset = _dataset()
        validation_dataset = _dataset()
        candidate_specs = [
            TransitionCandidateSpec(
                "bad_offset",
                lambda: OffsetTransitionModel(offset=3.0),
            ),
            TransitionCandidateSpec(
                "perfect_offset",
                lambda: OffsetTransitionModel(offset=0.0),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = train_transition_candidates(
                train_dataset,
                validation_dataset,
                candidate_specs,
                crop="tomato",
                models_root=tmp,
                selection_strategy=MeanRmseStrategy(),
                rollout_frame=_rollout_frame(),
                rollout_horizon_steps=(3,),
            )

            self.assertEqual(result.crop, "tomato")
            self.assertEqual(result.selection_result.selected_model_name, "perfect_offset")
            self.assertEqual(result.selected_candidate.model_name, "perfect_offset")
            self.assertTrue(result.selected_manifest_path.exists())
            self.assertEqual(
                load_selected_transition_model_manifest(tmp, "tomato")[
                    "selected_model_name"
                ],
                "perfect_offset",
            )

            for candidate in result.candidate_results:
                self.assertTrue(candidate.artifact.model_path.exists())
                self.assertTrue(candidate.artifact.manifest_path.exists())
                self.assertTrue(candidate.artifact.one_step_metrics_path.exists())
                self.assertTrue(candidate.artifact.rollout_metrics_path.exists())
                self.assertTrue(candidate.artifact.training_summary_path.exists())
                self.assertIn("15min", candidate.rollout_metrics)

    def test_train_candidates_requires_at_least_one_candidate(self):
        with self.assertRaises(ValueError):
            train_transition_candidates(
                _dataset(),
                _dataset(),
                [],
                crop="tomato",
                models_root="unused",
            )


if __name__ == "__main__":
    unittest.main()
