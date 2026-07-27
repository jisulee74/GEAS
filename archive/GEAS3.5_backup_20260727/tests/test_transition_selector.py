import unittest

from geas35.models.transition import (
    CustomScoreStrategy,
    HumidityPriorityStrategy,
    MeanRmseStrategy,
    RolloutWeightedRmseStrategy,
    TemperaturePriorityStrategy,
    TransitionModelSelectionCandidate,
    WeightedRmseStrategy,
    build_selection_strategy,
    default_selection_strategy_registry,
    select_transition_model,
)
from geas35.models.transition.evaluator import TransitionOneStepEvaluationReport


def _one_step_report(temp_rmse=1.0, humidity_rmse=2.0, mean_rmse=None):
    mean = (temp_rmse + humidity_rmse) / 2.0 if mean_rmse is None else mean_rmse
    return TransitionOneStepEvaluationReport(
        row_count=10,
        target_columns=("obs_indoor_temp_c", "obs_indoor_humidity_pct"),
        target_metrics={
            "obs_indoor_temp_c": {
                "r2": 0.0,
                "mae": temp_rmse,
                "rmse": temp_rmse,
                "q90": temp_rmse,
                "cvar90": temp_rmse,
            },
            "obs_indoor_humidity_pct": {
                "r2": 0.0,
                "mae": humidity_rmse,
                "rmse": humidity_rmse,
                "q90": humidity_rmse,
                "cvar90": humidity_rmse,
            },
        },
        aggregate_metrics={"mean_rmse": mean},
        target_group_metrics={},
    )


class TransitionSelectorTest(unittest.TestCase):
    def test_mean_rmse_strategy_selects_lowest_one_step_mean_rmse(self):
        result = select_transition_model(
            [
                TransitionModelSelectionCandidate(
                    "model_a",
                    one_step_report=_one_step_report(mean_rmse=2.0),
                ),
                TransitionModelSelectionCandidate(
                    "model_b",
                    one_step_report=_one_step_report(mean_rmse=1.0),
                ),
            ],
            strategy=MeanRmseStrategy(),
        )

        self.assertEqual(result.selected_model_name, "model_b")
        self.assertEqual(
            [score.model_name for score in result.candidate_scores],
            ["model_b", "model_a"],
        )
        self.assertFalse(result.higher_is_better)

    def test_rollout_weighted_rmse_strategy_uses_horizon_weights(self):
        result = select_transition_model(
            [
                TransitionModelSelectionCandidate(
                    "short_good_long_bad",
                    rollout_metrics={
                        "15min": {"trajectory_rmse": 0.1},
                        "30min": {"trajectory_rmse": 0.1},
                        "60min": {"trajectory_rmse": 10.0},
                    },
                ),
                TransitionModelSelectionCandidate(
                    "balanced",
                    rollout_metrics={
                        "15min": {"trajectory_rmse": 2.0},
                        "30min": {"trajectory_rmse": 2.0},
                        "60min": {"trajectory_rmse": 2.0},
                    },
                ),
            ],
            strategy=RolloutWeightedRmseStrategy(
                horizon_weights={"15min": 0.2, "30min": 0.3, "60min": 0.5}
            ),
        )

        self.assertEqual(result.selected_model_name, "balanced")
        self.assertAlmostEqual(result.selected_score, 2.0)

    def test_priority_strategies_apply_target_weights(self):
        candidates = [
            TransitionModelSelectionCandidate(
                "temp_good",
                one_step_report=_one_step_report(temp_rmse=1.0, humidity_rmse=5.0),
            ),
            TransitionModelSelectionCandidate(
                "humidity_good",
                one_step_report=_one_step_report(temp_rmse=3.0, humidity_rmse=1.0),
            ),
        ]

        temp_result = select_transition_model(
            candidates,
            strategy=TemperaturePriorityStrategy(),
        )
        humidity_result = select_transition_model(
            candidates,
            strategy=HumidityPriorityStrategy(),
        )

        self.assertEqual(temp_result.selected_model_name, "temp_good")
        self.assertEqual(humidity_result.selected_model_name, "humidity_good")

    def test_weighted_rmse_strategy_accepts_custom_weights(self):
        result = select_transition_model(
            [
                TransitionModelSelectionCandidate(
                    "a",
                    one_step_report=_one_step_report(temp_rmse=1.0, humidity_rmse=5.0),
                ),
                TransitionModelSelectionCandidate(
                    "b",
                    one_step_report=_one_step_report(temp_rmse=3.0, humidity_rmse=1.0),
                ),
            ],
            strategy=WeightedRmseStrategy(
                {"obs_indoor_temp_c": 1.0, "obs_indoor_humidity_pct": 5.0}
            ),
        )

        self.assertEqual(result.selected_model_name, "b")

    def test_custom_score_strategy_can_be_higher_is_better(self):
        result = select_transition_model(
            [
                TransitionModelSelectionCandidate("a", metadata={"score": 0.1}),
                TransitionModelSelectionCandidate("b", metadata={"score": 0.9}),
            ],
            strategy=CustomScoreStrategy(score_key="score", higher_is_better=True),
        )

        self.assertEqual(result.selected_model_name, "b")
        self.assertTrue(result.higher_is_better)

    def test_strategy_registry_builds_known_strategies(self):
        registry = default_selection_strategy_registry()

        self.assertIn("rollout_weighted_rmse", registry)
        self.assertIsInstance(build_selection_strategy("mean_rmse"), MeanRmseStrategy)
        self.assertIsInstance(
            build_selection_strategy("rollout_weighted_rmse"),
            RolloutWeightedRmseStrategy,
        )
        with self.assertRaises(ValueError):
            build_selection_strategy("missing_strategy")

    def test_selection_result_artifact_is_json_safe(self):
        result = select_transition_model(
            [
                TransitionModelSelectionCandidate(
                    "missing_rollout",
                    rollout_metrics={},
                )
            ]
        )

        artifact = result.to_artifact()

        self.assertEqual(artifact["stage"], "transition_model_selection")
        self.assertEqual(artifact["selected_model_name"], "missing_rollout")
        self.assertIsNone(artifact["selected_score"])
        self.assertIsNone(artifact["candidate_scores"][0]["score"])

    def test_select_transition_model_rejects_empty_candidates(self):
        with self.assertRaises(ValueError):
            select_transition_model([])


if __name__ == "__main__":
    unittest.main()
