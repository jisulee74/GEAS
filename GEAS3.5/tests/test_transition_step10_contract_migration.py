from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.cli import run_from_config
from geas35.models.transition import (
    ARTIFACT_INTEGRITY_JSON,
    BEST_CONFIG_FILENAME,
    CANDIDATE_METRICS_CSV,
    CANDIDATE_METRICS_JSON,
    CANDIDATE_RESOURCE_METRICS_CSV,
    CANDIDATE_RESOURCE_METRICS_JSON,
    CANDIDATE_ROLLOUT_METRICS_CSV,
    CANDIDATE_ROLLOUT_METRICS_JSON,
    CANDIDATE_TEST_METRICS_CSV,
    CANDIDATE_TEST_METRICS_JSON,
    FEATURE_SCHEMA_FILENAME,
    HPO_RESULTS_FILENAME,
    LinearRegressionTransitionModel,
    ONE_STEP_METRICS_FILENAME,
    PersistenceTransitionModel,
    RESOURCE_METRICS_FILENAME,
    ROLLOUT_METRICS_FILENAME,
    TEST_METRICS_FILENAME,
    TRAINING_SUMMARY_FILENAME,
    TRANSITION_MANIFEST_FILENAME,
    TRANSITION_MODEL_FILENAME,
    TRANSITION_SUMMARY_MD,
    load_transition_candidate_model,
    predict_next_observation_reward,
)
from geas35.rl import MDP_V1_ACTION_COLUMNS
from synthetic_experiment_fixtures import (
    SYNTHETIC_CROP,
    transition_rl_ready_frame,
)


OFFICIAL_CANDIDATES = (
    "persistence",
    "linear_regression",
    "linear_svr",
    "knn",
    "extra_trees",
    "lightgbm",
    "xgboost",
    "mlp",
)


def test_step10_synthetic_e2e_covers_v14_transition_contract(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "transition_rl_dataset" / SYNTHETIC_CROP
    output_dir = tmp_path / "transition_artifacts"
    dataset_dir.mkdir(parents=True)
    split_paths = {}
    for split, offset in (("train", 0.0), ("validation", 0.25), ("test", 0.5)):
        path = dataset_dir / f"{split}.parquet"
        transition_rl_ready_frame(offset=offset).to_parquet(path, index=False)
        split_paths[split] = path

    config_path = tmp_path / "transition_step10.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  crop: {SYNTHETIC_CROP}",
                f"  train: '{split_paths['train'].as_posix()}'",
                f"  validation: '{split_paths['validation'].as_posix()}'",
                f"  test: '{split_paths['test'].as_posix()}'",
                "experiment:",
                f"  output_dir: '{output_dir.as_posix()}'",
                "  random_seed: 11",
                "models:",
                "  candidates:",
                *[
                    line
                    for candidate_name in OFFICIAL_CANDIDATES
                    for line in (
                        f"    - name: {candidate_name}",
                        "      params: {}",
                    )
                ],
                "hpo:",
                "  enabled: true",
                "  budget: 1",
                "  random_seed: 42",
                "rollout:",
                "  enabled: true",
                "  horizon_steps: [3, 6, 12]",
                "  step_minutes: 5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = run_from_config(config_path, registry=_lightweight_registry())

    expected_root_files = {
        CANDIDATE_METRICS_CSV,
        CANDIDATE_METRICS_JSON,
        CANDIDATE_ROLLOUT_METRICS_CSV,
        CANDIDATE_ROLLOUT_METRICS_JSON,
        CANDIDATE_RESOURCE_METRICS_CSV,
        CANDIDATE_RESOURCE_METRICS_JSON,
        CANDIDATE_TEST_METRICS_CSV,
        CANDIDATE_TEST_METRICS_JSON,
        TRANSITION_SUMMARY_MD,
        ARTIFACT_INTEGRITY_JSON,
        "config.json",
        "experiment_summary.json",
    }
    assert all((output_dir / name).exists() for name in expected_root_files)

    expected_candidate_files = {
        TRANSITION_MODEL_FILENAME,
        TRANSITION_MANIFEST_FILENAME,
        FEATURE_SCHEMA_FILENAME,
        ONE_STEP_METRICS_FILENAME,
        ROLLOUT_METRICS_FILENAME,
        RESOURCE_METRICS_FILENAME,
        HPO_RESULTS_FILENAME,
        BEST_CONFIG_FILENAME,
        TRAINING_SUMMARY_FILENAME,
        TEST_METRICS_FILENAME,
    }
    for candidate_name in OFFICIAL_CANDIDATES:
        candidate_dir = output_dir / SYNTHETIC_CROP / candidate_name
        assert all((candidate_dir / name).exists() for name in expected_candidate_files)

    summary = json.loads((output_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    validation_ranks = {
        score["model_name"]: score["rank"]
        for score in summary["validation_ranking"]["candidate_scores"]
    }
    assert set(validation_ranks) == set(OFFICIAL_CANDIDATES)
    assert summary["automatic_model_selection"] is False
    assert summary["test_used_for_hpo"] is False
    assert summary["test_used_for_validation_ranking"] is False
    assert summary["test_used_for_selection"] is False

    candidate_payload = json.loads(
        (output_dir / CANDIDATE_METRICS_JSON).read_text(encoding="utf-8")
    )
    test_payload = json.loads(
        (output_dir / CANDIDATE_TEST_METRICS_JSON).read_text(encoding="utf-8")
    )
    assert {row["candidate_name"] for row in candidate_payload["rows"]} == set(
        OFFICIAL_CANDIDATES
    )
    assert {row["candidate_name"] for row in test_payload["rows"]} == set(
        OFFICIAL_CANDIDATES
    )
    assert test_payload["validation_ranking_unchanged"] is True
    assert {
        row["candidate_name"]: row["validation_rank"] for row in test_payload["rows"]
    } == validation_ranks
    assert all(
        row["test_used_for_validation_ranking"] is False
        for row in test_payload["rows"]
    )

    forbidden_paths = (
        output_dir / SYNTHETIC_CROP / "selected_transition_model.json",
        output_dir / SYNTHETIC_CROP / "selected_transition_model_test_metrics.json",
        output_dir / "selection_report.json",
        output_dir / "selected_model",
    )
    assert not any(path.exists() for path in forbidden_paths)
    assert _forbidden_keys(summary).isdisjoint(
        {
            "selected_model",
            "selected_model_name",
            "winner",
            "recommended_model",
            "selection_report",
        }
    )

    loaded = load_transition_candidate_model(
        output_dir,
        SYNTHETIC_CROP,
        model_name="xgboost",
    )
    current_row = _reward_ready_row(transition_rl_ready_frame(offset=0.5).iloc[0])
    action = {column: float(current_row[column]) for column in MDP_V1_ACTION_COLUMNS}
    reward_prediction = predict_next_observation_reward(loaded, current_row, action)
    assert reward_prediction.prediction.target_columns
    assert np.isfinite(reward_prediction.reward)
    assert reward_prediction.reward_terms

    integrity = json.loads(
        (output_dir / ARTIFACT_INTEGRITY_JSON).read_text(encoding="utf-8")
    )
    assert integrity["passed"] is True
    assert result.experiment_result.test_output_paths


def test_step10_transition_readme_uses_candidate_evaluation_language() -> None:
    readme = (
        PROJECT_ROOT / "experiments" / "transition_model_selection" / "README.md"
    ).read_text(encoding="utf-8")
    script = (
        PROJECT_ROOT / "experiments" / "transition_model_selection" / "scripts" / "run.py"
    ).read_text(encoding="utf-8")

    assert "Transition Model Candidate Evaluation" in readme
    assert "Test metrics are computed for every candidate" in readme
    assert "selected_transition_model.json" in readme
    assert "candidate_test_metrics.csv/json" in readme
    assert "Test metrics are not computed in this step" not in readme
    assert "candidate-evaluation experiment" in script


def _lightweight_registry() -> dict[str, Any]:
    registry = {
        name: _linear_factory
        for name in OFFICIAL_CANDIDATES
    }
    registry["persistence"] = lambda **_: PersistenceTransitionModel()
    return registry


def _linear_factory(**_: Any) -> LinearRegressionTransitionModel:
    return LinearRegressionTransitionModel()


def _reward_ready_row(row: Any) -> dict[str, float]:
    out = dict(row)
    out.update(
        {
            "obs_outdoor_temp_c": 12.0,
            "obs_outdoor_humidity_pct": 70.0,
            "obs_outdoor_light": 200.0,
            "obs_outdoor_wind_speed": 1.0,
            "obs_rain_flag": 0.0,
            "obs_is_daytime": 1.0,
            "obs_target_day_temp_c": 24.0,
            "obs_target_night_temp_c": 12.0,
            "obs_current_target_temp_min_c": 23.0,
            "obs_current_target_temp_max_c": 25.0,
            "obs_current_target_temp_c": 24.0,
            "obs_current_vpd_kpa": 0.8,
            "obs_current_dewpoint_c": 14.0,
            "obs_current_condensation_margin_c": 2.0,
        }
    )
    return out


def _forbidden_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        out = set(value)
        for item in value.values():
            out.update(_forbidden_keys(item))
        return out
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out.update(_forbidden_keys(item))
        return out
    return set()
