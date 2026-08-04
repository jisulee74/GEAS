from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.cli import run_from_config
from geas35.models.transition import (
    TransitionOneStepEvaluationReport,
    validation_rollout_weighted_score,
)
from synthetic_experiment_fixtures import (
    SYNTHETIC_CROP,
    transition_rl_ready_frame,
)


def test_validation_rollout_weighted_score_uses_step5_formula() -> None:
    report = TransitionOneStepEvaluationReport(
        row_count=3,
        target_columns=("obs_indoor_temp_c",),
        target_metrics={},
        aggregate_metrics={"normalized_mean_rmse": 1.0},
        target_group_metrics={},
    )
    rollout_metrics = {
        "15min": {"trajectory_nrmse": 2.0, "drift_slope_nmae": 0.1},
        "30min": {"trajectory_nrmse": 3.0, "drift_slope_nmae": 0.2},
        "60min": {"trajectory_nrmse": 4.0, "drift_slope_nmae": 0.3},
    }

    score, components = validation_rollout_weighted_score(report, rollout_metrics)

    assert components == {
        "one_step": 1.0,
        "15min": 2.0,
        "30min": 3.0,
        "60min": 4.0,
        "drift": pytest.approx(0.2),
    }
    assert score == pytest.approx(
        0.10 * 1.0 + 0.10 * 2.0 + 0.25 * 3.0 + 0.45 * 4.0 + 0.10 * 0.2
    )


def test_transition_hpo_writes_trainable_and_persistence_artifacts(
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

    config_path = tmp_path / "transition_step5.yaml"
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
                "    - name: persistence",
                "      params: {}",
                "    - name: linear_regression",
                "      params: {}",
                "hpo:",
                "  enabled: true",
                "  budget: 2",
                "  random_seed: 42",
                "  search_spaces:",
                "    model_specific:",
                "      linear_regression:",
                "        fit_intercept:",
                "          values: [true, false]",
                "rollout:",
                "  enabled: true",
                "  horizon_steps: [3, 6, 12]",
                "  step_minutes: 5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_from_config(config_path)

    linear_dir = output_dir / SYNTHETIC_CROP / "linear_regression"
    persistence_dir = output_dir / SYNTHETIC_CROP / "persistence"
    linear_hpo = json.loads((linear_dir / "hpo_results.json").read_text(encoding="utf-8"))
    linear_best = json.loads((linear_dir / "best_config.json").read_text(encoding="utf-8"))
    persistence_hpo = json.loads(
        (persistence_dir / "hpo_results.json").read_text(encoding="utf-8")
    )
    persistence_best = json.loads(
        (persistence_dir / "best_config.json").read_text(encoding="utf-8")
    )

    assert linear_hpo["status"] == "completed"
    assert linear_hpo["objective"]["name"] == "validation_rollout_weighted_score"
    assert linear_hpo["objective"]["test_used_for_hpo"] is False
    assert len(linear_hpo["trials"]) == 2
    assert linear_hpo["best_config"] == linear_best["best_config"]
    assert linear_best["test_used_for_hpo"] is False
    assert all(
        set(("15min", "30min", "60min")).issubset(
            trial["validation_rollout_metrics"]
        )
        for trial in linear_hpo["trials"]
    )
    assert all(
        "validation_rollout_weighted_score" in trial
        for trial in linear_hpo["trials"]
    )

    assert persistence_hpo["status"] == "not_applicable"
    assert persistence_hpo["hpo_applied"] is False
    assert "persistence_baseline" in persistence_hpo["reason"]
    assert persistence_hpo["final_validation_rollout_weighted_score"] is not None
    assert persistence_best["status"] == "not_applicable"
    assert persistence_best["validation_rollout_weighted_score"] is not None
