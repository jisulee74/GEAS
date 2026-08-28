from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.cli import run_from_config
from geas35.models.transition import (
    ARTIFACT_INTEGRITY_JSON,
    CANDIDATE_METRICS_CSV,
    CANDIDATE_METRICS_JSON,
    CANDIDATE_RESOURCE_METRICS_CSV,
    CANDIDATE_RESOURCE_METRICS_JSON,
    CANDIDATE_ROLLOUT_METRICS_CSV,
    CANDIDATE_ROLLOUT_METRICS_JSON,
    NO_AUTO_SELECTION_NOTICE,
    TRANSITION_SUMMARY_MD,
    validate_no_auto_selection_integrity,
)
from synthetic_experiment_fixtures import (
    SYNTHETIC_CROP,
    transition_rl_ready_frame,
)


def test_step8_writes_candidate_comparison_outputs(
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

    config_path = tmp_path / "transition_step8.yaml"
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

    result = run_from_config(config_path)

    expected_files = {
        CANDIDATE_METRICS_CSV,
        CANDIDATE_METRICS_JSON,
        CANDIDATE_ROLLOUT_METRICS_CSV,
        CANDIDATE_ROLLOUT_METRICS_JSON,
        CANDIDATE_RESOURCE_METRICS_CSV,
        CANDIDATE_RESOURCE_METRICS_JSON,
        TRANSITION_SUMMARY_MD,
        ARTIFACT_INTEGRITY_JSON,
    }
    output_paths = result.experiment_result.comparison_output_paths
    assert set(output_paths) == expected_files
    assert all((output_dir / name).exists() for name in expected_files)
    assert "candidate_test_metrics.csv" not in output_paths
    assert "candidate_test_metrics.json" not in output_paths

    candidate_payload = json.loads(
        (output_dir / CANDIDATE_METRICS_JSON).read_text(encoding="utf-8")
    )
    candidate_rows = candidate_payload["rows"]
    assert {row["candidate_name"] for row in candidate_rows} == {
        "persistence",
        "linear_regression",
    }
    assert all(
        "validation_rollout_weighted_score_rank" in row for row in candidate_rows
    )
    assert all(
        "persistence_improvement_absolute" in row for row in candidate_rows
    )
    assert candidate_payload["automatic_model_selection"] is False
    assert candidate_payload["test_used_for_validation_ranking"] is False

    rollout_payload = json.loads(
        (output_dir / CANDIDATE_ROLLOUT_METRICS_JSON).read_text(encoding="utf-8")
    )
    assert all("60min_trajectory_rmse_rank" in row for row in rollout_payload["rows"])
    assert all("nan_inf_status" in row for row in rollout_payload["rows"])

    resource_payload = json.loads(
        (output_dir / CANDIDATE_RESOURCE_METRICS_JSON).read_text(encoding="utf-8")
    )
    assert all(
        row["resource_metrics_used_in_validation_score"] is False
        for row in resource_payload["rows"]
    )
    assert all("inference_latency_p95_ms_rank" in row for row in resource_payload["rows"])

    summary_md = (output_dir / TRANSITION_SUMMARY_MD).read_text(encoding="utf-8")
    assert NO_AUTO_SELECTION_NOTICE in summary_md
    assert "Validation Ranking" in summary_md
    assert "Deployment Trade-offs" in summary_md

    integrity = json.loads(
        (output_dir / ARTIFACT_INTEGRITY_JSON).read_text(encoding="utf-8")
    )
    assert integrity["passed"] is True
    assert integrity["automatic_model_selection"] is False
    assert integrity["errors"] == []

    assert not (output_dir / "all_candidate_test_report.json").exists()


def test_step8_integrity_flags_forbidden_auto_selection_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "transition_artifacts"
    output_dir.mkdir()
    (output_dir / "selected_transition_model.json").write_text(
        "{}",
        encoding="utf-8",
    )

    integrity = validate_no_auto_selection_integrity(output_dir)

    assert integrity["passed"] is False
    assert integrity["automatic_model_selection"] is False
    assert integrity["errors"]
