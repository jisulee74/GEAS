from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.cli import run_from_config
from synthetic_experiment_fixtures import write_transition_smoke_fixture


def test_step6_candidate_accuracy_rollout_and_resource_artifacts(
    tmp_path: Path,
) -> None:
    fixture = write_transition_smoke_fixture(tmp_path)
    fixture.config_path.write_text(
        fixture.config_path.read_text(encoding="utf-8").replace(
            "  enabled: false",
            "  enabled: true",
        ),
        encoding="utf-8",
    )
    result = run_from_config(fixture.config_path)

    artifact_dir = fixture.output_dir / fixture.crop / "linear_regression"
    one_step = json.loads((artifact_dir / "one_step_metrics.json").read_text(encoding="utf-8"))
    rollout = json.loads((artifact_dir / "rollout_metrics.json").read_text(encoding="utf-8"))
    resource = json.loads((artifact_dir / "resource_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads(Path(result.experiment_summary_path).read_text(encoding="utf-8"))

    target_metrics = one_step["target_metrics"]["obs_indoor_temp_c"]
    aggregate_metrics = one_step["aggregate_metrics"]
    assert {"r2", "mae", "rmse", "q90", "cvar90", "nrmse"}.issubset(
        target_metrics
    )
    assert "mean_nrmse" in aggregate_metrics
    assert "normalized_mean_rmse" in aggregate_metrics

    assert {"15min", "30min", "60min"}.issubset(rollout)
    for horizon_metrics in rollout.values():
        assert "trajectory_rmse" in horizon_metrics
        assert "trajectory_nrmse" in horizon_metrics
        assert "final_step_rmse" in horizon_metrics
        assert "drift_slope_mae" in horizon_metrics
        assert "physical_violation_rate" in horizon_metrics
        assert "nan_inf_count" in horizon_metrics

    assert resource["stage"] == "transition_candidate_resource_benchmark"
    assert resource["benchmark_status"] == "completed"
    assert resource["protocol"]["inference_benchmark"] == "subprocess_cold_load_predict"
    assert resource["protocol"]["resource_metrics_used_in_validation_score"] is False
    assert resource["training_time_seconds"] is not None
    assert resource["hpo_total_time_seconds"] is not None
    assert resource["inference_latency_median_ms"] is not None
    assert resource["inference_latency_p95_ms"] is not None
    assert resource["peak_cpu_rss_bytes"] is not None
    assert resource["peak_gpu_allocated_bytes"] is None
    assert resource["serialized_model_size_bytes"] > 0
    assert resource["artifact_dir_size_bytes"] > 0
    assert "library_versions" in resource
    assert "device" in resource

    assert summary["hpo"]["test_used_for_hpo"] is False
