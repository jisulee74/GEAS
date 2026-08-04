"""Candidate comparison outputs for transition model experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd

from geas35.io_utils import write_json
from geas35.models.transition.artifacts import TEST_METRICS_FILENAME
from geas35.models.transition.base import TransitionDataset
from geas35.models.transition.evaluator import evaluate_transition_model_one_step
from geas35.models.transition.hyperparameters import (
    enrich_rollout_metrics_with_normalized_errors,
    transition_target_normalization_scales,
)
from geas35.models.transition.rollout import LoggedActionProvider, TransitionRolloutSimulator
from geas35.models.transition.trainer import TransitionTrainingResult


CANDIDATE_METRICS_CSV = "candidate_metrics.csv"
CANDIDATE_METRICS_JSON = "candidate_metrics.json"
CANDIDATE_ROLLOUT_METRICS_CSV = "candidate_rollout_metrics.csv"
CANDIDATE_ROLLOUT_METRICS_JSON = "candidate_rollout_metrics.json"
CANDIDATE_RESOURCE_METRICS_CSV = "candidate_resource_metrics.csv"
CANDIDATE_RESOURCE_METRICS_JSON = "candidate_resource_metrics.json"
CANDIDATE_TEST_METRICS_CSV = "candidate_test_metrics.csv"
CANDIDATE_TEST_METRICS_JSON = "candidate_test_metrics.json"
TRANSITION_SUMMARY_MD = "transition_summary.md"
ARTIFACT_INTEGRITY_JSON = "artifact_integrity.json"
NO_AUTO_SELECTION_NOTICE = (
    "This framework intentionally does not perform automatic model selection.\n"
    "Final transition-model selection is left to the researcher after considering "
    "rollout accuracy, resource usage, deployment constraints, and practical trade-offs."
)
FORBIDDEN_SELECTION_KEYS = {
    "selected_model",
    "selected_model_name",
    "winner",
    "recommended_model",
    "best_transition_model",
    "selection_report",
}
FORBIDDEN_SELECTION_FILENAMES = {
    "selection_report.json",
    "selected_transition_model.json",
    "selected_manifest",
    "selected_model",
    "selected_model_test_metrics.json",
    "selected_transition_model_test_metrics.json",
}


def write_transition_candidate_comparison_outputs(
    output_root: Path | str,
    training_result: TransitionTrainingResult,
    *,
    config_payload: Mapping[str, Any],
    dataset_payload: Mapping[str, Any],
) -> dict[str, str]:
    """Write Step 8 candidate comparison artifacts."""

    root = Path(output_root)
    candidate_rows = _candidate_metric_rows(training_result)
    rollout_rows = _candidate_rollout_rows(training_result)
    resource_rows = _candidate_resource_rows(training_result)

    candidate_rows = _with_ranks(
        candidate_rows,
        rank_specs={
            "validation_rollout_weighted_score": False,
            "validation_one_step_mae": False,
            "validation_one_step_rmse": False,
            "validation_one_step_nrmse": False,
            "validation_one_step_q90": False,
            "validation_one_step_cvar90": False,
            "validation_one_step_r2": True,
        },
    )
    rollout_rows = _with_ranks(
        rollout_rows,
        rank_specs={
            "15min_trajectory_rmse": False,
            "30min_trajectory_rmse": False,
            "60min_trajectory_rmse": False,
            "15min_final_step_rmse": False,
            "30min_final_step_rmse": False,
            "60min_final_step_rmse": False,
            "drift_slope_nmae": False,
        },
    )
    resource_rows = _with_ranks(
        resource_rows,
        rank_specs={
            "training_time_seconds": False,
            "hpo_total_time_seconds": False,
            "inference_latency_median_ms": False,
            "inference_latency_p95_ms": False,
            "peak_cpu_rss_bytes": False,
            "serialized_model_size_bytes": False,
            "artifact_dir_size_bytes": False,
        },
    )

    paths = {
        CANDIDATE_METRICS_CSV: root / CANDIDATE_METRICS_CSV,
        CANDIDATE_METRICS_JSON: root / CANDIDATE_METRICS_JSON,
        CANDIDATE_ROLLOUT_METRICS_CSV: root / CANDIDATE_ROLLOUT_METRICS_CSV,
        CANDIDATE_ROLLOUT_METRICS_JSON: root / CANDIDATE_ROLLOUT_METRICS_JSON,
        CANDIDATE_RESOURCE_METRICS_CSV: root / CANDIDATE_RESOURCE_METRICS_CSV,
        CANDIDATE_RESOURCE_METRICS_JSON: root / CANDIDATE_RESOURCE_METRICS_JSON,
        TRANSITION_SUMMARY_MD: root / TRANSITION_SUMMARY_MD,
        ARTIFACT_INTEGRITY_JSON: root / ARTIFACT_INTEGRITY_JSON,
    }
    _write_table(paths[CANDIDATE_METRICS_CSV], candidate_rows)
    write_json(
        paths[CANDIDATE_METRICS_JSON],
        _table_payload("transition_candidate_metrics", candidate_rows),
        convert=True,
    )
    _write_table(paths[CANDIDATE_ROLLOUT_METRICS_CSV], rollout_rows)
    write_json(
        paths[CANDIDATE_ROLLOUT_METRICS_JSON],
        _table_payload("transition_candidate_rollout_metrics", rollout_rows),
        convert=True,
    )
    _write_table(paths[CANDIDATE_RESOURCE_METRICS_CSV], resource_rows)
    write_json(
        paths[CANDIDATE_RESOURCE_METRICS_JSON],
        _table_payload("transition_candidate_resource_metrics", resource_rows),
        convert=True,
    )
    paths[TRANSITION_SUMMARY_MD].write_text(
        _markdown_summary(
            config_payload=config_payload,
            dataset_payload=dataset_payload,
            candidate_rows=candidate_rows,
            rollout_rows=rollout_rows,
            resource_rows=resource_rows,
        ),
        encoding="utf-8",
    )
    integrity = validate_no_auto_selection_integrity(
        root,
        payloads=[
            config_payload,
            dataset_payload,
            _table_payload("transition_candidate_metrics", candidate_rows),
            _table_payload("transition_candidate_rollout_metrics", rollout_rows),
            _table_payload("transition_candidate_resource_metrics", resource_rows),
        ],
    )
    write_json(paths[ARTIFACT_INTEGRITY_JSON], integrity, convert=True)
    if not integrity["passed"]:
        raise ValueError("Transition no-auto-selection integrity check failed.")
    return {name: str(path) for name, path in paths.items()}


def write_transition_candidate_test_outputs(
    output_root: Path | str,
    training_result: TransitionTrainingResult,
    *,
    test_dataset: TransitionDataset,
    test_frame: pd.DataFrame,
    rollout_horizon_steps: Sequence[int],
    rollout_step_minutes: int,
) -> dict[str, str]:
    """Write Step 9 all-candidate test evaluation artifacts."""

    root = Path(output_root)
    rows = _candidate_test_rows(
        training_result,
        test_dataset=test_dataset,
        test_frame=test_frame,
        rollout_horizon_steps=rollout_horizon_steps,
        rollout_step_minutes=rollout_step_minutes,
    )
    rows = _with_ranks(
        rows,
        rank_specs={
            "test_one_step_mae": False,
            "test_one_step_rmse": False,
            "test_one_step_nrmse": False,
            "test_one_step_q90": False,
            "test_one_step_cvar90": False,
            "test_one_step_r2": True,
            "test_15min_trajectory_rmse": False,
            "test_30min_trajectory_rmse": False,
            "test_60min_trajectory_rmse": False,
        },
        rank_suffix="_test_descriptive_rank",
    )
    csv_path = root / CANDIDATE_TEST_METRICS_CSV
    json_path = root / CANDIDATE_TEST_METRICS_JSON
    _write_table(csv_path, rows)
    payload = _table_payload("transition_candidate_test_metrics", rows)
    payload["validation_ranking_unchanged"] = True
    write_json(json_path, payload, convert=True)
    _append_test_section_to_summary(root / TRANSITION_SUMMARY_MD, rows)
    integrity = validate_no_auto_selection_integrity(root, payloads=[payload])
    write_json(root / ARTIFACT_INTEGRITY_JSON, integrity, convert=True)
    if not integrity["passed"]:
        raise ValueError("Transition no-auto-selection integrity check failed.")
    return {
        CANDIDATE_TEST_METRICS_CSV: str(csv_path),
        CANDIDATE_TEST_METRICS_JSON: str(json_path),
        **{
            f"{result.model_name}/{TEST_METRICS_FILENAME}": str(
                result.artifact.test_metrics_path
            )
            for result in training_result.candidate_results
        },
    }


def validate_no_auto_selection_integrity(
    output_root: Path | str,
    *,
    payloads: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return integrity result and flag selected/winner/recommended artifacts."""

    root = Path(output_root)
    errors = []
    for forbidden in FORBIDDEN_SELECTION_FILENAMES:
        path = root / forbidden
        if path.exists():
            errors.append(f"Forbidden selected artifact exists: {path}")
    for path in root.rglob("*"):
        if path.name in FORBIDDEN_SELECTION_FILENAMES:
            errors.append(f"Forbidden selected artifact exists: {path}")
    for index, payload in enumerate(payloads):
        forbidden_keys = sorted(_forbidden_keys(payload))
        for key in forbidden_keys:
            errors.append(f"Forbidden selected field in payload {index}: {key}")
    return {
        "stage": "transition_artifact_integrity",
        "passed": not errors,
        "automatic_model_selection": False,
        "checked_forbidden_keys": sorted(FORBIDDEN_SELECTION_KEYS),
        "checked_forbidden_artifacts": sorted(FORBIDDEN_SELECTION_FILENAMES),
        "errors": errors,
    }


def _candidate_test_rows(
    training_result: TransitionTrainingResult,
    *,
    test_dataset: TransitionDataset,
    test_frame: pd.DataFrame,
    rollout_horizon_steps: Sequence[int],
    rollout_step_minutes: int,
) -> list[dict[str, Any]]:
    normalization_scales = transition_target_normalization_scales(
        test_dataset.y,
        test_dataset.target_columns,
    )
    validation_rank_by_model = {
        score.model_name: int(score.rank)
        for score in training_result.validation_ranking.candidate_scores
    }
    rows = []
    for result in training_result.candidate_results:
        one_step_report = evaluate_transition_model_one_step(
            result.model,
            test_dataset,
            normalization_scales=normalization_scales,
        )
        rollout_metrics = _evaluate_test_rollouts(
            result.model,
            test_frame=test_frame,
            rollout_horizon_steps=rollout_horizon_steps,
            rollout_step_minutes=rollout_step_minutes,
            normalization_scales=normalization_scales,
        )
        aggregate = one_step_report.aggregate_metrics
        row = {
            "candidate_name": result.model_name,
            "validation_rank": validation_rank_by_model.get(result.model_name),
            "test_one_step_mae": aggregate.get("mean_mae"),
            "test_one_step_rmse": aggregate.get("mean_rmse"),
            "test_one_step_r2": aggregate.get("mean_r2"),
            "test_one_step_q90": aggregate.get("mean_q90"),
            "test_one_step_cvar90": aggregate.get("mean_cvar90"),
            "test_one_step_nrmse": aggregate.get("mean_nrmse"),
            "test_used_for_hpo": False,
            "test_used_for_validation_ranking": False,
            "test_used_for_selection": False,
        }
        for horizon in ("15min", "30min", "60min"):
            metrics = rollout_metrics.get(horizon, {})
            row[f"test_{horizon}_trajectory_rmse"] = metrics.get("trajectory_rmse")
            row[f"test_{horizon}_trajectory_mae"] = metrics.get("trajectory_mae")
            row[f"test_{horizon}_trajectory_nrmse"] = metrics.get("trajectory_nrmse")
            row[f"test_{horizon}_final_step_rmse"] = metrics.get("final_step_rmse")
            row[f"test_{horizon}_physical_violation_rate"] = metrics.get(
                "physical_violation_rate"
            )
            row[f"test_{horizon}_nan_inf_count"] = metrics.get("nan_inf_count")
        write_json(
            result.artifact.test_metrics_path,
            {
                "stage": "transition_candidate_test_evaluation",
                "model_name": result.model_name,
                "validation_rank": validation_rank_by_model.get(result.model_name),
                "test_one_step_metrics": one_step_report.to_artifact(),
                "test_rollout_metrics": rollout_metrics,
                "test_used_for_hpo": False,
                "test_used_for_validation_ranking": False,
                "test_used_for_selection": False,
            },
            convert=True,
        )
        rows.append(row)
    return rows


def _evaluate_test_rollouts(
    model: Any,
    *,
    test_frame: pd.DataFrame,
    rollout_horizon_steps: Sequence[int],
    rollout_step_minutes: int,
    normalization_scales: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for horizon_steps in rollout_horizon_steps:
        if horizon_steps <= 0:
            raise ValueError("rollout_horizon_steps must contain positive values.")
        if len(test_frame.index) < int(horizon_steps):
            continue
        result = TransitionRolloutSimulator(
            model,
            action_provider=LoggedActionProvider(),
        ).simulate(
            test_frame,
            start_index=0,
            horizon_steps=int(horizon_steps),
        )
        horizon_key = f"{int(horizon_steps) * int(rollout_step_minutes)}min"
        metrics[horizon_key] = enrich_rollout_metrics_with_normalized_errors(
            dict(result.metrics),
            result.errors,
            normalization_scales,
        )
    return metrics


def _candidate_metric_rows(training_result: TransitionTrainingResult) -> list[dict[str, Any]]:
    baseline_score = None
    for result in training_result.candidate_results:
        if result.model_name == "persistence":
            baseline_score = _metadata_float(
                result.ranking_candidate.metadata,
                "validation_rollout_weighted_score",
            )
            break
    rows = []
    for result in training_result.candidate_results:
        aggregate = result.one_step_report.aggregate_metrics
        score = _metadata_float(
            result.ranking_candidate.metadata,
            "validation_rollout_weighted_score",
        )
        improvement_abs = (
            None
            if baseline_score is None or score is None
            else baseline_score - score
        )
        improvement_rel = (
            None
            if baseline_score in (None, 0.0) or improvement_abs is None
            else improvement_abs / baseline_score
        )
        rows.append(
            {
                "candidate_name": result.model_name,
                "model_family": _model_family(result.model_name),
                "candidate_status": "completed",
                "hpo_status": result.hpo_result.status,
                "validation_one_step_mae": aggregate.get("mean_mae"),
                "validation_one_step_rmse": aggregate.get("mean_rmse"),
                "validation_one_step_r2": aggregate.get("mean_r2"),
                "validation_one_step_q90": aggregate.get("mean_q90"),
                "validation_one_step_cvar90": aggregate.get("mean_cvar90"),
                "validation_one_step_nrmse": aggregate.get("mean_nrmse"),
                "validation_rollout_weighted_score": score,
                "persistence_improvement_absolute": improvement_abs,
                "persistence_improvement_relative": improvement_rel,
                "test_used_for_hpo": False,
                "test_used_for_validation_ranking": False,
                "test_used_for_selection": False,
            }
        )
    return rows


def _candidate_rollout_rows(training_result: TransitionTrainingResult) -> list[dict[str, Any]]:
    rows = []
    for result in training_result.candidate_results:
        row: dict[str, Any] = {"candidate_name": result.model_name}
        drift_values = []
        violation_values = []
        nan_inf_values = []
        for horizon in ("15min", "30min", "60min"):
            metrics = result.rollout_metrics.get(horizon, {})
            row[f"{horizon}_final_step_rmse"] = metrics.get("final_step_rmse")
            row[f"{horizon}_final_step_mae"] = metrics.get("final_step_mae")
            row[f"{horizon}_trajectory_rmse"] = metrics.get("trajectory_rmse")
            row[f"{horizon}_trajectory_mae"] = metrics.get("trajectory_mae")
            row[f"{horizon}_trajectory_nrmse"] = metrics.get("trajectory_nrmse")
            row[f"{horizon}_physical_violation_rate"] = metrics.get(
                "physical_violation_rate"
            )
            row[f"{horizon}_nan_inf_count"] = metrics.get("nan_inf_count")
            if metrics.get("drift_slope_nmae") is not None:
                drift_values.append(metrics.get("drift_slope_nmae"))
            if metrics.get("physical_violation_rate") is not None:
                violation_values.append(metrics.get("physical_violation_rate"))
            if metrics.get("nan_inf_count") is not None:
                nan_inf_values.append(metrics.get("nan_inf_count"))
        row["drift_slope_nmae"] = _finite_mean(drift_values)
        row["physical_violation_rate"] = _finite_mean(violation_values)
        row["nan_inf_count"] = _finite_sum(nan_inf_values)
        row["nan_inf_status"] = "ok" if (row["nan_inf_count"] in (None, 0.0)) else "invalid"
        rows.append(row)
    return rows


def _candidate_resource_rows(training_result: TransitionTrainingResult) -> list[dict[str, Any]]:
    rows = []
    for result in training_result.candidate_results:
        metrics = result.resource_metrics
        rows.append(
            {
                "candidate_name": result.model_name,
                "benchmark_status": metrics.get("benchmark_status"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "hpo_total_time_seconds": metrics.get("hpo_total_time_seconds"),
                "inference_latency_median_ms": metrics.get(
                    "inference_latency_median_ms"
                ),
                "inference_latency_p95_ms": metrics.get("inference_latency_p95_ms"),
                "peak_cpu_rss_bytes": metrics.get("peak_cpu_rss_bytes"),
                "peak_gpu_allocated_bytes": metrics.get("peak_gpu_allocated_bytes"),
                "serialized_model_size_bytes": metrics.get(
                    "serialized_model_size_bytes"
                ),
                "artifact_dir_size_bytes": metrics.get("artifact_dir_size_bytes"),
                "resource_metrics_used_in_validation_score": False,
                "device": json.dumps(metrics.get("device", {}), ensure_ascii=False),
                "library_versions": json.dumps(
                    metrics.get("library_versions", {}),
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def _with_ranks(
    rows: list[dict[str, Any]],
    *,
    rank_specs: Mapping[str, bool],
    rank_suffix: str = "_rank",
) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    for metric, higher_is_better in rank_specs.items():
        values = [_numeric(row.get(metric)) for row in out]
        finite = sorted(
            ((value, index) for index, value in enumerate(values) if np.isfinite(value)),
            reverse=higher_is_better,
        )
        ranks = [None] * len(out)
        for rank, (_, index) in enumerate(finite, start=1):
            ranks[index] = rank
        rank_key = f"{metric}{rank_suffix}"
        for row, rank in zip(out, ranks):
            row[rank_key] = rank
    return out


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _table_payload(stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": stage,
        "automatic_model_selection": False,
        "test_used_for_hpo": False,
        "test_used_for_validation_ranking": False,
        "test_used_for_selection": False,
        "rows": rows,
    }


def _markdown_summary(
    *,
    config_payload: Mapping[str, Any],
    dataset_payload: Mapping[str, Any],
    candidate_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
) -> str:
    ordered = sorted(
        candidate_rows,
        key=lambda row: (
            row.get("validation_rollout_weighted_score_rank") is None,
            row.get("validation_rollout_weighted_score_rank") or 10**9,
        ),
    )
    lines = [
        "# GEAS Transition Candidate Comparison",
        "",
        "## Experiment Config",
        "",
        f"- crop: {config_payload.get('crop')}",
        f"- output_root: {config_payload.get('output_root')}",
        f"- dataset: {dataset_payload}",
        "",
        "## Validation Ranking",
        "",
        "| Rank | Candidate | Validation Rollout Weighted Score | One-step RMSE | One-step NRMSE | Persistence Improvement |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| {rank} | {name} | {score} | {rmse} | {nrmse} | {imp} |".format(
                rank=row.get("validation_rollout_weighted_score_rank"),
                name=row.get("candidate_name"),
                score=_fmt(row.get("validation_rollout_weighted_score")),
                rmse=_fmt(row.get("validation_one_step_rmse")),
                nrmse=_fmt(row.get("validation_one_step_nrmse")),
                imp=_fmt(row.get("persistence_improvement_relative")),
            )
        )
    lines.extend(
        [
            "",
            "## Rollout Metrics",
            "",
            "| Candidate | 15min RMSE | 30min RMSE | 60min RMSE | Drift | Physical Violation Rate | NaN/Inf |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rollout_rows:
        lines.append(
            "| {name} | {r15} | {r30} | {r60} | {drift} | {viol} | {nan} |".format(
                name=row.get("candidate_name"),
                r15=_fmt(row.get("15min_trajectory_rmse")),
                r30=_fmt(row.get("30min_trajectory_rmse")),
                r60=_fmt(row.get("60min_trajectory_rmse")),
                drift=_fmt(row.get("drift_slope_nmae")),
                viol=_fmt(row.get("physical_violation_rate")),
                nan=_fmt(row.get("nan_inf_count")),
            )
        )
    lines.extend(
        [
            "",
            "## Resource Usage",
            "",
            "| Candidate | Train s | HPO s | Latency median ms | Latency p95 ms | Model bytes | Artifact bytes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in resource_rows:
        lines.append(
            "| {name} | {train} | {hpo} | {med} | {p95} | {model} | {artifact} |".format(
                name=row.get("candidate_name"),
                train=_fmt(row.get("training_time_seconds")),
                hpo=_fmt(row.get("hpo_total_time_seconds")),
                med=_fmt(row.get("inference_latency_median_ms")),
                p95=_fmt(row.get("inference_latency_p95_ms")),
                model=_fmt(row.get("serialized_model_size_bytes")),
                artifact=_fmt(row.get("artifact_dir_size_bytes")),
            )
        )
    lines.extend(
        [
            "",
            "## Invalid or Failed Candidates",
            "",
            "Candidates with failed resource benchmarks or invalid numeric metrics are retained with status fields rather than dropped.",
            "",
            "## Deployment Trade-offs",
            "",
            "Use the validation, rollout, drift, and resource tables together to judge deployment trade-offs. Resource metrics are reported separately and are not part of the validation score.",
            "",
            NO_AUTO_SELECTION_NOTICE,
            "",
        ]
    )
    return "\n".join(lines)


def _append_test_section_to_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not path.exists():
        return
    lines = [
        "",
        "## Test Evaluation",
        "",
        "| Validation Rank | Candidate | Test RMSE | Test NRMSE | Test R2 | 60min Test Rollout RMSE | Test Descriptive Rank |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("validation_rank") is None,
            row.get("validation_rank") or 10**9,
        ),
    )
    for row in ordered:
        lines.append(
            "| {validation_rank} | {name} | {rmse} | {nrmse} | {r2} | {rollout} | {test_rank} |".format(
                validation_rank=row.get("validation_rank"),
                name=row.get("candidate_name"),
                rmse=_fmt(row.get("test_one_step_rmse")),
                nrmse=_fmt(row.get("test_one_step_nrmse")),
                r2=_fmt(row.get("test_one_step_r2")),
                rollout=_fmt(row.get("test_60min_trajectory_rmse")),
                test_rank=row.get("test_one_step_rmse_test_descriptive_rank"),
            )
        )
    lines.extend(
        [
            "",
            "Test metrics are descriptive only. They were not used for HPO, validation ranking, or automatic selection.",
            "",
            NO_AUTO_SELECTION_NOTICE,
            "",
        ]
    )
    existing = path.read_text(encoding="utf-8").rstrip()
    path.write_text(existing + "\n" + "\n".join(lines), encoding="utf-8")


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_SELECTION_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


def _metadata_float(metadata: Mapping[str, Any], key: str) -> float | None:
    return _none_if_nan(_numeric(metadata.get(key)))


def _numeric(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _none_if_nan(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _finite_mean(values: Sequence[Any]) -> float | None:
    numeric = [_numeric(value) for value in values]
    finite = [value for value in numeric if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _finite_sum(values: Sequence[Any]) -> float | None:
    numeric = [_numeric(value) for value in values]
    finite = [value for value in numeric if np.isfinite(value)]
    return float(np.sum(finite)) if finite else None


def _fmt(value: Any) -> str:
    numeric = _numeric(value)
    if np.isfinite(numeric):
        if abs(numeric) >= 1000:
            return f"{numeric:.0f}"
        return f"{numeric:.4f}"
    return ""


def _model_family(model_name: str) -> str:
    if model_name in {"linear_regression", "linear_svr"}:
        return "linear"
    if model_name in {"extra_trees", "lightgbm", "xgboost"}:
        return "tree_ensemble"
    if model_name == "mlp":
        return "neural_network"
    if model_name == "persistence":
        return "baseline"
    return "instance_based" if model_name == "knn" else "unknown"


__all__ = [
    "ARTIFACT_INTEGRITY_JSON",
    "CANDIDATE_METRICS_CSV",
    "CANDIDATE_METRICS_JSON",
    "CANDIDATE_RESOURCE_METRICS_CSV",
    "CANDIDATE_RESOURCE_METRICS_JSON",
    "CANDIDATE_ROLLOUT_METRICS_CSV",
    "CANDIDATE_ROLLOUT_METRICS_JSON",
    "CANDIDATE_TEST_METRICS_CSV",
    "CANDIDATE_TEST_METRICS_JSON",
    "NO_AUTO_SELECTION_NOTICE",
    "TRANSITION_SUMMARY_MD",
    "validate_no_auto_selection_integrity",
    "write_transition_candidate_test_outputs",
    "write_transition_candidate_comparison_outputs",
]
