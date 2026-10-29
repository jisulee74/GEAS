"""Rollout root-cause controls for transition experiment v1.4.

This module never opens Test data. It reproduces five controlled cucumber MLP
conditions and writes only to the dedicated v1.4 diagnostic root.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from geas35.experiments.transition.config import (
    load_transition_experiment_config, read_configured_transition_frames,
)
from geas35.io_utils import write_json
from geas35.models.transition.causal_state import build_rollout_windows, load_frozen_observation_scaler
from geas35.models.transition.datasets import build_transition_dataset_from_rl_frame
from geas35.models.transition.evaluator import evaluate_transition_model_one_step
from geas35.models.transition.feature_variants import (
    CONDENSATION_PREDICTION_COLUMN, TransitionDiagnosticFeatureSchema,
    diagnostic_feature_contract, observation_columns_for_diagnostic_schema,
    prepare_frame_for_feature_variant, project_frozen_observation_scaler,
)
from geas35.models.transition.hyperparameters import transition_target_normalization_scales
from geas35.models.transition.rollout import TransitionRolloutSimulator
from geas35.models.transition.sklearn_models import MLPTransitionModel
from geas35.models.transition.trainer import _evaluate_rollouts
from geas35.models.transition.base import TransitionTargetFormulation
from geas35.rl import MDP_V1_ACTION_COLUMNS


class RolloutStabilityStatus(str, Enum):
    STABLE = "stable"
    PHYSICAL_VIOLATION = "physical_violation"
    NON_FINITE = "non_finite"
    EXPLOSIVE_ERROR = "explosive_error"


@dataclass(frozen=True)
class ControlledCondition:
    code: str
    schema: TransitionDiagnosticFeatureSchema
    parameter_source: str


CONDITIONS = (
    ControlledCondition("R", TransitionDiagnosticFeatureSchema.LEGACY_V1_1, "v1_1"),
    ControlledCondition("A", TransitionDiagnosticFeatureSchema.DEDUPLICATED_WITHOUT_CONDENSATION_PREDICTION, "v1_1"),
    ControlledCondition("B", TransitionDiagnosticFeatureSchema.DEDUPLICATED_WITH_CONDENSATION_PREDICTION, "v1_1"),
    ControlledCondition("C", TransitionDiagnosticFeatureSchema.DEDUPLICATED_WITHOUT_CONDENSATION_PREDICTION, "v1_2"),
    ControlledCondition("D", TransitionDiagnosticFeatureSchema.DEDUPLICATED_WITH_CONDENSATION_PREDICTION, "v1_2"),
)
DEFAULT_SEEDS = (42, 43, 44)


def run_cucumber_controlled_comparison(
    project_root: str | Path, *, seeds: Sequence[int] = DEFAULT_SEEDS,
    condition_codes: Sequence[str] | None = None,
    output_root: str | Path | None = None, resume: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve() if output_root else root / "experiments/transition_model_selection/artifacts/step12_v1_4_root_cause"
    comparison_root = output / "cucumber_controlled_comparison"
    trace_root = output / "failure_traces"
    comparison_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    config_path = root / "experiments/transition_model_selection/configs/cucumber_v1_2_with_flags.yaml"
    loaded = load_transition_experiment_config(config_path)
    train, validation, test, rollout = read_configured_transition_frames(loaded, include_test=False)
    if test is not None:
        raise AssertionError("Test must remain unopened during v1.4 diagnosis.")
    rollout = validation if rollout is None else rollout
    source_scaler, source_scaler_hash = load_frozen_observation_scaler(loaded.scaler_path)
    params = {
        "v1_1": _best_params(root / "experiments/transition_model_selection/artifacts/step12_v1_1/cucumber/cucumber/mlp/best_config.json"),
        "v1_2": _best_params(root / "experiments/transition_model_selection/artifacts/step12_v1_2/deduplicated_with_quality_flags/cucumber/cucumber/mlp/best_config.json"),
    }
    selected_codes = set(condition_codes or (condition.code for condition in CONDITIONS))
    unknown = selected_codes.difference(condition.code for condition in CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown controlled conditions: {sorted(unknown)}")
    results: list[dict[str, Any]] = []
    for condition in (item for item in CONDITIONS if item.code in selected_codes):
        columns = observation_columns_for_diagnostic_schema(condition.schema)
        contract = diagnostic_feature_contract(condition.schema, action_columns=MDP_V1_ACTION_COLUMNS)
        frames = [_prepare_diagnostic_frame(frame, condition.schema) for frame in (train, validation, rollout)]
        scaler = project_frozen_observation_scaler(source_scaler, columns)
        train_build = build_transition_dataset_from_rl_frame(frames[0], observation_columns=columns)
        validation_build = build_transition_dataset_from_rl_frame(frames[1], observation_columns=columns)
        scales = transition_target_normalization_scales(validation_build.dataset.y, validation_build.dataset.target_columns)
        _require_finite_dataset(train_build.dataset.x, label=f"{condition.code}:train")
        _require_finite_dataset(validation_build.dataset.x, label=f"{condition.code}:validation")
        for seed in seeds:
            destination = comparison_root / condition.code / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            cached = destination / "metrics.json"
            if resume and cached.is_file():
                results.append(json.loads(cached.read_text(encoding="utf-8")))
                continue
            try:
                model_params = _params_for_seed(params[condition.parameter_source], int(seed))
                model = MLPTransitionModel(**model_params).configure_target_formulation(
                    TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA,
                    observation_scaler=scaler,
                )
                model.fit(train_build.dataset, validation_dataset=validation_build.dataset)
                one = evaluate_transition_model_one_step(model, validation_build.dataset, normalization_scales=scales)
                multi = _evaluate_rollouts(
                    model, rollout_frame=frames[2], rollout_horizon_steps=(3, 6, 12),
                    rollout_step_minutes=5, action_provider_factory=None,
                    normalization_scales=scales, observation_scaler=scaler, hpo_mode=False,
                )
                trace_summary = _materialize_rollout_failure_trace(
                    model, frames[2], scaler, scales,
                    destination=trace_root / condition.code / f"seed_{seed}",
                )
                record = {
                    "condition": condition.code, "seed": int(seed), "status": "evaluated",
                    "schema": contract, "parameter_source": condition.parameter_source,
                    "params": model_params, "one_step": one.to_artifact(), "rollout": multi,
                    "failure_trace": trace_summary, "test_opened": False,
                }
                write_json(destination / "metrics.json", record, convert=True)
                results.append(record)
            except Exception as exc:
                failure = {
                    "condition": condition.code, "seed": int(seed), "status": "execution_error",
                    "error_type": type(exc).__name__, "message": str(exc),
                    "schema": contract, "parameter_source": condition.parameter_source,
                    "test_opened": False,
                }
                write_json(destination / "failure.json", failure, convert=True)
                results.append(failure)
    # Merge all completed conditions from disk so a D-only continuation keeps
    # the original R reference and produces a complete five-condition report.
    by_key = {(str(r["condition"]), int(r["seed"])): r for r in results}
    for path in comparison_root.glob("*/seed_*/metrics.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        by_key[(str(record["condition"]), int(record["seed"]))] = record
    results = [by_key[key] for key in sorted(by_key)]
    reference = [
        _nrmse60(r) for r in results if r["condition"] == "R" and r["status"] == "evaluated"
    ]
    reference_median = float(np.median(reference)) if reference else float("nan")
    classified = [_classify(record, reference_median) for record in results]
    for record in classified:
        if record.get("stability") != RolloutStabilityStatus.STABLE.value and record.get("status") == "evaluated":
            _write_failure_traces(
                record, frames=None, output=trace_root,
                comparison_root=comparison_root,
            )
    summary = {
        "stage": "transition_v1_4_cucumber_controlled_comparison",
        "research_flow_stage": "before_crop_specific_transition_selection_and_before_ppo",
        "conditions": [c.__dict__ | {"schema": c.schema.value} for c in CONDITIONS],
        "reference_60min_nrmse_median": reference_median,
        "results": classified,
        "cause_assessment": assess_controlled_cause(classified),
        "source_scaler_sha256": source_scaler_hash,
        "dataset_sha256": {
            "train": _sha256(loaded.train_path), "validation": _sha256(loaded.validation_path),
        },
        "test_opened": False,
    }
    write_json(comparison_root / "controlled_comparison.json", summary, convert=True)
    pd.DataFrame([_flat_result(r) for r in classified]).to_csv(comparison_root / "controlled_comparison.csv", index=False)
    return summary


def audit_melon_inputs(project_root: str | Path, *, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve() if output_root else root / "experiments/transition_model_selection/artifacts/step12_v1_4_root_cause"
    audit_root = output / "melon_input_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for variant in ("with", "without"):
        cfg = root / f"experiments/transition_model_selection/configs/melon_v1_2_{variant}_flags.yaml"
        loaded = load_transition_experiment_config(cfg)
        train, validation, test, rollout = read_configured_transition_frames(loaded, include_test=False)
        for split, frame in (("train", train), ("validation", validation), ("rollout", validation if rollout is None else rollout)):
            numeric = frame.select_dtypes(include=[np.number])
            arr = numeric.to_numpy(dtype=np.float64)
            bad = ~np.isfinite(arr)
            locations = []
            for row, col in np.argwhere(bad)[:100]:
                locations.append({"row_position": int(row), "column": str(numeric.columns[col]), "value": str(arr[row, col])})
            max_by_column = {
                str(c): float(np.nanmax(np.abs(pd.to_numeric(numeric[c], errors="coerce"))))
                for c in numeric.columns if len(numeric.index)
            }
            reports.append({
                "variant": variant, "split": split, "rows": len(frame),
                "non_finite_count": int(bad.sum()), "first_locations": locations,
                "maximum_absolute_value": max(max_by_column.values(), default=0.0),
                "maximum_absolute_value_by_column": max_by_column,
            })
    payload = {"stage": "transition_v1_4_melon_input_audit", "reports": reports, "test_opened": False}
    write_json(audit_root / "input_audit.json", payload, convert=True)
    return payload


def assess_controlled_cause(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stable = {code: _condition_majority(results, code) for code in "RABCD"}
    mixed = {code: _condition_is_mixed(results, code) for code in "RABCD"}
    seed_counts = {code: len([r for r in results if r.get("condition") == code]) for code in "RABCD"}
    pending = [code for code in "RABCD" if mixed[code] and seed_counts[code] < 5]
    if pending:
        cause = "ambiguous_seed_dependent_result"
    elif stable["R"] is not True:
        cause = "rollout_evaluator_or_dataset_change"
    elif stable["A"] is not True:
        cause = "schema_or_scaler_migration"
    elif stable["B"] is False:
        cause = "condensation_prediction_recursive_feedback"
    elif stable["C"] is False:
        cause = "v1_2_mlp_hyperparameters"
    elif stable["B"] is True and stable["C"] is True and stable["D"] is False:
        cause = "feature_hyperparameter_interaction"
    elif all(value is True for value in stable.values()):
        cause = "not_reproduced_after_scaler_contract_repair"
    else:
        cause = "ambiguous_seed_dependent_result"
    return {
        "condition_stability_majority": stable,
        "condition_seed_disagreement": mixed,
        "seed_counts": seed_counts,
        "cause": cause,
        "additional_seeds_required": bool(pending),
        "additional_seed_conditions": pending,
    }


def _condition_is_mixed(results: Sequence[Mapping[str, Any]], code: str) -> bool:
    values = [
        r.get("stability") == RolloutStabilityStatus.STABLE.value
        for r in results if r.get("condition") == code
    ]
    return bool(values) and any(values) and not all(values)


def _materialize_rollout_failure_trace(model, frame, scaler, scales, *, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    windows = build_rollout_windows(frame, horizon_steps=12, stride_steps=12)
    worst_score = float("-inf")
    worst = None
    first_failure = None
    for window in windows:
        try:
            result = TransitionRolloutSimulator(model, scaler=scaler).simulate(
                frame, start_index=window.start_position, horizon_steps=12
            )
            per_step = []
            first_error = None
            for step in range(len(result.errors.index)):
                normalized = []
                for target in result.target_columns:
                    scale = float(scales.get(target, 1.0)) or 1.0
                    normalized.append(abs(float(result.errors.iloc[step][target])) / scale)
                per_step.append(float(np.mean(normalized)))
                if step and first_error is None and per_step[step] > max(10.0 * per_step[0], 10.0):
                    first_error = step + 1
            score = float(np.sqrt(np.mean(np.square(per_step)))) if per_step else float("nan")
            violation = float(result.metrics.get("physical_violation_rate", 0.0)) > 0
            nonfinite = float(result.metrics.get("nan_inf_count", 0.0)) > 0
            entry = {
                "window_id": f"{window.crop}|{window.episode_id}|{window.start_position}",
                "start_position": int(window.start_position), "nrmse": score,
                "first_rapid_error_step": first_error,
                "physical_violation": violation, "non_finite": nonfinite,
                "trace": result.trace,
            }
            if np.isfinite(score) and score > worst_score:
                worst_score, worst = score, entry
            if first_failure is None and (violation or nonfinite or first_error is not None):
                first_failure = entry
        except Exception as exc:
            if first_failure is None:
                first_failure = {
                    "window_id": f"{window.crop}|{window.episode_id}|{window.start_position}",
                    "start_position": int(window.start_position), "error_type": type(exc).__name__,
                    "message": str(exc), "trace": pd.DataFrame(),
                }
    def save(name: str, entry):
        if entry is None:
            return None
        saved = dict(entry)
        trace = saved.pop("trace")
        path = destination / f"{name}.csv"
        trace.to_csv(path, index=False)
        saved["trace_path"] = str(path)
        return saved
    payload = {
        "window_count": len(windows),
        "first_failure": save("first_failure_trace", first_failure),
        "worst_window": save("worst_window_trace", worst),
    }
    write_json(destination / "trace_summary.json", payload, convert=True)
    return payload

def _prepare_diagnostic_frame(frame: pd.DataFrame, schema: TransitionDiagnosticFeatureSchema) -> pd.DataFrame:
    out = frame.copy()
    if schema is TransitionDiagnosticFeatureSchema.DEDUPLICATED_WITH_CONDENSATION_PREDICTION:
        out = prepare_frame_for_feature_variant(out, "deduplicated_with_quality_flags")
    missing = [c for c in observation_columns_for_diagnostic_schema(schema) if c not in out]
    if missing:
        raise ValueError(f"Diagnostic schema {schema.value} is missing {missing}")
    return out


def _best_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["best_config"])


def _params_for_seed(params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    out = json.loads(json.dumps(params))
    out["random_state"] = int(seed)
    return out


def _classify(record: Mapping[str, Any], reference_median: float) -> dict[str, Any]:
    out = dict(record)
    if record.get("status") != "evaluated":
        out["stability"] = RolloutStabilityStatus.NON_FINITE.value
        return out
    metrics = record["rollout"]["60min"]
    if float(metrics.get("nan_inf_count", 0.0)) > 0:
        status = RolloutStabilityStatus.NON_FINITE
    elif float(metrics.get("physical_violation_rate", 0.0)) > 0:
        status = RolloutStabilityStatus.PHYSICAL_VIOLATION
    elif np.isfinite(reference_median) and _nrmse60(record) > 10.0 * reference_median:
        status = RolloutStabilityStatus.EXPLOSIVE_ERROR
    else:
        status = RolloutStabilityStatus.STABLE
    out["stability"] = status.value
    out["reference_ratio_60min_nrmse"] = _nrmse60(record) / reference_median if reference_median > 0 else None
    return out


def _nrmse60(record: Mapping[str, Any]) -> float:
    return float(record["rollout"]["60min"].get("trajectory_nrmse", float("nan")))


def _condition_majority(results: Sequence[Mapping[str, Any]], code: str) -> bool | None:
    values = [r.get("stability") == RolloutStabilityStatus.STABLE.value for r in results if r.get("condition") == code]
    if not values:
        return None
    yes = sum(values)
    if yes * 2 == len(values):
        return None
    return yes * 2 > len(values)


def _flat_result(record: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: record.get(k) for k in ("condition", "seed", "status", "stability", "parameter_source", "reference_ratio_60min_nrmse")}
    if record.get("status") == "evaluated":
        out["one_step_nrmse"] = record["one_step"]["aggregate_metrics"].get("normalized_mean_rmse")
        for horizon in ("15min", "30min", "60min"):
            metrics = record["rollout"][horizon]
            out[f"{horizon}_nrmse"] = metrics.get("trajectory_nrmse")
            out[f"{horizon}_r2"] = metrics.get("trajectory_r2")
            out[f"{horizon}_physical_violation_rate"] = metrics.get("physical_violation_rate")
    return out


def _write_failure_traces(record: Mapping[str, Any], *, frames, output: Path, comparison_root: Path) -> None:
    # The official metrics artifact contains per-window NRMSE samples and IDs.
    # Copy the worst-window index here; a replay command can materialize full
    # state traces without bloating every successful diagnostic run.
    metrics = record["rollout"]["60min"]
    samples = np.asarray(metrics.get("window_trajectory_nrmse_samples", []), dtype=float)
    ids = list(metrics.get("window_ids", []))
    order = np.argsort(np.nan_to_num(samples, nan=np.inf))[::-1][:10]
    payload = {
        "condition": record["condition"], "seed": record["seed"],
        "first_divergent_target": _first_divergent_target(metrics),
        "worst_windows": [{"window_id": ids[i], "nrmse": float(samples[i])} for i in order if i < len(ids)],
        "metrics_path": str(comparison_root / record["condition"] / f"seed_{record['seed']}" / "metrics.json"),
    }
    write_json(output / f"{record['condition']}_seed_{record['seed']}.json", payload, convert=True)


def _first_divergent_target(metrics: Mapping[str, Any]) -> str | None:
    pairs = [(k, float(v)) for k, v in metrics.items() if k.endswith("_trajectory_nrmse") and not k.startswith("pooled_") and isinstance(v, (int, float))]
    return max(pairs, key=lambda item: item[1])[0].removesuffix("_trajectory_nrmse") if pairs else None


def _require_finite_dataset(frame: pd.DataFrame, *, label: str) -> None:
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        row, col = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(f"{label} non-finite input at row={row}, column={frame.columns[col]}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["CONDITIONS", "ControlledCondition", "RolloutStabilityStatus", "assess_controlled_cause", "audit_melon_inputs", "run_cucumber_controlled_comparison"]
