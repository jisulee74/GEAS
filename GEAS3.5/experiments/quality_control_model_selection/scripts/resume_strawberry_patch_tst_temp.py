"""Resume strawberry quality-model selection without repeating completed HPO.

ModernTCN and TimesNet HPO artifacts are loaded from disk. PatchTST HPO is
executed only when its complete 50-trial artifact is absent. All three best
configurations then continue through calibration, evaluation, benchmarking,
reporting, PNG generation, and integrity validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_src_on_path() -> None:
    src = _project_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_hpo_result(path: Path, *, expected_trials: int):
    from geas35.models.quality import (
        CandidateTrainingResult,
        EarlyStoppingConfig,
        HyperparameterCandidate,
        HyperparameterObjective,
        HyperparameterOptimizationResult,
        MetricGoal,
    )

    payload = _read_json(path)
    rows = payload.get("candidate_results", [])
    if len(rows) != expected_trials:
        raise ValueError(
            f"Incomplete HPO artifact {path}: expected {expected_trials} trials, "
            f"found {len(rows)}."
        )

    candidates = []
    for row in rows:
        raw_candidate = row["candidate"]
        candidate = HyperparameterCandidate(
            name=str(raw_candidate["name"]),
            common=dict(raw_candidate.get("common", {})),
            model_specific=dict(raw_candidate.get("model_specific", {})),
        )
        candidates.append(
            CandidateTrainingResult(
                candidate=candidate,
                validation_metrics=dict(row.get("validation_metrics", {})),
                training_history=tuple(row.get("training_history", [])),
                model_artifact=row.get("model_artifact"),
            )
        )

    objective_payload = payload.get("objective", {})
    primary_payload = objective_payload.get("primary", {})
    secondary_payloads = objective_payload.get("secondary", [])
    objective = HyperparameterObjective(
        primary=MetricGoal(
            str(primary_payload.get("name", "validation_synthetic_masking_rmse")),
            str(primary_payload.get("mode", "min")),
        ),
        secondary=tuple(
            MetricGoal(str(goal["name"]), str(goal.get("mode", "min")))
            for goal in secondary_payloads
        ),
    )
    early_payload = payload.get("early_stopping", {})
    early_stopping = EarlyStoppingConfig(
        monitor=str(early_payload.get("monitor", "validation_reconstruction_loss")),
        mode=str(early_payload.get("mode", "min")),
        patience=int(early_payload.get("patience", 10)),
        min_delta=float(early_payload.get("min_delta", 0.0)),
    )
    best_name = str(payload["best_candidate"]["name"])
    try:
        best_result = next(row for row in candidates if row.candidate.name == best_name)
    except StopIteration as exc:
        raise ValueError(f"Best candidate {best_name} is absent from {path}") from exc
    return HyperparameterOptimizationResult(
        best_result=best_result,
        candidate_results=candidates,
        objective=objective,
        early_stopping=early_stopping,
    )


def _write_hpo_aggregates(output_root: Path, model_results) -> None:
    hpo_payload = {
        "stage": "hyperparameter_optimization_summary",
        "threshold_calibration_included": False,
        "threshold_is_hpo_parameter": False,
        "models": {
            result.model_name: result.hpo_result.to_artifact()
            for result in model_results
        },
    }
    best_payload = {
        "stage": "best_hyperparameter_configurations",
        "threshold_calibration_included": False,
        "models": {
            result.model_name: result.hpo_result.best_candidate.to_artifact()
            for result in model_results
        },
    }
    for name, payload in (
        ("hpo_results.json", hpo_payload),
        ("best_config.json", best_payload),
    ):
        (output_root / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    history_rows: list[dict[str, Any]] = []
    for result in model_results:
        for row in result.hpo_result.best_result.training_history:
            history_rows.append(
                {
                    "model_name": result.model_name,
                    "candidate_name": result.hpo_result.best_candidate.name,
                    **dict(row),
                }
            )
    fieldnames = sorted({key for row in history_rows for key in row})
    with (output_root / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def run(config_path: Path) -> int:
    _ensure_src_on_path()
    from geas35.experiments.quality.benchmark_runner import (
        quality_online_benchmark_config_from_experiment_config,
        run_quality_online_benchmark,
    )
    from geas35.experiments.quality.calibration_runner import (
        quality_threshold_config_from_experiment_config,
        run_quality_threshold_calibration,
    )
    from geas35.experiments.quality.config import (
        load_quality_experiment_config,
        read_configured_datasets,
    )
    from geas35.experiments.quality.evaluation_runner import (
        quality_evaluation_config_from_experiment_config,
        run_quality_validation_test_evaluation,
    )
    from geas35.experiments.quality.hpo_runner import (
        ModelHPOResult,
        QualityHPOResult,
        run_model_hpo,
    )
    from geas35.experiments.quality.integrity_runner import (
        quality_artifact_integrity_config_from_experiment_config,
        run_quality_artifact_integrity_check,
    )
    from geas35.experiments.quality.report_runner import (
        quality_report_config_from_experiment_config,
        run_quality_report_generation,
    )
    from geas35.experiments.quality.runner import default_quality_model_registry
    from geas35.experiments.quality.visualization_runner import (
        quality_visualization_config_from_experiment_config,
        run_quality_visualization_generation,
    )

    loaded = load_quality_experiment_config(config_path)
    config = loaded.experiment_config
    output_root = Path(config.output_root)
    if output_root.name != "strawberry":
        raise ValueError(f"This temporary resume script requires strawberry output: {output_root}")
    train_df, validation_df, test_df, columns = read_configured_datasets(loaded)
    specs = {spec.model_name: spec for spec in config.models}
    expected_models = ("modern_tcn", "timesnet", "patch_tst")
    if tuple(spec.model_name for spec in config.models) != expected_models:
        raise ValueError(f"Unexpected model order/configuration: {tuple(specs)}")

    registry = default_quality_model_registry()
    model_results = []
    for model_name in expected_models:
        spec = specs[model_name]
        artifact_path = output_root / model_name / "hpo_results.json"
        if model_name in {"modern_tcn", "timesnet"}:
            hpo = _load_hpo_result(artifact_path, expected_trials=spec.hpo_budget)
            print(
                f"[strawberry][{model_name}] 기존 HPO 재사용 "
                f"({len(hpo.candidate_results)} trials, best={hpo.best_candidate.name})",
                flush=True,
            )
            model_results.append(
                ModelHPOResult(model_name, hpo, str(output_root / model_name))
            )
        elif artifact_path.exists():
            hpo = _load_hpo_result(artifact_path, expected_trials=spec.hpo_budget)
            print(
                f"[strawberry][patch_tst] 완료된 HPO 재사용 "
                f"({len(hpo.candidate_results)} trials, best={hpo.best_candidate.name})",
                flush=True,
            )
            model_results.append(
                ModelHPOResult(model_name, hpo, str(output_root / model_name))
            )
        else:
            print("[strawberry][patch_tst] 중단 지점부터 HPO 재실행", flush=True)
            model_results.append(
                run_model_hpo(
                    model_spec=spec,
                    implementation=registry[model_name],
                    train_df=train_df,
                    validation_df=validation_df,
                    observation_columns=columns,
                    early_stopping=config.early_stopping,
                    output_root=output_root,
                )
            )

    _write_hpo_aggregates(output_root, model_results)
    hpo_result = QualityHPOResult(
        output_root=str(output_root),
        model_results=model_results,
        hpo_results_path=str(output_root / "hpo_results.json"),
        best_config_path=str(output_root / "best_config.json"),
        training_history_path=str(output_root / "training_history.csv"),
    )
    print("[strawberry] 세 모델 HPO 결과 통합 완료", flush=True)

    threshold_result = run_quality_threshold_calibration(
        config=quality_threshold_config_from_experiment_config(config),
        hpo_result=hpo_result,
        train_df=train_df,
        validation_df=validation_df,
        observation_columns=columns,
        registry=registry,
    )
    evaluation_result = run_quality_validation_test_evaluation(
        config=quality_evaluation_config_from_experiment_config(config),
        calibration_result=threshold_result,
        reference_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        observation_columns=columns,
    )
    benchmark_result = run_quality_online_benchmark(
        config=quality_online_benchmark_config_from_experiment_config(config),
        calibration_result=threshold_result,
        validation_df=validation_df,
        test_df=test_df,
        observation_columns=columns,
    )
    run_quality_report_generation(
        config=quality_report_config_from_experiment_config(config),
        evaluation_result=evaluation_result,
        online_benchmark_result=benchmark_result,
    )
    if loaded.generate_figures:
        run_quality_visualization_generation(
            config=quality_visualization_config_from_experiment_config(config)
        )
    integrity_config = quality_artifact_integrity_config_from_experiment_config(config)
    integrity_config = type(integrity_config)(
        output_root=integrity_config.output_root,
        require_figures=loaded.generate_figures,
    )
    integrity = run_quality_artifact_integrity_check(config=integrity_config)
    print(
        f"[strawberry] Resume 파이프라인 완료: integrity_passed={integrity.passed}",
        flush=True,
    )
    return 0 if integrity.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_experiment_root() / "configs" / "strawberry.yaml",
    )
    args = parser.parse_args()
    return run(args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
