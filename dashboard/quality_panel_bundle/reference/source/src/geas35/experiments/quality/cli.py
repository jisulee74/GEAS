"""Command-line entry point for quality-model experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from geas35.experiments.quality.benchmark_runner import (
    QualityOnlineBenchmarkResult,
    quality_online_benchmark_config_from_experiment_config,
    run_quality_online_benchmark,
)
from geas35.experiments.quality.calibration_runner import (
    QualityThresholdCalibrationResult,
    quality_threshold_config_from_experiment_config,
    run_quality_threshold_calibration,
)
from geas35.experiments.quality.config import (
    LoadedQualityExperimentConfig,
    load_quality_experiment_config,
    read_configured_datasets,
)
from geas35.experiments.quality.evaluation_runner import (
    QualityEvaluationResult,
    quality_evaluation_config_from_experiment_config,
    run_quality_validation_test_evaluation,
)
from geas35.experiments.quality.hpo_runner import (
    QualityHPOResult,
    quality_hpo_config_from_experiment_config,
    run_quality_hpo,
)
from geas35.experiments.quality.integrity_runner import (
    QualityArtifactIntegrityResult,
    quality_artifact_integrity_config_from_experiment_config,
    run_quality_artifact_integrity_check,
)
from geas35.experiments.quality.report_runner import (
    QualityReportResult,
    quality_report_config_from_experiment_config,
    run_quality_report_generation,
)
from geas35.experiments.quality.runner import ModelImplementation
from geas35.experiments.quality.visualization_runner import (
    QualityVisualizationResult,
    quality_visualization_config_from_experiment_config,
    run_quality_visualization_generation,
)


@dataclass(frozen=True)
class QualityEndToEndExperimentResult:
    """Step 9 end-to-end experiment result."""

    output_root: str
    loaded_config: LoadedQualityExperimentConfig
    hpo_result: QualityHPOResult
    threshold_result: QualityThresholdCalibrationResult
    evaluation_result: QualityEvaluationResult
    online_benchmark_result: QualityOnlineBenchmarkResult
    report_result: QualityReportResult
    visualization_result: QualityVisualizationResult | None
    integrity_result: QualityArtifactIntegrityResult

    @property
    def comparison_csv_path(self) -> str:
        return self.report_result.model_comparison_csv_path

    @property
    def markdown_summary_path(self) -> str:
        return self.report_result.markdown_summary_path

    @property
    def artifact_integrity_path(self) -> str:
        return self.integrity_result.artifact_integrity_path


def run_from_config(
    config_path: str | Path,
    *,
    registry: Mapping[str, ModelImplementation] | None = None,
) -> QualityEndToEndExperimentResult:
    """Run Step 1 through Step 8 from a YAML config file."""

    loaded = load_quality_experiment_config(config_path)
    train_df, validation_df, test_df, observation_columns = read_configured_datasets(
        loaded
    )
    experiment_config = loaded.experiment_config
    crop_name = experiment_config.output_root.name
    print(
        f"[{crop_name}] 전체 파이프라인 시작",
        flush=True,
    )

    hpo_result = run_quality_hpo(
        config=quality_hpo_config_from_experiment_config(experiment_config),
        train_df=train_df,
        validation_df=validation_df,
        observation_columns=observation_columns,
        registry=registry,
    )
    print(f"[{crop_name}] 전체 모델 HPO 완료", flush=True)
    threshold_result = run_quality_threshold_calibration(
        config=quality_threshold_config_from_experiment_config(experiment_config),
        hpo_result=hpo_result,
        train_df=train_df,
        validation_df=validation_df,
        observation_columns=observation_columns,
        registry=registry,
    )
    print(f"[{crop_name}] 전체 모델 Threshold Calibration 완료", flush=True)
    evaluation_result = run_quality_validation_test_evaluation(
        config=quality_evaluation_config_from_experiment_config(experiment_config),
        calibration_result=threshold_result,
        reference_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        observation_columns=observation_columns,
    )
    print(f"[{crop_name}] 전체 모델 Validation/Test 평가 완료", flush=True)
    online_benchmark_result = run_quality_online_benchmark(
        config=quality_online_benchmark_config_from_experiment_config(
            experiment_config
        ),
        calibration_result=threshold_result,
        validation_df=validation_df,
        test_df=test_df,
        observation_columns=observation_columns,
    )
    print(f"[{crop_name}] 전체 모델 Online Benchmark 완료", flush=True)
    print(f"[{crop_name}] Report 생성 시작", flush=True)
    report_result = run_quality_report_generation(
        config=quality_report_config_from_experiment_config(experiment_config),
        evaluation_result=evaluation_result,
        online_benchmark_result=online_benchmark_result,
    )
    print(f"[{crop_name}] Report 생성 완료", flush=True)
    if loaded.generate_figures:
        print(f"[{crop_name}] PNG Figure 생성 시작", flush=True)
    visualization_result = (
        run_quality_visualization_generation(
            config=quality_visualization_config_from_experiment_config(
                experiment_config
            )
        )
        if loaded.generate_figures
        else None
    )
    if visualization_result is not None:
        print(f"[{crop_name}] PNG Figure 생성 완료", flush=True)
    print(f"[{crop_name}] Artifact 무결성 검사 시작", flush=True)
    integrity_config = quality_artifact_integrity_config_from_experiment_config(
        experiment_config
    )
    integrity_config = type(integrity_config)(
        output_root=integrity_config.output_root,
        require_figures=loaded.generate_figures,
    )
    integrity_result = run_quality_artifact_integrity_check(config=integrity_config)
    print(
        f"[{crop_name}] Artifact 무결성 검사 완료: "
        f"passed={integrity_result.passed}",
        flush=True,
    )
    print(f"[{crop_name}] 전체 파이프라인 완료", flush=True)
    return QualityEndToEndExperimentResult(
        output_root=str(experiment_config.output_root),
        loaded_config=loaded,
        hpo_result=hpo_result,
        threshold_result=threshold_result,
        evaluation_result=evaluation_result,
        online_benchmark_result=online_benchmark_result,
        report_result=report_result,
        visualization_result=visualization_result,
        integrity_result=integrity_result,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the GEAS AI Quality Model experiment from YAML."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML experiment config.",
    )
    args = parser.parse_args(argv)
    result = run_from_config(args.config)
    print(f"Experiment output: {result.output_root}")
    print(f"Model comparison CSV: {result.comparison_csv_path}")
    print(f"Markdown summary: {result.markdown_summary_path}")
    print(f"Artifact integrity: {result.artifact_integrity_path}")
    if not result.integrity_result.passed:
        print("Artifact integrity check failed.")
        return 1
    return 0


__all__ = ["QualityEndToEndExperimentResult", "main", "run_from_config"]
