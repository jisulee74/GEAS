"""End-to-end offline preprocessing runner for the GEAS quality stage.

Step 8 orchestrates the AI Quality Model boundary only:

* fit candidate quality models on train;
* select threshold/model on validation;
* transform train/validation/test with the selected model and threshold;
* write quality-model artifacts and preprocessing manifests.

RL dataset creation is intentionally out of scope for this runner.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from geas35.models.quality import (
    BaseQualityModel,
    GridSearchThresholdOptimizer,
    MedianQualityModel,
    ModernTCNQualityModel,
    PatchTSTQualityModel,
    RuleOnlyQualityModel,
    TimesNetQualityModel,
    ThresholdOptimizationResult,
)
from geas35.offline.prepare_missing_outliers_handled_splits import (
    MissingOutliersHandledSplitSummary,
    prepare_missing_outliers_handled_split_file,
)
from geas35.offline._common import existing_crops, write_json
from geas35.preprocessing import (
    DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
    DEFAULT_DOMAIN_RANGE_PATH,
    STATE_COLUMNS,
    missing_flag_column,
    prepare_missing_outliers_handled_features,
    rule_outlier_flag_column,
)


DEFAULT_DATASET_ROOT = Path("../datasets/iot97_historical")
DEFAULT_INPUT_DIR_NAME = "4_preprocessed/2_unit_canonicalized"
DEFAULT_OUTPUT_DIR_NAME = "4_preprocessed/3_missing_outliers_handled"
DEFAULT_ARTIFACT_DIR_NAME = "_artifacts"
DEFAULT_TRANSFORM_SPLITS = ("train", "validation", "test")
DEFAULT_CANDIDATE_MODELS = ("rule_only", "median")
DEFAULT_CANDIDATE_THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class CandidateSelectionReport:
    model_name: str
    best_threshold: float
    best_objective_value: float
    objective_metric: str
    masking_mae: float | None
    masking_rmse: float | None
    selected: bool


@dataclass(frozen=True)
class QualityPipelineCropSummary:
    crop: str
    selected_model: str
    selected_threshold: float
    model_artifact_path: str
    validation_report_path: str
    split_summaries: list[MissingOutliersHandledSplitSummary]
    candidate_reports: list[CandidateSelectionReport]


def _quality_model_name(model: BaseQualityModel) -> str:
    return str(getattr(model, "model_name", model.__class__.__name__))


def _make_quality_model(model_name: str) -> BaseQualityModel:
    normalized = model_name.strip().lower()
    if normalized == "rule_only":
        return RuleOnlyQualityModel()
    if normalized == "median":
        return MedianQualityModel()
    if normalized in {"modern_tcn", "moderntcn"}:
        return ModernTCNQualityModel()
    if normalized == "timesnet":
        return TimesNetQualityModel()
    if normalized in {"patch_tst", "patchtst"}:
        return PatchTSTQualityModel()
    raise ValueError(f"Unsupported quality model candidate: {model_name}")


def _valid_observation_mask(
    df: pd.DataFrame,
    observation_columns: tuple[str, ...],
) -> pd.DataFrame:
    mask = pd.DataFrame(True, index=df.index, columns=list(observation_columns))
    for col in observation_columns:
        mask[col] &= pd.to_numeric(df[col], errors="coerce").notna()
        missing_col = missing_flag_column(col)
        rule_col = rule_outlier_flag_column(col)
        if missing_col in df.columns:
            mask[col] &= ~(
                pd.to_numeric(df[missing_col], errors="coerce")
                .fillna(0)
                .astype(bool)
            )
        if rule_col in df.columns:
            mask[col] &= ~(
                pd.to_numeric(df[rule_col], errors="coerce")
                .fillna(0)
                .astype(bool)
            )
    return mask


def _selection_frame(
    input_path: Path,
    *,
    domain_csv_path: Path,
    observation_columns: tuple[str, ...],
) -> pd.DataFrame:
    df_input = pd.read_parquet(input_path)
    return prepare_missing_outliers_handled_features(
        df_input,
        quality_model=RuleOnlyQualityModel(),
        domain_csv_path=domain_csv_path,
        observation_columns=observation_columns,
        keep_extra_columns=True,
    )


def _fit_model(
    model_name: str,
    train_df: pd.DataFrame,
    observation_columns: tuple[str, ...],
) -> BaseQualityModel:
    model = _make_quality_model(model_name)
    return model.fit(
        train_df,
        observation_columns,
        valid_mask=_valid_observation_mask(train_df, observation_columns),
    )


def _masking_mae(result: ThresholdOptimizationResult) -> float | None:
    if result.masking_result is None:
        return None
    return result.masking_result.mae


def _masking_rmse(result: ThresholdOptimizationResult) -> float | None:
    if result.masking_result is None:
        return None
    return result.masking_result.rmse


def _selection_sort_key(
    item: tuple[str, BaseQualityModel, ThresholdOptimizationResult],
) -> tuple[float, float]:
    _, _, result = item
    objective = result.best_objective_value
    rmse = _masking_rmse(result)
    rmse_score = -float("inf") if rmse is None or not np.isfinite(rmse) else -float(rmse)
    return (float(objective), rmse_score)


def _confusion_matrix_dict(metrics: object) -> dict[str, int]:
    cm = metrics.confusion_matrix
    return {
        "true_positive": cm.true_positive,
        "false_positive": cm.false_positive,
        "true_negative": cm.true_negative,
        "false_negative": cm.false_negative,
    }


def _metrics_dict(metrics: object) -> dict[str, object]:
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "confusion_matrix": _confusion_matrix_dict(metrics),
    }


def _detection_result_dict(result: object) -> dict[str, object]:
    return {
        "threshold": result.threshold,
        "injected_cells": result.injected_cells,
        "metrics": _metrics_dict(result.metrics),
        "per_column": {
            col: _metrics_dict(metrics)
            for col, metrics in result.per_column.items()
        },
    }


def _threshold_result_dict(result: ThresholdOptimizationResult) -> dict[str, object]:
    return {
        "best_threshold": result.best_threshold,
        "best_objective_value": result.best_objective_value,
        "objective_metric": result.objective_metric,
        "best_detection_result": _detection_result_dict(result.best_detection_result),
        "detection_results": [
            _detection_result_dict(item) for item in result.detection_results
        ],
        "masking_result": None
        if result.masking_result is None
        else {
            "masked_cells": result.masking_result.masked_cells,
            "mae": result.masking_result.mae,
            "rmse": result.masking_result.rmse,
            "per_column": result.masking_result.per_column,
        },
    }


def _model_artifact_dict(
    model: BaseQualityModel,
    *,
    threshold: float,
    imputation_confidence_threshold: float,
    observation_columns: tuple[str, ...],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_name": _quality_model_name(model),
        "observation_columns": list(observation_columns),
        "threshold": threshold,
        "imputation_confidence_threshold": imputation_confidence_threshold,
        "action_policy": "actions_are_never_quality_model_imputed",
    }
    if isinstance(model, MedianQualityModel):
        medians = {} if model.medians_ is None else model.medians_.to_dict()
        payload["medians"] = medians
    if isinstance(
        model,
        (ModernTCNQualityModel, TimesNetQualityModel, PatchTSTQualityModel),
    ):
        payload.update(model.to_artifact())
    return payload


def _candidate_report(
    model_name: str,
    result: ThresholdOptimizationResult,
    *,
    selected: bool,
) -> CandidateSelectionReport:
    return CandidateSelectionReport(
        model_name=model_name,
        best_threshold=result.best_threshold,
        best_objective_value=result.best_objective_value,
        objective_metric=result.objective_metric,
        masking_mae=_masking_mae(result),
        masking_rmse=_masking_rmse(result),
        selected=selected,
    )


def run_preprocessing_pipeline_for_crop(
    *,
    dataset_root: Path,
    crop: str,
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    artifact_dir_name: str = DEFAULT_ARTIFACT_DIR_NAME,
    transform_splits: Iterable[str] = DEFAULT_TRANSFORM_SPLITS,
    candidate_model_names: Sequence[str] = DEFAULT_CANDIDATE_MODELS,
    candidate_thresholds: Sequence[float] = DEFAULT_CANDIDATE_THRESHOLDS,
    imputation_confidence_threshold: float = DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
    observation_columns: Iterable[str] | None = None,
    domain_csv_path: Path = DEFAULT_DOMAIN_RANGE_PATH,
) -> QualityPipelineCropSummary:
    """Fit/select/apply a quality model for one crop."""

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name
    artifact_root = output_root / artifact_dir_name / crop
    split_names = list(transform_splits)
    obs_cols = tuple(STATE_COLUMNS if observation_columns is None else observation_columns)

    train_path = input_root / crop / "train.parquet"
    validation_path = input_root / crop / "validation.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation parquet not found: {validation_path}")

    train_for_fit = _selection_frame(
        train_path,
        domain_csv_path=domain_csv_path,
        observation_columns=obs_cols,
    )
    validation_for_selection = _selection_frame(
        validation_path,
        domain_csv_path=domain_csv_path,
        observation_columns=obs_cols,
    )

    optimizer = GridSearchThresholdOptimizer(
        candidate_thresholds,
        objective_metric="f1_score",
        anomaly_fraction=0.1,
        anomaly_scale=8.0,
        mask_fraction=0.1,
        random_state=0,
    )
    candidates: list[tuple[str, BaseQualityModel, ThresholdOptimizationResult]] = []
    for model_name in candidate_model_names:
        model = _fit_model(model_name, train_for_fit, obs_cols)
        threshold_result = optimizer.optimize(
            model,
            validation_for_selection,
            obs_cols,
        )
        candidates.append((model_name, model, threshold_result))

    selected_name, selected_model, selected_result = max(
        candidates,
        key=_selection_sort_key,
    )
    selected_threshold = float(selected_result.best_threshold)

    split_summaries: list[MissingOutliersHandledSplitSummary] = []
    for split in split_names:
        input_path = input_root / crop / f"{split}.parquet"
        if not input_path.exists():
            raise FileNotFoundError(f"Split parquet not found: {input_path}")
        output_path = output_root / crop / f"{split}.parquet"
        split_summaries.append(
            prepare_missing_outliers_handled_split_file(
                input_path,
                output_path,
                crop=crop,
                split=split,
                quality_model=selected_model,
                thresholds=selected_threshold,
                domain_csv_path=domain_csv_path,
                imputation_confidence_threshold=imputation_confidence_threshold,
            )
        )

    candidate_reports = [
        _candidate_report(
            model_name,
            result,
            selected=model_name == selected_name,
        )
        for model_name, _, result in candidates
    ]

    model_artifact_path = artifact_root / "selected_quality_model.json"
    validation_report_path = artifact_root / "validation_selection_report.json"
    write_json(
        model_artifact_path,
        _model_artifact_dict(
            selected_model,
            threshold=selected_threshold,
            imputation_confidence_threshold=imputation_confidence_threshold,
            observation_columns=obs_cols,
        ),
        convert=True,
    )
    write_json(
        validation_report_path,
        {
            "crop": crop,
            "selection_split": "validation",
            "test_used_for_selection": False,
            "selected_model": selected_name,
            "selected_threshold": selected_threshold,
            "imputation_confidence_threshold": imputation_confidence_threshold,
            "candidate_reports": [asdict(item) for item in candidate_reports],
            "threshold_results": {
                model_name: _threshold_result_dict(result)
                for model_name, _, result in candidates
            },
        },
        convert=True,
    )

    return QualityPipelineCropSummary(
        crop=crop,
        selected_model=selected_name,
        selected_threshold=selected_threshold,
        model_artifact_path=str(model_artifact_path),
        validation_report_path=str(validation_report_path),
        split_summaries=split_summaries,
        candidate_reports=candidate_reports,
    )


def run_preprocessing_pipeline(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    artifact_dir_name: str = DEFAULT_ARTIFACT_DIR_NAME,
    crops: Iterable[str] | None = None,
    transform_splits: Iterable[str] = DEFAULT_TRANSFORM_SPLITS,
    candidate_model_names: Sequence[str] = DEFAULT_CANDIDATE_MODELS,
    candidate_thresholds: Sequence[float] = DEFAULT_CANDIDATE_THRESHOLDS,
    imputation_confidence_threshold: float = DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
    observation_columns: Iterable[str] | None = None,
    domain_csv_path: Path = DEFAULT_DOMAIN_RANGE_PATH,
) -> list[QualityPipelineCropSummary]:
    """Run Step 8 quality preprocessing orchestration for selected crops."""

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name
    crop_names = list(crops) if crops is not None else existing_crops(input_root)

    summaries = [
        run_preprocessing_pipeline_for_crop(
            dataset_root=dataset_root,
            crop=crop,
            input_dir_name=input_dir_name,
            output_dir_name=output_dir_name,
            artifact_dir_name=artifact_dir_name,
            transform_splits=transform_splits,
            candidate_model_names=candidate_model_names,
            candidate_thresholds=candidate_thresholds,
            imputation_confidence_threshold=imputation_confidence_threshold,
            observation_columns=observation_columns,
            domain_csv_path=domain_csv_path,
        )
        for crop in crop_names
    ]

    manifest_path = output_root / "preprocessing_pipeline_manifest.json"
    write_json(
        manifest_path,
        {
            "stage": "preprocessing_pipeline",
            "scope": "quality_model_only",
            "input_root": str(input_root),
            "output_root": str(output_root),
            "artifact_root": str(output_root / artifact_dir_name),
            "domain_csv_path": str(domain_csv_path),
            "candidate_models": list(candidate_model_names),
            "candidate_thresholds": list(candidate_thresholds),
            "imputation_confidence_threshold": imputation_confidence_threshold,
            "transform_splits": list(transform_splits),
            "selection_policy": {
                "fit_split": "train",
                "selection_split": "validation",
                "test_used_for_selection": False,
                "primary_objective": "validation_f1_score",
                "tie_breaker": "lower_validation_synthetic_masking_rmse",
            },
            "crops": [asdict(summary) for summary in summaries],
        },
        convert=True,
    )
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Step 8 GEAS quality preprocessing pipeline."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root of the iot97_historical dataset directory.",
    )
    parser.add_argument(
        "--input-dir-name",
        default=DEFAULT_INPUT_DIR_NAME,
        help="Unit-canonicalized input directory under dataset root.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help="Missing/outlier handled output directory under dataset root.",
    )
    parser.add_argument(
        "--artifact-dir-name",
        default=DEFAULT_ARTIFACT_DIR_NAME,
        help="Artifact directory under output root.",
    )
    parser.add_argument(
        "--domain-csv-path",
        type=Path,
        default=DEFAULT_DOMAIN_RANGE_PATH,
        help="Domain range CSV used for rule-based outlier flags.",
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        default=None,
        help="Crop folders to process. Defaults to all crop folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_TRANSFORM_SPLITS),
        help="Split names to transform after model selection.",
    )
    parser.add_argument(
        "--candidate-models",
        nargs="+",
        default=list(DEFAULT_CANDIDATE_MODELS),
        choices=["rule_only", "median", "modern_tcn", "timesnet", "patch_tst"],
        help="Quality model candidates to fit and compare.",
    )
    parser.add_argument(
        "--candidate-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_CANDIDATE_THRESHOLDS),
        help="Validation threshold candidates.",
    )
    parser.add_argument(
        "--imputation-confidence-threshold",
        type=float,
        default=DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
        help=(
            "Confidence gate for AI-only outlier imputation. Missing/rule "
            "invalid cells remain imputation candidates regardless of this value."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summaries = run_preprocessing_pipeline(
        dataset_root=args.dataset_root,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        artifact_dir_name=args.artifact_dir_name,
        crops=args.crops,
        transform_splits=args.splits,
        candidate_model_names=args.candidate_models,
        candidate_thresholds=args.candidate_thresholds,
        imputation_confidence_threshold=args.imputation_confidence_threshold,
        domain_csv_path=args.domain_csv_path,
    )

    for summary in summaries:
        print(
            f"[preprocessing_pipeline] {summary.crop}: "
            f"selected {summary.selected_model} "
            f"threshold={summary.selected_threshold}, "
            f"wrote {len(summary.split_summaries)} splits"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
