"""Apply GEAS quality control to offline split datasets.

This script is the offline dataset preparation entry point for the shared
preprocessing modules:

1. input schema preparation;
2. unit canonicalization;
3. combined missing/outlier handling.

Quality model selection is intentionally outside this script. By default, this
entry point uses the rule-only quality model from ``geas35.preprocessing``. When
``--quality-model-artifact`` is provided, it applies that human-selected model
and threshold to train/validation/test without performing model selection here.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets"


DEFAULT_INPUT_DIR_NAME = "02_split"
DEFAULT_OUTPUT_DIR_NAME = "03_quality_controlled"
DEFAULT_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class QualityControlledSplitSummary:
    crop: str
    split: str
    input_path: str
    output_path: str
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    missing_cells: int
    restored_action_cells: int
    rule_outlier_rows: int
    rule_outlier_cells: int
    ai_outlier_rows: int
    invalid_rows: int
    invalid_cells: int
    imputed_rows: int
    imputed_cells: int


@dataclass(frozen=True)
class QualityControlApplication:
    quality_model: object | None
    thresholds: object | None
    observation_columns: tuple[str, ...] | None
    imputation_confidence_threshold: float | None
    artifact_path: Path | None
    raw_payload: dict[str, object] | None


def _ensure_src_on_path() -> None:
    src_path = _project_root() / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _load_quality_control_application(
    artifact_path: Path | None,
) -> QualityControlApplication:
    if artifact_path is None:
        return QualityControlApplication(
            quality_model=None,
            thresholds=None,
            observation_columns=None,
            imputation_confidence_threshold=None,
            artifact_path=None,
            raw_payload=None,
        )

    from geas35.models.quality import load_quality_model_application

    application = load_quality_model_application(artifact_path)
    return QualityControlApplication(
        quality_model=application.model,
        thresholds=application.thresholds,
        observation_columns=application.observation_columns,
        imputation_confidence_threshold=application.imputation_confidence_threshold,
        artifact_path=application.artifact_path,
        raw_payload=application.raw_payload,
    )


def _prepare_quality_controlled_frame(
    df_raw: pd.DataFrame,
    *,
    domain_csv_path: Path,
    add_flags: bool,
    imputation_confidence_threshold: float,
    application: QualityControlApplication,
) -> pd.DataFrame:
    from geas35.preprocessing import (
        canonicalize_feature_units,
        prepare_geas_input_schema,
        prepare_missing_outliers_handled_features,
    )

    df = prepare_geas_input_schema(df_raw, keep_extra_columns=True)
    df = canonicalize_feature_units(df)
    return prepare_missing_outliers_handled_features(
        df,
        df_control_log=None,
        quality_model=application.quality_model,
        thresholds=application.thresholds,
        domain_csv_path=domain_csv_path,
        observation_columns=application.observation_columns,
        keep_extra_columns=True,
        add_flags=add_flags,
        imputation_confidence_threshold=(
            application.imputation_confidence_threshold
            if application.imputation_confidence_threshold is not None
            else imputation_confidence_threshold
        ),
    )


def _prepare_split_file(
    input_path: Path,
    output_path: Path,
    *,
    crop: str,
    split: str,
    domain_csv_path: Path,
    add_flags: bool,
    imputation_confidence_threshold: float,
    application: QualityControlApplication,
) -> QualityControlledSplitSummary:
    from geas35.preprocessing import (
        ACTION_RESTORED_FLAG_SUFFIX,
        DEFAULT_RULE_AGGREGATE_FLAG,
        MISSING_FLAG_SUFFIX,
        QUALITY_AI_OUTLIER_FLAG_COLUMN,
        QUALITY_IMPUTED_FLAG_COLUMN,
        QUALITY_INVALID_FLAG_COLUMN,
    )
    from geas35.io_utils import flag_cell_count, flag_sum

    df_input = pd.read_parquet(input_path)
    df_output = _prepare_quality_controlled_frame(
        df_input,
        domain_csv_path=domain_csv_path,
        add_flags=add_flags,
        imputation_confidence_threshold=imputation_confidence_threshold,
        application=application,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_parquet(output_path, index=False)

    return QualityControlledSplitSummary(
        crop=crop,
        split=split,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=len(df_input),
        output_rows=len(df_output),
        input_columns=len(df_input.columns),
        output_columns=len(df_output.columns),
        missing_cells=flag_cell_count(df_output, MISSING_FLAG_SUFFIX),
        restored_action_cells=flag_cell_count(
            df_output,
            ACTION_RESTORED_FLAG_SUFFIX,
        ),
        rule_outlier_rows=flag_sum(df_output, DEFAULT_RULE_AGGREGATE_FLAG),
        rule_outlier_cells=flag_cell_count(
            df_output,
            "_rule_outlier_flag",
            exclude={DEFAULT_RULE_AGGREGATE_FLAG},
        ),
        ai_outlier_rows=flag_sum(df_output, QUALITY_AI_OUTLIER_FLAG_COLUMN),
        invalid_rows=flag_sum(df_output, QUALITY_INVALID_FLAG_COLUMN),
        invalid_cells=flag_cell_count(
            df_output,
            "_invalid_flag",
            exclude={QUALITY_INVALID_FLAG_COLUMN},
        ),
        imputed_rows=flag_sum(df_output, QUALITY_IMPUTED_FLAG_COLUMN),
        imputed_cells=flag_cell_count(
            df_output,
            "_imputed_flag",
            exclude={QUALITY_IMPUTED_FLAG_COLUMN},
        ),
    )


def prepare_quality_controlled_splits(
    *,
    dataset_root: Path = _dataset_root(),
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    crops: Iterable[str] | None = None,
    splits: Iterable[str] = DEFAULT_SPLITS,
    domain_csv_path: Path | None = None,
    add_flags: bool = True,
    imputation_confidence_threshold: float | None = None,
    quality_model_artifact: Path | None = None,
) -> list[QualityControlledSplitSummary]:
    from geas35.preprocessing import (
        DEFAULT_DOMAIN_RANGE_PATH,
        DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
    )
    from geas35.io_utils import existing_crops, write_json

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name
    domain_csv_path = DEFAULT_DOMAIN_RANGE_PATH if domain_csv_path is None else domain_csv_path
    imputation_confidence_threshold = (
        DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD
        if imputation_confidence_threshold is None
        else imputation_confidence_threshold
    )
    application = _load_quality_control_application(quality_model_artifact)

    crop_names = list(crops) if crops is not None else existing_crops(input_root)
    split_names = list(splits)
    summaries: list[QualityControlledSplitSummary] = []

    for crop in crop_names:
        for split in split_names:
            input_path = input_root / crop / f"{split}.parquet"
            if not input_path.exists():
                raise FileNotFoundError(f"Split parquet not found: {input_path}")
            output_path = output_root / crop / f"{split}.parquet"
            summaries.append(
                _prepare_split_file(
                    input_path,
                    output_path,
                    crop=crop,
                    split=split,
                    domain_csv_path=domain_csv_path,
                    add_flags=add_flags,
                    imputation_confidence_threshold=imputation_confidence_threshold,
                    application=application,
                )
            )

    model_name = (
        "rule_only"
        if application.quality_model is None
        else str(
            getattr(
                application.quality_model,
                "model_name",
                application.quality_model.__class__.__name__,
            )
        )
    )
    effective_imputation_confidence_threshold = (
        application.imputation_confidence_threshold
        if application.imputation_confidence_threshold is not None
        else imputation_confidence_threshold
    )
    write_json(
        output_root / "quality_controlled_manifest.json",
        {
            "stage": "quality_controlled",
            "input_root": input_root,
            "output_root": output_root,
            "input_schema_stage": "geas35.preprocessing.1_input_schema_preparation",
            "unit_canonicalization_stage": "geas35.preprocessing.2_unit_canonicalization",
            "missing_outlier_stage": "geas35.preprocessing.3_missing_outliers_handling",
            "quality_model": model_name,
            "selected_quality_model_artifact": application.artifact_path,
            "thresholds": application.thresholds,
            "observation_columns": application.observation_columns,
            "domain_csv_path": domain_csv_path,
            "add_flags": add_flags,
            "imputation_confidence_threshold": effective_imputation_confidence_threshold,
            "splits": summaries,
        },
        convert=True,
    )
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    _ensure_src_on_path()
    from geas35.preprocessing import (
        DEFAULT_DOMAIN_RANGE_PATH,
        DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
    )

    parser = argparse.ArgumentParser(
        description="Build 03_quality_controlled datasets from 02_split parquet files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_dataset_root(),
        help="Root of the offline dataset directory.",
    )
    parser.add_argument(
        "--input-dir-name",
        default=DEFAULT_INPUT_DIR_NAME,
        help="Input directory name under dataset root.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help="Output directory name under dataset root.",
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
        default=list(DEFAULT_SPLITS),
        help="Split names to process. Defaults to train validation test.",
    )
    parser.add_argument(
        "--no-flags",
        action="store_true",
        help="Do not add missing/restored/quality flag columns.",
    )
    parser.add_argument(
        "--imputation-confidence-threshold",
        type=float,
        default=DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
        help="Confidence gate reserved for selected AI quality models.",
    )
    parser.add_argument(
        "--quality-model-artifact",
        type=Path,
        default=None,
        help=(
            "Optional human-selected quality model artifact JSON. Supports "
            "rule_only, median medians, or model_pickle_path for fitted models."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    args = build_arg_parser().parse_args(argv)
    summaries = prepare_quality_controlled_splits(
        dataset_root=args.dataset_root,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        crops=args.crops,
        splits=args.splits,
        domain_csv_path=args.domain_csv_path,
        add_flags=not args.no_flags,
        imputation_confidence_threshold=args.imputation_confidence_threshold,
        quality_model_artifact=args.quality_model_artifact,
    )

    for summary in summaries:
        print(
            f"[quality_controlled] {summary.crop}/{summary.split}: "
            f"{summary.input_rows} -> {summary.output_rows} rows, "
            f"{summary.invalid_rows} invalid rows, "
            f"{summary.imputed_cells} cells imputed, wrote {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
