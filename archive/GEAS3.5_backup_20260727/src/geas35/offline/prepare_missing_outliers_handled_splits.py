"""Prepare combined missing/outlier handled parquet files for offline splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from geas35.models.quality import BaseQualityModel
from geas35.offline._common import (
    existing_crops,
    flag_cell_count,
    flag_sum,
    write_json,
)
from geas35.preprocessing import (
    ACTION_RESTORED_FLAG_SUFFIX,
    DEFAULT_DOMAIN_RANGE_PATH,
    DEFAULT_RULE_AGGREGATE_FLAG,
    FEATURE_COLUMNS,
    MISSING_FLAG_SUFFIX,
    DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
    QUALITY_AI_OUTLIER_FLAG_COLUMN,
    QUALITY_IMPUTED_FLAG_COLUMN,
    QUALITY_INVALID_FLAG_COLUMN,
    prepare_missing_outliers_handled_features,
)


DEFAULT_DATASET_ROOT = Path("../datasets/iot97_historical")
DEFAULT_INPUT_DIR_NAME = "4_preprocessed/2_unit_canonicalized"
DEFAULT_OUTPUT_DIR_NAME = "4_preprocessed/3_missing_outliers_handled"
DEFAULT_SPLITS = ("train",)


@dataclass(frozen=True)
class MissingOutliersHandledSplitSummary:
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
    unresolved_missing_feature_cells: int


def _quality_model_name(model: BaseQualityModel | None) -> str:
    if model is None:
        return "rule_only"
    return str(getattr(model, "model_name", model.__class__.__name__))


def prepare_missing_outliers_handled_split_file(
    input_path: Path,
    output_path: Path,
    *,
    crop: str,
    split: str,
    quality_model: BaseQualityModel | None = None,
    thresholds: object | None = None,
    domain_csv_path: Path = DEFAULT_DOMAIN_RANGE_PATH,
    add_flags: bool = True,
    imputation_confidence_threshold: float = DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
) -> MissingOutliersHandledSplitSummary:
    """Prepare one unit-canonicalized split parquet and write clean output."""

    df_input = pd.read_parquet(input_path)
    df_prepared = prepare_missing_outliers_handled_features(
        df_input,
        df_control_log=None,
        quality_model=quality_model,
        thresholds=thresholds,
        domain_csv_path=domain_csv_path,
        keep_extra_columns=True,
        add_flags=add_flags,
        imputation_confidence_threshold=imputation_confidence_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_prepared.to_parquet(output_path, index=False)

    return MissingOutliersHandledSplitSummary(
        crop=crop,
        split=split,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=len(df_input),
        output_rows=len(df_prepared),
        input_columns=len(df_input.columns),
        output_columns=len(df_prepared.columns),
        missing_cells=flag_cell_count(df_prepared, MISSING_FLAG_SUFFIX),
        restored_action_cells=flag_cell_count(
            df_prepared,
            ACTION_RESTORED_FLAG_SUFFIX,
        ),
        rule_outlier_rows=flag_sum(df_prepared, DEFAULT_RULE_AGGREGATE_FLAG),
        rule_outlier_cells=flag_cell_count(
            df_prepared,
            "_rule_outlier_flag",
            exclude={DEFAULT_RULE_AGGREGATE_FLAG},
        ),
        ai_outlier_rows=flag_sum(df_prepared, QUALITY_AI_OUTLIER_FLAG_COLUMN),
        invalid_rows=flag_sum(df_prepared, QUALITY_INVALID_FLAG_COLUMN),
        invalid_cells=flag_cell_count(
            df_prepared,
            "_invalid_flag",
            exclude={QUALITY_INVALID_FLAG_COLUMN},
        ),
        imputed_rows=flag_sum(df_prepared, QUALITY_IMPUTED_FLAG_COLUMN),
        imputed_cells=flag_cell_count(
            df_prepared,
            "_imputed_flag",
            exclude={QUALITY_IMPUTED_FLAG_COLUMN},
        ),
        unresolved_missing_feature_cells=int(df_prepared[FEATURE_COLUMNS].isna().sum().sum()),
    )


def prepare_missing_outliers_handled_splits(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    crops: Iterable[str] | None = None,
    splits: Iterable[str] = DEFAULT_SPLITS,
    quality_model: BaseQualityModel | None = None,
    thresholds: object | None = None,
    domain_csv_path: Path = DEFAULT_DOMAIN_RANGE_PATH,
    add_flags: bool = True,
    imputation_confidence_threshold: float = DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD,
) -> list[MissingOutliersHandledSplitSummary]:
    """Prepare combined missing/outlier handled parquet files."""

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name

    crop_names = list(crops) if crops is not None else existing_crops(input_root)
    split_names = list(splits)
    summaries: list[MissingOutliersHandledSplitSummary] = []

    for crop in crop_names:
        for split in split_names:
            input_path = input_root / crop / f"{split}.parquet"
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Unit-canonicalized parquet not found: {input_path}"
                )

            output_path = output_root / crop / f"{split}.parquet"
            summaries.append(
                prepare_missing_outliers_handled_split_file(
                    input_path,
                    output_path,
                    crop=crop,
                    split=split,
                    quality_model=quality_model,
                    thresholds=thresholds,
                    domain_csv_path=domain_csv_path,
                    add_flags=add_flags,
                    imputation_confidence_threshold=imputation_confidence_threshold,
                )
            )

    manifest_path = output_root / "missing_outliers_handled_manifest.json"
    manifest = {
        "stage": "missing_outliers_handled",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "domain_csv_path": str(domain_csv_path),
        "quality_model": _quality_model_name(quality_model),
        "thresholds": thresholds,
        "imputation_confidence_threshold": imputation_confidence_threshold,
        "add_flags": add_flags,
        "rule_aggregate_flag": DEFAULT_RULE_AGGREGATE_FLAG,
        "ai_outlier_flag": QUALITY_AI_OUTLIER_FLAG_COLUMN,
        "invalid_flag": QUALITY_INVALID_FLAG_COLUMN,
        "imputed_flag": QUALITY_IMPUTED_FLAG_COLUMN,
        "splits": [asdict(summary) for summary in summaries],
    }
    write_json(manifest_path, manifest)
    return summaries


prepare_quality_handled_splits = prepare_missing_outliers_handled_splits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare combined missing/outlier handled parquet files from "
            "unit-canonicalized datasets."
        )
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
        help="Split names to process. Defaults to train.",
    )
    parser.add_argument(
        "--no-flags",
        action="store_true",
        help="Do not add missing/restored flag columns.",
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
    summaries = prepare_missing_outliers_handled_splits(
        dataset_root=args.dataset_root,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        crops=args.crops,
        splits=args.splits,
        domain_csv_path=args.domain_csv_path,
        add_flags=not args.no_flags,
        imputation_confidence_threshold=args.imputation_confidence_threshold,
    )

    for summary in summaries:
        print(
            f"[missing_outliers_handled] {summary.crop}/{summary.split}: "
            f"{summary.input_rows} -> {summary.output_rows} rows, "
            f"{summary.invalid_rows} invalid rows, "
            f"{summary.imputed_cells} cells imputed, wrote {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
