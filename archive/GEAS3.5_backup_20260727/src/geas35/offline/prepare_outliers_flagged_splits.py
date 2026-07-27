"""Prepare outlier-flagged parquet files for offline split datasets.

This runner reads unit-canonicalized intermediate files and writes the next
dataset after rule-based and optional TCN-based outlier flagging. Feature values
are not modified in this stage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from geas35.offline._common import existing_crops, flag_sum, write_json
from geas35.preprocessing import (
    DEFAULT_AGGREGATE_FLAG,
    DEFAULT_AI_OUTLIER_FLAG_COLUMN,
    DEFAULT_AI_WARNING_FLAG_COLUMN,
    DEFAULT_DOMAIN_RANGE_PATH,
    DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN,
    DEFAULT_IMPUTED_FLAG_COLUMN,
    DEFAULT_RULE_AGGREGATE_FLAG,
    DEFAULT_TCN_FLAG_COLUMN,
    DEFAULT_TCN_PRED_VALUE_COLUMN,
    DEFAULT_TCN_SCORE_COLUMN,
    DEFAULT_TCN_WARNING_FLAG_COLUMN,
    apply_outlier_flags,
)


DEFAULT_DATASET_ROOT = Path("../datasets/iot97_historical")
DEFAULT_INPUT_DIR_NAME = "4_preprocessed/3_unit_canonicalized"
DEFAULT_OUTPUT_DIR_NAME = "4_preprocessed/4_outliers_flagged"
DEFAULT_SPLITS = ("train",)


@dataclass(frozen=True)
class OutliersFlaggedSplitSummary:
    crop: str
    split: str
    input_path: str
    output_path: str
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    rule_outlier_rows: int
    tcn_outlier_rows: int
    ai_warning_rows: int
    ai_outlier_rows: int
    imputed_rows: int
    high_confidence_outlier_rows: int
    outlier_rows: int
    rule_outlier_cells: int
    imputed_cells: int


def _rule_outlier_cells(df: pd.DataFrame) -> int:
    flag_cols = [
        col
        for col in df.columns
        if col.endswith("_rule_outlier_flag")
        and col != DEFAULT_RULE_AGGREGATE_FLAG
    ]
    if not flag_cols:
        return 0
    return int(
        df[flag_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum()
        .sum()
    )


def _imputed_cells(df: pd.DataFrame) -> int:
    flag_cols = [
        col
        for col in df.columns
        if col.endswith("_outlier_imputed_flag")
        and col != DEFAULT_IMPUTED_FLAG_COLUMN
    ]
    if not flag_cols:
        return 0
    return int(
        df[flag_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum()
        .sum()
    )


def prepare_outliers_flagged_split_file(
    input_path: Path,
    output_path: Path,
    *,
    crop: str,
    split: str,
    domain_csv_path: Path = DEFAULT_DOMAIN_RANGE_PATH,
) -> OutliersFlaggedSplitSummary:
    """Prepare one unit-canonicalized split parquet and write flagged output."""

    df_input = pd.read_parquet(input_path)
    df_prepared = apply_outlier_flags(
        df_input,
        domain_csv_path=domain_csv_path,
        tcn_model=None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_prepared.to_parquet(output_path, index=False)

    return OutliersFlaggedSplitSummary(
        crop=crop,
        split=split,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=len(df_input),
        output_rows=len(df_prepared),
        input_columns=len(df_input.columns),
        output_columns=len(df_prepared.columns),
        rule_outlier_rows=flag_sum(df_prepared, DEFAULT_RULE_AGGREGATE_FLAG),
        tcn_outlier_rows=flag_sum(df_prepared, DEFAULT_TCN_FLAG_COLUMN),
        ai_warning_rows=flag_sum(df_prepared, DEFAULT_AI_WARNING_FLAG_COLUMN),
        ai_outlier_rows=flag_sum(df_prepared, DEFAULT_AI_OUTLIER_FLAG_COLUMN),
        imputed_rows=flag_sum(df_prepared, DEFAULT_IMPUTED_FLAG_COLUMN),
        high_confidence_outlier_rows=flag_sum(
            df_prepared,
            DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN,
        ),
        outlier_rows=flag_sum(df_prepared, DEFAULT_AGGREGATE_FLAG),
        rule_outlier_cells=_rule_outlier_cells(df_prepared),
        imputed_cells=_imputed_cells(df_prepared),
    )


def prepare_outliers_flagged_splits(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    crops: Iterable[str] | None = None,
    splits: Iterable[str] = DEFAULT_SPLITS,
    domain_csv_path: Path = DEFAULT_DOMAIN_RANGE_PATH,
) -> list[OutliersFlaggedSplitSummary]:
    """Prepare outlier-flagged intermediate parquet files for selected splits."""

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name

    crop_names = list(crops) if crops is not None else existing_crops(input_root)
    split_names = list(splits)
    summaries: list[OutliersFlaggedSplitSummary] = []

    for crop in crop_names:
        for split in split_names:
            input_path = input_root / crop / f"{split}.parquet"
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Unit-canonicalized parquet not found: {input_path}"
                )

            output_path = output_root / crop / f"{split}.parquet"
            summaries.append(
                prepare_outliers_flagged_split_file(
                    input_path,
                    output_path,
                    crop=crop,
                    split=split,
                    domain_csv_path=domain_csv_path,
                )
            )

    manifest_path = output_root / "outliers_flagged_manifest.json"
    manifest = {
        "stage": "outliers_flagged",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "domain_csv_path": str(domain_csv_path),
        "rule_aggregate_flag": DEFAULT_RULE_AGGREGATE_FLAG,
        "tcn_score_column": DEFAULT_TCN_SCORE_COLUMN,
        "tcn_warning_flag_column": DEFAULT_TCN_WARNING_FLAG_COLUMN,
        "tcn_flag_column": DEFAULT_TCN_FLAG_COLUMN,
        "tcn_pred_value_column": DEFAULT_TCN_PRED_VALUE_COLUMN,
        "ai_warning_flag": DEFAULT_AI_WARNING_FLAG_COLUMN,
        "ai_outlier_flag": DEFAULT_AI_OUTLIER_FLAG_COLUMN,
        "imputed_flag": DEFAULT_IMPUTED_FLAG_COLUMN,
        "high_confidence_outlier_flag": DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN,
        "aggregate_flag": DEFAULT_AGGREGATE_FLAG,
        "tcn_model": None,
        "splits": [asdict(summary) for summary in summaries],
    }
    write_json(manifest_path, manifest)
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare outlier-flagged parquet files from unit-canonicalized datasets."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summaries = prepare_outliers_flagged_splits(
        dataset_root=args.dataset_root,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        crops=args.crops,
        splits=args.splits,
        domain_csv_path=args.domain_csv_path,
    )

    for summary in summaries:
        print(
            f"[outliers_flagged] {summary.crop}/{summary.split}: "
            f"{summary.input_rows} -> {summary.output_rows} rows, "
            f"{summary.outlier_rows} outlier rows, wrote {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
