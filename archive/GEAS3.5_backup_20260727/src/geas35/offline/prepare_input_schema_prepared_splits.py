"""Prepare GEAS input-schema parquet files for offline split datasets.

This runner reads split parquet files produced by ``datasets/iot97_historical``
and writes an intermediate dataset where each row conforms to the GEAS 39-feature
input schema. It does not impute missing values, apply QC flags, or generate
features.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from geas35.core import DEFAULT_GROWTH_STAGE_RULE_PATH
from geas35.offline._common import existing_crops, write_json
from geas35.preprocessing import (
    FEATURE_COLUMNS,
    GROWTH_STAGE_DAT_COLUMN,
    GROWTH_STAGE_ORDER_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    TIME_COLUMN,
    add_growth_stage_columns,
    growth_stage_rules_sha256,
    prepare_geas_input_schema,
)


DEFAULT_DATASET_ROOT = Path("../datasets/iot97_historical")
DEFAULT_INPUT_DIR_NAME = "3_splits"
DEFAULT_OUTPUT_DIR_NAME = "4_preprocessed/1_input_schema_prepared"
DEFAULT_SPLITS = ("train",)


@dataclass(frozen=True)
class InputSchemaPreparedSplitSummary:
    crop: str
    split: str
    input_path: str
    output_path: str
    input_rows: int
    output_rows: int
    dropped_invalid_time_rows: int
    input_columns: int
    output_columns: int
    added_feature_columns: list[str]
    growth_stage_dat_missing_rows: int
    growth_stage_order_missing_rows: int


def _load_series_transplant_dates(dataset_root: Path) -> dict[str, pd.Timestamp]:
    """Load series-level transplant/start dates from the raw dataset manifest."""

    manifest_path = dataset_root / "1_raw" / "manifest.csv"
    if not manifest_path.exists():
        return {}

    manifest = pd.read_csv(manifest_path)
    required_columns = {"series_id", "actual_start"}
    if not required_columns.issubset(manifest.columns):
        return {}

    mapping: dict[str, pd.Timestamp] = {}
    for row in manifest[["series_id", "actual_start"]].itertuples(index=False):
        timestamp = pd.to_datetime(row.actual_start, errors="coerce")
        if pd.notna(timestamp):
            mapping[str(row.series_id)] = timestamp
    return mapping


def prepare_split_file(
    input_path: Path,
    output_path: Path,
    *,
    crop: str,
    split: str,
    keep_extra_columns: bool = True,
    add_growth_stage: bool = True,
    transplant_date_by_series: Mapping[object, object] | None = None,
) -> InputSchemaPreparedSplitSummary:
    """Prepare one split parquet and write the intermediate parquet file."""

    df_raw = pd.read_parquet(input_path)
    added_feature_columns = [
        col for col in NUMERIC_FEATURE_COLUMNS if col not in df_raw.columns
    ]

    df_prepared = prepare_geas_input_schema(
        df_raw,
        keep_extra_columns=keep_extra_columns,
        drop_invalid_time=True,
        sort_by_time=True,
    )
    if add_growth_stage:
        df_prepared = add_growth_stage_columns(
            df_prepared,
            crop=crop,
            time_col=TIME_COLUMN,
            transplant_date_by_series=transplant_date_by_series,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_prepared.to_parquet(output_path, index=False)

    return InputSchemaPreparedSplitSummary(
        crop=crop,
        split=split,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=len(df_raw),
        output_rows=len(df_prepared),
        dropped_invalid_time_rows=len(df_raw) - len(df_prepared),
        input_columns=len(df_raw.columns),
        output_columns=len(df_prepared.columns),
        added_feature_columns=added_feature_columns,
        growth_stage_dat_missing_rows=(
            int(df_prepared[GROWTH_STAGE_DAT_COLUMN].isna().sum())
            if add_growth_stage
            else 0
        ),
        growth_stage_order_missing_rows=(
            int(df_prepared[GROWTH_STAGE_ORDER_COLUMN].isna().sum())
            if add_growth_stage
            else 0
        ),
    )


def prepare_input_schema_prepared_splits(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    crops: Iterable[str] | None = None,
    splits: Iterable[str] = DEFAULT_SPLITS,
    keep_extra_columns: bool = True,
    add_growth_stage: bool = True,
) -> list[InputSchemaPreparedSplitSummary]:
    """Prepare input-schema intermediate parquet files for selected splits."""

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name

    crop_names = list(crops) if crops is not None else existing_crops(input_root)
    split_names = list(splits)
    transplant_date_by_series = _load_series_transplant_dates(dataset_root)
    summaries: list[InputSchemaPreparedSplitSummary] = []

    for crop in crop_names:
        for split in split_names:
            input_path = input_root / crop / f"{split}.parquet"
            if not input_path.exists():
                raise FileNotFoundError(f"Split parquet not found: {input_path}")

            output_path = output_root / crop / f"{split}.parquet"
            summaries.append(
                prepare_split_file(
                    input_path,
                    output_path,
                    crop=crop,
                    split=split,
                    keep_extra_columns=keep_extra_columns,
                    add_growth_stage=add_growth_stage,
                    transplant_date_by_series=transplant_date_by_series,
                )
            )

    manifest_path = output_root / "input_schema_manifest.json"
    manifest = {
        "stage": "input_schema",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "feature_columns": FEATURE_COLUMNS,
        "keep_extra_columns": keep_extra_columns,
        "growth_stage_columns": (
            [GROWTH_STAGE_ORDER_COLUMN, GROWTH_STAGE_DAT_COLUMN]
            if add_growth_stage
            else []
        ),
        "growth_stage_name_storage": "configs/crops/growth_stage_rules.csv",
        "growth_stage_rules_path": str(DEFAULT_GROWTH_STAGE_RULE_PATH),
        "growth_stage_rules_sha256": (
            growth_stage_rules_sha256(DEFAULT_GROWTH_STAGE_RULE_PATH)
            if add_growth_stage
            else None
        ),
        "growth_stage_transplant_source": (
            str(dataset_root / "1_raw" / "manifest.csv")
            if transplant_date_by_series
            else "inferred_from_min_reg_date"
        ),
        "splits": [asdict(summary) for summary in summaries],
    }
    write_json(manifest_path, manifest)
    return summaries


prepare_input_schema_splits = prepare_input_schema_prepared_splits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare GEAS input-schema parquet files from split datasets."
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
        help="Input split directory name under dataset root.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help="Output directory name under dataset root.",
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
        "--drop-extra-columns",
        action="store_true",
        help="Write only the 39 GEAS feature columns.",
    )
    parser.add_argument(
        "--no-growth-stage-columns",
        action="store_true",
        help="Do not add growth_stage_order and growth_stage_dat columns.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summaries = prepare_input_schema_prepared_splits(
        dataset_root=args.dataset_root,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        crops=args.crops,
        splits=args.splits,
        keep_extra_columns=not args.drop_extra_columns,
        add_growth_stage=not args.no_growth_stage_columns,
    )

    for summary in summaries:
        print(
            f"[input_schema] {summary.crop}/{summary.split}: "
            f"{summary.input_rows} -> {summary.output_rows} rows, "
            f"wrote {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
