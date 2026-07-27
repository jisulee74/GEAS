"""Prepare missing-value handled parquet files for offline split datasets.

This runner reads the intermediate files produced by
``prepare_input_schema_prepared_splits.py`` and writes the next intermediate
dataset after GEAS missing-value handling. It assumes ``reg_date`` and the 39
GEAS input features have already been prepared by
``1_input_schema_preparation.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from geas35.offline._common import existing_crops, flag_cell_count, write_json
from geas35.preprocessing import (
    ACTION_RESTORED_FLAG_SUFFIX,
    FEATURE_COLUMNS,
    MISSING_FLAG_SUFFIX,
    prepare_missing_features,
)


DEFAULT_DATASET_ROOT = Path("../datasets/iot97_historical")
DEFAULT_INPUT_DIR_NAME = "4_preprocessed/1_input_schema_prepared"
DEFAULT_OUTPUT_DIR_NAME = "4_preprocessed/2_missing_values_handled"
DEFAULT_SPLITS = ("train",)


@dataclass(frozen=True)
class MissingValuesHandledSplitSummary:
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
    unresolved_missing_feature_cells: int


def prepare_missing_split_file(
    input_path: Path,
    output_path: Path,
    *,
    crop: str,
    split: str,
    add_flags: bool = True,
) -> MissingValuesHandledSplitSummary:
    """Prepare one input-schema split parquet and write missing-handled output."""

    df_input = pd.read_parquet(input_path)
    df_prepared = prepare_missing_features(
        df_input,
        df_control_log=None,
        keep_extra_columns=True,
        add_flags=add_flags,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_prepared.to_parquet(output_path, index=False)

    unresolved_missing = int(df_prepared[FEATURE_COLUMNS].isna().sum().sum())

    return MissingValuesHandledSplitSummary(
        crop=crop,
        split=split,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=len(df_input),
        output_rows=len(df_prepared),
        input_columns=len(df_input.columns),
        output_columns=len(df_prepared.columns),
        missing_cells=flag_cell_count(
            df_prepared,
            MISSING_FLAG_SUFFIX,
            numeric_only=True,
        ),
        restored_action_cells=flag_cell_count(
            df_prepared,
            ACTION_RESTORED_FLAG_SUFFIX,
            numeric_only=True,
        ),
        unresolved_missing_feature_cells=unresolved_missing,
    )


def prepare_missing_values_handled_splits(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    crops: Iterable[str] | None = None,
    splits: Iterable[str] = DEFAULT_SPLITS,
    add_flags: bool = True,
) -> list[MissingValuesHandledSplitSummary]:
    """Prepare missing-handled intermediate parquet files for selected splits."""

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name

    crop_names = list(crops) if crops is not None else existing_crops(input_root)
    split_names = list(splits)
    summaries: list[MissingValuesHandledSplitSummary] = []

    for crop in crop_names:
        for split in split_names:
            input_path = input_root / crop / f"{split}.parquet"
            if not input_path.exists():
                raise FileNotFoundError(f"Input-schema parquet not found: {input_path}")

            output_path = output_root / crop / f"{split}.parquet"
            summaries.append(
                prepare_missing_split_file(
                    input_path,
                    output_path,
                    crop=crop,
                    split=split,
                    add_flags=add_flags,
                )
            )

    manifest_path = output_root / "missing_manifest.json"
    manifest = {
        "stage": "missing",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "feature_columns": FEATURE_COLUMNS,
        "add_flags": add_flags,
        "splits": [asdict(summary) for summary in summaries],
    }
    write_json(manifest_path, manifest)
    return summaries


prepare_missing_splits = prepare_missing_values_handled_splits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare missing-handled parquet files from input-schema datasets."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summaries = prepare_missing_values_handled_splits(
        dataset_root=args.dataset_root,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        crops=args.crops,
        splits=args.splits,
        add_flags=not args.no_flags,
    )

    for summary in summaries:
        print(
            f"[missing] {summary.crop}/{summary.split}: "
            f"{summary.input_rows} -> {summary.output_rows} rows, "
            f"{summary.missing_cells} missing cells flagged, "
            f"{summary.restored_action_cells} action cells restored, "
            f"wrote {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
