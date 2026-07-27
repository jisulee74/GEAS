"""Prepare unit-canonicalized parquet files for offline split datasets.

This runner reads input-schema intermediate files and writes the next dataset
after GEAS unit/scale canonicalization. It does not clip values and does not
attach QC flags.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from geas35.offline._common import existing_crops, write_json
from geas35.preprocessing import FEATURE_COLUMNS, PERCENT_ACTION_COLUMNS
from geas35.preprocessing import canonicalize_feature_units


DEFAULT_DATASET_ROOT = Path("../datasets/iot97_historical")
DEFAULT_INPUT_DIR_NAME = "4_preprocessed/1_input_schema_prepared"
DEFAULT_OUTPUT_DIR_NAME = "4_preprocessed/2_unit_canonicalized"
DEFAULT_SPLITS = ("train",)


@dataclass(frozen=True)
class UnitCanonicalizedSplitSummary:
    crop: str
    split: str
    input_path: str
    output_path: str
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    changed_percent_action_cells: int
    unresolved_missing_feature_cells: int


def _changed_percent_action_cells(before: pd.DataFrame, after: pd.DataFrame) -> int:
    changed = 0
    for col in PERCENT_ACTION_COLUMNS:
        if col not in before.columns or col not in after.columns:
            continue
        before_values = pd.to_numeric(before[col], errors="coerce")
        after_values = pd.to_numeric(after[col], errors="coerce")
        changed += int(
            (
                before_values.notna()
                & after_values.notna()
                & before_values.ne(after_values)
            ).sum()
        )
    return changed


def prepare_unit_canonicalized_split_file(
    input_path: Path,
    output_path: Path,
    *,
    crop: str,
    split: str,
) -> UnitCanonicalizedSplitSummary:
    """Prepare one input-schema split parquet and write canonicalized output."""

    df_input = pd.read_parquet(input_path)
    df_prepared = canonicalize_feature_units(df_input)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_prepared.to_parquet(output_path, index=False)

    unresolved_missing = int(df_prepared[FEATURE_COLUMNS].isna().sum().sum())

    return UnitCanonicalizedSplitSummary(
        crop=crop,
        split=split,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=len(df_input),
        output_rows=len(df_prepared),
        input_columns=len(df_input.columns),
        output_columns=len(df_prepared.columns),
        changed_percent_action_cells=_changed_percent_action_cells(
            df_input,
            df_prepared,
        ),
        unresolved_missing_feature_cells=unresolved_missing,
    )


def prepare_unit_canonicalized_splits(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    input_dir_name: str = DEFAULT_INPUT_DIR_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    crops: Iterable[str] | None = None,
    splits: Iterable[str] = DEFAULT_SPLITS,
) -> list[UnitCanonicalizedSplitSummary]:
    """Prepare unit-canonicalized intermediate parquet files for selected splits."""

    dataset_root = Path(dataset_root)
    input_root = dataset_root / input_dir_name
    output_root = dataset_root / output_dir_name

    crop_names = list(crops) if crops is not None else existing_crops(input_root)
    split_names = list(splits)
    summaries: list[UnitCanonicalizedSplitSummary] = []

    for crop in crop_names:
        for split in split_names:
            input_path = input_root / crop / f"{split}.parquet"
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Input-schema parquet not found: {input_path}"
                )

            output_path = output_root / crop / f"{split}.parquet"
            summaries.append(
                prepare_unit_canonicalized_split_file(
                    input_path,
                    output_path,
                    crop=crop,
                    split=split,
                )
            )

    manifest_path = output_root / "unit_canonicalized_manifest.json"
    manifest = {
        "stage": "unit_canonicalized",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "feature_columns": FEATURE_COLUMNS,
        "percent_action_columns": PERCENT_ACTION_COLUMNS,
        "splits": [asdict(summary) for summary in summaries],
    }
    write_json(manifest_path, manifest)
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare unit-canonicalized parquet files from input-schema datasets."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summaries = prepare_unit_canonicalized_splits(
        dataset_root=args.dataset_root,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        crops=args.crops,
        splits=args.splits,
    )

    for summary in summaries:
        print(
            f"[unit_canonicalized] {summary.crop}/{summary.split}: "
            f"{summary.input_rows} -> {summary.output_rows} rows, "
            f"{summary.changed_percent_action_cells} percent-action cells changed, "
            f"wrote {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
