"""Shared helpers for parquet-based offline runners."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def existing_crops(input_root: Path) -> list[str]:
    """Return crop folder names under an input dataset root."""

    return sorted(path.name for path in input_root.iterdir() if path.is_dir())


def jsonable(value: Any) -> Any:
    """Convert common numpy/path/dataclass objects to JSON-safe values."""

    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    return str(value)


def write_json(path: Path, payload: object, *, convert: bool = False) -> None:
    """Write a UTF-8 JSON file with the repository's manifest formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = jsonable(payload) if convert else payload
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def flag_sum(df: pd.DataFrame, column: str) -> int:
    """Return numeric sum of a single flag column, or 0 when absent."""

    if column not in df.columns:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def flag_cell_count(
    df: pd.DataFrame,
    suffix: str,
    *,
    exclude: set[str] | None = None,
    numeric_only: bool = False,
) -> int:
    """Return numeric sum across columns ending with ``suffix``."""

    excluded = set() if exclude is None else exclude
    flag_cols = [
        col
        for col in df.columns
        if col.endswith(suffix) and col not in excluded
    ]
    if not flag_cols:
        return 0

    values = df[flag_cols]
    if numeric_only:
        return int(values.sum(numeric_only=True).sum())
    return int(values.apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum())


__all__ = [
    "existing_crops",
    "flag_cell_count",
    "flag_sum",
    "jsonable",
    "write_json",
]
