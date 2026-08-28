from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "offline_dataset_preparation/scripts/02_split.py"
SPEC = importlib.util.spec_from_file_location("geas35_offline_split", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _cycles() -> pd.DataFrame:
    return pd.DataFrame({
        "crop_cycle_id": [1, 2],
        "iot_data_idx": [97, 97],
        "crop": ["strawberry", "strawberry"],
        "transplant_date": ["2025-01-01 00:00", "2025-03-01 00:00"],
        "effective_crop_end_date": ["2025-01-31 23:55", "2025-03-31 23:55"],
    })


def test_crop_cycle_filter_is_inclusive_and_excludes_gap() -> None:
    frame = pd.DataFrame({
        "reg_date": pd.to_datetime([
            "2024-12-31 23:55", "2025-01-01 00:00", "2025-01-31 23:55",
            "2025-02-01 00:00", "2025-03-01 00:00", "2025-03-31 23:55",
            "2025-04-01 00:00",
        ]),
        "crop": ["strawberry"] * 7,
        "iot_data_idx": [97, 97, 97, 97, pd.NA, 97, 97],
    })
    filtered, audit = MODULE.filter_to_effective_crop_cycles(frame, _cycles())
    assert filtered["reg_date"].tolist() == pd.to_datetime([
        "2025-01-01 00:00", "2025-01-31 23:55",
        "2025-03-01 00:00", "2025-03-31 23:55",
    ]).tolist()
    assert filtered["crop_cycle_id"].tolist() == [1, 1, 2, 2]
    assert audit.loc[0, "excluded_outside_cycle_rows"] == 3
    assert audit.loc[0, "matched_crop_cycles"] == 2


def test_crop_cycle_filter_fails_on_overlapping_intervals() -> None:
    cycles = _cycles()
    cycles.loc[0, "effective_crop_end_date"] = "2025-03-02 00:00"
    frame = pd.DataFrame({
        "reg_date": pd.to_datetime(["2025-03-01 12:00"]),
        "crop": ["strawberry"], "iot_data_idx": [97],
    })
    with pytest.raises(ValueError, match="overlapping effective crop cycles"):
        MODULE.filter_to_effective_crop_cycles(frame, cycles)


def test_episode_split_remains_chronological_approximately_70_15_15() -> None:
    episodes = pd.DataFrame({
        "crop": ["strawberry"] * 20,
        "episode_start": pd.date_range("2025-01-01", periods=20, freq="D"),
        "is_valid_episode": [True] * 20,
    })
    result = MODULE.assign_splits(episodes)
    assert result["split"].value_counts().to_dict() == {
        "train": 14, "validation": 3, "test": 3,
    }
    assert result.loc[result["split"] == "train", "episode_start"].max() < result.loc[result["split"] == "validation", "episode_start"].min()
    assert result.loc[result["split"] == "validation", "episode_start"].max() < result.loc[result["split"] == "test", "episode_start"].min()
