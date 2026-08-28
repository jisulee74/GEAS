from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geas35.preprocessing import ACTION_COLUMNS, STATE_COLUMNS
from geas35.rl.datasets import (
    EXCLUSION_REASON_ORDER,
    MDP_V1_REQUIRED_SOURCE_ACTION_COLUMNS,
    _transition_exclusion_masks,
    prepare_rl_dataset_splits,
)
from geas35.rl.mdp_v1 import MDP_V1_TRANSITION_TARGET_COLUMNS


def _qc_frame(crop: str, offset: float = 0.0, rows: int = 8) -> pd.DataFrame:
    data = {column: np.zeros(rows) for column in (*STATE_COLUMNS, *ACTION_COLUMNS)}
    data.update({
        "reg_date": pd.date_range("2026-06-01", periods=rows, freq="5min"),
        "crop": [crop] * rows,
        "series_id": [f"{crop}_series"] * rows,
        "segment_id": [f"{crop}_segment"] * rows,
        "episode_id": [f"{crop}_episode"] * rows,
        "growth_stage_dat": pd.Series([10] * rows, dtype="Int64"),
        "growth_stage_order": pd.Series([1] * rows, dtype="Int64"),
        "in_temp_representative": np.arange(rows) * 0.2 + 20.0 + offset,
        "in_hum_representative": np.arange(rows) * 0.3 + 60.0 + offset,
        "in_co2_representative": np.arange(rows) * 5.0 + 500.0 + offset * 10.0,
        "out_temp": np.full(rows, 15.0 + offset),
        "out_hum": np.full(rows, 55.0 + offset),
        "out_light": np.full(rows, 100.0 + offset),
        "out_light_sum": np.arange(rows) + 1.0,
        "out_windsp": np.full(rows, 1.0),
        "out_rain": np.zeros(rows),
    })
    return pd.DataFrame(data)


def test_exclusion_masks_keep_reasons_separate() -> None:
    rows = 6
    frame = pd.DataFrame({
        "reg_date": pd.to_datetime([
            "2026-06-01 00:00", "2026-06-01 00:05", "2026-06-01 00:20",
            "2026-06-01 00:25", "2026-06-01 00:30", "2026-06-01 00:35",
        ]),
        "series_id": ["a", "a", "a", "b", "b", "b"],
        "obs": [1.0, 1.0, 1.0, np.nan, 1.0, 1.0],
        "target_next_indoor_temp_c": [1.0] * 5 + [np.nan],
        "target_next_indoor_humidity_pct": [1.0] * 5 + [np.nan],
        "target_next_indoor_co2_ppm": [1.0] * 5 + [np.nan],
    })
    for column in MDP_V1_REQUIRED_SOURCE_ACTION_COLUMNS:
        frame[column] = 0.0
    frame.loc[0, "cont_fan_run"] = np.nan
    masks = _transition_exclusion_masks(
        frame, ["obs"], MDP_V1_REQUIRED_SOURCE_ACTION_COLUMNS
    )
    assert tuple(masks) == EXCLUSION_REASON_ORDER
    assert masks["action_nan"].tolist() == [True, False, False, False, False, False]
    assert masks["time_discontinuity"].tolist() == [False, True, False, False, False, False]
    assert masks["episode_or_series_boundary"].tolist() == [False, False, True, False, False, False]
    assert masks["required_state_missing"].tolist() == [False, False, False, True, False, False]
    assert masks["target_missing"].tolist() == [False, False, False, False, False, True]


def test_step11_5_builds_train_scaled_three_crop_outputs(tmp_path: Path) -> None:
    input_root = tmp_path / "03_quality_controlled"
    crops = ("strawberry", "melon", "cucumber")
    offsets = {"train": 0.0, "validation": 10.0, "test": 20.0}
    for crop in crops:
        crop_root = input_root / crop
        crop_root.mkdir(parents=True)
        for split, offset in offsets.items():
            _qc_frame(crop, offset=offset).to_parquet(crop_root / f"{split}.parquet", index=False)

    summaries = prepare_rl_dataset_splits(dataset_root=tmp_path, crops=crops)
    assert len(summaries) == 9
    assert all(summary.output_rows > 0 for summary in summaries)
    assert all(summary.transition_target_columns == list(MDP_V1_TRANSITION_TARGET_COLUMNS) for summary in summaries)

    manifest = json.loads((tmp_path / "5_rl_dataset" / "rl_dataset_manifest.json").read_text())
    assert manifest["status"] == "passed"
    assert manifest["transition_target_columns"] == list(MDP_V1_TRANSITION_TARGET_COLUMNS)
    assert "code_version" not in manifest
    schemas = []
    for crop in crops:
        scaler = json.loads((tmp_path / "5_rl_dataset" / crop / "observation_scaler.json").read_text())
        assert scaler["fit_split"] == "train"
        assert len(scaler["raw_train"]["sha256"]) == 64
        assert set(scaler["columns"]) == set(scaler["mean"]) == set(scaler["scale"])
        train = pd.read_parquet(tmp_path / "5_rl_dataset" / crop / "train.parquet")
        validation = pd.read_parquet(tmp_path / "5_rl_dataset" / crop / "validation.parquet")
        assert train["obs_indoor_temp_c"].mean() == pytest.approx(0.0, abs=1e-12)
        assert abs(validation["obs_indoor_temp_c"].mean()) > 1.0
        assert validation["target_next_indoor_temp_c"].mean() > 20.0
        schemas.extend([tuple(train.columns), tuple(validation.columns)])
    assert len(set(schemas)) == 1


def test_step11_5_rejects_partial_split_sets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly train, validation, and test"):
        prepare_rl_dataset_splits(dataset_root=tmp_path, crops=[], splits=("train",))
