from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.cli import run_from_config
from geas35.experiments.transition.config import (
    load_transition_experiment_config,
    read_configured_transition_frames,
)
from geas35.models.transition import load_transition_candidate_model
from synthetic_experiment_fixtures import (
    write_transition_smoke_config,
    write_transition_smoke_fixture,
)


def test_transition_experiment_writes_candidate_artifacts_without_selection(
    tmp_path: Path,
) -> None:
    fixture = write_transition_smoke_fixture(tmp_path)
    result = run_from_config(fixture.config_path)

    candidate_dir = fixture.output_dir / fixture.crop / "linear_regression"
    summary = fixture.output_dir / "experiment_summary.json"

    assert candidate_dir.exists()
    assert not (fixture.output_dir / fixture.crop / "selected_transition_model.json").exists()
    assert not (
        fixture.output_dir / fixture.crop / "selected_transition_model_test_metrics.json"
    ).exists()
    assert summary.exists()

    loaded = load_transition_candidate_model(
        fixture.output_dir,
        fixture.crop,
        model_name="linear_regression",
    )
    assert loaded.model_name == "linear_regression"

    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload["automatic_model_selection"] is False
    assert summary_payload["test_used_for_selection"] is False
    assert "validation_ranking" in summary_payload
    assert "selected_model_name" not in summary_payload
    assert "selection_result" not in summary_payload
    assert "selected_test_evaluation" not in summary_payload


def test_transition_config_fails_fast_for_missing_rl_dataset(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_transition.yaml"
    missing_dir = tmp_path / "missing"
    write_transition_smoke_config(
        config_path,
        train_path=missing_dir / "train.parquet",
        validation_path=missing_dir / "validation.parquet",
        test_path=missing_dir / "test.parquet",
        output_dir=tmp_path / "out",
    )

    loaded = load_transition_experiment_config(config_path)
    with pytest.raises(FileNotFoundError) as exc:
        read_configured_transition_frames(loaded)

    message = str(exc.value)
    assert "5_rl_dataset" in message
    assert "03_control_quality.py" in message
    assert "04_prepare_rl_dataset.py" in message


def test_transition_config_fails_fast_for_non_rl_ready_columns(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "bad_dataset"
    dataset_dir.mkdir()
    bad_frame = pd.DataFrame(
        {
            "reg_date": pd.date_range("2024-01-01", periods=4, freq="5min"),
            "obs_indoor_temp_c": [20.0, 20.1, 20.2, 20.3],
        }
    )
    for split in ("train", "validation", "test"):
        bad_frame.to_parquet(dataset_dir / f"{split}.parquet", index=False)

    config_path = tmp_path / "bad_transition.yaml"
    write_transition_smoke_config(
        config_path,
        train_path=dataset_dir / "train.parquet",
        validation_path=dataset_dir / "validation.parquet",
        test_path=dataset_dir / "test.parquet",
        output_dir=tmp_path / "out",
    )

    loaded = load_transition_experiment_config(config_path)
    with pytest.raises(ValueError) as exc:
        read_configured_transition_frames(loaded)

    message = str(exc.value)
    assert "not RL-ready" in message
    assert "rl_valid_transition" in message
    assert "04_prepare_rl_dataset.py" in message
