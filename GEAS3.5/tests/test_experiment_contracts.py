from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.quality.config import load_quality_experiment_config
from geas35.experiments.transition.config import load_transition_experiment_config
from synthetic_experiment_fixtures import (
    write_quality_smoke_fixture,
    write_transition_smoke_fixture,
)


@pytest.mark.parametrize(
    "script",
    (
        "experiments/quality_control_model_selection/scripts/run.py",
        "experiments/transition_model_selection/scripts/run.py",
        "offline_dataset_preparation/scripts/00_extract_raw.py",
        "offline_dataset_preparation/scripts/03_control_quality.py",
        "offline_dataset_preparation/scripts/04_prepare_rl_dataset.py",
    ),
)
def test_entrypoint_help_does_not_require_datasets(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_default_experiment_configs_load_without_reading_datasets() -> None:
    quality = load_quality_experiment_config(
        PROJECT_ROOT
        / "experiments"
        / "quality_control_model_selection"
        / "configs"
        / "cucumber.yaml"
    )
    transition = load_transition_experiment_config(
        PROJECT_ROOT
        / "experiments"
        / "transition_model_selection"
        / "configs"
        / "default.yaml"
    )

    assert quality.train_path.name == "train.parquet"
    assert quality.validation_path.name == "validation.parquet"
    assert quality.test_path.name == "test.parquet"
    assert quality.experiment_config.output_root == (
        PROJECT_ROOT
        / "experiments"
        / "quality_control_model_selection"
        / "artifacts"
        / "cucumber"
    )

    assert transition.train_path.name == "train.parquet"
    assert transition.validation_path.name == "validation.parquet"
    assert transition.test_path.name == "test.parquet"
    assert "5_rl_dataset" in transition.train_path.parts
    assert transition.experiment_config.output_root == (
        PROJECT_ROOT
        / "experiments"
        / "transition_model_selection"
        / "artifacts"
        / "default"
    )


def test_synthetic_smoke_fixtures_are_tmp_scoped(tmp_path: Path) -> None:
    quality = write_quality_smoke_fixture(tmp_path)
    transition = write_transition_smoke_fixture(tmp_path)
    tmp_root = tmp_path.resolve()

    generated_paths = [
        quality.config_path,
        quality.dataset_dir,
        quality.output_dir,
        *quality.split_paths.values(),
        transition.config_path,
        transition.dataset_dir,
        transition.output_dir,
        *transition.split_paths.values(),
    ]

    for path in generated_paths:
        path.resolve().relative_to(tmp_root)
