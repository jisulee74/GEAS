from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.quality.config import (
    load_quality_experiment_config,
    read_configured_datasets,
)
from geas35.experiments.transition.config import (
    load_transition_experiment_config,
    read_configured_transition_frames,
)


def test_optional_quality_sliced_parquet_contract() -> None:
    config_path = _optional_config_path("GEAS35_QUALITY_SMOKE_CONFIG")

    loaded = load_quality_experiment_config(config_path)
    train_df, validation_df, test_df, observation_columns = read_configured_datasets(
        loaded
    )

    assert not train_df.empty
    assert not validation_df.empty
    assert not test_df.empty
    assert observation_columns


def test_optional_transition_sliced_parquet_contract() -> None:
    config_path = _optional_config_path("GEAS35_TRANSITION_SMOKE_CONFIG")

    loaded = load_transition_experiment_config(config_path)
    train_df, validation_df, test_df, _rollout_df = read_configured_transition_frames(
        loaded
    )

    assert not train_df.empty
    assert not validation_df.empty
    assert not test_df.empty


def _optional_config_path(env_var: str) -> Path:
    value = os.environ.get(env_var)
    if not value:
        pytest.skip(f"Set {env_var} to opt into sliced parquet contract validation.")
    return Path(value).expanduser()
