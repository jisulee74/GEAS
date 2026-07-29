"""Deterministic synthetic fixtures for local experiment smoke tests."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.quality.runner import ModelImplementation
from geas35.models.quality import MedianQualityModel
from geas35.models.quality.hyperparameters import CandidateTrainingResult
from geas35.rl import MDP_V1_ACTION_COLUMNS


QUALITY_OBSERVATION_COLUMNS = ("in_temp", "in_hum", "out_temp")
SYNTHETIC_CROP = "cucumber"


@dataclass(frozen=True)
class QualitySmokeFixture:
    config_path: Path
    dataset_dir: Path
    output_dir: Path
    split_paths: Mapping[str, Path]
    observation_columns: tuple[str, ...]


@dataclass(frozen=True)
class TransitionSmokeFixture:
    config_path: Path
    dataset_dir: Path
    output_dir: Path
    split_paths: Mapping[str, Path]
    crop: str


def write_quality_smoke_fixture(root: Path) -> QualitySmokeFixture:
    """Write a tiny train/validation/test quality experiment under ``root``."""

    dataset_dir = root / "quality_dataset"
    output_dir = root / "quality_artifacts"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    split_paths: dict[str, Path] = {}
    for split, offset in (("train", 0.0), ("validation", 0.25), ("test", 0.5)):
        path = dataset_dir / f"{split}.parquet"
        _quality_frame(offset=offset).to_parquet(path, index=False)
        split_paths[split] = path

    config_path = root / "quality_smoke.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  train: '{split_paths['train'].as_posix()}'",
                f"  validation: '{split_paths['validation'].as_posix()}'",
                f"  test: '{split_paths['test'].as_posix()}'",
                "  observation_columns: [in_temp, in_hum, out_temp]",
                "experiment:",
                f"  output_dir: '{output_dir.as_posix()}'",
                "  random_seed: 7",
                "hpo:",
                "  budget: 1",
                "  models: [median_smoke]",
                "  early_stopping:",
                "    monitor: validation_reconstruction_loss",
                "    mode: min",
                "    patience: 0",
                "    min_delta: 0.0",
                "  search_spaces:",
                "    common:",
                "      random_state:",
                "        value: 7",
                "    model_specific:",
                "      median_smoke: {}",
                "threshold:",
                "  enabled: true",
                "  candidate_generation: validation_reconstruction_error_linspace",
                "  candidate_count: 100",
                "evaluation:",
                "  mask_fraction: 0.2",
                "  anomaly_fraction: 0.2",
                "  anomaly_scale: 4.0",
                "  random_seed: 7",
                "  efficiency_repeats: 1",
                "report:",
                "  generate_figures: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return QualitySmokeFixture(
        config_path=config_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        split_paths=split_paths,
        observation_columns=QUALITY_OBSERVATION_COLUMNS,
    )


def median_smoke_registry() -> Mapping[str, ModelImplementation]:
    return {"median_smoke": _median_smoke_implementation()}


def write_transition_smoke_fixture(root: Path) -> TransitionSmokeFixture:
    """Write a tiny RL-ready transition experiment under ``root``."""

    dataset_dir = root / "transition_rl_dataset" / SYNTHETIC_CROP
    output_dir = root / "transition_artifacts"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    split_paths: dict[str, Path] = {}
    for split, offset in (("train", 0.0), ("validation", 0.25), ("test", 0.5)):
        path = dataset_dir / f"{split}.parquet"
        transition_rl_ready_frame(offset=offset).to_parquet(path, index=False)
        split_paths[split] = path

    config_path = root / "transition_smoke.yaml"
    write_transition_smoke_config(
        config_path,
        train_path=split_paths["train"],
        validation_path=split_paths["validation"],
        test_path=split_paths["test"],
        output_dir=output_dir,
        crop=SYNTHETIC_CROP,
    )

    return TransitionSmokeFixture(
        config_path=config_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        split_paths=split_paths,
        crop=SYNTHETIC_CROP,
    )


def write_transition_smoke_config(
    config_path: Path,
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    output_dir: Path,
    crop: str = SYNTHETIC_CROP,
) -> None:
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  crop: {crop}",
                f"  train: '{train_path.as_posix()}'",
                f"  validation: '{validation_path.as_posix()}'",
                f"  test: '{test_path.as_posix()}'",
                "experiment:",
                f"  output_dir: '{output_dir.as_posix()}'",
                "  random_seed: 11",
                "models:",
                "  candidates:",
                "    - name: linear_regression",
                "      params: {}",
                "selection:",
                "  strategy: mean_rmse",
                "  params: {}",
                "rollout:",
                "  enabled: false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def transition_rl_ready_frame(*, offset: float) -> pd.DataFrame:
    rows = 12
    index = np.arange(rows, dtype=float)
    indoor_temp = 20.0 + offset + index * 0.2
    indoor_humidity = 65.0 + offset + index * 0.3
    frame = pd.DataFrame(
        {
            "reg_date": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "crop": SYNTHETIC_CROP,
            "series_id": "synthetic_transition",
            "segment_id": "synthetic_transition_seg00",
            "episode_id": "synthetic_transition_ep00",
            "mdp_v1_rollout_id": 1,
            "mdp_v1_valid_transition": 1,
            "rl_valid_transition": 1,
            "done": 0,
            "obs_indoor_temp_c": indoor_temp,
            "obs_indoor_humidity_pct": indoor_humidity,
            "next_obs_indoor_temp_c": indoor_temp + 0.15,
            "next_obs_indoor_humidity_pct": indoor_humidity + 0.2,
            "target_next_indoor_temp_c": indoor_temp + 0.15,
            "target_next_indoor_humidity_pct": indoor_humidity + 0.2,
        }
    )
    for action_index, action_col in enumerate(MDP_V1_ACTION_COLUMNS):
        frame[action_col] = 0.1 * (action_index + 1) + index * 0.01
    return frame


def _median_smoke_implementation() -> ModelImplementation:
    def train_candidate(
        candidate,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        observation_columns,
        early_stopping,
    ) -> CandidateTrainingResult:
        del validation_df, early_stopping
        model = MedianQualityModel().fit(train_df, observation_columns)
        reconstruction = model.reconstruct(train_df, observation_columns)
        errors = (
            train_df.loc[:, list(observation_columns)].astype(float)
            - reconstruction.loc[:, list(observation_columns)].astype(float)
        ).abs()
        rmse = float(np.sqrt(np.nanmean(np.square(errors.to_numpy(dtype=float)))))
        mae = float(np.nanmean(errors.to_numpy(dtype=float)))
        return CandidateTrainingResult(
            candidate=candidate,
            validation_metrics={
                "validation_synthetic_masking_rmse": rmse,
                "validation_synthetic_masking_mae": mae,
                "validation_anomaly_pr_auc": 1.0,
            },
            training_history=(
                {
                    "epoch": 1,
                    "validation_reconstruction_loss": rmse,
                },
            ),
            model_artifact={"model_name": "median_smoke"},
        )

    def build_model(candidate) -> MedianQualityModel:
        del candidate
        return MedianQualityModel()

    return ModelImplementation(
        train_candidate=train_candidate,
        build_model=build_model,
    )


def _quality_frame(*, offset: float) -> pd.DataFrame:
    rows = 24
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "reg_date": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "crop": SYNTHETIC_CROP,
            "series_id": "synthetic_quality",
            "segment_id": "synthetic_quality_seg00",
            "episode_id": "synthetic_quality_ep00",
            "in_temp": 20.0 + offset + index * 0.1,
            "in_hum": 65.0 + offset + np.sin(index / 3.0),
            "out_temp": 12.0 + offset + index * 0.05,
        }
    )
