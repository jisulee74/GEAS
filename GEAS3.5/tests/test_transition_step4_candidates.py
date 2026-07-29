from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.config import load_transition_experiment_config
from geas35.models import transition
from geas35.models.transition import (
    ExtraTreesTransitionModel,
    PersistenceTransitionModel,
    TransitionDataset,
    build_transition_model,
    default_transition_model_registry,
)


OFFICIAL_CANDIDATES = (
    "persistence",
    "linear_regression",
    "linear_svr",
    "knn",
    "extra_trees",
    "lightgbm",
    "xgboost",
    "mlp",
)


def _dataset() -> TransitionDataset:
    x = pd.DataFrame(
        {
            "obs_indoor_temp_c": [20.0, 20.5, 21.0, 21.5],
            "obs_indoor_humidity_pct": [60.0, 61.0, 62.0, 63.0],
            "vent_pct": [0.0, 10.0, 20.0, 30.0],
        }
    )
    y = pd.DataFrame(
        {
            "obs_indoor_temp_c": [20.2, 20.7, 21.2, 21.7],
            "obs_indoor_humidity_pct": [60.5, 61.5, 62.5, 63.5],
        }
    )
    return TransitionDataset(
        x=x,
        y=y,
        input_columns=tuple(x.columns),
        target_columns=tuple(y.columns),
        observation_columns=(
            "obs_indoor_temp_c",
            "obs_indoor_humidity_pct",
        ),
        action_columns=("vent_pct",),
    )


def test_official_transition_registry_contains_exactly_eight_candidates() -> None:
    assert tuple(default_transition_model_registry()) == OFFICIAL_CANDIDATES


@pytest.mark.parametrize("removed_name", ("tcn", "catboost"))
def test_removed_transition_candidate_is_rejected(removed_name: str) -> None:
    with pytest.raises(ValueError, match=f"Unknown transition model: {removed_name}"):
        build_transition_model(removed_name)


def test_removed_transition_models_are_not_public_exports() -> None:
    assert not hasattr(transition, "TCNTransitionModel")
    assert not hasattr(transition, "CatBoostTransitionModel")


def test_default_config_uses_exactly_official_candidates() -> None:
    loaded = load_transition_experiment_config(
        PROJECT_ROOT
        / "experiments"
        / "transition_model_selection"
        / "configs"
        / "default.yaml"
    )
    assert tuple(
        candidate.model_name for candidate in loaded.experiment_config.models
    ) == OFFICIAL_CANDIDATES


def test_persistence_carries_dynamic_observations_forward() -> None:
    dataset = _dataset()
    model = PersistenceTransitionModel().fit(dataset)

    prediction = model.predict(dataset)

    assert prediction.target_columns == dataset.target_columns
    pd.testing.assert_frame_equal(
        prediction.next_observation,
        dataset.x.loc[:, list(dataset.target_columns)],
    )


def test_extra_trees_uses_independent_target_prediction_schema() -> None:
    dataset = _dataset()
    model = ExtraTreesTransitionModel(
        n_estimators=8,
        random_state=42,
    ).fit(dataset)

    prediction = model.predict(dataset)

    assert prediction.target_columns == dataset.target_columns
    assert tuple(prediction.next_observation.columns) == dataset.target_columns
    assert prediction.next_observation.shape == dataset.y.shape
