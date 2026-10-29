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

from geas35.models.transition import (
    LinearRegressionTransitionModel,
    PersistenceTransitionModel,
    ResidualTransitionAdapter,
    TransitionDataset,
    TransitionHPOConfig,
    TransitionTargetFormulation,
    load_transition_model_artifact,
    sample_transition_hpo_params,
    save_transition_model_artifact,
)
from geas35.models.transition.hpo_staged import build_local_hpo_params
from geas35.rl import MdpV1ObservationScaler


TARGETS = ("obs_indoor_temp_c", "obs_indoor_humidity_pct", "obs_indoor_co2_ppm")


def _scaler() -> MdpV1ObservationScaler:
    return MdpV1ObservationScaler(
        columns=TARGETS,
        means={name: value for name, value in zip(TARGETS, (20.0, 60.0, 500.0))},
        scales={name: value for name, value in zip(TARGETS, (2.0, 5.0, 100.0))},
    )


def test_residual_adapter_adds_physical_delta_after_inverse_scaling() -> None:
    scaler = _scaler()
    scaled = pd.DataFrame({name: [1.0] for name in TARGETS})
    delta = pd.DataFrame({name: [value] for name, value in zip(TARGETS, (0.5, -1.0, 10.0))})
    result = ResidualTransitionAdapter(scaler).to_physical_next_state(scaled, delta, TARGETS)
    assert result.iloc[0].to_dict() == {
        "obs_indoor_temp_c": 22.5,
        "obs_indoor_humidity_pct": 64.0,
        "obs_indoor_co2_ppm": 610.0,
    }


def test_zero_residual_equals_persistence() -> None:
    scaler = _scaler()
    x = pd.DataFrame({name: [0.0, 1.0, 2.0] for name in TARGETS})
    current = scaler.inverse_transform_frame(x)
    dataset = TransitionDataset(x=x, y=current.copy(), input_columns=TARGETS, target_columns=TARGETS)
    residual = LinearRegressionTransitionModel().configure_target_formulation(
        TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA,
        observation_scaler=scaler,
    ).fit(dataset).predict(dataset).next_observation
    persistence = PersistenceTransitionModel(observation_scaler=scaler).fit(dataset).predict(dataset).next_observation
    pd.testing.assert_frame_equal(residual, persistence, check_exact=False, atol=1e-12)


def test_artifact_loader_rejects_absolute_as_residual(tmp_path: Path) -> None:
    x = pd.DataFrame({name: [0.0, 1.0] for name in TARGETS})
    model = LinearRegressionTransitionModel().fit(
        TransitionDataset(x=x, y=x.copy(), input_columns=TARGETS, target_columns=TARGETS)
    )
    save_transition_model_artifact(model, tmp_path, "melon", model_name="linear_regression")
    with pytest.raises(ValueError, match="formulation mismatch"):
        load_transition_model_artifact(
            tmp_path,
            "melon",
            model_name="linear_regression",
            expected_target_formulation=TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value,
        )


def test_staged_hpo_has_unique_trials_and_preserves_n_jobs() -> None:
    config = TransitionHPOConfig(staged_nonlinear=True, nonlinear_budget=40)
    broad = sample_transition_hpo_params("extra_trees", {"n_jobs": 8}, config)
    local = build_local_hpo_params(broad[:3], count=20, seed=43, forbidden=broad[:20])
    assert len(broad) == 40 == len({repr(item) for item in broad})
    assert len(local) == 20 == len({repr(item) for item in local})
    assert not ({repr(item) for item in broad[:20]} & {repr(item) for item in local})
    assert {item["n_jobs"] for item in local} == {8}


def test_mlp_forwards_residual_contract_to_independent_estimators() -> None:
    from geas35.models.transition import MLPTransitionModel

    scaler = _scaler()
    x = pd.DataFrame({name: [0.0, 0.5, 1.0, 1.5] for name in TARGETS})
    current = scaler.inverse_transform_frame(x)
    y = current.add(pd.DataFrame({name: [1.0] * 4 for name in TARGETS}))
    dataset = TransitionDataset(x=x, y=y, input_columns=TARGETS, target_columns=TARGETS)
    model = MLPTransitionModel(
        hidden_layer_sizes=(4,), max_iter=5, random_state=42
    ).configure_target_formulation(
        TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA,
        observation_scaler=scaler,
    ).fit(dataset)
    assert model.independent_model_ is not None
    assert model.independent_model_.target_formulation == TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA.value
    assert model.independent_model_.observation_scaler is scaler
    assert model.predict(dataset).next_observation.shape == (4, 3)
