from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geas35.preprocessing import ACTION_COLUMNS, STATE_COLUMNS
from geas35.rl import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_OBSERVATION_COLUMNS,
    MDP_V1_TRANSITION_OBSERVATION_COLUMNS,
    MDP_V1_TRANSITION_TARGET_COLUMNS,
    MDP_V1_VALID_TRANSITION_COLUMN,
    MDP_V1_ZSCORE_OBSERVATION_COLUMNS,
    prepare_mdp_v1_frame,
)
from geas35.models.transition import (
    ForecastWeatherProvider,
    OFFICIAL_TRANSITION_TARGET_COLUMNS,
    RecordedWeatherProvider,
    TransitionDatasetBuilder,
    TransitionHPOConfig,
    TransitionTargetPolicy,
    LinearRegressionTransitionModel,
    sample_transition_hpo_params,
)
from geas35.models.transition.base import TransitionDataset
from geas35.models.transition.evaluator import evaluate_transition_predictions


def _qc_frame() -> pd.DataFrame:
    rows = 4
    data = {column: np.zeros(rows) for column in (*STATE_COLUMNS, *ACTION_COLUMNS)}
    data.update({
        "reg_date": pd.date_range("2026-06-01", periods=rows, freq="5min"),
        "crop": ["cucumber"] * rows,
        "series_id": ["series"] * rows,
        "segment_id": ["segment"] * rows,
        "episode_id": ["episode"] * rows,
        "growth_stage_dat": pd.Series([10] * rows, dtype="Int64"),
        "growth_stage_order": pd.Series([1] * rows, dtype="Int64"),
        "in_temp": [99.0] * rows,
        "in_hum": [1.0] * rows,
        "in_co2": [9999.0] * rows,
        "in_temp_representative": [22.0, 22.2, 22.4, 22.6],
        "in_hum_representative": [70.0, 70.5, 71.0, 71.5],
        "in_co2_representative": [500.0, 510.0, 520.0, 530.0],
        "out_temp": [15.0] * rows, "out_hum": [60.0] * rows,
        "out_light": [100.0] * rows, "out_light_sum": [1.0, 4.0, 7.0, 10.0],
        "out_windsp": [1.0] * rows, "out_rain": [0.0] * rows,
    })
    return pd.DataFrame(data)


def _rl_frame() -> pd.DataFrame:
    rows = 5
    frame = pd.DataFrame({
        "obs_indoor_temp_c": np.arange(rows) + 20.0,
        "obs_indoor_humidity_pct": np.arange(rows) + 60.0,
        "obs_indoor_co2_ppm": np.arange(rows) * 10.0 + 500.0,
        "next_obs_indoor_temp_c": np.arange(rows) + 20.1,
        "next_obs_indoor_humidity_pct": np.arange(rows) + 60.2,
        "next_obs_indoor_co2_ppm": np.arange(rows) * 10.0 + 502.0,
        "rl_valid_transition": [1] * rows,
    })
    for column in MDP_V1_ACTION_COLUMNS[:3]:
        frame[column] = 0.25
    for column in MDP_V1_ACTION_COLUMNS[3:]:
        frame[column] = 0.0
    return frame


def test_canonical_mdp_uses_representative_co2_and_three_targets() -> None:
    result = prepare_mdp_v1_frame(_qc_frame())
    assert MDP_V1_TRANSITION_OBSERVATION_COLUMNS == list(OFFICIAL_TRANSITION_TARGET_COLUMNS)
    assert "obs_indoor_co2_ppm" in MDP_V1_OBSERVATION_COLUMNS
    assert "obs_indoor_co2_ppm" in MDP_V1_ZSCORE_OBSERVATION_COLUMNS
    assert result["obs_indoor_co2_ppm"].tolist() == [500.0, 510.0, 520.0, 530.0]
    assert result["target_next_indoor_co2_ppm"].iloc[:3].tolist() == [510.0, 520.0, 530.0]
    assert MDP_V1_TRANSITION_TARGET_COLUMNS == [
        "target_next_indoor_temp_c", "target_next_indoor_humidity_pct",
        "target_next_indoor_co2_ppm",
    ]
    assert result[MDP_V1_VALID_TRANSITION_COLUMN].tolist() == [1, 1, 1, 0]


def test_official_dataset_builder_requires_exact_three_targets_and_action_ranges() -> None:
    result = TransitionDatasetBuilder().from_rl_dataset_frame(_rl_frame())
    assert result.dataset.target_columns == OFFICIAL_TRANSITION_TARGET_COLUMNS
    broken = _rl_frame().drop(columns=["next_obs_indoor_co2_ppm"])
    with pytest.raises(ValueError, match="three-target transition contract is incomplete"):
        TransitionDatasetBuilder().from_rl_dataset_frame(broken)
    invalid_action = _rl_frame()
    invalid_action["heat_run"] = 0.5
    with pytest.raises(ValueError, match="Binary action"):
        TransitionDatasetBuilder().from_rl_dataset_frame(invalid_action)


def test_two_target_support_is_explicitly_non_official() -> None:
    frame = _rl_frame().drop(columns=["obs_indoor_co2_ppm", "next_obs_indoor_co2_ppm"])
    policy = TransitionTargetPolicy.legacy_compatibility(
        ("obs_indoor_temp_c", "obs_indoor_humidity_pct")
    )
    result = TransitionDatasetBuilder(target_policy=policy).from_rl_dataset_frame(frame)
    assert result.dataset.target_columns == ("obs_indoor_temp_c", "obs_indoor_humidity_pct")


def test_co2_is_reported_per_target_and_in_macro_metrics() -> None:
    truth = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in OFFICIAL_TRANSITION_TARGET_COLUMNS})
    pred = truth + 0.5
    report = evaluate_transition_predictions(truth, pred)
    assert set(report.target_metrics) == set(OFFICIAL_TRANSITION_TARGET_COLUMNS)
    assert report.aggregate_metrics["indoor_co2_rmse"] == pytest.approx(0.5)
    assert "indoor_environment" in report.target_group_metrics


def test_independent_estimators_accept_target_specific_configuration() -> None:
    built = TransitionDatasetBuilder().from_rl_dataset_frame(_rl_frame()).dataset
    model = LinearRegressionTransitionModel(target_estimator_kwargs={
        "obs_indoor_temp_c": {"fit_intercept": False},
        "obs_indoor_co2_ppm": {"fit_intercept": True},
    }).fit(built)
    assert model.estimators_["obs_indoor_temp_c"].get_params()["fit_intercept"] is False
    assert model.estimators_["obs_indoor_co2_ppm"].get_params()["fit_intercept"] is True
    config = TransitionHPOConfig(budget=1, target_search_spaces={
        "linear_regression": {
            "obs_indoor_co2_ppm": {"fit_intercept": {"value": False}}
        }
    })
    params = sample_transition_hpo_params(
        "linear_regression", {}, config,
        target_columns=OFFICIAL_TRANSITION_TARGET_COLUMNS,
    )[0]
    assert params["target_estimator_kwargs"]["obs_indoor_co2_ppm"]["fit_intercept"] is False


def test_recorded_and_forecast_weather_providers_are_explicit() -> None:
    source = pd.DataFrame({
        "reg_date": pd.date_range("2026-06-01", periods=2, freq="5min"),
        "obs_outdoor_temp_c": [10.0, 11.0], "obs_outdoor_humidity_pct": [60.0, 61.0],
        "obs_outdoor_light": [100.0, 110.0], "obs_outdoor_wind_speed": [1.0, 1.1],
        "obs_rain_flag": [0.0, 0.0],
    })
    recorded = RecordedWeatherProvider().next_row(
        source_frame=source, next_source_position=1, recorded_next_row=source.iloc[1]
    )
    assert recorded["obs_outdoor_temp_c"] == 11.0
    forecast_frame = source.copy()
    forecast_frame.loc[1, "obs_outdoor_temp_c"] = 15.0
    provider = ForecastWeatherProvider(forecast_frame)
    provider.reset(None)  # reset does not consume rollout state
    forecast = provider.next_row(
        source_frame=source, next_source_position=1, recorded_next_row=source.iloc[1]
    )
    assert provider.mode == "forecast_weather"
    assert forecast["obs_outdoor_temp_c"] == 15.0
