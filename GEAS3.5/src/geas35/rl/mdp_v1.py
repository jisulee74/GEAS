"""MDP v1 environment for GEAS offline RL experiments.

This module provides a dependency-light replay environment. It follows an
observed greenhouse trajectory and computes reward for a candidate action using
the observed next state. It is intended for feature-selection, transition-model,
and offline policy-evaluation experiments before a learned simulator exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from geas35.features import (
    DAYLIGHT_CONDITION_VALUES,
    MDP_DERIVED_CONTINUOUS_COLUMNS,
    MDP_DERIVED_OBSERVATION_COLUMNS,
    SOLAR_PERIOD_VALUES,
    add_mdp_context_features,
    add_mdp_derived_features,
)
from geas35.preprocessing import (
    GROWTH_STAGE_DAT_COLUMN,
    GROWTH_STAGE_ORDER_COLUMN,
    TIME_COLUMN,
    canonicalize_feature_units,
    prepare_geas_input_schema,
)

MDP_V1_MAX_GROWTH_STAGE_ORDER = 6
MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS = [
    f"obs_growth_stage_{stage_order}"
    for stage_order in range(1, MDP_V1_MAX_GROWTH_STAGE_ORDER + 1)
]
MDP_V1_SOLAR_PERIOD_ONEHOT_COLUMNS = [
    f"obs_solar_period_{value}"
    for value in SOLAR_PERIOD_VALUES
]
MDP_V1_DAYLIGHT_CONDITION_ONEHOT_COLUMNS = [
    f"obs_daylight_condition_{value}"
    for value in DAYLIGHT_CONDITION_VALUES
]

MDP_V1_OBSERVATION_COLUMNS = [
    "obs_indoor_temp_c",
    "obs_indoor_humidity_pct",
    "obs_outdoor_temp_c",
    "obs_outdoor_humidity_pct",
    "obs_outdoor_light",
    "obs_outdoor_wind_speed",
    "obs_rain_flag",
    "obs_hour_sin",
    "obs_hour_cos",
    "obs_is_daytime",
    *MDP_V1_SOLAR_PERIOD_ONEHOT_COLUMNS,
    *MDP_V1_DAYLIGHT_CONDITION_ONEHOT_COLUMNS,
    *MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS,
    "obs_growth_stage_dat",
    "obs_target_day_temp_c",
    "obs_target_night_temp_c",
    "obs_current_target_temp_min_c",
    "obs_current_target_temp_max_c",
    "obs_current_target_temp_c",
    "obs_current_vpd_kpa",
    "obs_current_dewpoint_c",
    "obs_current_condensation_margin_c",
    *MDP_DERIVED_OBSERVATION_COLUMNS,
    "obs_prev_vent_pct",
    "obs_prev_shade_curtain_pct",
    "obs_prev_thermal_curtain_pct",
    "obs_prev_heat_run",
    "obs_prev_cool_run",
    "obs_prev_fan_run",
]

MDP_V1_ACTION_COLUMNS = [
    "vent_pct",
    "shade_curtain_pct",
    "thermal_curtain_pct",
    "heat_run",
    "cool_run",
    "fan_run",
]

MDP_V1_CONTINUOUS_ACTION_COLUMNS = [
    "vent_pct",
    "shade_curtain_pct",
    "thermal_curtain_pct",
]

MDP_V1_BINARY_ACTION_COLUMNS = [
    "heat_run",
    "cool_run",
    "fan_run",
]

MDP_V1_TRANSITION_TARGET_COLUMNS = [
    "target_next_indoor_temp_c",
    "target_next_indoor_humidity_pct",
]

MDP_V1_TRANSITION_METRIC_KEYS = [
    "r2",
    "mae",
    "rmse",
    "q90",
    "cvar90",
]

MDP_V1_QUALITY_FLAG_PREFIX = "obs_quality_"
MDP_V1_AGGREGATE_QUALITY_FLAG_COLUMNS = [
    "missing_imputed_flag",
    "outlier_flag",
]
MDP_V1_QUALITY_FLAG_FALLBACKS = {
    "missing_imputed_flag": (
        "missing_imputed_flag",
        "imputed_flag",
        "outlier_imputed_flag",
    ),
    "outlier_flag": (
        "outlier_flag",
        "invalid_flag",
        "rule_outlier_flag",
        "ai_outlier_flag",
    ),
}
MDP_V1_VALID_TRANSITION_COLUMN = "mdp_v1_valid_transition"
MDP_V1_ROLLOUT_ID_COLUMN = "mdp_v1_rollout_id"
MDP_V1_STEP_MINUTES = 5.0
MDP_V1_REWARD_NORMALIZER_QUANTILE = 0.95

MDP_V1_ZSCORE_OBSERVATION_COLUMNS = [
    "obs_indoor_temp_c",
    "obs_indoor_humidity_pct",
    "obs_outdoor_temp_c",
    "obs_outdoor_humidity_pct",
    "obs_outdoor_light",
    "obs_outdoor_wind_speed",
    "obs_growth_stage_dat",
    "obs_target_day_temp_c",
    "obs_target_night_temp_c",
    "obs_current_target_temp_min_c",
    "obs_current_target_temp_max_c",
    "obs_current_target_temp_c",
    "obs_current_vpd_kpa",
    "obs_current_dewpoint_c",
    "obs_current_condensation_margin_c",
    *MDP_DERIVED_CONTINUOUS_COLUMNS,
]

MDP_V1_REWARD_TERM_KEYS = [
    "temp",
    "rh",
    "vpd",
    "cond",
    "hv",
    "act",
]


@dataclass(frozen=True)
class MdpV1Config:
    """Configuration for the GEAS MDP v1."""

    default_day_target_temp_c: float = 24.0
    default_night_target_temp_c: float = 12.0
    temp_band_c: float = 1.0
    vpd_min_kpa: float = 0.3
    vpd_max_kpa: float | None = None
    rh_max_pct: float = 90.0
    condensation_margin_min_c: float = 0.8
    heat_vent_alpha: float = 0.25
    temp_penalty_clip_c: float = 3.0
    rh_penalty_clip_pct: float = 10.0
    vpd_penalty_clip_kpa: float = 0.3
    cond_penalty_clip_c: float = 0.8
    hv_q_max: float = 5.0
    hv_excess_heat_fraction_clip: float = 0.5
    continuous_action_reversal_weight: float = 1.0
    binary_action_reversal_weight: float = 2.0
    physics_k_heat: float = 15000.0
    physics_rho_cp: float = 1200.0
    physics_greenhouse_volume_m3: float = 300.0
    physics_ach_base: float = 0.0
    physics_ach_vent_coef: float = 3.0
    physics_ach_wind_coef: float = 1.0
    physics_ach_max: float = 15.0
    rain_vent_open_cap: float = 0.0
    wind_cap_threshold: float = 5.0
    wind_vent_open_cap: float = 0.2
    daylight_light_threshold: float = 10.0
    latitude: float | None = None
    longitude: float | None = None
    timezone: str = "Asia/Seoul"
    early_night_hours: float = 6.0
    ramp_limit_pct: float = 15.0
    reward_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            key: 1.0 / len(MDP_V1_REWARD_TERM_KEYS)
            for key in MDP_V1_REWARD_TERM_KEYS
        }
    )


@dataclass(frozen=True)
class MdpV1ObservationScaler:
    """Train-split z-score scaler for MDP v1 continuous observations."""

    columns: tuple[str, ...]
    means: Mapping[str, float]
    scales: Mapping[str, float]

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in self.columns:
            if col not in out.columns:
                continue
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = (values - float(self.means[col])) / float(self.scales[col])
        return out


@dataclass(frozen=True)
class MdpV1RewardNormalizer:
    """Compatibility no-op for MDP v1 domain-bounded reward penalties.

    MDP v1 now scales each reward term by fixed domain thresholds inside the
    reward formula itself. The class is kept so older callers can still pass a
    normalizer object without changing the 0-1 bounded penalty semantics.
    """

    scales: Mapping[str, float]
    method: str = "domain_bounded"
    epsilon: float = 1e-8

    def normalize(self, raw_penalties: Mapping[str, float]) -> dict[str, float]:
        return {
            key: float(np.clip(value, 0.0, 1.0))
            for key, value in raw_penalties.items()
        }


def _sat_vapor_pressure_kpa(temp_c: pd.Series | np.ndarray | float) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))


def _vpd_kpa(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_c, errors="coerce")
    rh = pd.to_numeric(rh_pct, errors="coerce").clip(0.0, 100.0)
    es = _sat_vapor_pressure_kpa(temp)
    ea = es * (rh.to_numpy(dtype=float) / 100.0)
    return pd.Series(np.maximum(es - ea, 0.0), index=temp.index)


def _dewpoint_c(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_c, errors="coerce")
    rh = pd.to_numeric(rh_pct, errors="coerce").clip(1e-6, 100.0)
    a = 17.62
    b = 243.12
    gamma = np.log(rh.to_numpy(dtype=float) / 100.0) + (
        a * temp.to_numpy(dtype=float)
    ) / (b + temp.to_numpy(dtype=float))
    dewpoint = (b * gamma) / (a - gamma)
    return pd.Series(dewpoint, index=temp.index)


def _first_numeric(df: pd.DataFrame, columns: Sequence[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype="float64")
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _mean_numeric(df: pd.DataFrame, columns: Sequence[str], default: float = 0.0) -> pd.Series:
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series(default, index=df.index, dtype="float64")
    values = df[present].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1, skipna=True).fillna(default)


def _binary(series: pd.Series, threshold: float = 0.5) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce").fillna(0.0) > threshold).astype(float)


def _quality_flag_observation_name(column: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in column).strip("_")
    return f"{MDP_V1_QUALITY_FLAG_PREFIX}{safe}"


def _quality_flag_series(df: pd.DataFrame, canonical_column: str) -> pd.Series:
    for col in MDP_V1_QUALITY_FLAG_FALLBACKS.get(
        canonical_column,
        (canonical_column,),
    ):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(0.0, index=df.index, dtype="float64")


def mdp_v1_observation_columns(df: pd.DataFrame) -> list[str]:
    quality_cols = sorted(
        col for col in df.columns if col.startswith(MDP_V1_QUALITY_FLAG_PREFIX)
    )
    return [*MDP_V1_OBSERVATION_COLUMNS, *quality_cols]


def fit_mdp_v1_observation_scaler(
    df: pd.DataFrame,
    config: MdpV1Config | None = None,
    *,
    observation_columns: Sequence[str] | None = None,
) -> MdpV1ObservationScaler:
    """Fit a z-score scaler for continuous MDP v1 observations.

    Fit this only on the train split. Binary, one-hot, quality-flag, time sine/
    cosine, and already-bounded opening-ratio columns are intentionally excluded.
    """

    if "obs_indoor_temp_c" in df.columns and MDP_V1_VALID_TRANSITION_COLUMN in df.columns:
        frame = df
    else:
        frame = prepare_mdp_v1_frame(df, config=config)

    obs_cols = list(observation_columns or mdp_v1_observation_columns(frame))
    zscore_cols = [
        col
        for col in MDP_V1_ZSCORE_OBSERVATION_COLUMNS
        if col in frame.columns and col in obs_cols
    ]
    valid = (
        frame[MDP_V1_VALID_TRANSITION_COLUMN].astype(bool)
        if MDP_V1_VALID_TRANSITION_COLUMN in frame.columns
        else pd.Series(True, index=frame.index)
    )
    train_rows = frame.loc[valid, zscore_cols]

    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for col in zscore_cols:
        values = pd.to_numeric(train_rows[col], errors="coerce")
        mean = float(values.mean(skipna=True))
        scale = float(values.std(skipna=True, ddof=0))
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        means[col] = mean
        scales[col] = scale
    return MdpV1ObservationScaler(
        columns=tuple(zscore_cols),
        means=means,
        scales=scales,
    )


def _transition_metric_values(
    y_true: pd.Series | np.ndarray | Sequence[float],
    y_pred: pd.Series | np.ndarray | Sequence[float],
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred)
    if not valid.any():
        return {key: float("nan") for key in MDP_V1_TRANSITION_METRIC_KEYS}

    true = true[valid]
    pred = pred[valid]
    err = pred - true
    abs_err = np.abs(err)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float("nan") if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    q90 = float(np.quantile(abs_err, 0.9))
    tail = abs_err[abs_err >= q90]
    cvar90 = float(np.mean(tail)) if len(tail) else q90
    return {
        "r2": r2,
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "q90": q90,
        "cvar90": cvar90,
    }


def evaluate_mdp_v1_transition_predictions(
    y_true: pd.DataFrame | np.ndarray | Mapping[str, Sequence[float]],
    y_pred: pd.DataFrame | np.ndarray | Mapping[str, Sequence[float]],
    *,
    target_columns: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute transition-model prediction metrics per target column.

    Metrics include average errors (MAE/RMSE), explained variance (R2), and
    tail-error diagnostics (Q90/CVaR90). Q90 is the 90th percentile of absolute
    error. CVaR90 is the mean absolute error within the worst 10% tail.
    """

    true_df = pd.DataFrame(y_true)
    pred_df = pd.DataFrame(y_pred)

    if target_columns is None:
        target_columns = [
            col
            for col in true_df.columns
            if col in pred_df.columns
        ]
    if not target_columns:
        raise ValueError("No transition target columns are available for evaluation.")

    metrics: dict[str, dict[str, float]] = {}
    for col in target_columns:
        if col not in true_df.columns:
            raise KeyError(f"Missing true target column: {col}")
        if col not in pred_df.columns:
            raise KeyError(f"Missing predicted target column: {col}")
        metrics[str(col)] = _transition_metric_values(true_df[col], pred_df[col])
    return metrics


def _same_next_value(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    current = df[column].astype("string")
    next_value = current.shift(-1)
    return current.eq(next_value).fillna(False)


def _finite_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series(True, index=df.index)
    values = df[present].apply(pd.to_numeric, errors="coerce")
    return np.isfinite(values.to_numpy(dtype=float)).all(axis=1)


def _build_valid_transition_mask(df: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    step_minutes = timestamps.shift(-1).sub(timestamps).dt.total_seconds().div(60.0)
    valid = step_minutes.eq(MDP_V1_STEP_MINUTES).fillna(False)

    for group_col in ["crop", "series_id", "segment_id", "episode_id"]:
        valid &= _same_next_value(df, group_col)
    valid &= timestamps.dt.date.eq(timestamps.shift(-1).dt.date).fillna(False)

    observation_columns = mdp_v1_observation_columns(df)
    valid &= _finite_columns(df, observation_columns)
    valid &= _finite_columns(df, ["obs_growth_stage_order", "obs_growth_stage_dat"])
    valid &= _finite_columns(df, MDP_V1_TRANSITION_TARGET_COLUMNS)
    return valid.astype(bool)


def logged_mdp_v1_action(row: pd.Series | Mapping[str, object]) -> dict[str, float]:
    """Return the MDP v1 action representation from GEAS action columns."""

    data = pd.Series(row)

    def _num(key: str, default: float = 0.0) -> float:
        value = pd.to_numeric(pd.Series([data.get(key, default)]), errors="coerce").iloc[0]
        if pd.isna(value):
            return float(default)
        return float(value)

    vent_values = [
        _num("cont_skyl_vol", default=np.nan),
        _num("cont_skyr_vol", default=np.nan),
    ]
    valid_vent_values = [value for value in vent_values if np.isfinite(value)]
    vent_pct = float(np.mean(valid_vent_values)) if valid_vent_values else 0.0
    return {
        "vent_pct": float(np.clip(vent_pct, 0.0, 1.0)),
        "shade_curtain_pct": float(np.clip(_num("cont_cur_vol"), 0.0, 1.0)),
        "thermal_curtain_pct": float(np.clip(_num("cont_kwcur_vol"), 0.0, 1.0)),
        "heat_run": float(_num("cont_heater_run") > 0.5),
        "cool_run": float(_num("cont_cooler_run") > 0.5),
        "fan_run": float(_num("cont_fan_run") > 0.5),
    }


def _normalize_action(action: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(action, Mapping):
        values = {key: float(action.get(key, 0.0)) for key in MDP_V1_ACTION_COLUMNS}
    else:
        if len(action) != len(MDP_V1_ACTION_COLUMNS):
            raise ValueError(
                f"Action must contain {len(MDP_V1_ACTION_COLUMNS)} values: "
                f"{MDP_V1_ACTION_COLUMNS}"
            )
        values = {key: float(value) for key, value in zip(MDP_V1_ACTION_COLUMNS, action)}

    for key in MDP_V1_CONTINUOUS_ACTION_COLUMNS:
        values[key] = float(np.clip(values[key], 0.0, 1.0))
    for key in MDP_V1_BINARY_ACTION_COLUMNS:
        values[key] = float(values[key] > 0.5)
    return values


def _project_action_for_weather_constraints(
    action: Mapping[str, float],
    row: pd.Series | Mapping[str, object],
    config: MdpV1Config,
) -> tuple[dict[str, float], dict[str, float | bool]]:
    projected = dict(action)
    rain = pd.to_numeric(pd.Series([row.get("obs_rain_flag", 0.0)]), errors="coerce").iloc[0]
    wind = pd.to_numeric(
        pd.Series([row.get("obs_outdoor_wind_speed", np.nan)]),
        errors="coerce",
    ).iloc[0]
    rain_active = bool(pd.notna(rain) and float(rain) > 0.0)
    wind_active = bool(
        pd.notna(wind)
        and np.isfinite(float(wind))
        and float(wind) > config.wind_cap_threshold
    )

    vent_cap = 1.0
    if rain_active:
        vent_cap = min(vent_cap, config.rain_vent_open_cap)
    if wind_active:
        vent_cap = min(vent_cap, config.wind_vent_open_cap)
    vent_cap = float(np.clip(vent_cap, 0.0, 1.0))
    projected["vent_pct"] = float(np.clip(projected["vent_pct"], 0.0, vent_cap))

    return projected, {
        "weather_constraint_applied": bool(rain_active or wind_active),
        "rain_vent_constraint_applied": rain_active,
        "wind_vent_constraint_applied": wind_active,
        "vent_open_cap": vent_cap,
    }


def _act_stability_penalty(
    action: Mapping[str, float],
    prev_action: Mapping[str, float],
    prev_prev_action: Mapping[str, float],
    config: MdpV1Config,
) -> float:
    weighted_reversal_sum = 0.0
    weight_sum = 0.0

    for key in MDP_V1_CONTINUOUS_ACTION_COLUMNS:
        prev_delta = prev_action[key] - prev_prev_action[key]
        current_delta = action[key] - prev_action[key]
        term = 0.0
        if prev_delta * current_delta < 0.0:
            term = (abs(current_delta) + abs(prev_delta)) / 2.0
        weighted_reversal_sum += config.continuous_action_reversal_weight * term
        weight_sum += config.continuous_action_reversal_weight

    for key in MDP_V1_BINARY_ACTION_COLUMNS:
        prev_delta = prev_action[key] - prev_prev_action[key]
        current_delta = action[key] - prev_action[key]
        term = 1.0 if prev_delta * current_delta < 0.0 else 0.0
        weighted_reversal_sum += config.binary_action_reversal_weight * term
        weight_sum += config.binary_action_reversal_weight

    if weight_sum <= 0.0:
        return 0.0
    return float(np.clip(weighted_reversal_sum / weight_sum, 0.0, 1.0))


def _bounded_penalty(value: float, scale: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return float(np.clip(value, 0.0, 1.0))
    return float(np.clip(value / scale, 0.0, 1.0))


def _reward_penalty_label(key: str) -> str:
    return f"{key}_penalty"


def _target_temp_for_row(row: pd.Series, config: MdpV1Config) -> float:
    if pd.notna(row.get("obs_current_target_temp_c", np.nan)):
        return float(row["obs_current_target_temp_c"])
    if pd.notna(row.get("current_target_temp_c", np.nan)):
        return float(row["current_target_temp_c"])
    if bool(row.get("obs_is_daytime", False)):
        value = row.get("obs_target_day_temp_c", config.default_day_target_temp_c)
        return float(value) if pd.notna(value) else config.default_day_target_temp_c
    value = row.get("obs_target_night_temp_c", config.default_night_target_temp_c)
    return float(value) if pd.notna(value) else config.default_night_target_temp_c


def _target_temp_bounds_for_row(
    row: pd.Series,
    config: MdpV1Config,
) -> tuple[float, float, float]:
    target = _target_temp_for_row(row, config)
    lower = row.get("obs_current_target_temp_min_c", np.nan)
    upper = row.get("obs_current_target_temp_max_c", np.nan)
    lower = float(lower) if pd.notna(lower) else target - config.temp_band_c
    upper = float(upper) if pd.notna(upper) else target + config.temp_band_c
    if not np.isfinite(lower):
        lower = target - config.temp_band_c
    if not np.isfinite(upper):
        upper = target + config.temp_band_c
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper, target


def _heat_vent_physics_for_row(
    row: pd.Series,
    action: Mapping[str, float],
    config: MdpV1Config,
) -> dict[str, float]:
    tin = float(row.get("obs_indoor_temp_c", np.nan))
    tout = float(row.get("obs_outdoor_temp_c", np.nan))
    wind = float(row.get("obs_outdoor_wind_speed", 0.0))
    if not np.isfinite(wind):
        wind = 0.0

    ach = (
        config.physics_ach_base
        + config.physics_ach_vent_coef * action["vent_pct"]
        + config.physics_ach_wind_coef * wind * action["vent_pct"]
    )
    ach = float(np.clip(ach, 0.0, config.physics_ach_max))
    q_heat = float(config.physics_k_heat * action["heat_run"])
    q_ventloss = 0.0
    if np.isfinite(tin) and np.isfinite(tout):
        q_ventloss = float(
            max(
                0.0,
                config.physics_rho_cp
                * config.physics_greenhouse_volume_m3
                * (ach / 3600.0)
                * max(0.0, tin - tout),
            )
        )

    threshold = config.heat_vent_alpha * q_heat
    if (
        action["heat_run"] <= 0.5
        or q_heat <= config.hv_q_max
        or threshold <= 0.0
    ):
        excess_ratio = 0.0
        excess_heat_fraction = 0.0
    else:
        q_excess = max(q_ventloss - threshold, 0.0)
        excess_ratio = q_excess / max(threshold, 1e-6)
        excess_heat_fraction = q_excess / max(q_heat, 1e-6)

    return {
        "ach": float(ach),
        "q_heat": float(q_heat),
        "q_ventloss": float(q_ventloss),
        "heat_vent_threshold": float(threshold),
        "heat_vent_excess_ratio": float(excess_ratio),
        "heat_vent_excess_heat_fraction": float(excess_heat_fraction),
    }


def _compute_mdp_v1_raw_penalties(
    next_row: pd.Series,
    action: Mapping[str, float] | Sequence[float],
    *,
    prev_action: Mapping[str, float] | None = None,
    prev_prev_action: Mapping[str, float] | None = None,
    config: MdpV1Config | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    config = config or MdpV1Config()
    action_dict = _normalize_action(action)
    prev_action_dict = (
        _normalize_action(prev_action)
        if prev_action is not None
        else {key: 0.0 for key in MDP_V1_ACTION_COLUMNS}
    )
    prev_prev_action_dict = (
        _normalize_action(prev_prev_action)
        if prev_prev_action is not None
        else dict(prev_action_dict)
    )

    target_temp_min, target_temp_max, target_temp = _target_temp_bounds_for_row(
        next_row,
        config,
    )
    temp = float(next_row.get("obs_indoor_temp_c", np.nan))
    rh = float(next_row.get("obs_indoor_humidity_pct", np.nan))
    vpd = float(next_row.get("obs_current_vpd_kpa", np.nan))
    cond_margin = float(next_row.get("obs_current_condensation_margin_c", np.nan))

    temp_error = (
        0.0
        if not np.isfinite(temp)
        else max(target_temp_min - temp, 0.0) + max(temp - target_temp_max, 0.0)
    )
    rh_excess = 0.0 if not np.isfinite(rh) else max(rh - config.rh_max_pct, 0.0)
    vpd_error = 0.0 if not np.isfinite(vpd) else max(config.vpd_min_kpa - vpd, 0.0)
    delta_t_cond = (
        0.0
        if not np.isfinite(cond_margin)
        else max(config.condensation_margin_min_c - cond_margin, 0.0)
    )
    hv_terms = _heat_vent_physics_for_row(next_row, action_dict, config)
    act_penalty = _act_stability_penalty(
        action_dict,
        prev_action_dict,
        prev_prev_action_dict,
        config,
    )

    raw_penalties = {
        "temp": _bounded_penalty(temp_error, config.temp_penalty_clip_c),
        "rh": _bounded_penalty(
            rh_excess,
            config.rh_penalty_clip_pct,
        ),
        "vpd": _bounded_penalty(vpd_error, config.vpd_penalty_clip_kpa),
        "cond": _bounded_penalty(
            delta_t_cond,
            config.cond_penalty_clip_c,
        ),
        "hv": _bounded_penalty(
            hv_terms["heat_vent_excess_heat_fraction"],
            config.hv_excess_heat_fraction_clip,
        ),
        "act": act_penalty,
    }
    metadata = {
        "target_temp_min_c": float(target_temp_min),
        "target_temp_max_c": float(target_temp_max),
        "target_temp_c": float(target_temp),
        "temp_violation_c": float(temp_error),
        "rh_excess_pct": float(rh_excess),
        "vpd_deficit_kpa": float(vpd_error),
        "delta_t_cond_c": float(delta_t_cond),
        "heat_vent_ach": hv_terms["ach"],
        "heat_vent_q_heat": hv_terms["q_heat"],
        "heat_vent_q_ventloss": hv_terms["q_ventloss"],
        "heat_vent_q_max": float(config.hv_q_max),
        "heat_vent_threshold": hv_terms["heat_vent_threshold"],
        "heat_vent_excess_ratio": hv_terms["heat_vent_excess_ratio"],
        "heat_vent_excess_heat_fraction": hv_terms[
            "heat_vent_excess_heat_fraction"
        ],
        "act_unbounded": float(act_penalty),
    }
    return raw_penalties, metadata


def fit_mdp_v1_reward_normalizer(
    df: pd.DataFrame,
    config: MdpV1Config | None = None,
    *,
    quantile: float = MDP_V1_REWARD_NORMALIZER_QUANTILE,
    min_scale: float = 1e-6,
) -> MdpV1RewardNormalizer:
    """Return the domain-bounded reward normalizer compatibility object.

    MDP v1 reward terms are already scaled by fixed domain thresholds and
    clipped to 0-1 inside the reward function. The data argument and quantile
    parameter are accepted for backward compatibility with earlier scripts.
    """

    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in the interval (0, 1].")
    return MdpV1RewardNormalizer(
        scales={key: 1.0 for key in MDP_V1_REWARD_TERM_KEYS},
        method="domain_bounded",
    )


def prepare_mdp_v1_frame(
    df_raw: pd.DataFrame,
    config: MdpV1Config | None = None,
) -> pd.DataFrame:
    """Create observation and transition-target columns for the MDP v1."""

    config = config or MdpV1Config()
    if df_raw is None:
        return pd.DataFrame()

    df = prepare_geas_input_schema(df_raw, keep_extra_columns=True)
    df = canonicalize_feature_units(df)
    if df.empty:
        return df

    t = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    hours = t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0
    hour_angle = 2.0 * np.pi * hours / 24.0
    light = pd.to_numeric(df.get("out_light"), errors="coerce")
    fallback_daytime = (hours >= 6.0) & (hours < 18.0)
    is_daytime = np.where(
        light.notna(),
        light >= config.daylight_light_threshold,
        fallback_daytime,
    )

    indoor_temp = pd.to_numeric(df["in_temp"], errors="coerce")
    indoor_humidity = pd.to_numeric(df["in_hum"], errors="coerce")
    vpd = _vpd_kpa(indoor_temp, indoor_humidity)
    dewpoint = _dewpoint_c(indoor_temp, indoor_humidity)
    cond_margin = indoor_temp - dewpoint

    df["obs_indoor_temp_c"] = indoor_temp
    df["obs_indoor_humidity_pct"] = indoor_humidity
    df["obs_outdoor_temp_c"] = pd.to_numeric(df["out_temp"], errors="coerce")
    df["obs_outdoor_humidity_pct"] = pd.to_numeric(df["out_hum"], errors="coerce")
    df["obs_outdoor_light"] = light
    df["obs_outdoor_wind_speed"] = pd.to_numeric(df["out_windsp"], errors="coerce")
    df["obs_rain_flag"] = pd.to_numeric(df["out_rain"], errors="coerce").fillna(0.0)
    df["obs_hour_sin"] = np.sin(hour_angle)
    df["obs_hour_cos"] = np.cos(hour_angle)
    df["obs_is_daytime"] = pd.Series(is_daytime, index=df.index).astype(float)
    df = add_mdp_context_features(
        df,
        time_col=TIME_COLUMN,
        lat=config.latitude,
        lon=config.longitude,
        tz=config.timezone,
        early_night_hours=config.early_night_hours,
        prefix="obs",
    )
    growth_stage_order = (
        pd.to_numeric(df[GROWTH_STAGE_ORDER_COLUMN], errors="coerce")
        if GROWTH_STAGE_ORDER_COLUMN in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    df["obs_growth_stage_order"] = growth_stage_order
    for stage_order, col in enumerate(MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS, start=1):
        df[col] = (growth_stage_order == stage_order).astype(float)
    df["obs_growth_stage_dat"] = (
        pd.to_numeric(df[GROWTH_STAGE_DAT_COLUMN], errors="coerce")
        if GROWTH_STAGE_DAT_COLUMN in df.columns
        else np.nan
    )
    df["obs_target_day_temp_c"] = _first_numeric(
        df,
        ["target_day_temp_c", "day_target_temp_c"],
        default=config.default_day_target_temp_c,
    )
    df["obs_target_night_temp_c"] = _first_numeric(
        df,
        ["target_night_temp_c", "night_target_temp_c"],
        default=config.default_night_target_temp_c,
    )
    current_target = _first_numeric(
        df,
        ["current_target_temp_c", "target_temp_c"],
        default=np.nan,
    )
    day_min = _first_numeric(
        df,
        ["target_day_temp_min_c", "day_target_temp_min_c", "day_temp_min_c"],
        default=np.nan,
    )
    day_max = _first_numeric(
        df,
        ["target_day_temp_max_c", "day_target_temp_max_c", "day_temp_max_c"],
        default=np.nan,
    )
    night_min = _first_numeric(
        df,
        ["target_night_temp_min_c", "night_target_temp_min_c", "night_temp_min_c"],
        default=np.nan,
    )
    night_max = _first_numeric(
        df,
        ["target_night_temp_max_c", "night_target_temp_max_c", "night_temp_max_c"],
        default=np.nan,
    )
    current_min = _first_numeric(
        df,
        [
            "current_target_temp_min_c",
            "target_temp_min_c",
            "current_temp_min_c",
        ],
        default=np.nan,
    )
    current_max = _first_numeric(
        df,
        [
            "current_target_temp_max_c",
            "target_temp_max_c",
            "current_temp_max_c",
        ],
        default=np.nan,
    )
    day_mask = pd.Series(is_daytime, index=df.index).astype(bool)
    period_min = day_min.where(day_mask, night_min)
    period_max = day_max.where(day_mask, night_max)
    period_target = df["obs_target_day_temp_c"].where(day_mask, df["obs_target_night_temp_c"])
    current_target = current_target.where(current_target.notna(), period_target)
    df["obs_current_target_temp_c"] = current_target
    df["obs_current_target_temp_min_c"] = current_min.where(
        current_min.notna(),
        period_min,
    )
    df["obs_current_target_temp_max_c"] = current_max.where(
        current_max.notna(),
        period_max,
    )
    df["obs_current_target_temp_min_c"] = df["obs_current_target_temp_min_c"].where(
        df["obs_current_target_temp_min_c"].notna(),
        df["obs_current_target_temp_c"] - config.temp_band_c,
    )
    df["obs_current_target_temp_max_c"] = df["obs_current_target_temp_max_c"].where(
        df["obs_current_target_temp_max_c"].notna(),
        df["obs_current_target_temp_c"] + config.temp_band_c,
    )
    df["obs_current_vpd_kpa"] = vpd
    df["obs_current_dewpoint_c"] = dewpoint
    df["obs_current_condensation_margin_c"] = cond_margin

    df["_mdp_v1_logged_vent_pct"] = _mean_numeric(df, ["cont_skyl_vol", "cont_skyr_vol"])
    df["_mdp_v1_logged_shade_curtain_pct"] = pd.to_numeric(
        df["cont_cur_vol"], errors="coerce"
    ).fillna(0.0)
    df["_mdp_v1_logged_thermal_curtain_pct"] = pd.to_numeric(
        df["cont_kwcur_vol"], errors="coerce"
    ).fillna(0.0)
    df["_mdp_v1_logged_heat_run"] = _binary(df["cont_heater_run"])
    df["_mdp_v1_logged_cool_run"] = _binary(df["cont_cooler_run"])
    df["_mdp_v1_logged_fan_run"] = _binary(df["cont_fan_run"])
    df = add_mdp_derived_features(
        df,
        time_col=TIME_COLUMN,
        condensation_margin_min_c=config.condensation_margin_min_c,
        rh_high_pct=config.rh_max_pct,
        ramp_limit_pct=config.ramp_limit_pct,
    )

    quality_flags = {
        _quality_flag_observation_name(col): _quality_flag_series(df, col)
        for col in MDP_V1_AGGREGATE_QUALITY_FLAG_COLUMNS
    }
    df = pd.concat([df, pd.DataFrame(quality_flags, index=df.index)], axis=1)

    for action_col in MDP_V1_ACTION_COLUMNS:
        logged_col = f"_mdp_v1_logged_{action_col}"
        df[f"obs_prev_{action_col}"] = df[logged_col].shift(1).fillna(df[logged_col])

    df["target_next_indoor_temp_c"] = df["obs_indoor_temp_c"].shift(-1)
    df["target_next_indoor_humidity_pct"] = df["obs_indoor_humidity_pct"].shift(-1)
    df[MDP_V1_VALID_TRANSITION_COLUMN] = _build_valid_transition_mask(df).astype(int)
    invalid_previous = ~df[MDP_V1_VALID_TRANSITION_COLUMN].shift(1).fillna(0).astype(bool)
    df[MDP_V1_ROLLOUT_ID_COLUMN] = invalid_previous.cumsum().astype(int)
    return df


def compute_mdp_v1_reward(
    next_row: pd.Series,
    action: Mapping[str, float] | Sequence[float],
    *,
    prev_action: Mapping[str, float] | None = None,
    prev_prev_action: Mapping[str, float] | None = None,
    config: MdpV1Config | None = None,
    reward_normalizer: MdpV1RewardNormalizer | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute MDP v1 reward from the observed next row and chosen action."""

    config = config or MdpV1Config()
    weights = config.reward_weights
    raw_penalties, metadata = _compute_mdp_v1_raw_penalties(
        next_row,
        action,
        prev_action=prev_action,
        prev_prev_action=prev_prev_action,
        config=config,
    )
    normalized_penalties = (
        reward_normalizer.normalize(raw_penalties)
        if reward_normalizer is not None
        else dict(raw_penalties)
    )

    terms: dict[str, float] = {}
    for key in MDP_V1_REWARD_TERM_KEYS:
        label = _reward_penalty_label(key)
        terms[f"raw_{label}"] = float(raw_penalties[key])
        terms[f"normalized_{label}"] = float(normalized_penalties[key])
        terms[label] = float(weights[key] * normalized_penalties[key])
    terms.update(metadata)
    reward = -float(sum(terms[_reward_penalty_label(key)] for key in MDP_V1_REWARD_TERM_KEYS))
    return reward, terms


class MdpV1Env:
    """Gym-like replay environment for the GEAS MDP v1."""

    observation_columns = MDP_V1_OBSERVATION_COLUMNS
    action_columns = MDP_V1_ACTION_COLUMNS
    transition_target_columns = MDP_V1_TRANSITION_TARGET_COLUMNS

    def __init__(
        self,
        df_raw: pd.DataFrame,
        config: MdpV1Config | None = None,
        observation_scaler: MdpV1ObservationScaler | None = None,
        reward_normalizer: MdpV1RewardNormalizer | None = None,
    ) -> None:
        self.config = config or MdpV1Config()
        self.observation_scaler = observation_scaler
        self.reward_normalizer = reward_normalizer
        self.frame = prepare_mdp_v1_frame(df_raw, self.config)
        if len(self.frame.index) < 2:
            raise ValueError("MdpV1Env requires at least two valid timestamped rows.")
        if not bool(self.frame[MDP_V1_VALID_TRANSITION_COLUMN].any()):
            raise ValueError("MdpV1Env requires at least one valid 5-minute transition.")
        self.observation_columns = mdp_v1_observation_columns(self.frame)
        self._observation_frame = (
            self.observation_scaler.transform_frame(self.frame)
            if self.observation_scaler is not None
            else self.frame
        )
        valid_positions = np.flatnonzero(
            self.frame[MDP_V1_VALID_TRANSITION_COLUMN].to_numpy(dtype=bool)
        )
        self._index = int(valid_positions[0])
        self._last_action = logged_mdp_v1_action(self.frame.iloc[self._index])
        self._last_last_action = (
            logged_mdp_v1_action(self.frame.iloc[self._index - 1])
            if self._index > 0
            else dict(self._last_action)
        )

    def reset(self, *, start_index: int | None = None) -> np.ndarray:
        if start_index is None:
            valid_positions = np.flatnonzero(
                self.frame[MDP_V1_VALID_TRANSITION_COLUMN].to_numpy(dtype=bool)
            )
            start_index = int(valid_positions[0])
        if start_index < 0 or start_index >= len(self.frame.index) - 1:
            raise ValueError("start_index must leave at least one transition step.")
        if not bool(self.frame.iloc[start_index][MDP_V1_VALID_TRANSITION_COLUMN]):
            raise ValueError("start_index must point to a valid 5-minute transition.")
        self._index = int(start_index)
        self._last_action = logged_mdp_v1_action(self.frame.iloc[self._index])
        self._last_last_action = (
            logged_mdp_v1_action(self.frame.iloc[self._index - 1])
            if self._index > 0
            else dict(self._last_action)
        )
        return self.observation()

    def observation(self) -> np.ndarray:
        row = self._observation_frame.iloc[self._index]
        return row[self.observation_columns].to_numpy(dtype=float)

    def transition_target(self) -> dict[str, float]:
        row = self.frame.iloc[self._index]
        return {
            key: float(row[key])
            for key in self.transition_target_columns
            if pd.notna(row.get(key, np.nan))
        }

    def logged_action(self) -> dict[str, float]:
        return logged_mdp_v1_action(self.frame.iloc[self._index])

    def step(
        self,
        action: Mapping[str, float] | Sequence[float],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        normalized_action = _normalize_action(action)
        if self._index >= len(self.frame.index) - 1:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")
        if not bool(self.frame.iloc[self._index][MDP_V1_VALID_TRANSITION_COLUMN]):
            raise RuntimeError("Current row is not a valid transition start. Call reset().")

        current_row = self.frame.iloc[self._index]
        action_dict, constraint_info = _project_action_for_weather_constraints(
            normalized_action,
            current_row,
            self.config,
        )
        next_index = self._index + 1
        next_row = self.frame.iloc[next_index]
        reward, reward_terms = compute_mdp_v1_reward(
            next_row,
            action_dict,
            prev_action=self._last_action,
            prev_prev_action=self._last_last_action,
            config=self.config,
            reward_normalizer=self.reward_normalizer,
        )
        self._index = next_index
        self._last_last_action = self._last_action
        self._last_action = action_dict

        terminated = (
            self._index >= len(self.frame.index) - 1
            or not bool(self.frame.iloc[self._index][MDP_V1_VALID_TRANSITION_COLUMN])
        )
        info: dict[str, object] = {
            "timestamp": current_row[TIME_COLUMN],
            "next_timestamp": next_row[TIME_COLUMN],
            "valid_transition": bool(current_row[MDP_V1_VALID_TRANSITION_COLUMN]),
            "rollout_id": int(current_row[MDP_V1_ROLLOUT_ID_COLUMN]),
            "action": action_dict,
            "raw_action": normalized_action,
            "constraint_info": constraint_info,
            "logged_action": logged_mdp_v1_action(current_row),
            "transition_target": {
                key: float(current_row[key])
                for key in self.transition_target_columns
                if pd.notna(current_row.get(key, np.nan))
            },
            "reward_terms": reward_terms,
        }
        return self.observation(), reward, terminated, False, info


__all__ = [
    "MDP_V1_ACTION_COLUMNS",
    "MDP_V1_AGGREGATE_QUALITY_FLAG_COLUMNS",
    "MDP_V1_DAYLIGHT_CONDITION_ONEHOT_COLUMNS",
    "MDP_V1_GROWTH_STAGE_ONEHOT_COLUMNS",
    "MDP_V1_MAX_GROWTH_STAGE_ORDER",
    "MDP_V1_OBSERVATION_COLUMNS",
    "MDP_V1_QUALITY_FLAG_PREFIX",
    "MDP_V1_REWARD_TERM_KEYS",
    "MDP_V1_ROLLOUT_ID_COLUMN",
    "MDP_V1_SOLAR_PERIOD_ONEHOT_COLUMNS",
    "MDP_V1_STEP_MINUTES",
    "MDP_V1_TRANSITION_METRIC_KEYS",
    "MDP_V1_TRANSITION_TARGET_COLUMNS",
    "MDP_V1_VALID_TRANSITION_COLUMN",
    "MDP_V1_ZSCORE_OBSERVATION_COLUMNS",
    "MdpV1Config",
    "MdpV1Env",
    "MdpV1ObservationScaler",
    "MdpV1RewardNormalizer",
    "compute_mdp_v1_reward",
    "fit_mdp_v1_observation_scaler",
    "fit_mdp_v1_reward_normalizer",
    "logged_mdp_v1_action",
    "mdp_v1_observation_columns",
    "prepare_mdp_v1_frame",
]
