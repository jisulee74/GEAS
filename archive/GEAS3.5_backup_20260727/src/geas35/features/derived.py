"""Derived feature builders for MDP and transition-model inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


MDP_DERIVED_CONTINUOUS_COLUMNS = [
    "obs_derived_vpd_kpa",
    "obs_derived_dewpoint_c",
    "obs_derived_condensation_margin_c",
    "obs_derived_rh90_minutes_1h",
    "obs_derived_low_vpd_minutes_1h",
    "obs_derived_condensation_margin_10m_c",
    "obs_derived_dli_mol_m2",
    "obs_derived_light_sum",
    "obs_derived_clear_sky_sum",
    "obs_derived_light_sum_ratio",
    "obs_derived_light_eta_minutes",
    "obs_derived_vent_loss_proxy",
    "obs_derived_temp_diff_in_out_c",
    "obs_derived_ach_min_day",
    "obs_derived_ach_min_night",
    "obs_derived_ramp_limit_pct",
    "obs_derived_heating_degree_minutes",
    "obs_derived_cooling_degree_minutes",
]
MDP_DERIVED_BINARY_COLUMNS = [
    "obs_derived_condensation_risk_10m",
]
MDP_DERIVED_OBSERVATION_COLUMNS = [
    *MDP_DERIVED_CONTINUOUS_COLUMNS,
    *MDP_DERIVED_BINARY_COLUMNS,
]


def _sat_vapor_pressure_kpa(temp_c: pd.Series | np.ndarray | float) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))


def vpd_kpa(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_c, errors="coerce")
    rh = pd.to_numeric(rh_pct, errors="coerce").clip(0.0, 100.0)
    es = _sat_vapor_pressure_kpa(temp)
    ea = es * (rh.to_numpy(dtype=float) / 100.0)
    return pd.Series(np.maximum(es - ea, 0.0), index=temp.index)


def dewpoint_c(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
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
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        out = out.where(out.notna(), values)
    return out.fillna(default)


def _rolling_minutes(
    timestamps: pd.Series,
    condition: pd.Series,
    *,
    window: str = "60min",
    default_step_minutes: float = 5.0,
) -> pd.Series:
    values = pd.to_numeric(condition, errors="coerce").fillna(0.0).astype(float)
    if len(values) <= 1:
        return values * default_step_minutes
    indexed = pd.Series(values.to_numpy(dtype=float), index=pd.to_datetime(timestamps, errors="coerce"))
    if indexed.index.isna().any():
        return values * default_step_minutes
    return indexed.rolling(window, min_periods=1).sum().to_numpy() * default_step_minutes


def _condensation_margin_forecast_10m(
    timestamps: pd.Series,
    margin: pd.Series,
    *,
    lookback_minutes: float = 30.0,
    horizon_minutes: float = 10.0,
) -> pd.Series:
    t = pd.to_datetime(timestamps, errors="coerce")
    y = pd.to_numeric(margin, errors="coerce")
    out = []
    for idx, now in enumerate(t):
        current = y.iloc[idx]
        if pd.isna(now) or not np.isfinite(current):
            out.append(np.nan)
            continue
        mask = (t >= now - pd.Timedelta(minutes=lookback_minutes)) & (t <= now)
        hist_t = t.loc[mask]
        hist_y = y.loc[mask]
        valid = hist_t.notna() & np.isfinite(hist_y.to_numpy(dtype=float))
        hist_t = hist_t.loc[valid]
        hist_y = hist_y.loc[valid]
        if len(hist_y) < 2:
            out.append(float(current))
            continue
        seconds = (hist_t - hist_t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        values = hist_y.to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(seconds), seconds])
        beta, *_ = np.linalg.lstsq(design, values, rcond=None)
        pred_t = (now - hist_t.iloc[0]).total_seconds() + horizon_minutes * 60.0
        out.append(float(beta[0] + beta[1] * pred_t))
    return pd.Series(out, index=margin.index, dtype="float64")


def _cumulative_light_dli(timestamps: pd.Series, light: pd.Series) -> pd.Series:
    t = pd.to_datetime(timestamps, errors="coerce")
    q = pd.to_numeric(light, errors="coerce").fillna(0.0)
    dt_sec = t.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    return (q * dt_sec).groupby(t.dt.date).cumsum() / 1e6


def _cumulative_degree_minutes(
    timestamps: pd.Series,
    indoor_temp: pd.Series,
    target_temp: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    t = pd.to_datetime(timestamps, errors="coerce")
    temp = pd.to_numeric(indoor_temp, errors="coerce")
    target = pd.to_numeric(target_temp, errors="coerce")
    dt_min = t.diff().dt.total_seconds().fillna(0.0).div(60.0).clip(lower=0.0)
    heat = (target - temp).clip(lower=0.0).fillna(0.0) * dt_min
    cool = (temp - target).clip(lower=0.0).fillna(0.0) * dt_min
    return heat.groupby(t.dt.date).cumsum(), cool.groupby(t.dt.date).cumsum()


def add_mdp_derived_features(
    df_raw: pd.DataFrame,
    *,
    time_col: str = "reg_date",
    indoor_temp_col: str = "obs_indoor_temp_c",
    indoor_humidity_col: str = "obs_indoor_humidity_pct",
    outdoor_temp_col: str = "obs_outdoor_temp_c",
    outdoor_light_col: str = "obs_outdoor_light",
    vent_col: str = "_mdp_v1_logged_vent_pct",
    target_temp_col: str = "obs_current_target_temp_c",
    condensation_margin_min_c: float = 0.8,
    vpd_low_kpa: float = 0.5,
    rh_high_pct: float = 90.0,
    ach_min_day_base: float = 0.10,
    ach_min_night_base: float = 0.05,
    ramp_limit_pct: float = 15.0,
) -> pd.DataFrame:
    """Add backup-inspired derived features for model inputs."""

    if df_raw is None:
        return pd.DataFrame()
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()
    t = pd.to_datetime(df[time_col], errors="coerce")
    indoor_temp = pd.to_numeric(df[indoor_temp_col], errors="coerce")
    indoor_humidity = pd.to_numeric(df[indoor_humidity_col], errors="coerce")
    outdoor_temp = pd.to_numeric(df[outdoor_temp_col], errors="coerce")
    light = pd.to_numeric(df[outdoor_light_col], errors="coerce")
    target_temp = pd.to_numeric(df[target_temp_col], errors="coerce")

    vpd = vpd_kpa(indoor_temp, indoor_humidity)
    dewpoint = dewpoint_c(indoor_temp, indoor_humidity)
    cond_margin = indoor_temp - dewpoint
    cond_10m = _condensation_margin_forecast_10m(t, cond_margin)

    df["obs_derived_vpd_kpa"] = vpd
    df["obs_derived_dewpoint_c"] = dewpoint
    df["obs_derived_condensation_margin_c"] = cond_margin
    df["obs_derived_rh90_minutes_1h"] = _rolling_minutes(t, indoor_humidity >= rh_high_pct)
    df["obs_derived_low_vpd_minutes_1h"] = _rolling_minutes(t, vpd < vpd_low_kpa)
    df["obs_derived_condensation_margin_10m_c"] = cond_10m.where(cond_10m.notna(), cond_margin)
    df["obs_derived_condensation_risk_10m"] = (
        df["obs_derived_condensation_margin_10m_c"] < condensation_margin_min_c
    ).astype(float)

    df["obs_derived_dli_mol_m2"] = _cumulative_light_dli(t, light).fillna(0.0)
    df["obs_derived_light_sum"] = _first_numeric(
        df,
        ["out_light_sum", "light_sum_jcm2", "measured_sum_jcm2"],
        default=0.0,
    ).fillna(0.0)
    df["obs_derived_clear_sky_sum"] = _first_numeric(
        df,
        ["cs_sum_jcm2", "clear_sky_sum_jcm2"],
        default=0.0,
    ).fillna(0.0)
    target_light_sum = _first_numeric(
        df,
        ["target_jcm2", "max_sun_light", "target_light_sum_jcm2"],
        default=np.nan,
    )
    denominator = target_light_sum.where(target_light_sum > 0.0, df["obs_derived_clear_sky_sum"])
    df["obs_derived_light_sum_ratio"] = (
        df["obs_derived_light_sum"] / denominator.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    light_eta = pd.to_datetime(
        _first_numeric(df, ["light_eta"], default=np.nan),
        errors="coerce",
    )
    if "LightETA" in df.columns:
        light_eta = pd.to_datetime(df["LightETA"], errors="coerce")
    df["obs_derived_light_eta_minutes"] = (
        (light_eta - t).dt.total_seconds().div(60.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vent = pd.to_numeric(df[vent_col], errors="coerce").fillna(0.0) if vent_col in df.columns else 0.0
    df["obs_derived_temp_diff_in_out_c"] = indoor_temp - outdoor_temp
    df["obs_derived_vent_loss_proxy"] = (
        pd.Series(vent, index=df.index, dtype="float64")
        * df["obs_derived_temp_diff_in_out_c"].clip(lower=0.0)
    ).fillna(0.0)
    rh90 = df["obs_derived_rh90_minutes_1h"]
    low_vpd = df["obs_derived_low_vpd_minutes_1h"]
    ach_day = pd.Series(ach_min_day_base, index=df.index, dtype="float64")
    ach_night = pd.Series(ach_min_night_base, index=df.index, dtype="float64")
    severe = (rh90 >= 20.0) | (low_vpd >= 20.0)
    moderate = ((rh90 >= 10.0) | (low_vpd >= 10.0)) & ~severe
    ach_day.loc[severe] *= 1.5
    ach_night.loc[severe] *= 1.3
    ach_day.loc[moderate] *= 1.2
    ach_night.loc[moderate] *= 1.1
    df["obs_derived_ach_min_day"] = ach_day
    df["obs_derived_ach_min_night"] = ach_night
    df["obs_derived_ramp_limit_pct"] = float(ramp_limit_pct)
    heat_deg, cool_deg = _cumulative_degree_minutes(t, indoor_temp, target_temp)
    df["obs_derived_heating_degree_minutes"] = heat_deg.fillna(0.0)
    df["obs_derived_cooling_degree_minutes"] = cool_deg.fillna(0.0)

    for column in MDP_DERIVED_CONTINUOUS_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    for column in MDP_DERIVED_BINARY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
    return df


__all__ = [
    "MDP_DERIVED_BINARY_COLUMNS",
    "MDP_DERIVED_CONTINUOUS_COLUMNS",
    "MDP_DERIVED_OBSERVATION_COLUMNS",
    "add_mdp_derived_features",
    "dewpoint_c",
    "vpd_kpa",
]
