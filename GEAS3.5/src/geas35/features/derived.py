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
    "dTcond_pred_10m",
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
    boundary_frame: pd.DataFrame | None = None,
    expected_step_minutes: float = 5.0,
) -> pd.Series:
    """Extrapolate the local condensation-margin trend without crossing gaps.

    The numerical definition is unchanged for a continuous 5-minute sequence:
    fit a least-squares line to the finite values in the inclusive trailing
    30-minute window and evaluate it 10 minutes after the current timestamp.
    A moving left pointer avoids the previous full-timeline scan at every row.
    """

    t = pd.to_datetime(timestamps, errors="coerce")
    values = pd.to_numeric(margin, errors="coerce").to_numpy(dtype=float)
    # Pandas 3 may retain ``datetime64[us]`` internally, while Timedelta.value
    # is expressed in nanoseconds. Normalize explicitly before integer math so
    # window and gap comparisons are invariant across pandas versions.
    time_ns = t.to_numpy(dtype="datetime64[ns]").view("int64")
    nat_value = np.iinfo(np.int64).min
    lookback_ns = int(pd.Timedelta(minutes=lookback_minutes).value)
    expected_step_ns = int(pd.Timedelta(minutes=expected_step_minutes).value)
    horizon_seconds = horizon_minutes * 60.0
    groups = []
    if boundary_frame is not None:
        groups = [
            boundary_frame[column].astype("string").fillna("<NA>").to_numpy()
            for column in boundary_frame.columns
        ]

    out = np.full(len(t), np.nan, dtype=float)
    segment_start = 0
    window_left = 0
    for idx in range(len(t)):
        now_ns = time_ns[idx]
        if now_ns == nat_value:
            segment_start = idx + 1
            window_left = segment_start
            continue

        if idx > 0:
            previous_ns = time_ns[idx - 1]
            group_changed = any(group[idx] != group[idx - 1] for group in groups)
            date_changed = (
                previous_ns == nat_value
                or t.iloc[idx].date() != t.iloc[idx - 1].date()
            )
            time_discontinuity = (
                previous_ns == nat_value or now_ns - previous_ns != expected_step_ns
            )
            if group_changed or date_changed or time_discontinuity:
                segment_start = idx
                window_left = idx

        window_left = max(window_left, segment_start)
        lower_bound = now_ns - lookback_ns
        while window_left < idx and time_ns[window_left] < lower_bound:
            window_left += 1

        current = values[idx]
        if not np.isfinite(current):
            continue
        hist_ns = time_ns[window_left : idx + 1]
        hist_values = values[window_left : idx + 1]
        finite = np.isfinite(hist_values)
        hist_ns = hist_ns[finite]
        hist_values = hist_values[finite]
        if len(hist_values) < 2:
            out[idx] = current
            continue

        seconds = (hist_ns - hist_ns[0]).astype(float) / 1e9
        design = np.column_stack([np.ones_like(seconds), seconds])
        beta, *_ = np.linalg.lstsq(design, hist_values, rcond=None)
        prediction_seconds = (now_ns - hist_ns[0]) / 1e9 + horizon_seconds
        out[idx] = float(beta[0] + beta[1] * prediction_seconds)

    return pd.Series(out, index=margin.index, dtype="float64")


def _daily_step_seconds(timestamps: pd.Series) -> pd.Series:
    t = pd.to_datetime(timestamps, errors="coerce")
    return t.groupby(t.dt.date).diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)


def _cumulative_light_dli(
    timestamps: pd.Series,
    outdoor_solar_w_m2: pd.Series,
    *,
    solar_to_ppfd_umol_per_j: float,
) -> pd.Series:
    """Estimate daily DLI from broadband outdoor solar radiation."""
    t = pd.to_datetime(timestamps, errors="coerce")
    solar = pd.to_numeric(outdoor_solar_w_m2, errors="coerce").fillna(0.0).clip(lower=0.0)
    photon_umol_m2 = solar * _daily_step_seconds(t) * float(solar_to_ppfd_umol_per_j)
    return photon_umol_m2.groupby(t.dt.date).cumsum() / 1e6


def _cumulative_solar_energy_j_cm2(timestamps: pd.Series, outdoor_solar_w_m2: pd.Series) -> pd.Series:
    t = pd.to_datetime(timestamps, errors="coerce")
    solar = pd.to_numeric(outdoor_solar_w_m2, errors="coerce").fillna(0.0).clip(lower=0.0)
    return (solar * _daily_step_seconds(t) / 1e4).groupby(t.dt.date).cumsum()


def _haurwitz_clear_sky_ghi_w_m2(
    timestamps: pd.Series,
    *,
    latitude: float,
    longitude: float,
    timezone: str,
) -> pd.Series:
    """Approximate clear-sky GHI with NOAA solar geometry and Haurwitz."""
    local = pd.to_datetime(timestamps, errors="coerce")
    if local.dt.tz is None:
        aware = local.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    else:
        aware = local.dt.tz_convert(timezone)
    offsets = aware.map(
        lambda value: value.utcoffset().total_seconds() / 3600.0
        if pd.notna(value) and value.utcoffset() is not None else np.nan
    ).to_numpy(dtype=float)
    day = aware.dt.dayofyear.to_numpy(dtype=float)
    hour = (aware.dt.hour.to_numpy(dtype=float) + aware.dt.minute.to_numpy(dtype=float) / 60.0
            + aware.dt.second.to_numpy(dtype=float) / 3600.0)
    gamma = 2.0 * np.pi / 365.0 * (day - 1.0 + (hour - 12.0) / 24.0)
    equation_of_time = 229.18 * (0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
        - 0.014615*np.cos(2.0*gamma) - 0.040849*np.sin(2.0*gamma))
    declination = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
        - 0.006758*np.cos(2.0*gamma) + 0.000907*np.sin(2.0*gamma)
        - 0.002697*np.cos(3.0*gamma) + 0.00148*np.sin(3.0*gamma))
    true_solar_minutes = hour*60.0 + equation_of_time + 4.0*float(longitude) - 60.0*offsets
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    latitude_rad = np.deg2rad(float(latitude))
    cos_zenith = (np.sin(latitude_rad)*np.sin(declination)
        + np.cos(latitude_rad)*np.cos(declination)*np.cos(hour_angle))
    positive = np.clip(np.where(np.isfinite(cos_zenith), cos_zenith, 0.0), 0.0, None)
    ghi = np.zeros_like(positive)
    daylight = positive > 0.0
    ghi[daylight] = 1098.0 * positive[daylight] * np.exp(-0.059 / positive[daylight])
    return pd.Series(ghi, index=timestamps.index, dtype="float64")


def _light_eta_from_clear_sky_curve(
    timestamps: pd.Series,
    clear_sky_sum_j_cm2: pd.Series,
    target_sum_j_cm2: pd.Series,
) -> pd.Series:
    """Inverse lookup of the daily clear-sky cumulative curve."""
    t = pd.to_datetime(timestamps, errors="coerce")
    clear_sum = pd.to_numeric(clear_sky_sum_j_cm2, errors="coerce")
    target = pd.to_numeric(target_sum_j_cm2, errors="coerce")
    result = pd.Series(np.nan, index=timestamps.index, dtype="float64")
    groups = pd.Series(np.arange(len(t)), index=t.index).groupby(t.dt.date)
    for _, positions in groups:
        pos = positions.to_numpy(dtype=int)
        times = t.iloc[pos]
        curve = np.maximum.accumulate(np.where(np.isfinite(clear_sum.iloc[pos]), clear_sum.iloc[pos], 0.0))
        targets = target.iloc[pos].to_numpy(dtype=float)
        for local_index, absolute_position in enumerate(pos):
            if not np.isfinite(targets[local_index]) or pd.isna(times.iloc[local_index]):
                continue
            match = max(int(np.searchsorted(curve, targets[local_index], side="left")), local_index)
            if match < len(pos) and pd.notna(times.iloc[match]):
                result.iloc[absolute_position] = max(
                    (times.iloc[match] - times.iloc[local_index]).total_seconds() / 60.0, 0.0)
    return result


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
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str = "Asia/Seoul",
    target_light_sum_j_cm2: float = 1200.0,
    solar_to_ppfd_umol_per_j: float = 2.02,
    vent_loss_solar_reference_w_m2: float = 1000.0,
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
    boundary_columns = [
        column
        for column in ("crop", "series_id", "segment_id", "episode_id")
        if column in df.columns
    ]
    cond_10m = _condensation_margin_forecast_10m(
        t,
        cond_margin,
        boundary_frame=df.loc[:, boundary_columns],
    )

    df["obs_derived_vpd_kpa"] = vpd
    df["obs_derived_dewpoint_c"] = dewpoint
    df["obs_derived_condensation_margin_c"] = cond_margin
    df["obs_derived_rh90_minutes_1h"] = _rolling_minutes(t, indoor_humidity >= rh_high_pct)
    df["obs_derived_low_vpd_minutes_1h"] = _rolling_minutes(t, vpd < vpd_low_kpa)
    df["dTcond_pred_10m"] = cond_10m.where(cond_10m.notna(), cond_margin)
    df["obs_derived_condensation_risk_10m"] = (
        df["dTcond_pred_10m"] < condensation_margin_min_c
    ).astype(float)

    df["obs_derived_dli_mol_m2"] = _cumulative_light_dli(
        t, light, solar_to_ppfd_umol_per_j=solar_to_ppfd_umol_per_j
    ).fillna(0.0)

    measured_input = _first_numeric(
        df, ["out_light_sum", "light_sum_jcm2", "measured_sum_jcm2"], default=np.nan
    )
    measured_integrated = _cumulative_solar_energy_j_cm2(t, light)
    df["obs_derived_light_sum"] = measured_input.where(
        measured_input.notna(), measured_integrated
    ).fillna(0.0)

    clear_input = _first_numeric(df, ["cs_sum_jcm2", "clear_sky_sum_jcm2"], default=np.nan)
    if clear_input.notna().any():
        clear_sum = clear_input
    elif latitude is not None and longitude is not None:
        clear_ghi = _haurwitz_clear_sky_ghi_w_m2(
            t, latitude=latitude, longitude=longitude, timezone=timezone
        )
        clear_sum = _cumulative_solar_energy_j_cm2(t, clear_ghi)
    else:
        clear_sum = pd.Series(np.nan, index=df.index, dtype="float64")
    df["obs_derived_clear_sky_sum"] = clear_sum.fillna(0.0)

    target_light_sum = _first_numeric(
        df, ["target_jcm2", "max_sun_light", "target_light_sum_jcm2"],
        default=float(target_light_sum_j_cm2),
    )
    df["obs_derived_light_sum_ratio"] = (
        df["obs_derived_light_sum"] / clear_sum.where(clear_sum > 0.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["obs_derived_light_eta_minutes"] = _light_eta_from_clear_sky_curve(
        t, clear_sum, target_light_sum
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vent = (pd.to_numeric(df[vent_col], errors="coerce").fillna(0.0)
            if vent_col in df.columns else pd.Series(0.0, index=df.index))
    vent_fraction = pd.Series(vent, index=df.index, dtype="float64")
    if vent_fraction.abs().max() > 1.0:
        vent_fraction = vent_fraction / 100.0
    vent_fraction = vent_fraction.clip(0.0, 1.0)
    df["obs_derived_temp_diff_in_out_c"] = indoor_temp - outdoor_temp
    solar_relief = 1.0 - (
        light.clip(lower=0.0) / float(vent_loss_solar_reference_w_m2)
    ).clip(0.0, 1.0)
    df["obs_derived_vent_loss_proxy"] = (
        vent_fraction * df["obs_derived_temp_diff_in_out_c"].clip(lower=0.0) * solar_relief
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
