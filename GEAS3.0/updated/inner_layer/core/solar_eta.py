from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd

try:
    import pvlib
    _HAS_PVLIB = True
    
except ImportError:
    _HAS_PVLIB = False

def get_sun_times(
    lat: float,
    lon: float,
    when: pd.Timestamp,
    tz: str = "Asia/Seoul",
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    
    when = pd.to_datetime(when)
    if when.tzinfo is None:
        when = when.tz_localize(tz)
    else:
        when = when.tz_convert(tz)

    date = when.normalize()

    if _HAS_PVLIB:
        loc = pvlib.location.Location(latitude=lat, longitude=lon, tz=tz)
        times = pd.date_range(
            start=date, periods=24 * 4, freq="15min", tz=tz
        )
        sp = pvlib.solarposition.get_solarposition(times, lat, lon)

        above = sp["apparent_elevation"] > 0
        if above.any():
            idx = np.where(above)[0]
            sunrise = times[idx[0]]
            sunset = times[idx[-1]]
        else:
            # polar night / extreme case
            sunrise = date + pd.Timedelta(hours=7)
            sunset = date + pd.Timedelta(hours=18)
    else:
        sunrise = date + pd.Timedelta(hours=7)
        sunset = date + pd.Timedelta(hours=18)

    return sunrise, sunset


def _clear_sky_ghi_series(
    lat: float,
    lon: float,
    date: pd.Timestamp,
    tz: str = "Asia/Seoul",
    freq: str = "5min",
    ) -> pd.DataFrame:

    date = pd.to_datetime(date)
    if date.tzinfo is None:
        date = date.tz_localize(tz)
    else:
        date = date.tz_convert(tz)

    day = date.normalize()

    if _HAS_PVLIB:
        loc = pvlib.location.Location(latitude=lat, longitude=lon, tz=tz)
        times = pd.date_range(start=day, end=day + pd.Timedelta(days=1), freq=freq, tz=tz)
        cs = loc.get_clearsky(times)  # GHI/DHI/DNI in W/m^2
        ghi = cs["ghi"].astype(float)
    else:
        # simple cosine-shaped daylight curve as fallback
        sunrise = day + pd.Timedelta(hours=7)
        sunset = day + pd.Timedelta(hours=18)
        times = pd.date_range(start=day, end=day + pd.Timedelta(days=1), freq=freq, tz=tz)
        t_hours = (times - sunrise) / pd.Timedelta(hours=1)
        day_length = (sunset - sunrise) / pd.Timedelta(hours=1)
        x = np.clip(t_hours / max(day_length, 1e-6), 0, 1)
        # bell-shaped approximate GHI, peak ~800 W/m²
        ghi = np.asarray(800.0 * np.sin(np.pi * x), dtype=float)
        ghi[ghi < 0] = 0.0
        ghi = pd.Series(ghi, index=times, name="ghi")

    # integrate to cumulative J/cm²
    # GHI [W/m²] = J/(s·m²)
    # For each step: energy_step_J_per_m2 = ghi * dt_seconds
    # Convert to J/cm²: divide by 10_000
    dt_s = np.diff(ghi.index.view("int64")) / 1e9
    if len(dt_s) == 0:
        cum = np.zeros_like(ghi.values, dtype=float)
    else:
        # assume last dt same as previous
        dt_s = np.concatenate([dt_s, dt_s[-1:]])
        step_jcm2 = ghi.values * dt_s / 10000.0
        cum = np.cumsum(step_jcm2)

    df = pd.DataFrame({"ghi": ghi.values, "sum_jcm2": cum}, index=ghi.index)
    return df

def compute_eta(
    lat: float,
    lon: float,
    target_jcm2: float,
    df_all: pd.DataFrame,
    now_ts: pd.Timestamp,
    tz: str = "Asia/Seoul",
    ) -> Tuple[pd.Timestamp | pd.NaT, pd.Timestamp, pd.Timestamp, pd.DataFrame]:

    now_ts = pd.to_datetime(now_ts)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize(tz)
    else:
        now_ts = now_ts.tz_convert(tz)

    date = now_ts.normalize()

    sunrise, sunset = get_sun_times(lat, lon, now_ts, tz=tz)
    cs_df = _clear_sky_ghi_series(lat, lon, date, tz=tz, freq="5min")

    eta: pd.Timestamp | pd.NaT
    if target_jcm2 <= 0 or cs_df["sum_jcm2"].max() < target_jcm2:
        eta = pd.NaT
    else:
        idx = np.searchsorted(cs_df["sum_jcm2"].values, target_jcm2)
        idx = int(np.clip(idx, 0, len(cs_df) - 1))
        eta = cs_df.index[idx]

    return eta, sunrise, sunset, cs_df

def integrate_measured_to_jcm2(
    df: pd.DataFrame,
    col: str = "out_light",
    time_col: str = "reg_date",
    ) -> pd.Series:

    if df.empty or col not in df.columns or time_col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, name=f"{col}_sum")

    s = df[[time_col, col]].copy()
    s[time_col] = pd.to_datetime(s[time_col])
    s = s.sort_values(time_col).reset_index(drop=True)

    vals = s[col].astype(float).to_numpy()
    t_ns = s[time_col].astype("int64").to_numpy()

    if len(vals) < 2:
        return pd.Series(np.zeros(len(df)), index=df.index, name=f"{col}_sum")

    dt_s = np.diff(t_ns) / 1e9
    # use median dt for first step to avoid 0
    dt0 = np.median(dt_s) if len(dt_s) > 0 else 0.0
    dt_s = np.concatenate([[dt0], dt_s])

    step_jcm2 = vals * dt_s / 10000.0
    cum = np.cumsum(step_jcm2)

    # align back to original index order
    out = pd.Series(cum, index=s.index, name=f"{col}_sum")
    out = out.reindex(df.sort_values(time_col).index)
    out = out.reindex(df.index)  # original order
    return out

if __name__ == "__main__":
    lat, lon = 36.46, 128.22
    now = pd.Timestamp.now(tz="Asia/Seoul")
    target_jcm2 = 5000.0

    # dummy df_all (just for signature compatibility; not actually used)
    times = pd.date_range(start=now.normalize(), periods=24, freq="1h", tz="Asia/Seoul")
    df_all = pd.DataFrame({"reg_date": times, "out_light": np.linspace(0, 800, len(times))})

    eta, sunrise, sunset, cs_df = compute_eta(lat, lon, target_jcm2, df_all, now)
    print("Sunrise:", sunrise)
    print("Sunset:", sunset)
    print("ETA (target_jcm2):", eta)
    print("Clear-sky head:\n", cs_df.head())
    print("Clear-sky tail:\n", cs_df.tail())
    print("Max clear-sky J/cm²:", cs_df["sum_jcm2"].max())

    # test integrate_measured_to_jcm2
    df_meas = df_all.copy()
    df_meas["reg_date"] = times
    df_meas["out_light_sum"] = integrate_measured_to_jcm2(df_meas, col="out_light", time_col="reg_date")

    print("\nMeasured integration test:")
    print(df_meas[["reg_date", "out_light", "out_light_sum"]].head())
    print(df_meas[["reg_date", "out_light", "out_light_sum"]].tail())
    print("Max measured J/cm²:", df_meas["out_light_sum"].max())
