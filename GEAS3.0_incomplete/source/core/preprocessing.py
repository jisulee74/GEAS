from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Callable

from core.timeutils import (
    ensure_datetime,
    today_window_kst,
    slice_today,
)

__all__ = [
    "ensure_datetime",
    "today_window_kst",
    "slice_today",
    "latest_row_for_controller",
    "normalize_for_derived",
    "quick_derived_data_report",
]

def latest_row_for_controller(df_today: pd.DataFrame) -> pd.DataFrame:

    required = {
        "reg_date": pd.Timestamp.now(tz="Asia/Seoul").tz_convert(None),
        "in_temp": np.nan,
        "out_temp": np.nan,
        "in_hum": np.nan,
        "in_rh": np.nan,
        "in_co2": np.nan,
        "out_light": 0.0,
        "out_light_sum": 0.0,
        "out_rain": 0.0,
        "wind_speed": 0.0,
        "window_pct": 0.0,
    }

    if df_today.empty:
        return pd.DataFrame([required])

    df = df_today.copy()

    for col, default in required.items():
        if col not in df.columns:
            df[col] = default

    if "in_hum" in df.columns:
        df["in_rh"] = df["in_hum"]

    if df["window_pct"].max() <= 1.5:
        df["window_pct"] = df["window_pct"] * 100.0

    df["window_pct"] = df["window_pct"].clip(0, 100)
    df["wind_speed"] = df["wind_speed"].astype(float).clip(lower=0.0)

    for col in ["out_temp", "out_light", "out_light_sum", "wind_speed"]:
        df[col] = df[col].ffill().bfill()

    return df[list(required.keys())].tail(1).reset_index(drop=True)

def normalize_for_derived(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_raw.copy()

    rename_map = {
        "in_hum": "in_rh",
        "out_windsp": "wind_speed",
        "cont_skyl_vol": "window_pct",
        "cont_heater_run": "heater_duty",
    }
    df = df.rename(columns=rename_map)

    if "reg_date" in df.columns:
        df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
        df = df.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)

    if "window_pct" in df.columns:
        w = df["window_pct"].astype(float)
        if w.max() <= 1.5:   
            w = w * 100.0
        df["window_pct"] = w.clip(0, 100)

    if "heater_duty" in df.columns:
        df["heater_duty"] = df["heater_duty"].astype(float).clip(0.0, 1.0)
    else:
        df["heater_duty"] = np.nan

    if "in_rh" in df.columns:
        df["in_rh"] = df["in_rh"].astype(float).clip(0.0, 100.0)

    if "wind_speed" in df.columns:
        df["wind_speed"] = df["wind_speed"].astype(float).clip(lower=0.0)

    for col in ["out_temp", "out_light", "out_light_sum"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    if "window_pct" not in df.columns:
        df["window_pct"] = 0.0
        
    if "wind_speed" not in df.columns:
        df["wind_speed"] = 0.0

    return df

def quick_derived_data_report(
    df: pd.DataFrame,
    light_thr: float = 10.0,
    min_points_ua_c: int = 200,
    min_points_ach: int = 300,
    run_estimator_fn: Optional[Callable] = None,
    cfg_obj: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:

    need = ["in_temp", "out_temp", "in_rh", "out_light", "window_pct", "wind_speed", "heater_duty"]
    have = {c: (c in df.columns and int(df[c].notna().sum())) for c in need}
    print("column availability:", have)

    out_light_series = df.get("out_light", pd.Series(0, index=df.index))
    wind_series = df.get("wind_speed", pd.Series(np.inf, index=df.index))
    winpct = df.get("window_pct", pd.Series(0, index=df.index))

    is_night = (out_light_series < light_thr)
    low_win = (winpct <= 5.0)
    day = (out_light_series >= light_thr)
    low_wind = (wind_series <= 1.5)

    n_ua_c = int((is_night & low_win).sum())
    n_gsolar = int((day & low_wind & (winpct <= 10)).sum())

    print(f"UA/C samples: {n_ua_c} / need ≥ {min_points_ua_c}")
    print(f"g_solar samples: {n_gsolar} / need ≥ 120")

    reasons: Dict[str, list] = {}

    reasons["UA/C"] = []
    if n_ua_c < min_points_ua_c:
        reasons["UA/C"].append(f"insufficient samples ({n_ua_c})")

    n_ach0 = int((is_night & (winpct <= 2.0)).sum())
    reasons["ACH0"] = []
    if n_ach0 < 60:
        reasons["ACH0"].append(f"insufficient samples (have {n_ach0}, need ≥ 60)")

    reasons["ACH(open,wind)"] = []
    if len(df) < min_points_ach:
        reasons["ACH(open,wind)"].append(f"too few samples ({len(df)})")

    reasons["g_solar"] = []
    if n_gsolar < 120:
        reasons["g_solar"].append(f"insufficient samples ({n_gsolar})")

    reasons["safety(VPD,cond)"] = []
    if "in_temp" not in df or "in_rh" not in df:
        reasons["safety(VPD,cond)"].append("missing humidity or temperature")

    for key, val in reasons.items():
        ok = "OK" if len(val) == 0 else "BLOCKED"
        print(f"{key}: {ok} | {', '.join(val) if val else ''}")

    if run_estimator_fn is None:
        print("Estimator function not provided.")
        return None

    if cfg_obj is None:
        class _Cfg:
            def __init__(self, light_threshold, min_points_ua_c, min_points_ach):
                self.light_threshold = light_threshold
                self.min_points_ua_c = min_points_ua_c
                self.min_points_ach = min_points_ach

        cfg_obj = _Cfg(light_thr, min_points_ua_c, min_points_ach)

    res = run_estimator_fn(df, cfg=cfg_obj, hist_daily_est=None)
    res["reasons"] = reasons
    return res
