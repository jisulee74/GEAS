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
    "prepare_for_sensor_qc",
    "recover_action_columns_from_log",
    "recover_external_state_columns_from_cache",
    "latest_row_for_controller",
    "normalize_for_derived",
    "quick_derived_data_report",
]

INTERNAL_STATE_COLUMNS = [
    "in_medium_temp1",
    "in_medium_temp2",
    "in_temp",
    "in_temp2",
    "in_water_hot",
    "in_water_cold",
    "in_hum",
    "in_hum2",
    "in_medium_hum1",
    "in_medium_hum2",
    "in_co2",
    "in_co2_2",
    "in_medium_ec1",
    "in_medium_ec2",
]

SYSTEM_STATE_COLUMNS = [
    "etc_blackout",
    "etc_plc_abnorm",
    "etc_plc_norm",
]

EXTERNAL_STATE_CACHE_MAP = {
    "out_temp": "out_temp",
    "out_hum": "out_hum",
    "out_winddirec": "out_winddirec",
    "out_windsp": "out_windsp",
    "out_light": "out_light",
    "out_light_sum": "out_light_sum",
    "out_rainfall": "out_rainfall",
    "out_rain": "out_rain",
    "out_airpress": "out_airpress",
}

ACTION_RECOVERY_MAP = {
    "cont_skyl_vol": "pred_ltw",
    "cont_skyr_vol": "pred_rtw",
    "cont_cur_vol": "pred_pc1",
    "cont_kwcur_vol": "pred_pc2",
    "cont_heater_run": "pred_heater",
    "cont_cooler_run": "pred_cooler",
    "cont_3way1_vol": "pred_tw1",
    "cont_3way2_vol": "pred_tw2",
    "cont_pump1_run": "pred_cp1",
    "cont_pump2_run": "pred_cp2",
    "cont_fan_run": "pred_fan",
}

def _to_numeric_if_present(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def recover_external_state_columns_from_cache(
    df_raw: pd.DataFrame,
    df_weather_cache: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Recover external-state columns from same-timestamp weather cache.

    Required external-state columns are created if missing. Existing non-missing
    raw values are preserved. If no same-timestamp cache value exists, the value
    remains NaN.
    """

    if df_raw.empty:
        return df_raw

    df = df_raw.copy()
    if "reg_date" not in df.columns:
        for state_col in EXTERNAL_STATE_CACHE_MAP:
            if state_col not in df.columns:
                df[state_col] = np.nan
        return df

    df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")

    for state_col in EXTERNAL_STATE_CACHE_MAP:
        if state_col not in df.columns:
            df[state_col] = np.nan

    if df_weather_cache is None or df_weather_cache.empty or "reg_date" not in df_weather_cache.columns:
        return df

    available_map = {
        state_col: cache_col
        for state_col, cache_col in EXTERNAL_STATE_CACHE_MAP.items()
        if cache_col in df_weather_cache.columns
    }
    if not available_map:
        return df

    cache_cols = ["reg_date", *available_map.values()]
    cache = df_weather_cache[cache_cols].copy()
    cache["reg_date"] = pd.to_datetime(cache["reg_date"], errors="coerce")
    cache = (
        cache.dropna(subset=["reg_date"])
        .sort_values("reg_date")
        .drop_duplicates(subset=["reg_date"], keep="last")
    )
    if cache.empty:
        return df

    for cache_col in available_map.values():
        cache[cache_col] = pd.to_numeric(cache[cache_col], errors="coerce")

    left = df[["reg_date"]].reset_index().dropna(subset=["reg_date"])
    if left.empty:
        return df

    matched = left.merge(cache, on="reg_date", how="left").set_index("index")

    for state_col, cache_col in available_map.items():
        recovered = matched[cache_col].reindex(df.index)
        current = pd.to_numeric(df[state_col], errors="coerce")
        df[state_col] = current.where(current.notna(), recovered)

    return df


def recover_action_columns_from_log(
    df_raw: pd.DataFrame,
    df_action_log: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Recover action columns from GEAS control output logs.

    Missing action columns are created from the control log at the same
    timestamp. Existing non-missing raw values are preserved. If no same-
    timestamp log exists, the value remains NaN.
    """

    if df_raw.empty:
        return df_raw

    df = df_raw.copy()
    if "reg_date" not in df.columns:
        for action_col in ACTION_RECOVERY_MAP:
            if action_col not in df.columns:
                df[action_col] = np.nan
        return df

    df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")

    for action_col in ACTION_RECOVERY_MAP:
        if action_col not in df.columns:
            df[action_col] = np.nan

    if df_action_log is None or df_action_log.empty or "reg_date" not in df_action_log.columns:
        return df

    available_map = {
        action_col: log_col
        for action_col, log_col in ACTION_RECOVERY_MAP.items()
        if log_col in df_action_log.columns
    }
    if not available_map:
        return df

    log_cols = ["reg_date", *available_map.values()]
    log = df_action_log[log_cols].copy()
    log["reg_date"] = pd.to_datetime(log["reg_date"], errors="coerce")
    log = (
        log.dropna(subset=["reg_date"])
        .sort_values("reg_date")
        .drop_duplicates(subset=["reg_date"], keep="last")
    )
    if log.empty:
        return df

    for log_col in available_map.values():
        log[log_col] = pd.to_numeric(log[log_col], errors="coerce")

    left = df[["reg_date"]].reset_index().dropna(subset=["reg_date"])
    if left.empty:
        return df

    matched = left.merge(log, on="reg_date", how="left").set_index("index")

    for action_col, log_col in available_map.items():
        recovered = matched[log_col].reindex(df.index)
        current = pd.to_numeric(df[action_col], errors="coerce")
        df[action_col] = current.where(current.notna(), recovered)

    return df


def prepare_for_sensor_qc(
    df_raw: pd.DataFrame,
    df_action_log: Optional[pd.DataFrame] = None,
    df_weather_cache: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply raw-column missing handling before sensor QC runs."""

    if df_raw.empty:
        return df_raw
    if "reg_date" not in df_raw.columns:
        return df_raw.iloc[0:0].copy()

    df = recover_external_state_columns_from_cache(df_raw, df_weather_cache)
    df = recover_action_columns_from_log(df, df_action_log)
    for col in [*INTERNAL_STATE_COLUMNS, *SYSTEM_STATE_COLUMNS]:
        if col not in df.columns:
            df[col] = np.nan

    df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
    df = df.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)

    _to_numeric_if_present(
        df,
        [
            *INTERNAL_STATE_COLUMNS,
            *SYSTEM_STATE_COLUMNS,
            *EXTERNAL_STATE_CACHE_MAP.keys(),
            *ACTION_RECOVERY_MAP.keys(),
        ],
    )

    for col in ["in_hum", "in_hum2", "in_medium_hum1", "in_medium_hum2"]:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 100.0)

    for col in SYSTEM_STATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)

    for col in [
        "cont_skyl_vol",
        "cont_skyr_vol",
        "cont_cur_vol",
        "cont_kwcur_vol",
        "cont_3way1_vol",
        "cont_3way2_vol",
    ]:
        if col in df.columns:
            w = df[col]
            if w.notna().any() and w.max() <= 1.5:
                w = w * 100.0
            df[col] = w.clip(0.0, 100.0)

    for col in ["cont_heater_run", "cont_cooler_run", "cont_pump1_run", "cont_pump2_run", "cont_fan_run"]:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)

    if "out_hum" in df.columns:
        df["out_hum"] = df["out_hum"].clip(0.0, 100.0)

    if "out_winddirec" in df.columns:
        df["out_winddirec"] = df["out_winddirec"].clip(0.0, 360.0)

    for col in ["out_windsp", "out_light", "out_light_sum", "out_rainfall"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    if "out_rain" in df.columns:
        df["out_rain"] = df["out_rain"].clip(0.0, 1.0)

    return df


def latest_row_for_controller(df_today: pd.DataFrame) -> pd.DataFrame:

    required = {
        "reg_date": pd.Timestamp.now(tz="Asia/Seoul").tz_convert(None),
        "in_temp": np.nan,
        "out_temp": np.nan,
        "in_hum": np.nan,
        "in_rh": np.nan,
        "in_co2": np.nan,
        "out_light": np.nan,
        "out_light_sum": np.nan,
        "out_rain": np.nan,
        "wind_speed": np.nan,
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

    if "window_pct" not in df.columns:
        df["window_pct"] = 0.0
        
    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan

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
