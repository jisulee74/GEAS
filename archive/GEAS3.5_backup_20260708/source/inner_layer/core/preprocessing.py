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
    "apply_realtime_outlier_flags",
    "apply_action_columns_from_api_history",
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

EXTERNAL_STATE_COLUMNS = [
    "out_temp",
    "out_hum",
    "out_winddirec",
    "out_windsp",
    "out_light",
    "out_light_sum",
    "out_rainfall",
    "out_rain",
    "out_airpress",
]

ACTION_API_CHANNEL_MAP = {
    1: "cont_skyl_vol",
    2: "cont_skyr_vol",
    4: "cont_cur_vol",
    5: "cont_kwcur_vol",
    6: "cont_co2_run",
    7: "cont_pump1_run",
    8: "cont_pump2_run",
    9: "cont_heater_run",
    10: "cont_cooler_run",
    11: "cont_3way1_vol",
    12: "cont_3way2_vol",
    13: "cont_fan_run",
}

ACTION_COLUMNS = list(ACTION_API_CHANNEL_MAP.values())

ACTION_CONTROL_LOG_MAP = {
    "cont_skyl_vol": "pred_ltw",
    "cont_skyr_vol": "pred_rtw",
    "cont_cur_vol": "pred_pc1",
    "cont_kwcur_vol": "pred_pc2",
    "cont_co2_run": "pred_co2",
    "cont_pump1_run": "pred_cp1",
    "cont_pump2_run": "pred_cp2",
    "cont_heater_run": "pred_heater",
    "cont_cooler_run": "pred_cooler",
    "cont_3way1_vol": "pred_tw1",
    "cont_3way2_vol": "pred_tw2",
    "cont_fan_run": "pred_fan",
}

CONTROL_API_HISTORY_CUTOFF = pd.Timestamp("2026-01-22 09:05:00")

SEASON_BY_MONTH = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}

# Domain-review sensor ranges. Each tuple is
# (lower, upper, lower_is_valid, upper_is_valid); None means no bound.
DOMAIN_NORMAL_RANGES = {
    "in_temp": (2.0, 50.0, True, False),
    "in_temp2": (2.0, 50.0, True, False),
    "in_hum": (0.0, 100.0, True, True),
    "in_hum2": (0.0, 100.0, True, True),
    "in_co2": (200.0, 5000.0, True, False),
    "in_co2_2": (200.0, 5000.0, True, False),
    "out_temp": (None, 50.0, True, False),
    "out_winddirec": (0.0, 359.0, True, True),
    "out_windsp": (0.0, 30.0, True, True),
    "out_rain": (0.0, 1.0, True, True),
    "out_light": (0.0, 1400.0, True, True),
    "out_light_sum": (None, 10000.0, True, False),
    "in_medium_hum1": (0.0, 100.0, True, True),
    "in_medium_hum2": (0.0, 100.0, True, True),
    "in_medium_temp1": (0.0, 41.0, False, False),
    "in_medium_temp2": (0.0, 41.0, False, False),
}

DOMAIN_ALLOWED_VALUES = {
    "out_rain": {0.0, 1.0},
}

SEASONAL_CONSISTENCY_Q999 = {
    ("in_temp", "in_temp2"): {
        "spring": 25.950000000000003,
        "summer": 2.049999999999997,
        "autumn": 29.300000000000004,
        "winter": 1.549999999999999,
    },
    ("in_hum", "in_hum2"): {
        "spring": 21.87,
        "summer": 9.579999999999998,
        "autumn": 34.96,
        "winter": 28.84,
    },
    ("in_co2", "in_co2_2"): {
        "spring": 1430.0,
        "summer": 158.0,
        "autumn": 2408.0,
        "winter": 4021.0,
    },
    ("in_medium_temp1", "in_medium_temp2"): {
        "spring": 42.1,
        "summer": 57.1,
        "autumn": 57.1,
        "winter": 26.9,
    },
    ("in_medium_hum1", "in_medium_hum2"): {
        "spring": 53.2,
        "summer": 48.5,
        "autumn": 79.7,
        "winter": 9.11,
    },
    ("in_medium_ec1", "in_medium_ec2"): {
        "spring": 7.14,
        "summer": 2.8,
        "autumn": 8.0,
        "winter": 0.2800000000000002,
    },
}

def _to_numeric_if_present(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _is_missing_text(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _domain_rule_flag(
    values: pd.Series,
    rule: tuple[float | None, float | None, bool, bool],
    allowed_values: set[float] | None = None,
) -> pd.Series:
    lower, upper, lower_is_valid, upper_is_valid = rule
    present = values.notna()
    flag = pd.Series(False, index=values.index)
    if lower is not None:
        flag |= present & ((values < lower) if lower_is_valid else (values <= lower))
    if upper is not None:
        flag |= present & ((values > upper) if upper_is_valid else (values >= upper))
    if allowed_values is not None:
        flag |= present & ~values.isin(allowed_values)
    return flag


def apply_realtime_outlier_flags(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Attach real-time sensor outlier flags without changing sensor values."""

    if df_raw.empty:
        return df_raw

    df = df_raw.copy()
    index = df.index
    domain_flag = pd.Series(False, index=index)
    consistency_flag = pd.Series(False, index=index)

    for col, rule in DOMAIN_NORMAL_RANGES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        col_flag = _domain_rule_flag(values, rule, DOMAIN_ALLOWED_VALUES.get(col))
        flag_col = f"outlier_domain_{col}"
        df[flag_col] = col_flag.fillna(False).astype(int)
        domain_flag |= col_flag.fillna(False)

    if "reg_date" in df.columns:
        seasons = pd.to_datetime(df["reg_date"], errors="coerce").dt.month.map(SEASON_BY_MONTH)
    else:
        seasons = pd.Series(pd.NA, index=index)

    for (var1, var2), thresholds in SEASONAL_CONSISTENCY_Q999.items():
        if var1 not in df.columns or var2 not in df.columns:
            continue

        s1 = pd.to_numeric(df[var1], errors="coerce")
        s2 = pd.to_numeric(df[var2], errors="coerce")
        diff = (s1 - s2).abs()
        pair_flag = pd.Series(False, index=index)

        for season, threshold in thresholds.items():
            season_mask = seasons.eq(season)
            valid = s1.notna() & s2.notna() & season_mask
            pair_flag |= valid & (diff > float(threshold))

        pair_col = f"outlier_consistency_{var1}_vs_{var2}"
        df[pair_col] = pair_flag.fillna(False).astype(int)
        consistency_flag |= pair_flag.fillna(False)

    df["outlier_domain_flag"] = domain_flag.astype(int)
    df["outlier_consistency_flag"] = consistency_flag.astype(int)
    df["outlier_flag"] = (domain_flag | consistency_flag).astype(int)
    return df


def _control_log_action_wide(df_control_log: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_control_log is None or df_control_log.empty:
        return pd.DataFrame()

    time_col = _first_existing_column(
        df_control_log,
        ["reg_date", "send_date", "request_time", "created_at", "create_date", "timestamp"],
    )
    if not time_col:
        return pd.DataFrame()

    available_map = {
        action_col: log_col
        for action_col, log_col in ACTION_CONTROL_LOG_MAP.items()
        if log_col in df_control_log.columns
    }
    if not available_map:
        return pd.DataFrame()

    cols = [time_col, *available_map.values()]
    wide = df_control_log[cols].copy()
    wide["reg_date"] = pd.to_datetime(wide[time_col], errors="coerce")
    wide = wide.dropna(subset=["reg_date"]).sort_values("reg_date")
    if wide.empty:
        return pd.DataFrame()

    for log_col in available_map.values():
        wide[log_col] = pd.to_numeric(wide[log_col], errors="coerce")
    wide = wide.rename(columns={log_col: action_col for action_col, log_col in available_map.items()})
    keep_cols = ["reg_date", *available_map.keys()]
    return wide[keep_cols].drop_duplicates(subset=["reg_date"], keep="last")


def apply_action_columns_from_api_history(
    df_raw: pd.DataFrame,
    df_action_history: Optional[pd.DataFrame] = None,
    df_control_log: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build action columns from the source that is valid for each period.

    Through CONTROL_API_HISTORY_CUTOFF, data_silla_enc cont_* values are used
    and missing values are recovered from the same-timestamp GEAS output log.
    After the cutoff, data_silla_enc cont_* values are ignored and valid API
    records use channel + ratio at the same timestamp. If channel, operation,
    or ratio is missing, the same-timestamp GEAS output log is used as fallback.
    If idx or n_idx is missing, that API record is not trusted and the
    corresponding action remains NaN.
    """

    if df_raw.empty:
        return df_raw

    df = df_raw.copy()
    if "reg_date" not in df.columns:
        for action_col in ACTION_COLUMNS:
            if action_col not in df.columns:
                df[action_col] = np.nan
        return df

    df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")

    raw_actions: dict[str, pd.Series] = {}
    for action_col in ACTION_COLUMNS:
        if action_col in df.columns:
            raw_actions[action_col] = pd.to_numeric(df[action_col], errors="coerce")
        else:
            raw_actions[action_col] = pd.Series(np.nan, index=df.index)
        df[action_col] = np.nan

    control_wide = _control_log_action_wide(df_control_log)
    left = df[["reg_date"]].reset_index().dropna(subset=["reg_date"])
    matched_control = (
        left.merge(control_wide, on="reg_date", how="left").set_index("index")
        if not control_wide.empty and not left.empty
        else pd.DataFrame(index=left["index"] if not left.empty else df.index)
    )

    pre_api_mask = df["reg_date"].notna() & (df["reg_date"] <= CONTROL_API_HISTORY_CUTOFF)
    post_api_mask = df["reg_date"].notna() & (df["reg_date"] > CONTROL_API_HISTORY_CUTOFF)

    for action_col in ACTION_COLUMNS:
        action = pd.Series(np.nan, index=df.index, dtype="float64")
        raw_action = raw_actions[action_col].reindex(df.index)
        action.loc[pre_api_mask] = raw_action.loc[pre_api_mask]
        if action_col in matched_control.columns:
            fallback = matched_control[action_col].reindex(df.index)
            action.loc[pre_api_mask] = action.loc[pre_api_mask].where(
                action.loc[pre_api_mask].notna(),
                fallback.loc[pre_api_mask],
            )
        df[action_col] = action

    if df_action_history is None or df_action_history.empty:
        return df

    time_col = _first_existing_column(
        df_action_history,
        ["reg_date", "send_date", "request_time", "created_at", "create_date", "timestamp"],
    )
    idx_col = _first_existing_column(df_action_history, ["idx", "id"])
    n_idx_col = _first_existing_column(df_action_history, ["n_idx"])
    channel_col = _first_existing_column(df_action_history, ["channel", "device_id"])
    operation_col = _first_existing_column(df_action_history, ["operation"])
    ratio_col = _first_existing_column(df_action_history, ["ratio", "target", "value"])
    if not time_col or not channel_col or not ratio_col:
        return df

    cols = [time_col, channel_col, ratio_col]
    for optional_col in [idx_col, n_idx_col, operation_col]:
        if optional_col and optional_col not in cols:
            cols.append(optional_col)

    log = df_action_history[cols].copy()
    log["reg_date"] = pd.to_datetime(log[time_col], errors="coerce")
    log["channel"] = pd.to_numeric(log[channel_col], errors="coerce").astype("Int64")
    log["ratio"] = pd.to_numeric(log[ratio_col], errors="coerce")
    log["idx_missing"] = True if idx_col is None else _is_missing_text(log[idx_col])
    log["n_idx_missing"] = True if n_idx_col is None else _is_missing_text(log[n_idx_col])
    log["operation_missing"] = True if operation_col is None else _is_missing_text(log[operation_col])
    log["action_col"] = log["channel"].map(ACTION_API_CHANNEL_MAP)

    has_time = log["reg_date"].notna()
    has_action = log["action_col"].notna()
    metadata_missing = log["idx_missing"] | log["n_idx_missing"]
    command_missing = log["channel"].isna() | log["operation_missing"] | log["ratio"].isna()
    valid_log = log[has_time & has_action & ~metadata_missing & ~command_missing].copy()
    blocked_log = log[has_time & has_action & metadata_missing].copy()

    if left.empty:
        return df

    if valid_log.empty:
        api_wide = pd.DataFrame(columns=["reg_date"])
    else:
        api_wide = (
            valid_log.sort_values("reg_date")
            .pivot_table(index="reg_date", columns="action_col", values="ratio", aggfunc="last")
            .reset_index()
        )

    if blocked_log.empty:
        blocked_wide = pd.DataFrame(columns=["reg_date"])
    else:
        blocked_log["blocked"] = True
        blocked_wide = (
            blocked_log.sort_values("reg_date")
            .pivot_table(index="reg_date", columns="action_col", values="blocked", aggfunc="last")
            .reset_index()
        )

    matched_api = left.merge(api_wide, on="reg_date", how="left").set_index("index")
    matched_blocked = (
        left.merge(blocked_wide, on="reg_date", how="left").set_index("index")
        if not blocked_wide.empty
        else pd.DataFrame(index=left["index"])
    )

    for action_col in ACTION_COLUMNS:
        api_action = (
            matched_api[action_col].reindex(df.index)
            if action_col in matched_api.columns
            else pd.Series(np.nan, index=df.index)
        )
        action = api_action.copy()
        if action_col in matched_control.columns:
            fallback = matched_control[action_col].reindex(df.index)
            action = action.where(action.notna(), fallback)
        if action_col in matched_blocked.columns:
            blocked = matched_blocked[action_col].reindex(df.index).fillna(False).astype(bool)
            action = action.mask(blocked & api_action.isna(), np.nan)
        df.loc[post_api_mask, action_col] = action.loc[post_api_mask]

    return df


def prepare_for_sensor_qc(
    df_raw: pd.DataFrame,
    df_action_history: Optional[pd.DataFrame] = None,
    df_control_log: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply raw-column missing handling before sensor QC runs."""

    if df_raw.empty:
        return df_raw
    if "reg_date" not in df_raw.columns:
        return df_raw.iloc[0:0].copy()

    df = df_raw.copy()
    df = apply_action_columns_from_api_history(df, df_action_history, df_control_log)
    for col in [*INTERNAL_STATE_COLUMNS, *SYSTEM_STATE_COLUMNS, *EXTERNAL_STATE_COLUMNS]:
        if col not in df.columns:
            df[col] = np.nan

    df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
    df = df.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)

    _to_numeric_if_present(
        df,
        [
            *INTERNAL_STATE_COLUMNS,
            *SYSTEM_STATE_COLUMNS,
            *EXTERNAL_STATE_COLUMNS,
            *ACTION_COLUMNS,
        ],
    )

    df = apply_realtime_outlier_flags(df)

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

    for col in [
        "cont_heater_run",
        "cont_cooler_run",
        "cont_co2_run",
        "cont_pump1_run",
        "cont_pump2_run",
        "cont_fan_run",
    ]:
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
        "outlier_flag": 0,
        "outlier_domain_flag": 0,
        "outlier_consistency_flag": 0,
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
