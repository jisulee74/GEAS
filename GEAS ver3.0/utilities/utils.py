from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import DbConfig
from infra.repository import Repository
from core.preprocessing import (
    normalize_for_derived,
    latest_row_for_controller,
    slice_today,
)
from core.timeutils import kst_now
from core.solar_eta import compute_eta, integrate_measured_to_jcm2
from core.features import compute_all_features
from policy.controller import PolicyState
from policy.profiles import get_profile
from policy.stage import stage as STAGE_TABLE

def load_policy_state(path: str) -> Optional[PolicyState]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return PolicyState(**d)
    except Exception:
        return None


def save_policy_state(path: str, st: PolicyState) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def get_stage_config() -> Dict[str, Any]:
    name = os.getenv("STAGE_NAME", "stage3")  # 기본값 stage3
    cfg = STAGE_TABLE.get(name)
    if cfg is None:
        raise ValueError(f"Unknown STAGE_NAME: {name}")
    return cfg

def get_base_temperature(stage_cfg: Dict[str, Any]) -> Tuple[float, float]:
    T_day = float(stage_cfg["temp"]["day"])
    T_night = float(stage_cfg["temp"]["night"])
    return T_day, T_night

def get_target_jcm2(stage_cfg: Dict[str, Any]) -> float:
    return float(stage_cfg.get("max_sun_light", 0.0))


def build_params() -> Dict[str, Any]:
    profile_name = os.getenv("CTRL_PROFILE", "safe_default")
    base = get_profile(profile_name)  

    fcu_mode = os.getenv("FCU_MODE", "").lower()
    if fcu_mode in ("cool", "heat"):
        base["fcu_mode"] = fcu_mode
    return base

def build_repository() -> Repository:
    cfg = DbConfig.from_env()
    return Repository(cfg)

def apply_controls(res: Dict[str, Any]) -> None:
    window_pct, _, win_cmd = res["window"]
    curt_mode, curt_size = res["curtain"]
    fcu_mode, fcu_state = res["fcu"]
    (fan_state,) = res["fan"]

    print(
        f"[ACT] window={window_pct:3d}% ({win_cmd}), "
        f"curtain={curt_mode}:{curt_size:3d}%, "
        f"fcu={fcu_mode}:{fcu_state}, "
        f"fan={fan_state}"
    )


def update_df_all(df_all: pd.DataFrame, row: Dict[str, Any]) -> pd.DataFrame:
    df_row = pd.DataFrame([row])
    return pd.concat([df_all, df_row], ignore_index=True)


def compute_today_context(
    df_all: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if "reg_date" not in df_all.columns:
        return None

    now = kst_now()
    df_norm = normalize_for_derived(df_all)
    df_today = slice_today(df_norm, now, col="reg_date")
    if df_today.empty:
        return None
    return df_today

def compute_light_and_kpis(
    df_today: pd.DataFrame,
    lat: float,
    lon: float,
    target_jcm2: float,
    T_day: float,
    T_night: float,
) -> Dict[str, Any]:

    now = kst_now()

    # out_light_sum 갱신
    df_today = df_today.copy()
    df_today["out_light_sum"] = integrate_measured_to_jcm2(
        df_today, col="out_light", time_col="reg_date"
    )

    eta, sunrise, sunset, cs_df = compute_eta(
        lat=lat,
        lon=lon,
        target_jcm2=target_jcm2,
        df_all=df_today,
        now_ts=now,
    )
    cs_sum = cs_df["sum_jcm2"].to_numpy(dtype=float)

    agro_kpis = compute_all_features(
        df_day=df_today,
        sunrise=sunrise,
        sunset=sunset,
        cs_sum_jcm2=cs_sum,
        target_jcm2=target_jcm2,
        light_eta=eta,
        T_day=T_day,
        T_night=T_night,
    )

    latest = latest_row_for_controller(df_today)

    return {
        "now": now,
        "df_today": df_today,
        "latest": latest,
        "sunrise": sunrise,
        "sunset": sunset,
        "eta": eta,
        "cs_sum": cs_sum,
        "agro_kpis": agro_kpis,
    }