"""
3.6.4. ?쒗븳???곗씠???섍꼍?먯꽌??遺遺?寃利?諛⑸쾿濡?

湲곗닠臾몄꽌???섎룄??留욎떠 ?ㅼ쓬 ?덉감瑜??섑뻾?쒕떎.

1. ?ㅼ륫 ?멸린/?쒖뼱 濡쒓렇瑜??꾩쿂由ы빐 怨좎젙??寃쎄퀎議곌굔(boundary)怨?愿痢??됰룞(action)??留뚮뱺??
2. ?뺣낫?됱씠 ?믪? ?대깽??援ш컙??異붿텧?쒕떎.
3. ?대깽??援ш컙?먯꽌??open-loop ?ъ깮 ?ㅼ감瑜??댁슜??臾쇰━ ?뚮씪誘명꽣 posterior瑜?洹쇱궗?쒕떎.
4. posterior ?섑뵆?????媛숈? ?멸린 寃쎄퀎議곌굔?먯꽌 AI ?뺤콉??counterfactual rollout ?쒕떎.
5. KPI 遺꾪룷, q90, CVaR90瑜???ν븳??

二쇱쓽:
- 臾몄꽌?먮뒗 Particle MCMC posterior媛 ?쒖떆?섏?留? 蹂?援ы쁽? ?꾩옣 ?쒖슜?깆쓣 ?꾪븳
  "importance-resampling 湲곕컲 posterior 洹쇱궗"瑜??ъ슜?쒕떎.
- ?듭떖 援ъ“??臾몄꽌? ?숈씪?섍쾶 "?ㅼ륫 濡쒓렇濡?遺덊솗?ㅼ꽦??異뺤냼???? 怨좎젙???멸린
  寃쎄퀎議곌굔?먯꽌 諛섏궗??KPI??瑗щ━?꾪뿕???됯?"?섎뒗 ?먮쫫?대떎.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sqlalchemy import create_engine
except Exception:  # pragma: no cover - optional dependency
    create_engine = None


_UPDATED_ROOT = Path(__file__).resolve().parents[1] / "source"
_INNER_ROOT = _UPDATED_ROOT / "inner_layer"
for _path in (_UPDATED_ROOT, _INNER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from policy.controller import (  # noqa: E402
    sat_vp_kpa_no_data as sat_vp_kpa,
    dewpoint_c_no_data as dewpoint_c,
    vpd_kpa_no_data as vpd_kpa,
    policy_optimal_no_data as policy_optimal,
)


_RESULTS_ROOT = Path(__file__).resolve().parent / "eval_results" / "limited_data"
_DOC_METRIC_COLUMNS = [
    "temp_viol_rate",
    "temp_viol_maxrun",
    "cond_viol_rate",
    "cond_viol_maxrun",
    "rh_viol_rate",
    "vpd_viol_rate",
    "hv_ineff_rate",
    "tv_vent",
    "sw_heat",
]
_THETA_COLUMNS = [
    "UA",
    "C",
    "eta",
    "k_heat",
    "a0",
    "a1",
    "a2",
    "k_evap",
    "k_photo",
    "rho_cp",
]


def connector(
    start_date,
    end_date=None,
    *,
    table: str = "data_silla_enc",
    iot_data_idx: int = 97,
    uri: str = "mysql+pymysql://<user>:<password>@<host>:<port>/<database>",
) -> pd.DataFrame:
    if create_engine is None:
        raise RuntimeError("sqlalchemy is not available in this environment.")
    if end_date is None:
        end_date = datetime.now()
    start_chr = pd.Timestamp(start_date).strftime("%Y-%m-%d %H:%M:%S")
    end_chr = pd.Timestamp(end_date).strftime("%Y-%m-%d %H:%M:%S")
    engine = create_engine(uri)
    query = f"""
        SELECT * FROM {table}
        WHERE iot_data_idx = {int(iot_data_idx)}
          AND reg_date >= '{start_chr}'
          AND reg_date <  '{end_chr}'
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def _format_period_label(start_date: Optional[str], end_date_exclusive: Optional[str]) -> str:
    if not start_date or not end_date_exclusive:
        return ""
    start_str = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_str = pd.Timestamp(end_date_exclusive).strftime("%Y%m%d")
    return f"_{start_str}_to_{end_str}"


def _ensure_results_dir(*, start_date: Optional[str] = None, end_date_exclusive: Optional[str] = None) -> Path:
    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    period_label = _format_period_label(start_date, end_date_exclusive)
    run_dir = _RESULTS_ROOT / f"{datetime.utcnow().strftime('run_%Y%m%d_%H%M%S')}{period_label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _round_floats(value, digits: int = 3):
    if isinstance(value, dict):
        return {str(k): _round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v, digits) for v in value]
    if isinstance(value, tuple):
        return tuple(_round_floats(v, digits) for v in value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, digits)
    return value


def numify(x):
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    x = x.astype(str).str.replace(",", "", regex=False).str.strip()
    x[x.isin(["", "NA", "NaN", "NULL", "None"])] = np.nan
    return pd.to_numeric(x, errors="coerce")


def sat_vp_kpa_arr(Tc) -> np.ndarray:
    Tc_arr = np.asarray(Tc, dtype=float)
    return 0.61078 * np.exp((17.2694 * Tc_arr) / (Tc_arr + 237.3))


def dewpoint_c_arr(Tc, RH) -> np.ndarray:
    Tc_arr = np.asarray(Tc, dtype=float)
    RH_arr = np.asarray(RH, dtype=float)
    es = sat_vp_kpa_arr(Tc_arr)
    e = np.clip(RH_arr * es, 1e-6, es)
    ln_ratio = np.log(e / 0.61078)
    return (237.3 * ln_ratio) / (17.2694 - ln_ratio)


def vpd_kpa_arr(Tc, RH) -> np.ndarray:
    Tc_arr = np.asarray(Tc, dtype=float)
    RH_arr = np.asarray(RH, dtype=float)
    es = sat_vp_kpa_arr(Tc_arr)
    ea = np.clip(RH_arr * es, 0.0, es)
    return np.maximum(0.0, es - ea)


def max_run(b: np.ndarray) -> int:
    if not b.any():
        return 0
    count = 0
    best = 0
    for v in b:
        count = count + 1 if bool(v) else 0
        best = max(best, count)
    return best


def cvar(x: np.ndarray, q: float = 0.9) -> float:
    x = x[np.isfinite(x)]
    thr = float(np.quantile(x, q))
    return float(x[x >= thr].mean())


def calc_metrics(
    df: pd.DataFrame,
    *,
    T_min=12.0,
    T_max=28.0,
    RH_max=0.90,
    VPD_min=0.30,
    dTcond_min=0.8,
    alpha=0.25,
) -> pd.DataFrame:
    temp_viol = (df["Tin"] < T_min) | (df["Tin"] > T_max)
    cond_viol = df["dTcond"] < dTcond_min
    rh_viol = df["RHin"] > RH_max
    vpd_viol = df["VPD"] < VPD_min

    idx_heat = df["u_heat"] > 0.5
    if idx_heat.any():
        hv_ineff = float(
            (
                df.loc[idx_heat, "Q_ventloss"]
                > alpha * df.loc[idx_heat, "Q_heat"].clip(lower=1e-6)
            ).mean()
        )
    else:
        hv_ineff = 0.0

    return pd.DataFrame(
        [
            {
                "temp_viol_rate": float(temp_viol.mean()),
                "temp_viol_maxrun": max_run(temp_viol.values),
                "cond_viol_rate": float(cond_viol.mean()),
                "cond_viol_maxrun": max_run(cond_viol.values),
                "rh_viol_rate": float(rh_viol.mean()),
                "vpd_viol_rate": float(vpd_viol.mean()),
                "hv_ineff_rate": hv_ineff,
                "tv_vent": float(df["x_vent"].diff().abs().sum()),
                "sw_heat": int((df["u_heat"].diff().abs() > 0).sum()),
            }
        ]
    )


def ach_model(x_vent: float, wind: float, a0: float, a1: float, a2: float, ACH_max: float = 15.0) -> float:
    ach = a0 + a1 * x_vent + a2 * wind * x_vent
    return float(np.clip(ach, 0.0, ACH_max))


def step_dynamics_dt(
    state: Dict[str, float],
    u: Dict[str, float],
    a: Dict[str, float],
    theta: Dict[str, float],
    *,
    dt_sec: float,
    A: float = 200.0,
    V: float = 300.0,
    Imax: float = 650.0,
    T0: float = 5.0,
) -> Dict[str, float]:
    Tin = float(state["Tin"])
    e_in = float(state["e_in"])
    CO2 = float(state["CO2"])

    ACH = ach_model(float(a["x_vent"]), float(u["wind"]), float(theta["a0"]), float(theta["a1"]), float(theta["a2"]))
    It_eff = float(u["It"]) * (1.0 - float(a["curtain"]))

    Q_trans = float(theta["UA"]) * (float(u["Tout"]) - Tin)
    Q_solar = float(theta["eta"]) * A * It_eff
    Q_heat = float(theta["k_heat"]) * float(a["u_heat"])
    Q_vent = float(theta["rho_cp"]) * V * (ACH / 3600.0) * (float(u["Tout"]) - Tin)
    Tin_next = Tin + (dt_sec / float(theta["C"])) * (Q_trans + Q_solar + Q_heat + Q_vent)

    e_out = float(u["RHout"]) * float(sat_vp_kpa(float(u["Tout"])))
    evap = float(theta["k_evap"]) * (It_eff / Imax) * max(0.0, Tin - T0)
    e_next = max(0.05, e_in + dt_sec * ((ACH / 3600.0) * (e_out - e_in) + evap))

    RHin = min(1.0, max(0.0, e_next / float(sat_vp_kpa(Tin_next))))
    VPD = float(vpd_kpa(Tin_next, RHin))
    gT = min(1.0, max(0.0, (Tin_next - 5.0) / 20.0))
    gV = min(1.0, max(0.0, VPD / 1.2))
    uptake = float(theta["k_photo"]) * It_eff * gT * gV * 1e6
    inj = 50.0 * float(a["u_co2"])
    CO2out = float(u.get("CO2out", 420.0))
    CO2_next = max(300.0, CO2 + dt_sec * ((ACH / 3600.0) * (CO2out - CO2) + inj - uptake))

    Q_ventloss = max(
        0.0,
        float(theta["rho_cp"]) * V * (ACH / 3600.0) * max(0.0, Tin_next - float(u["Tout"])),
    )
    return {
        "Tin": Tin_next,
        "e_in": e_next,
        "CO2": CO2_next,
        "ACH": ACH,
        "It_eff": It_eff,
        "Q_heat": Q_heat,
        "Q_ventloss": Q_ventloss,
    }


def project_action(a: Dict[str, float], prev_a: Dict[str, float], ramp_max: float = 0.15) -> Dict[str, float]:
    projected = dict(a)
    projected["u_heat"] = 1.0 if float(projected["u_heat"]) > 0.5 else 0.0
    projected["u_co2"] = 1.0 if float(projected["u_co2"]) > 0.5 else 0.0
    projected["x_vent"] = float(np.clip(projected["x_vent"], 0.0, 1.0))
    projected["curtain"] = float(np.clip(projected["curtain"], 0.0, 1.0))
    # 臾몄꽌???덉쟾?ъ쁺 痍⑥???留욎떠 ?쒕갑-?섍린 ?숈떆 ?ъ슜? 湲덉??쒕떎.
    if projected["u_heat"] > 0.5:
        projected["x_vent"] = 0.0
    dx = float(projected["x_vent"]) - float(prev_a["x_vent"])
    dx = float(np.clip(dx, -ramp_max, ramp_max))
    projected["x_vent"] = float(prev_a["x_vent"]) + dx
    return projected


def prepare_timeseries(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["reg_date"] = pd.to_datetime(df["reg_date"], utc=False).dt.tz_localize(None)
    df = df.sort_values("reg_date").reset_index(drop=True)

    numeric_cols = [
        "in_temp",
        "in_hum",
        "in_co2",
        "out_temp",
        "out_hum",
        "out_light",
        "out_windsp",
        "out_co2",
        "cont_heater_run",
        "cont_skyl_vol",
        "cont_cur_vol",
        "cont_co2_run",
        "etc_blackout",
        "etc_plc_abnorm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = numify(df[col])

    for col in ["etc_blackout", "etc_plc_abnorm"]:
        if col in df.columns:
            df = df[(df[col].isna()) | (df[col] == 0)]

    if "out_co2" not in df.columns or df["out_co2"].isna().all():
        df["out_co2"] = 420.0
    else:
        df["out_co2"] = numify(df["out_co2"]).fillna(420.0)

    required = ["in_temp", "in_hum", "out_temp", "out_hum", "out_light", "out_windsp"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["reg_date", "in_temp", "in_hum", "out_temp", "out_hum"]).copy()
    df = df.drop_duplicates(subset=["reg_date"]).reset_index(drop=True)

    df["dt_sec"] = df["reg_date"].diff().dt.total_seconds()
    med_dt = float(df["dt_sec"].median()) if df["dt_sec"].notna().any() else 300.0
    if not np.isfinite(med_dt) or med_dt <= 0:
        med_dt = 300.0
    df["dt_sec"] = df["dt_sec"].apply(lambda x: med_dt if (pd.isna(x) or x <= 0) else x)

    df["Tout"] = numify(df["out_temp"])
    df["RHout"] = np.clip(numify(df["out_hum"]) / 100.0, 0.0, 1.0)
    df["It"] = numify(df["out_light"]).clip(lower=0.0)
    df["wind"] = numify(df["out_windsp"]).fillna(0.5).clip(lower=0.1)
    df["CO2out"] = numify(df["out_co2"]).fillna(420.0)

    df["Tin_obs"] = numify(df["in_temp"])
    df["RHin_obs"] = np.clip(numify(df["in_hum"]) / 100.0, 0.0, 1.0)
    df["CO2_obs"] = numify(df["in_co2"]).ffill().bfill()
    if df["CO2_obs"].isna().all():
        df["CO2_obs"] = 900.0

    df["e_in_obs"] = df["RHin_obs"] * sat_vp_kpa_arr(df["Tin_obs"].values)
    df["VPD_obs"] = vpd_kpa_arr(df["Tin_obs"].values, df["RHin_obs"].values)
    df["Td_obs"] = dewpoint_c_arr(df["Tin_obs"].values, df["RHin_obs"].values)
    df["dTcond_obs"] = df["Tin_obs"] - df["Td_obs"]
    return df.reset_index(drop=True)


def make_inputs_actions(
    df: pd.DataFrame,
    *,
    vent_scale: float = 100.0,
    curtain_scale: float = 100.0,
) -> Dict[str, pd.DataFrame]:
    boundary = df.assign(k=np.arange(1, len(df) + 1))[[
        "k",
        "reg_date",
        "dt_sec",
        "Tout",
        "RHout",
        "It",
        "wind",
        "CO2out",
    ]].copy()

    action = pd.DataFrame(
        {
            "u_heat": (numify(df.get("cont_heater_run", 0)) > 0).astype(float),
            "x_vent": np.clip(numify(df.get("cont_skyl_vol", 0)).fillna(0.0) / vent_scale, 0.0, 1.0),
            "curtain": np.clip(numify(df.get("cont_cur_vol", 0)).fillna(0.0) / curtain_scale, 0.0, 1.0),
            "u_co2": (numify(df.get("cont_co2_run", 0)).fillna(0.0) > 0).astype(float),
        }
    )
    return {"boundary": boundary, "action": action}


def _build_episode_frame(
    boundary: pd.DataFrame,
    *,
    Tin: np.ndarray,
    e_in: np.ndarray,
    CO2: np.ndarray,
    u_heat: np.ndarray,
    x_vent: np.ndarray,
    curtain: np.ndarray,
    u_co2: np.ndarray,
    ACH: np.ndarray,
    Q_heat: np.ndarray,
    Q_ventloss: np.ndarray,
) -> pd.DataFrame:
    RHin = np.clip(e_in / sat_vp_kpa_arr(Tin), 0.0, 1.0)
    VPD = vpd_kpa_arr(Tin, RHin)
    Td = dewpoint_c_arr(Tin, RHin)
    dTcond = Tin - Td
    return pd.DataFrame(
        {
            "k": boundary["k"].values,
            "reg_date": boundary["reg_date"].values if "reg_date" in boundary else np.arange(len(boundary)),
            "Tout": boundary["Tout"].values,
            "RHout": boundary["RHout"].values,
            "It": boundary["It"].values,
            "wind": boundary["wind"].values,
            "CO2out": boundary["CO2out"].values,
            "Tin": Tin,
            "RHin": RHin,
            "e_in": e_in,
            "VPD": VPD,
            "dTcond": dTcond,
            "CO2": CO2,
            "u_heat": u_heat,
            "x_vent": x_vent,
            "curtain": curtain,
            "u_co2": u_co2,
            "ACH": ACH,
            "Q_heat": Q_heat,
            "Q_ventloss": Q_ventloss,
        }
    )


def simulate_openloop(
    boundary: pd.DataFrame,
    action: pd.DataFrame,
    theta: Dict[str, float],
    *,
    Tin0: float,
    RHin0: float,
    CO20: float,
    A: float = 200.0,
    V: float = 300.0,
    ramp_max: float = 0.15,
    show_progress: bool = False,
) -> pd.DataFrame:
    n = len(boundary)
    Tin = np.zeros(n)
    e_in = np.zeros(n)
    CO2 = np.zeros(n)
    Tin[0] = Tin0
    e_in[0] = RHin0 * float(sat_vp_kpa(Tin0))
    CO2[0] = CO20

    u_heat = np.zeros(n)
    x_vent = np.zeros(n)
    curtain = np.zeros(n)
    u_co2 = np.zeros(n)
    ACH = np.zeros(n)
    Q_heat = np.zeros(n)
    Q_ventloss = np.zeros(n)

    prev_a = {col: float(action[col].iloc[0]) for col in ["u_heat", "x_vent", "curtain", "u_co2"]}
    iterator: Iterable[int] = range(1, n)
    if show_progress:
        iterator = tqdm(iterator, desc="simulate_openloop")

    for t in iterator:
        state = {"Tin": Tin[t - 1], "e_in": e_in[t - 1], "CO2": CO2[t - 1]}
        u = boundary.iloc[t - 1].to_dict()
        a_raw = {col: float(action[col].iloc[t - 1]) for col in ["u_heat", "x_vent", "curtain", "u_co2"]}
        a = project_action(a_raw, prev_a, ramp_max=ramp_max)
        pred = step_dynamics_dt(state, u, a, theta, dt_sec=float(u["dt_sec"]), A=A, V=V)

        Tin[t] = pred["Tin"]
        e_in[t] = pred["e_in"]
        CO2[t] = pred["CO2"]
        u_heat[t] = a["u_heat"]
        x_vent[t] = a["x_vent"]
        curtain[t] = a["curtain"]
        u_co2[t] = a["u_co2"]
        ACH[t] = pred["ACH"]
        Q_heat[t] = pred["Q_heat"]
        Q_ventloss[t] = pred["Q_ventloss"]
        prev_a = a

    return _build_episode_frame(
        boundary,
        Tin=Tin,
        e_in=e_in,
        CO2=CO2,
        u_heat=u_heat,
        x_vent=x_vent,
        curtain=curtain,
        u_co2=u_co2,
        ACH=ACH,
        Q_heat=Q_heat,
        Q_ventloss=Q_ventloss,
    )


def simulate_counterfactual_policy(
    boundary: pd.DataFrame,
    theta: Dict[str, float],
    weights: Dict[str, float],
    *,
    Tin0: float,
    RHin0: float,
    CO20: float,
    A: float = 200.0,
    V: float = 300.0,
    Tref: Optional[float] = None,
    T_min: float = 12.0,
    T_max: float = 28.0,
    VPD_min: float = 0.30,
    dTcond_min: float = 0.8,
    alpha: float = 0.25,
    ramp_max: float = 0.15,
    show_progress: bool = False,
) -> pd.DataFrame:
    n = len(boundary)
    Tin = np.zeros(n)
    e_in = np.zeros(n)
    CO2 = np.zeros(n)
    Tin[0] = Tin0
    e_in[0] = RHin0 * float(sat_vp_kpa(Tin0))
    CO2[0] = CO20

    if Tref is None:
        Tref = 0.5 * (T_min + T_max)

    u_heat = np.zeros(n)
    x_vent = np.zeros(n)
    curtain = np.zeros(n)
    u_co2 = np.zeros(n)
    ACH = np.zeros(n)
    Q_heat = np.zeros(n)
    Q_ventloss = np.zeros(n)
    prev_a = {"u_heat": 0.0, "x_vent": 0.0, "curtain": 0.0, "u_co2": 0.0}

    iterator: Iterable[int] = range(1, n)
    if show_progress:
        iterator = tqdm(iterator, desc="simulate_counterfactual_policy")

    for t in iterator:
        state = {"Tin": Tin[t - 1], "e_in": e_in[t - 1], "CO2": CO2[t - 1]}
        u = boundary.iloc[t - 1].to_dict()
        dt_min = max(1, int(round(float(u["dt_sec"]) / 60.0)))
        a = policy_optimal(
            state,
            u,
            prev_a,
            theta,
            weights=weights,
            Tref=Tref,
            VPD_min=VPD_min,
            T_min=T_min,
            T_max=T_max,
            dTcond_min=dTcond_min,
            alpha=alpha,
            dt_min=dt_min,
            A=A,
            V=V,
            ramp_max=ramp_max,
        )
        pred = step_dynamics_dt(state, u, a, theta, dt_sec=float(u["dt_sec"]), A=A, V=V)

        Tin[t] = pred["Tin"]
        e_in[t] = pred["e_in"]
        CO2[t] = pred["CO2"]
        u_heat[t] = a["u_heat"]
        x_vent[t] = a["x_vent"]
        curtain[t] = a["curtain"]
        u_co2[t] = a["u_co2"]
        ACH[t] = pred["ACH"]
        Q_heat[t] = pred["Q_heat"]
        Q_ventloss[t] = pred["Q_ventloss"]
        prev_a = a

    return _build_episode_frame(
        boundary,
        Tin=Tin,
        e_in=e_in,
        CO2=CO2,
        u_heat=u_heat,
        x_vent=x_vent,
        curtain=curtain,
        u_co2=u_co2,
        ACH=ACH,
        Q_heat=Q_heat,
        Q_ventloss=Q_ventloss,
    )


def sample_theta_prior(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "UA": float(rng.lognormal(mean=np.log(3500.0), sigma=0.35)),
        "C": float(rng.lognormal(mean=np.log(1.5e7), sigma=0.55)),
        "eta": float(rng.beta(2.5, 2.5)),
        "k_heat": float(rng.lognormal(mean=np.log(30000.0), sigma=0.35)),
        "a0": float(np.clip(rng.lognormal(mean=np.log(0.18), sigma=0.45), 0.01, 1.5)),
        "a1": float(np.clip(rng.lognormal(mean=np.log(4.5), sigma=0.40), 0.2, 15.0)),
        "a2": float(np.clip(rng.lognormal(mean=np.log(0.9), sigma=0.45), 0.05, 4.0)),
        "k_evap": float(np.clip(rng.lognormal(mean=np.log(2.5e-6), sigma=0.40), 1e-7, 1e-4)),
        "k_photo": float(np.clip(rng.lognormal(mean=np.log(2.5e-5), sigma=0.45), 1e-6, 1e-3)),
        "rho_cp": 1.2 * 1005,
    }


def _theta_to_unconstrained(theta: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            np.log(float(theta["UA"])),
            np.log(float(theta["C"])),
            np.log(float(theta["k_heat"])),
            np.log(float(theta["a0"])),
            np.log(float(theta["a1"])),
            np.log(float(theta["a2"])),
            np.log(float(theta["k_evap"])),
            np.log(float(theta["k_photo"])),
            float(np.log(np.clip(float(theta["eta"]), 1e-6, 1 - 1e-6) / np.clip(1.0 - float(theta["eta"]), 1e-6, 1.0))),
        ],
        dtype=float,
    )


def _theta_from_unconstrained(z: np.ndarray) -> Dict[str, float]:
    eta = 1.0 / (1.0 + np.exp(-float(z[8])))
    return {
        "UA": float(np.exp(z[0])),
        "C": float(np.exp(z[1])),
        "k_heat": float(np.exp(z[2])),
        "a0": float(np.exp(z[3])),
        "a1": float(np.exp(z[4])),
        "a2": float(np.exp(z[5])),
        "k_evap": float(np.exp(z[6])),
        "k_photo": float(np.exp(z[7])),
        "eta": float(np.clip(eta, 1e-6, 1.0 - 1e-6)),
        "rho_cp": 1.2 * 1005,
    }


def _theta_log_prior(theta: Dict[str, float]) -> float:
    def _log_lognorm(x: float, mean_log: float, sigma: float) -> float:
        if not np.isfinite(x) or x <= 0:
            return -np.inf
        z = (np.log(x) - mean_log) / sigma
        return float(-np.log(x) - np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * z * z)

    def _log_beta(x: float, a: float, b: float) -> float:
        if not np.isfinite(x) or x <= 0 or x >= 1:
            return -np.inf
        return float((a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))

    return float(
        _log_lognorm(float(theta["UA"]), np.log(3500.0), 0.35)
        + _log_lognorm(float(theta["C"]), np.log(1.5e7), 0.55)
        + _log_beta(float(theta["eta"]), 2.5, 2.5)
        + _log_lognorm(float(theta["k_heat"]), np.log(30000.0), 0.35)
        + _log_lognorm(float(theta["a0"]), np.log(0.18), 0.45)
        + _log_lognorm(float(theta["a1"]), np.log(4.5), 0.40)
        + _log_lognorm(float(theta["a2"]), np.log(0.9), 0.45)
        + _log_lognorm(float(theta["k_evap"]), np.log(2.5e-6), 0.40)
        + _log_lognorm(float(theta["k_photo"]), np.log(2.5e-5), 0.45)
    )


def _normal_logpdf(x: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    z = (x - mean) / sigma
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * z * z


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    indexes = np.zeros(n, dtype=int)
    i = 0
    j = 0
    while i < n:
        if positions[i] < cumsum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def select_event_windows(
    df_ts: pd.DataFrame,
    action: pd.DataFrame,
    *,
    max_events: int = 12,
    window_steps: int = 12,
) -> pd.DataFrame:
    n = len(df_ts)
    if n < 4:
        raise ValueError("Need at least 4 rows to build event windows.")

    score = np.zeros(n, dtype=float)
    labels = [[] for _ in range(n)]

    heater_change = action["u_heat"].diff().abs().fillna(0.0)
    vent_change = action["x_vent"].diff().abs().fillna(0.0)
    curtain_change = action["curtain"].diff().abs().fillna(0.0)
    tout_jump = df_ts["Tout"].diff().abs().fillna(0.0)

    # 臾몄꽌???쒗븳???곗씠??遺遺??앸퀎 ?ㅻ챸?먯꽌 媛뺤“??援ш컙:
    # ?쇨컙 臾댁씪?? ?덊꽣 ?ㅽ뀦, ?섍린 ?ㅽ뀦, 留묒? 二쇨컙
    light_q90 = float(df_ts["It"].quantile(0.90))
    light_q50 = float(df_ts["It"].quantile(0.50))
    low_light = (df_ts["It"] <= max(1.0, light_q50 * 0.05)).astype(float)
    high_light = (df_ts["It"] >= max(light_q90, 1.0)).astype(float)
    night_no_solar = (low_light * (action["u_heat"] > 0).astype(float)).astype(float)
    clear_day = (high_light * (action["curtain"] <= 0.1).astype(float)).astype(float)

    components = [
        (heater_change * 6.0, "heater_step"),
        (vent_change * 5.0, "vent_step"),
        (night_no_solar * 4.0, "night_no_solar"),
        (clear_day * 3.5, "clear_day"),
        (curtain_change * 1.5, "curtain_step"),
        (tout_jump * 1.0, "tout_jump"),
    ]
    for values, label in components:
        idx = np.flatnonzero(values.values > 0)
        score += values.values
        for i in idx:
            labels[i].append(label)

    order = np.argsort(score)[::-1]
    chosen: List[Dict[str, object]] = []
    occupied: List[Tuple[int, int]] = []
    half = max(2, window_steps // 2)

    for idx in order:
        if score[idx] <= 0:
            break
        start = max(0, int(idx) - half)
        end = min(n, start + window_steps)
        start = max(0, end - window_steps)
        if end - start < 4:
            continue
        overlap = any(not (end <= s or start >= e) for s, e in occupied)
        if overlap:
            continue
        chosen.append(
            {
                "event_id": len(chosen) + 1,
                "start_idx": start,
                "end_idx": end,
                "event_score": float(score[idx]),
                "event_tags": ",".join(sorted(set(labels[idx]))),
                "start_time": df_ts["reg_date"].iloc[start],
                "end_time": df_ts["reg_date"].iloc[end - 1],
            }
        )
        occupied.append((start, end))
        if len(chosen) >= max_events:
            break

    if not chosen:
        chosen = [
            {
                "event_id": 1,
                "start_idx": 0,
                "end_idx": min(n, window_steps),
                "event_score": 1.0,
                "event_tags": "fallback",
                "start_time": df_ts["reg_date"].iloc[0],
                "end_time": df_ts["reg_date"].iloc[min(n, window_steps) - 1],
            }
        ]

    return pd.DataFrame(chosen)


def _slice_window(
    boundary: pd.DataFrame,
    action: pd.DataFrame,
    df_obs: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    boundary_w = boundary.iloc[start_idx:end_idx].reset_index(drop=True).copy()
    boundary_w["k"] = np.arange(1, len(boundary_w) + 1)
    action_w = action.iloc[start_idx:end_idx].reset_index(drop=True).copy()
    df_w = df_obs.iloc[start_idx:end_idx].reset_index(drop=True).copy()
    return boundary_w, action_w, df_w


def event_fit_details_openloop(
    boundary: pd.DataFrame,
    action: pd.DataFrame,
    df_obs: pd.DataFrame,
    events: pd.DataFrame,
    theta: Dict[str, float],
    *,
    A: float = 200.0,
    V: float = 300.0,
    ramp_max: float = 0.15,
    sigma_t: float = 1.5,
    sigma_e: float = 0.20,
    sigma_co2: float = 80.0,
) -> pd.DataFrame:
    details: List[Dict[str, float]] = []
    for row in events.itertuples(index=False):
        boundary_w, action_w, df_w = _slice_window(boundary, action, df_obs, int(row.start_idx), int(row.end_idx))
        sim = simulate_openloop(
            boundary_w,
            action_w,
            theta,
            Tin0=float(df_w["Tin_obs"].iloc[0]),
            RHin0=float(df_w["RHin_obs"].iloc[0]),
            CO20=float(df_w["CO2_obs"].iloc[0]),
            A=A,
            V=V,
            ramp_max=ramp_max,
            show_progress=False,
        )
        loss = float(
            np.nanmean(
                ((sim["Tin"].values - df_w["Tin_obs"].values) / sigma_t) ** 2
                + ((sim["e_in"].values - df_w["e_in_obs"].values) / sigma_e) ** 2
                + ((sim["CO2"].values - df_w["CO2_obs"].values) / sigma_co2) ** 2
            )
        )
        details.append(
            {
                "event_id": int(row.event_id),
                "start_idx": int(row.start_idx),
                "end_idx": int(row.end_idx),
                "event_score": float(row.event_score),
                "loss": loss,
                "rmse_t": float(np.sqrt(np.nanmean((sim["Tin"].values - df_w["Tin_obs"].values) ** 2))),
                "rmse_rh": float(np.sqrt(np.nanmean((sim["RHin"].values - df_w["RHin_obs"].values) ** 2))),
                "rmse_co2": float(np.sqrt(np.nanmean((sim["CO2"].values - df_w["CO2_obs"].values) ** 2))),
            }
        )
    return pd.DataFrame(details)


def particle_filter_loglik(
    boundary: pd.DataFrame,
    action: pd.DataFrame,
    df_obs: pd.DataFrame,
    events: pd.DataFrame,
    theta: Dict[str, float],
    *,
    n_particles: int,
    rng: np.random.Generator,
    A: float = 200.0,
    V: float = 300.0,
    ramp_max: float = 0.15,
    sigma_t_obs: float = 1.5,
    sigma_e_obs: float = 0.20,
    sigma_co2_obs: float = 80.0,
    sigma_t_proc: float = 0.30,
    sigma_e_proc: float = 0.03,
    sigma_co2_proc: float = 20.0,
    ess_ratio: float = 0.5,
) -> float:
    total_loglik = 0.0

    for row in events.itertuples(index=False):
        boundary_w, action_w, df_w = _slice_window(boundary, action, df_obs, int(row.start_idx), int(row.end_idx))
        m = len(boundary_w)
        if m < 2:
            continue

        tin = rng.normal(float(df_w["Tin_obs"].iloc[0]), sigma_t_obs, size=n_particles)
        e_in = rng.normal(float(df_w["e_in_obs"].iloc[0]), sigma_e_obs, size=n_particles)
        co2 = rng.normal(float(df_w["CO2_obs"].iloc[0]), sigma_co2_obs, size=n_particles)
        e_in = np.maximum(0.05, e_in)
        co2 = np.maximum(300.0, co2)
        weights = np.full(n_particles, 1.0 / n_particles)

        obs0_t = float(df_w["Tin_obs"].iloc[0])
        obs0_e = float(df_w["e_in_obs"].iloc[0])
        obs0_c = float(df_w["CO2_obs"].iloc[0])
        ll0 = (
            _normal_logpdf(tin, np.full(n_particles, obs0_t), sigma_t_obs)
            + _normal_logpdf(e_in, np.full(n_particles, obs0_e), sigma_e_obs)
            + _normal_logpdf(co2, np.full(n_particles, obs0_c), sigma_co2_obs)
        )
        max_ll0 = float(np.max(ll0))
        w0 = np.exp(ll0 - max_ll0)
        mean_w0 = float(np.mean(w0))
        if mean_w0 <= 0 or not np.isfinite(mean_w0):
            return -np.inf
        total_loglik += max_ll0 + np.log(mean_w0)
        weights = w0 / np.sum(w0)

        prev_u_heat = float(action_w["u_heat"].iloc[0])
        prev_x_vent = float(action_w["x_vent"].iloc[0])
        prev_curtain = float(action_w["curtain"].iloc[0])
        prev_u_co2 = float(action_w["u_co2"].iloc[0])

        for t in range(1, m):
            u = boundary_w.iloc[t - 1].to_dict()
            a_raw = {col: float(action_w[col].iloc[t - 1]) for col in ["u_heat", "x_vent", "curtain", "u_co2"]}

            new_tin = np.empty(n_particles)
            new_e_in = np.empty(n_particles)
            new_co2 = np.empty(n_particles)
            new_prev_u_heat = np.empty(n_particles)
            new_prev_x_vent = np.empty(n_particles)
            new_prev_curtain = np.empty(n_particles)
            new_prev_u_co2 = np.empty(n_particles)

            for i in range(n_particles):
                prev_a = {
                    "u_heat": float(prev_u_heat if np.isscalar(prev_u_heat) else prev_u_heat[i]),
                    "x_vent": float(prev_x_vent if np.isscalar(prev_x_vent) else prev_x_vent[i]),
                    "curtain": float(prev_curtain if np.isscalar(prev_curtain) else prev_curtain[i]),
                    "u_co2": float(prev_u_co2 if np.isscalar(prev_u_co2) else prev_u_co2[i]),
                }
                a = project_action(a_raw, prev_a, ramp_max=ramp_max)
                state = {"Tin": float(tin[i]), "e_in": float(e_in[i]), "CO2": float(co2[i])}
                pred = step_dynamics_dt(state, u, a, theta, dt_sec=float(u["dt_sec"]), A=A, V=V)

                new_tin[i] = pred["Tin"] + rng.normal(0.0, sigma_t_proc)
                new_e_in[i] = max(0.05, pred["e_in"] + rng.normal(0.0, sigma_e_proc))
                new_co2[i] = max(300.0, pred["CO2"] + rng.normal(0.0, sigma_co2_proc))
                new_prev_u_heat[i] = a["u_heat"]
                new_prev_x_vent[i] = a["x_vent"]
                new_prev_curtain[i] = a["curtain"]
                new_prev_u_co2[i] = a["u_co2"]

            tin = new_tin
            e_in = new_e_in
            co2 = new_co2
            prev_u_heat = new_prev_u_heat
            prev_x_vent = new_prev_x_vent
            prev_curtain = new_prev_curtain
            prev_u_co2 = new_prev_u_co2

            obs_t = float(df_w["Tin_obs"].iloc[t])
            obs_e = float(df_w["e_in_obs"].iloc[t])
            obs_c = float(df_w["CO2_obs"].iloc[t])
            ll = (
                _normal_logpdf(tin, np.full(n_particles, obs_t), sigma_t_obs)
                + _normal_logpdf(e_in, np.full(n_particles, obs_e), sigma_e_obs)
                + _normal_logpdf(co2, np.full(n_particles, obs_c), sigma_co2_obs)
            )
            max_ll = float(np.max(ll))
            w = weights * np.exp(ll - max_ll)
            sum_w = float(np.sum(w))
            if sum_w <= 0 or not np.isfinite(sum_w):
                return -np.inf
            total_loglik += max_ll + np.log(sum_w)
            weights = w / sum_w

            ess = 1.0 / float(np.sum(weights ** 2))
            if ess < ess_ratio * n_particles:
                idx = _systematic_resample(weights, rng)
                tin = tin[idx]
                e_in = e_in[idx]
                co2 = co2[idx]
                prev_u_heat = prev_u_heat[idx]
                prev_x_vent = prev_x_vent[idx]
                prev_curtain = prev_curtain[idx]
                prev_u_co2 = prev_u_co2[idx]
                weights = np.full(n_particles, 1.0 / n_particles)

    return float(total_loglik)


def fit_posterior_pmmh(
    boundary: pd.DataFrame,
    action: pd.DataFrame,
    df_obs: pd.DataFrame,
    events: pd.DataFrame,
    *,
    pmmh_iters: int = 180,
    burn_in: int = 60,
    thin: int = 2,
    n_particles: int = 64,
    seed: int = 42,
    proposal_scales: Optional[Dict[str, float]] = None,
    A: float = 200.0,
    V: float = 300.0,
    ramp_max: float = 0.15,
    sigma_t_obs: float = 1.5,
    sigma_e_obs: float = 0.20,
    sigma_co2_obs: float = 80.0,
    sigma_t_proc: float = 0.30,
    sigma_e_proc: float = 0.03,
    sigma_co2_proc: float = 20.0,
    ess_ratio: float = 0.5,
    show_progress: bool = True,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    if proposal_scales is None:
        proposal_scales = {
            "UA": 0.08,
            "C": 0.08,
            "k_heat": 0.08,
            "a0": 0.06,
            "a1": 0.06,
            "a2": 0.06,
            "k_evap": 0.08,
            "k_photo": 0.08,
            "eta": 0.12,
        }
    proposal_vec = np.array(
        [
            proposal_scales["UA"],
            proposal_scales["C"],
            proposal_scales["k_heat"],
            proposal_scales["a0"],
            proposal_scales["a1"],
            proposal_scales["a2"],
            proposal_scales["k_evap"],
            proposal_scales["k_photo"],
            proposal_scales["eta"],
        ],
        dtype=float,
    )

    current_theta = sample_theta_prior(rng)
    current_log_prior = _theta_log_prior(current_theta)
    current_loglik = particle_filter_loglik(
        boundary,
        action,
        df_obs,
        events,
        current_theta,
        n_particles=n_particles,
        rng=rng,
        A=A,
        V=V,
        ramp_max=ramp_max,
        sigma_t_obs=sigma_t_obs,
        sigma_e_obs=sigma_e_obs,
        sigma_co2_obs=sigma_co2_obs,
        sigma_t_proc=sigma_t_proc,
        sigma_e_proc=sigma_e_proc,
        sigma_co2_proc=sigma_co2_proc,
        ess_ratio=ess_ratio,
    )
    current_z = _theta_to_unconstrained(current_theta)

    rows: List[Dict[str, float]] = []
    accepted = 0
    iterator: Iterable[int] = range(pmmh_iters)
    if show_progress:
        iterator = tqdm(iterator, desc="posterior_pmmh")

    for i in iterator:
        proposal_z = current_z + rng.normal(0.0, proposal_vec)
        proposal_theta = _theta_from_unconstrained(proposal_z)
        proposal_log_prior = _theta_log_prior(proposal_theta)
        if np.isfinite(proposal_log_prior):
            proposal_loglik = particle_filter_loglik(
                boundary,
                action,
                df_obs,
                events,
                proposal_theta,
                n_particles=n_particles,
                rng=rng,
                A=A,
                V=V,
                ramp_max=ramp_max,
                sigma_t_obs=sigma_t_obs,
                sigma_e_obs=sigma_e_obs,
                sigma_co2_obs=sigma_co2_obs,
                sigma_t_proc=sigma_t_proc,
                sigma_e_proc=sigma_e_proc,
                sigma_co2_proc=sigma_co2_proc,
                ess_ratio=ess_ratio,
            )
        else:
            proposal_loglik = -np.inf

        log_accept_ratio = (proposal_log_prior + proposal_loglik) - (current_log_prior + current_loglik)
        is_accept = bool(np.log(rng.random()) < min(0.0, log_accept_ratio))
        if is_accept:
            current_theta = proposal_theta
            current_log_prior = proposal_log_prior
            current_loglik = proposal_loglik
            current_z = proposal_z
            accepted += 1

        row = {
            "iter": i + 1,
            "accepted": int(is_accept),
            "accept_rate_running": accepted / float(i + 1),
            "log_prior": float(current_log_prior),
            "loglik": float(current_loglik),
            "logpost": float(current_log_prior + current_loglik),
        }
        row.update({k: float(current_theta[k]) for k in _THETA_COLUMNS})
        rows.append(row)

    chain_df = pd.DataFrame(rows)
    posterior_df = chain_df.iloc[int(burn_in) :].copy().reset_index(drop=True)
    if thin > 1:
        posterior_df = posterior_df.iloc[:: int(thin)].reset_index(drop=True)
    posterior_df["posterior_draw_id"] = np.arange(1, len(posterior_df) + 1)

    if posterior_df.empty:
        raise RuntimeError("PMMH produced no posterior samples. Check burn_in/thin settings.")

    map_idx = int(chain_df["logpost"].idxmax())
    map_theta = {k: float(chain_df.loc[map_idx, k]) for k in _THETA_COLUMNS}
    event_fit_details = event_fit_details_openloop(
        boundary,
        action,
        df_obs,
        events,
        map_theta,
        A=A,
        V=V,
        ramp_max=ramp_max,
        sigma_t=sigma_t_obs,
        sigma_e=sigma_e_obs,
        sigma_co2=sigma_co2_obs,
    )
    return {
        "posterior_table": chain_df,
        "posterior_samples": posterior_df,
        "map_theta": map_theta,
        "event_fit_details": event_fit_details,
        "accept_rate": accepted / float(max(1, pmmh_iters)),
    }


def robust_counterfactual_eval(
    boundary: pd.DataFrame,
    posterior_samples: pd.DataFrame,
    *,
    Tin0: float,
    RHin0: float,
    CO20: float,
    weights: Dict[str, float],
    constraints: Dict[str, float],
    A: float = 200.0,
    V: float = 300.0,
    ramp_max: float = 0.15,
    show_progress: bool = True,
) -> Dict[str, object]:
    rows: List[Dict[str, float]] = []
    representative_episode: Optional[pd.DataFrame] = None
    representative_logpost = -np.inf

    iterator = posterior_samples.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(posterior_samples), desc="counterfactual_eval")

    for row in iterator:
        theta = {k: float(getattr(row, k)) for k in _THETA_COLUMNS}
        sim = simulate_counterfactual_policy(
            boundary,
            theta,
            weights,
            Tin0=Tin0,
            RHin0=RHin0,
            CO20=CO20,
            A=A,
            V=V,
            ramp_max=ramp_max,
            Tref=constraints.get("Tref"),
            T_min=float(constraints["T_min"]),
            T_max=float(constraints["T_max"]),
            VPD_min=float(constraints["VPD_min"]),
            dTcond_min=float(constraints["dTcond_min"]),
            alpha=float(constraints["alpha"]),
            show_progress=False,
        )
        metrics = calc_metrics(
            sim,
            T_min=float(constraints["T_min"]),
            T_max=float(constraints["T_max"]),
            RH_max=float(constraints["RH_max"]),
            VPD_min=float(constraints["VPD_min"]),
            dTcond_min=float(constraints["dTcond_min"]),
            alpha=float(constraints["alpha"]),
        ).iloc[0].to_dict()
        metrics["posterior_draw_id"] = int(getattr(row, "posterior_draw_id"))
        metrics["loglik"] = float(getattr(row, "loglik"))
        metrics["logpost"] = float(getattr(row, "logpost"))
        rows.append(metrics)

        if float(getattr(row, "logpost")) > representative_logpost:
            representative_logpost = float(getattr(row, "logpost"))
            representative_episode = sim

    metrics_df = pd.DataFrame(rows)
    cols = [c for c in _DOC_METRIC_COLUMNS if c in metrics_df.columns]
    q90 = {c: float(np.quantile(metrics_df[c], 0.9)) for c in cols}
    c90 = {c: cvar(metrics_df[c].values, 0.9) for c in cols}
    return {
        "metrics": metrics_df,
        "q90": q90,
        "cvar90": c90,
        "episode": representative_episode,
    }


def save_eval_outputs(
    output_dir: Path,
    *,
    robust_result: Dict[str, object],
    posterior_result: Dict[str, object],
    events_df: pd.DataFrame,
    config: Dict[str, object],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    q90_path = output_dir / "q90.json"
    cvar90_path = output_dir / "cvar90.json"
    episode_path = output_dir / "episode.csv"
    posterior_path = output_dir / "posterior_samples.csv"
    event_path = output_dir / "event_windows.csv"
    event_fit_path = output_dir / "event_fit_details.csv"
    period_path = output_dir / "evaluation_period.json"
    config_path = output_dir / "run_config.json"

    metrics_to_save = robust_result["metrics"].copy()

    episode = robust_result.get("episode")
    if isinstance(episode, pd.DataFrame):
        episode_to_save = episode.copy()
        float_cols = episode_to_save.select_dtypes(include=["float", "float16", "float32", "float64"]).columns
        episode_to_save[float_cols] = episode_to_save[float_cols].round(3)

    posterior_to_save = posterior_result["posterior_table"].copy()
    event_to_save = events_df.copy()
    event_fit_to_save = posterior_result["event_fit_details"].copy()

    for df_to_round in [metrics_to_save, posterior_to_save, event_to_save, event_fit_to_save]:
        float_cols = df_to_round.select_dtypes(include=["float", "float16", "float32", "float64"]).columns
        df_to_round[float_cols] = df_to_round[float_cols].round(3)

    metrics_to_save.to_csv(metrics_path, index=False)
    if isinstance(episode, pd.DataFrame):
        episode_to_save.to_csv(episode_path, index=False)
    posterior_to_save.to_csv(posterior_path, index=False)
    event_to_save.to_csv(event_path, index=False)
    event_fit_to_save.to_csv(event_fit_path, index=False)

    q90_path.write_text(
        json.dumps(_round_floats({k: robust_result["q90"][k] for k in _DOC_METRIC_COLUMNS}, 3), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cvar90_path.write_text(
        json.dumps(_round_floats({k: robust_result["cvar90"][k] for k in _DOC_METRIC_COLUMNS}, 3), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    period_path.write_text(
        json.dumps(
            _round_floats(
                {
                    "evaluation_start_date": config.get("evaluation_start_date"),
                    "evaluation_end_date_exclusive": config.get("evaluation_end_date_exclusive"),
                    "timeseries_start_time": config.get("start_time"),
                    "timeseries_end_time": config.get("end_time"),
                    "n_rows": config.get("n_rows"),
                },
                3,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(_round_floats(_to_builtin(config), 3), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "metrics_csv": str(metrics_path),
        "q90_json": str(q90_path),
        "cvar90_json": str(cvar90_path),
        "episode_csv": str(episode_path),
        "posterior_samples_csv": str(posterior_path),
        "event_windows_csv": str(event_path),
        "event_fit_details_csv": str(event_fit_path),
        "evaluation_period_json": str(period_path),
        "run_config_json": str(config_path),
    }


def run_limited_data_eval(
    df_raw: pd.DataFrame,
    *,
    evaluation_start_date: Optional[str] = None,
    evaluation_end_date_exclusive: Optional[str] = None,
    pmmh_iters: int = 180,
    burn_in: int = 60,
    thin: int = 2,
    n_particles: int = 64,
    max_events: int = 12,
    window_steps: int = 12,
    posterior_seed: int = 42,
    weights: Optional[Dict[str, float]] = None,
    constraints: Optional[Dict[str, float]] = None,
    A: float = 200.0,
    V: float = 300.0,
    ramp_max: float = 0.15,
    output_dir: Optional[Path] = None,
    show_progress: bool = True,
) -> Dict[str, object]:
    if weights is None:
        weights = {"wT": 1.0, "wVPD": 1.0, "wE": 1e-8, "wDx": 0.2, "wSlack": 50.0}
    if constraints is None:
        constraints = {
            "Tref": None,
            "T_min": 12.0,
            "T_max": 28.0,
            "RH_max": 0.90,
            "VPD_min": 0.30,
            "dTcond_min": 0.8,
            "alpha": 0.25,
        }
    if output_dir is None:
        output_dir = _ensure_results_dir(
            start_date=evaluation_start_date,
            end_date_exclusive=evaluation_end_date_exclusive,
        )

    df_ts = prepare_timeseries(df_raw)
    io = make_inputs_actions(df_ts)
    boundary = io["boundary"]
    action = io["action"]
    events_df = select_event_windows(df_ts, action, max_events=max_events, window_steps=window_steps)

    posterior_result = fit_posterior_pmmh(
        boundary,
        action,
        df_ts,
        events_df,
        pmmh_iters=pmmh_iters,
        burn_in=burn_in,
        thin=thin,
        n_particles=n_particles,
        seed=posterior_seed,
        A=A,
        V=V,
        ramp_max=ramp_max,
        show_progress=show_progress,
    )

    robust_result = robust_counterfactual_eval(
        boundary,
        posterior_result["posterior_samples"],
        Tin0=float(df_ts["Tin_obs"].iloc[0]),
        RHin0=float(df_ts["RHin_obs"].iloc[0]),
        CO20=float(df_ts["CO2_obs"].iloc[0]),
        weights=weights,
        constraints=constraints,
        A=A,
        V=V,
        ramp_max=ramp_max,
        show_progress=show_progress,
    )

    saved = save_eval_outputs(
        output_dir,
        robust_result=robust_result,
        posterior_result=posterior_result,
        events_df=events_df,
        config={
            "evaluation_start_date": evaluation_start_date,
            "evaluation_end_date_exclusive": evaluation_end_date_exclusive,
            "pmmh_iters": pmmh_iters,
            "burn_in": burn_in,
            "thin": thin,
            "n_particles": n_particles,
            "posterior_seed": posterior_seed,
            "max_events": max_events,
            "window_steps": window_steps,
            "A": A,
            "V": V,
            "ramp_max": ramp_max,
            "weights": weights,
            "constraints": constraints,
            "posterior_accept_rate": posterior_result["accept_rate"],
            "hyperparameters": {
                "event_weights": {
                    "heater_step": 6.0,
                    "vent_step": 5.0,
                    "night_no_solar": 4.0,
                    "clear_day": 3.5,
                    "curtain_step": 1.5,
                    "tout_jump": 1.0,
                },
                "observation_noise": {
                    "temperature_sigma_c": 1.5,
                    "absolute_humidity_sigma": 0.20,
                    "co2_sigma_ppm": 80.0,
                },
                "process_noise": {
                    "temperature_sigma_c": 0.30,
                    "absolute_humidity_sigma": 0.03,
                    "co2_sigma_ppm": 20.0,
                },
                "ess_ratio": 0.5,
                "proposal_scales": {
                    "UA": 0.08,
                    "C": 0.08,
                    "k_heat": 0.08,
                    "a0": 0.06,
                    "a1": 0.06,
                    "a2": 0.06,
                    "k_evap": 0.08,
                    "k_photo": 0.08,
                    "eta": 0.12,
                },
            },
            "n_rows": len(df_ts),
            "start_time": str(df_ts["reg_date"].iloc[0]),
            "end_time": str(df_ts["reg_date"].iloc[-1]),
        },
    )

    return {
        "saved": saved,
        "df_ts": df_ts,
        "boundary": boundary,
        "action": action,
        "events": events_df,
        "posterior": posterior_result,
        "robust": robust_result,
    }


if __name__ == "__main__":
    # start_date = "2024-08-27"
    # end_date = "2024-09-29"
    start_date = "2025-11-22"
    end_date = "2025-12-27"

    data = connector(start_date, end_date)
    res = run_limited_data_eval(
        data,
        evaluation_start_date=start_date,
        evaluation_end_date_exclusive=end_date,
        pmmh_iters=180,
        burn_in=60,
        thin=2,
        n_particles=64,
        max_events=12,
        window_steps=12,
        posterior_seed=42,
        show_progress=True,
    )
    print("saved_outputs:", res["saved"]["output_dir"])

