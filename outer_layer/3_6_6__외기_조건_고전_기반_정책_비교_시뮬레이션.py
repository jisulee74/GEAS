"""
3.6.6. 외기 조건 고전 기반 정책 비교 시뮬레이션
실측 데이터로 물리 파라미터(θ)를 피팅한 뒤,
Observed / Baseline / AI 세 가지 정책을 Monte Carlo로 비교·평가한다.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from physics_store_sync import approve_and_save_theta_case


# ── 1. DB 연결 ───────────────────────────────────────────────────────────────

def connector(start_date, end_date=None):
    import datetime
    if end_date is None:
        end_date = datetime.datetime.now()
    start_chr = pd.Timestamp(start_date).strftime("%Y-%m-%d %H:%M:%S")
    end_chr   = pd.Timestamp(end_date).strftime("%Y-%m-%d %H:%M:%S")
    from sqlalchemy import create_engine
    engine = create_engine(
        "mysql+pymysql://root:theimc#10!@211.195.9.227:3306/farmstom"
    )
    query = f"""
        SELECT * FROM data_silla_enc
        WHERE iot_data_idx = 97
          AND reg_date >= '{start_chr}'
          AND reg_date <  '{end_chr}'
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    print(df.columns.tolist())
    return df


# ── 2. 유틸리티 ──────────────────────────────────────────────────────────────

def numify(x):
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    x = x.astype(str).str.replace(",","",regex=False).str.strip()
    x[x.isin(["","NA","NaN","NULL"])] = np.nan
    return pd.to_numeric(x, errors="coerce")

def sat_vapor_pressure_kpa(Tc):
    return 0.6108 * np.exp((17.27 * Tc) / (Tc + 237.3))

def vpd_kpa(Tc, RH):
    es = sat_vapor_pressure_kpa(Tc)
    ea = es * (RH / 100)
    return np.maximum(0, es - ea)

def dewpoint_c(Tc, RH):
    a = 17.27; b = 237.3
    gamma = (a * Tc) / (b + Tc) + np.log(np.maximum(1e-6, RH / 100))
    return (b * gamma) / (a - gamma)

def abs_humidity_gm3(Tc, RH):
    es = sat_vapor_pressure_kpa(Tc) * 10
    e  = es * (RH / 100)
    return 216.7 * (e / (Tc + 273.15))

def ah_to_rh(Tc, AH):
    e_hpa  = (AH * (Tc + 273.15)) / 216.7
    es_hpa = sat_vapor_pressure_kpa(Tc) * 10
    return np.clip(100 * e_hpa / np.maximum(1e-6, es_hpa), 0, 100)

def normalize_01(x):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    if np.all(np.isnan(x)):
        return np.full_like(x, np.nan)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if (hi - lo) < 1e-9:
        return np.clip(x, 0, 1)
    return np.clip((x - lo) / (hi - lo), 0, 1)


# ── 3. 데이터 전처리 파이프라인 ─────────────────────────────────────────────

def map_columns_raw(dt: pd.DataFrame) -> pd.DataFrame:
    """DB 컬럼명을 표준 이름으로 변환하고, 환기·차광 정규화를 수행한다."""
    def gc(col, default=np.nan):
        if col in dt.columns:
            return pd.to_numeric(dt[col], errors="coerce")
        return pd.Series(default, index=dt.index, dtype=float)

    t_in1  = gc("in_temp"); t_in2  = gc("in_temp2")
    rh_in1 = gc("in_hum");  rh_in2 = gc("in_hum2")
    co2_in1 = gc("in_co2"); co2_in2 = gc("in_co2_2")
    t_out = gc("out_temp"); rh_out = gc("out_hum")
    wind  = gc("out_windsp", 0); solar = gc("out_light", 0)

    skyl = gc("cont_skyl_vol"); skyr = gc("cont_skyr_vol")
    vent_raw = pd.concat([skyl, skyr], axis=1).mean(axis=1, skipna=True)
    vent = pd.Series(normalize_01(vent_raw.values), index=dt.index)

    shade_raw   = gc("etc_blackout",    0).fillna(0)
    heat_raw    = gc("cont_heater_run", 0).fillna(0)
    co2inj_raw  = gc("cont_co2_run",   0).fillna(0)
    fan_raw     = gc("cont_fan_run",    0).fillna(0)

    return pd.DataFrame({
        "reg_date": dt["reg_date"],
        "t_in":    t_in1.where(t_in1.notna(), t_in2),
        "rh_in":   rh_in1.where(rh_in1.notna(), rh_in2),
        "co2_in":  co2_in1.where(co2_in1.notna(), co2_in2),
        "t_out":   t_out, "rh_out": rh_out,
        "wind":    wind.clip(lower=0), "solar": solar.clip(lower=0),
        "fan":    (fan_raw > 0).astype(float),
        "heat":   (heat_raw > 0).astype(float),
        "vent":   vent.fillna(0).clip(0, 1),
        "shade":  pd.Series(normalize_01(shade_raw.values), index=dt.index).fillna(0).clip(0,1),
        "co2inj": (co2inj_raw > 0).astype(float),
    })


def regularize_time_std(dt: pd.DataFrame, step_mins=10) -> pd.DataFrame:
    """타임스텝을 step_mins 단위로 정규화한다."""
    dt = dt.copy()
    dt["reg_date"] = pd.to_datetime(dt["reg_date"])
    epoch = pd.Timestamp("1970-01-01")
    step_sec = step_mins * 60
    dt["time"] = pd.to_datetime(
        (dt["reg_date"] - epoch).dt.total_seconds()
        // step_sec * step_sec, unit="s"
    )

    agg = dt.groupby("time").agg({
        "t_in":"mean","rh_in":"mean","co2_in":"mean",
        "t_out":"mean","rh_out":"mean","wind":"mean","solar":"mean",
        "fan":"mean","heat":"mean","vent":"mean","shade":"mean","co2inj":"mean",
    }).reset_index()

    agg["heat"]   = (agg["heat"]   > 0.5).astype(float)
    agg["co2inj"] = (agg["co2inj"] > 0.5).astype(float)
    agg["fan"]    = (agg["fan"]    > 0.5).astype(float)
    agg["vent"]   = agg["vent"].clip(0, 1)
    agg["shade"]  = agg["shade"].clip(0, 1)

    return agg.sort_values("time").reset_index(drop=True)


def derive_features(dt: pd.DataFrame) -> pd.DataFrame:
    dt = dt.copy()
    dt["ah_in"]  = abs_humidity_gm3(dt["t_in"],  dt["rh_in"])
    dt["ah_out"] = abs_humidity_gm3(dt["t_out"], dt["rh_out"])
    dt["vpd_in"] = vpd_kpa(dt["t_in"], dt["rh_in"])
    dt["td_in"]  = dewpoint_c(dt["t_in"], dt["rh_in"])
    dt["solar"]  = dt["solar"].clip(lower=0)
    return dt


# ── 4. 물리 모델 ─────────────────────────────────────────────────────────────

def actuator_to_ach(vent, wind=0, fan=0,
                    ach_min=0.2, ach_max=25, k_wind=0.06, k_fan=0.30):
    v = np.clip(vent, 0, 1)
    w = np.where(np.isnan(wind), 0, np.maximum(0, wind))
    f = np.where(np.isnan(fan),  0, np.maximum(0, fan))
    g = (v**1.2) * (1 + k_wind*w) * (1 + k_fan*f)
    return np.maximum(0, ach_min + (ach_max - ach_min) * (g / (1 + g)))

def effective_solar(solar, shade=0, k_shade=0.7):
    s  = np.maximum(0, solar)
    sh = np.clip(shade, 0, 1)
    return s * (1 - k_shade * sh)

def theta_default() -> dict:
    return {
        "C": 2.5e6, "UA": 350,
        "ach_min": 0.2, "ach_max": 25, "k_wind": 0.06, "k_fan": 0.30,
        "q_heat": 8000, "eta_heat": 1.0, "k_shade": 0.7,
        "k_solar_to_heat": 0.6,
        "k_transp": 1.2e-6, "k_co2_assim": 2.0e-6, "k_co2_inj": 2.0e-3,
    }

def step_model(x: dict, w: dict, u: dict, theta: dict,
               dt_sec: float, eps: dict = None) -> dict:
    """한 타임스텝의 온실 상태 변화를 계산한다."""
    if eps is None:
        eps = {"T": 0, "AH": 0, "CO2": 0}

    T = x["T"]; AH = x["AH"]; CO2 = x["CO2"]
    ACH  = float(actuator_to_ach(u["vent"], w["wind"], w["fan"],
                                  theta["ach_min"], theta["ach_max"],
                                  theta["k_wind"], theta["k_fan"]))
    Ieff = float(effective_solar(w["solar"], u["shade"], theta["k_shade"]))

    rho_cp = 1200
    Qsolar = theta["k_solar_to_heat"] * Ieff
    Qheat  = theta["eta_heat"] * theta["q_heat"] * max(0, u["heat"])

    dT     = (Qsolar + Qheat - theta["UA"]*(T - w["T_out"])
              - rho_cp*ACH*(T - w["T_out"])) / max(1e-6, theta["C"])
    T_next = float(np.clip(T + dt_sec*dT + eps["T"], -10, 60))

    transp  = theta["k_transp"] * max(0, Ieff)
    dAH     = transp - ACH*(AH - w["AH_out"])
    AH_next = float(np.clip(AH + dt_sec*dAH + eps["AH"], 0, 35))

    assim   = theta["k_co2_assim"] * max(0, Ieff) * max(0, CO2/(CO2+200))
    inj     = theta["k_co2_inj"] * max(0, u["co2inj"])
    dCO2    = inj - assim - ACH*(CO2 - w["CO2_out"])
    CO2_next = float(np.clip(CO2 + dt_sec*dCO2 + eps["CO2"], 0, 5000))

    return {"T": T_next, "AH": AH_next, "CO2": CO2_next, "ACH": ACH, "Ieff": Ieff}


# ── 5. 행동 투영 ─────────────────────────────────────────────────────────────

def project_action(u_raw, u_prev, limits, ramp, rules):
    u = dict(u_raw)
    for nm in limits:
        if nm in u:
            u[nm] = float(np.clip(u[nm], limits[nm][0], limits[nm][1]))
    for nm in ramp:
        if nm in u and nm in u_prev:
            lo = u_prev[nm] - ramp[nm]
            hi = u_prev[nm] + ramp[nm]
            u[nm] = float(np.clip(u[nm], lo, hi))
    u["heat"] = float(u.get("heat", 0))
    u["vent"] = float(u.get("vent", 0))
    if rules.get("no_heat_vent") and u["heat"] > 0 and u["vent"] > rules.get("vent_thr", 0.2):
        u["vent"] = rules["vent_thr"]
    return u

def prepare_limits_ramp_rules():
    return {
        "limits": {"heat":(0,1),"vent":(0,1),"shade":(0,1),"co2inj":(0,1)},
        "ramp":   {"heat":0.3,"vent":0.10,"shade":1.0,"co2inj":1.0},
        "rules":  {"no_heat_vent":False,"vent_thr":0.2},
    }


# ── 6. 목표값 생성 ───────────────────────────────────────────────────────────

def make_targets(w_dt: pd.DataFrame, dt_ref=None) -> dict:
    """시간대에 따른 온도·VPD·CO2 목표값 딕셔너리를 반환한다."""
    hr       = pd.to_datetime(w_dt["time"]).dt.hour
    is_night = (hr < 6) | (hr >= 18)

    if dt_ref is None:
        T_day, T_night = 23, 18
    else:
        T_day   = float(np.nanmedian(dt_ref.loc[~is_night, "t_in"])) if (~is_night).any() else 23
        T_night = float(np.nanmedian(dt_ref.loc[is_night,  "t_in"])) if  is_night.any()  else 18
        if not np.isfinite(T_day):   T_day   = 23
        if not np.isfinite(T_night): T_night = 18

    n = len(w_dt)
    return {
        "is_night":          is_night.values,
        "T_sp":              np.where(is_night, T_night, T_day),
        "deadband":          np.where(is_night, 1.5 if dt_ref is None else 2.5,
                                               1.0 if dt_ref is None else 2.5),
        "vpd_min":           np.where(is_night, 0.2, 0.4 if dt_ref is None else 0.3),
        "vpd_max":           np.where(is_night, 1.2, 1.6 if dt_ref is None else 2.0),
        "CO2_sp":            np.where(is_night, 450, 800),
        "AH_sp":             np.where(is_night, 9,   10),
        "solar_shade_thr":   500,
        "solar_co2_thr":     150,
        "temp_soft_margin":  2 if dt_ref is None else 3,
    }


# ── 7. 세 가지 정책 ──────────────────────────────────────────────────────────

def policy_observed(t, x, w, ctx):
    row = ctx["uobs"].iloc[t]
    return {"heat": float(row["heat"]), "vent": float(row["vent"]),
            "shade": float(row["shade"]), "co2inj": float(row["co2inj"])}

def policy_baseline(t, x, w, ctx):
    T     = x["T"]
    T_sp  = ctx["targets"]["T_sp"][t]
    dead  = ctx["targets"]["deadband"][t]
    solar = ctx["w"]["solar"].iloc[t]
    night = ctx["targets"]["is_night"][t]

    vent   = max(0, min(1, (T - (T_sp + dead)) / 3)) if T > T_sp + dead else 0
    heat   = max(0, min(1, ((T_sp - dead) - T) / 3)) if T < T_sp - dead else 0
    shade  = 1.0 if solar > ctx["targets"]["solar_shade_thr"] else 0.0
    co2inj = 1.0 if (not night) and solar > ctx["targets"]["solar_co2_thr"] else 0.0
    return {"heat": heat, "vent": vent, "shade": shade, "co2inj": co2inj}

def default_grid():
    return {"heat": [0, 0.25, 0.5, 0.75, 1.0],
            "vent": [0, 0.25, 0.5, 0.75, 1.0],
            "shade": [0, 1], "co2inj": [0, 1]}

def default_weights():
    return {"wT": 1.0, "wAH": 0.01, "wCO2": 0.001, "wAct": 0.1, "wSwitch": 0.5}

def policy_ai(t, x, w, ctx):
    """그리드 탐색으로 비용 함수를 최소화하는 행동을 선택한다."""
    from itertools import product as iproduct

    grid    = ctx["grid"]
    theta   = ctx["theta"]
    dt_sec  = ctx["dt_sec"]
    targets = ctx["targets"]
    weights = ctx["weights"]
    u_prev  = ctx["u_prev"]
    rules   = ctx["rules"]

    w_list = {
        "T_out": ctx["w"]["T_out"].iloc[t],
        "AH_out": ctx["w"]["AH_out"].iloc[t],
        "CO2_out": ctx["w"]["CO2_out"].iloc[t],
        "solar": ctx["w"]["solar"].iloc[t],
        "wind":  ctx["w"]["wind"].iloc[t],
        "fan":   ctx["w"]["fan"].iloc[t],
    }

    vpd_min = targets["vpd_min"][t]
    vpd_max = targets["vpd_max"][t]
    co2_sp  = targets["CO2_sp"][t]
    ah_sp   = targets["AH_sp"][t]

    best_cost = np.inf
    best_u    = {"heat": 0, "vent": 0, "shade": 0, "co2inj": 0}

    for combo in iproduct(grid["heat"], grid["vent"], grid["shade"], grid["co2inj"]):
        u = {"heat": combo[0], "vent": combo[1], "shade": combo[2], "co2inj": combo[3]}
        if rules.get("no_heat_vent") and u["heat"] > 0 and u["vent"] > rules["vent_thr"]:
            continue

        st = step_model(x, w_list, u, theta, dt_sec)
        Tn   = st["T"]
        AHn  = st["AH"]
        CO2n = st["CO2"]
        RHn  = float(ah_to_rh(Tn, AHn))
        VPDn = float(vpd_kpa(Tn, RHn))
        if not np.isfinite(RHn): RHn = 50
        if not np.isfinite(VPDn): VPDn = 0.8

        cost = 0
        cost += weights["wT"]    * (max(0, 6 - Tn)**2 + max(0, Tn - 25)**2)
        cost += weights["wAH"]   * (AHn - ah_sp)**2
        cost += weights["wCO2"]  * max(0, co2_sp - CO2n)**2
        cost += weights["wAct"]  * (abs(u["heat"] - u_prev["heat"]) +
                                    abs(u["vent"]  - u_prev["vent"]) +
                                    abs(u["shade"] - u_prev["shade"]) +
                                    abs(u["co2inj"]- u_prev["co2inj"]))
        cost += weights["wSwitch"] * (
            float((u["heat"] > 0) != (u_prev["heat"] > 0)) +
            float((u["vent"] > 0)  != (u_prev["vent"]  > 0))
        )
        cost += 1000 * (max(0, 6 - Tn) + max(0, Tn - 25))
        cost += 1000 * (max(0, vpd_min - VPDn) + max(0, VPDn - vpd_max))
        cost += 5.0  * u["heat"] * u["vent"]

        if cost < best_cost:
            best_cost = cost; best_u = dict(u)

    return best_u


# ── 8. KPI 계산 ──────────────────────────────────────────────────────────────

def compute_all_kpis(traj: pd.DataFrame, u: pd.DataFrame,
                     targets: dict, rules: dict) -> dict:
    T  = traj["T"].values; RH = traj["RH"].values
    T[~np.isfinite(T)] = np.nan; RH[~np.isfinite(RH)] = np.nan

    viol = (T < 6) | (T > 25)
    td   = dewpoint_c(T, RH)
    cond_risk = (T - td) < 1.0

    def max_run_arr(b):
        b = np.where(np.isnan(b), False, b).astype(bool)
        if not b.any(): return 0
        count, best = 0, 0
        for v in b:
            count = count+1 if v else 0
            best  = max(best, count)
        return best

    return {
        "temp_violation_rate":       float(np.nanmean(viol)),
        "max_consec_temp_violation": max_run_arr(viol),
        "condensation_violation_rate": float(np.nanmean(cond_risk)),
        "high_humidity_violation_rate": float(np.nanmean(RH > 90)),
        "heat_vent_inefficiency":    float(np.nanmean(
            (u["heat"].values > 0) & (u["vent"].values > rules.get("vent_thr", 0.2))
        )),
        "vent_total_movement":       float(np.nansum(np.abs(np.diff(
            np.where(np.isnan(u["vent"].values), 0, u["vent"].values)
        )))),
        "heat_switch_count":         int(np.nansum(np.abs(np.diff(
            (np.where(np.isnan(u["heat"].values), 0, u["heat"].values) > 0.2).astype(float)
        )) > 0)),
        "vent_switch_count":         int(np.nansum(np.abs(np.diff(
            (np.where(np.isnan(u["vent"].values), 0, u["vent"].values) > 0.2).astype(float)
        )) > 0)),
        "temp_mae":                  float(np.nanmean(np.abs(T - (6+25)/2))),
        "temp_excess":               float(np.nanmean(np.maximum(0, 6-T) + np.maximum(0, T-25))),
    }


# ── 9. θ 피팅 ───────────────────────────────────────────────────────────────

def temp_sse_given_theta(dt, theta, dt_sec, pr):
    limits, ramp, rules = pr["limits"], pr["ramp"], pr["rules"]
    w = {
        "T_out": dt["t_out"].values, "AH_out": dt["ah_out"].values,
        "CO2_out": np.full(len(dt), 420.0),
        "solar": dt["solar"].values, "wind": dt["wind"].values, "fan": dt["fan"].values,
    }
    x = {"T": float(dt["t_in"].iloc[0]),
         "AH": float(dt["ah_in"].iloc[0]),
         "CO2": float(dt["co2_in"].iloc[0]) if "co2_in" in dt.columns else 600.0}
    u_prev = {"heat": 0, "vent": 0, "shade": 0, "co2inj": 0}

    sse, n = 0.0, 0
    for t in range(len(dt) - 1):
        w_t = {k: float(v[t]) for k, v in w.items()}
        u_raw = {col: float(dt[col].iloc[t])
                 for col in ["heat","vent","shade","co2inj"] if col in dt.columns}
        u = project_action(u_raw, u_prev, limits, ramp, rules)
        pred = step_model(x, w_t, u, theta, dt_sec)
        T_next = dt["t_in"].iloc[t+1]
        if np.isfinite(pred["T"]) and np.isfinite(T_next):
            sse += (T_next - pred["T"])**2; n += 1
        x = {"T": T_next,
             "AH": dt["ah_in"].iloc[t+1],
             "CO2": dt["co2_in"].iloc[t+1] if "co2_in" in dt.columns else 600.0}
        u_prev = u

    return sse / n if n > 0 else np.inf

def fit_theta_temperature(dt, dt_sec, pr, maxit=600, show_progress=True,
                          auto_save: bool = False,
                          farm_sn: int | None = None,
                          stage_name: str | None = None,
                          model_name: str = "default",
                          store_path: str | None = None):
    """Nelder-Mead로 UA, q_heat, ach_max를 최적화한다."""
    base = theta_default()

    feval = [0]
    def obj(par):
        feval[0] += 1
        th = dict(base)
        th["UA"]      = float(np.clip(np.exp(par[0]), 10, 5000))
        th["q_heat"]  = float(np.clip(np.exp(par[1]), 0, 100000))
        th["ach_max"] = float(np.clip(np.exp(par[2]),
                                      th["ach_min"] + 1, 80))
        return temp_sse_given_theta(dt, th, dt_sec, pr)

    p0 = np.log([base["UA"], base["q_heat"], base["ach_max"]])
    res = minimize(obj, p0, method="Nelder-Mead",
                   options={"maxiter": maxit, "disp": show_progress})

    th_hat = dict(base)
    th_hat["UA"]      = float(np.exp(res.x[0]))
    th_hat["q_heat"]  = float(np.exp(res.x[1]))
    th_hat["ach_max"] = float(np.exp(res.x[2]))

    result = {"theta": th_hat, "loss": res.fun, "conv": res.status}

    valid_from = None
    valid_to = None
    try:
        if 'reg_date' in dt.columns and len(dt.index) > 0:
            ts = pd.to_datetime(dt['reg_date']).dropna()
            if not ts.empty:
                valid_from = ts.iloc[0].isoformat()
                valid_to = ts.iloc[-1].isoformat()
    except Exception:
        pass

    if auto_save:
        mapped = {
            "C": float(th_hat.get("C", 2.5e6)),
            "UA": float(th_hat.get("UA", 350.0)),
            "k_heat": float(th_hat.get("q_heat", 8000.0)),
            "eta": float(th_hat.get("k_solar_to_heat", 0.6)),
            "a0": float(th_hat.get("ach_min", 0.2)),
            "a1": float(max(th_hat.get("ach_max", 25.0) - th_hat.get("ach_min", 0.2), 0.0)),
            "a2": float(th_hat.get("k_wind", 0.06)),
            "a3": float(th_hat.get("k_fan", 0.30)),
            "k_shade": float(th_hat.get("k_shade", 0.7)),
        }
        saved = approve_and_save_theta_case(
            mapped,
            farm_sn=farm_sn,
            stage_name=stage_name,
            data_case="sufficient",
            model_name=model_name,
            valid_from=valid_from,
            valid_to=valid_to,
            store_path=store_path,
            source="modeling_fit_theta_temperature",
            metadata={
                "raw_theta": dict(th_hat),
                "loss": float(res.fun),
                "conv": int(res.status),
                "dt_sec": float(dt_sec),
                "maxit": int(maxit),
                "valid_from": valid_from,
                "valid_to": valid_to,
            },
        )
        result["approval"] = saved["approval"]
        if "record" in saved:
            result["store_key"] = saved["record"].key

    return result


# ── 10. 잔차 샘플러 ──────────────────────────────────────────────────────────

def build_residuals_from_observed(dt, theta, dt_sec, pr):
    limits, ramp, rules = pr["limits"], pr["ramp"], pr["rules"]
    w_arr = {
        "T_out":  dt["t_out"].values,  "AH_out": dt["ah_out"].values,
        "CO2_out": np.full(len(dt), 420.0),
        "solar":  dt["solar"].values,  "wind": dt["wind"].values, "fan": dt["fan"].values,
    }
    x = {"T": float(dt["t_in"].iloc[0]),
         "AH": float(dt["ah_in"].iloc[0]),
         "CO2": 600.0}
    u_prev = {"heat": 0, "vent": 0, "shade": 0, "co2inj": 0}

    eT, eAH, eCO2 = [], [], []
    for t in range(len(dt) - 1):
        w_t   = {k: float(v[t]) for k, v in w_arr.items()}
        u_raw = {col: float(dt[col].iloc[t])
                 for col in ["heat","vent","shade","co2inj"] if col in dt.columns}
        u   = project_action(u_raw, u_prev, limits, ramp, rules)
        pred = step_model(x, w_t, u, theta, dt_sec)
        eT.append(dt["t_in"].iloc[t+1]  - pred["T"])
        eAH.append(dt["ah_in"].iloc[t+1] - pred["AH"])
        eCO2.append((dt["co2_in"].iloc[t+1] if "co2_in" in dt.columns else 600.0) - pred["CO2"])
        x = {"T": dt["t_in"].iloc[t+1], "AH": dt["ah_in"].iloc[t+1], "CO2": 600.0}
        u_prev = u

    return pd.DataFrame({"eT": eT, "eAH": eAH, "eCO2": eCO2})

def residual_sampler_boot(res_dt: pd.DataFrame):
    rng = np.random.default_rng(99)
    def sampler(t):
        row = res_dt.iloc[rng.integers(0, len(res_dt))]
        return {"T": float(row["eT"]), "AH": float(row["eAH"]), "CO2": float(row["eCO2"])}
    return sampler


# ── 11. 윈도우 시뮬레이션 및 비교 ───────────────────────────────────────────

def run_window_sim(policy_fn, dt_win, theta_hat, dt_sec,
                   pr, grid, weights, eps_sampler=None) -> dict:
    limits, ramp, rules = pr["limits"], pr["ramp"], pr["rules"]
    w_df = pd.DataFrame({
        "time":    dt_win["time"],
        "T_out":   dt_win["t_out"],
        "AH_out":  dt_win["ah_out"],
        "CO2_out": np.full(len(dt_win), 420.0),
        "solar":   dt_win["solar"],
        "wind":    dt_win["wind"],
        "fan":     dt_win["fan"],
    })
    targets = make_targets(w_df, dt_ref=dt_win)
    uobs    = dt_win[["heat","vent","shade","co2inj"]].reset_index(drop=True)

    x = {"T": float(dt_win["t_in"].iloc[0]),
         "AH": float(dt_win["ah_in"].iloc[0]),
         "CO2": float(dt_win["co2_in"].iloc[0]) if "co2_in" in dt_win.columns else 600.0}
    u_prev = {"heat": 0, "vent": 0, "shade": 0, "co2inj": 0}

    traj_list, u_list = [], []
    ctx = {"w": w_df, "targets": targets, "rules": rules, "grid": grid,
           "weights": weights, "dt_sec": dt_sec, "theta": theta_hat,
           "u_prev": u_prev, "uobs": uobs}

    for t in range(len(w_df)):
        ctx["u_prev"] = u_prev
        w_t  = {k: float(w_df[k].iloc[t]) for k in ["T_out","AH_out","CO2_out","solar","wind","fan"]}
        u_raw = policy_fn(t, x, w_t, ctx)
        u     = project_action(u_raw, u_prev, limits, ramp, rules)
        eps   = eps_sampler(t) if eps_sampler else {"T": 0, "AH": 0, "CO2": 0}
        st    = step_model(x, w_t, u, theta_hat, dt_sec, eps=eps)

        RH = float(ah_to_rh(x["T"], x["AH"]))
        traj_list.append({"time": w_df["time"].iloc[t], "T": x["T"], "AH": x["AH"],
                           "CO2": x["CO2"], "RH": RH})
        u_list.append({"time": w_df["time"].iloc[t],
                       "heat": u["heat"], "vent": u["vent"],
                       "shade": u["shade"], "co2inj": u["co2inj"]})

        x = {"T": st["T"], "AH": st["AH"], "CO2": st["CO2"]}
        u_prev = u

    return {"traj": pd.DataFrame(traj_list), "u": pd.DataFrame(u_list), "targets": targets}


def cvar(x, alpha=0.9):
    x = x[np.isfinite(x)]
    if len(x) == 0: return np.nan
    q = float(np.quantile(x, alpha))
    return float(x[x >= q].mean())

def summarize_runs(kpi_list, alpha=0.9) -> pd.DataFrame:
    df = pd.DataFrame(kpi_list)
    out = {}
    for nm in df.columns:
        v = df[nm].values.astype(float)
        out[f"{nm}_mean"] = float(np.nanmean(v))
        out[f"{nm}_p{int(alpha*100)}"] = float(np.nanquantile(v, alpha))
        out[f"{nm}_cvar{int(alpha*100)}"] = cvar(v, alpha)
    return pd.DataFrame([out])

def slice_dt_window(dt, start, end):
    mask = (dt["time"] >= pd.Timestamp(start)) & (dt["time"] < pd.Timestamp(end))
    return dt[mask].reset_index(drop=True)


# ── 12. 메인 워크플로우 ──────────────────────────────────────────────────────

def run_fit_then_window(dt_full, window_start, window_end,
                        n_mc=20, alpha=0.9,
                        auto_save_theta: bool = False,
                        farm_sn: int | None = None,
                        stage_name: str | None = None,
                        model_name: str = "default",
                        store_path: str | None = None) -> dict:
    dt_sec  = 600
    pr      = prepare_limits_ramp_rules()
    grid    = default_grid()
    weights = default_weights()

    fit       = fit_theta_temperature(
        dt_full, dt_sec, pr, show_progress=True,
        auto_save=auto_save_theta,
        farm_sn=farm_sn,
        stage_name=stage_name,
        model_name=model_name,
        store_path=store_path,
    )
    theta_hat = fit["theta"]

    dt_win      = slice_dt_window(dt_full, window_start, window_end)
    eps_sampler = None
    if n_mc > 1:
        res_dt      = build_residuals_from_observed(dt_full, theta_hat, dt_sec, pr)
        eps_sampler = residual_sampler_boot(res_dt)

    policies = {
        "Observed": policy_observed,
        "Baseline": policy_baseline,
        "AI":       policy_ai,
    }
    results = {}

    for pname, pfn in policies.items():
        print(f"\n정책 평가 중: {pname}")
        kpis = []
        for _ in tqdm(range(n_mc)):
            run = run_window_sim(pfn, dt_win, theta_hat, dt_sec,
                                 pr, grid, weights, eps_sampler)
            kpis.append(compute_all_kpis(run["traj"], run["u"], run["targets"], pr["rules"]))
        results[pname] = {
            "summary": summarize_runs(kpis, alpha=alpha),
            "raw":     kpis,
        }

    summary = pd.concat(
        [res["summary"].assign(policy=nm) for nm, res in results.items()],
        ignore_index=True
    )
    return {"theta_hat": theta_hat, "fit_loss": fit["loss"], "fit_conv": fit["conv"],
            "window": {"start": window_start, "end": window_end, "n": len(dt_win)},
            "summary": summary, "results": results}


def print_fit_and_window(out, step_mins=10, alpha=0.9):
    print("\n===== FIT RESULT (theta_hat) =====")
    for k, v in out["theta_hat"].items():
        print(f"  {k}: {v:.4g}")
    print(f"  fit_loss={out['fit_loss']:.4f}  fit_conv={out['fit_conv']}")
    print("\n===== WINDOW =====")
    print(out["window"])
    print("\n===== KPI SUMMARY =====")
    print(out["summary"].to_string(index=False))


# ── 13. 메인 실행 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_date = "2024-01-01"
    end_date   = "2025-11-15"

    # data_raw = connector(start_date, end_date)
    # dt0      = data_raw.copy()
    # dt0["reg_date"] = pd.to_datetime(dt0["reg_date"])
    # dt0 = dt0.sort_values("reg_date").reset_index(drop=True)
    # dt_std = map_columns_raw(dt0)
    # dt     = regularize_time_std(dt_std, step_mins=10)
    # dt     = derive_features(dt)

    # out = run_fit_then_window(
    #     dt_full=dt,
    #     window_start="2025-10-25 00:00:00",
    #     window_end  ="2025-11-01 00:00:00",
    #     n_mc=20, alpha=0.9,
    # )
    # print_fit_and_window(out)

    print("모듈 로드 완료.")
