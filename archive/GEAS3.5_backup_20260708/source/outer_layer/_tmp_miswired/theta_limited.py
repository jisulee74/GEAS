"""
3.6.4. ?쒗븳???곗씠???섍꼍?먯꽌??遺遺?寃利?諛⑸쾿濡?
?ㅼ륫 ?곗씠?곕? ?쒖슜??open-loop ?쒕??덉씠??+ Monte Carlo濡?
AI ?쒖뼱 ?뺤콉??KPI瑜?遺遺?寃利앺븳??
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from physics_store_sync import approve_and_save_theta_case


# ?? 1. DB ?곌껐 ???????????????????????????????????????????????????????????????

def connector(start_date, end_date=None):
    import datetime
    if end_date is None:
        end_date = datetime.datetime.now()
    start_chr = pd.Timestamp(start_date).strftime("%Y-%m-%d %H:%M:%S")
    end_chr   = pd.Timestamp(end_date).strftime("%Y-%m-%d %H:%M:%S")
    from sqlalchemy import create_engine
    engine = create_engine(
        "mysql+pymysql://<user>:<password>@<host>:<port>/<database>"
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


# ?? 2. ?좏떥由ы떚 ??????????????????????????????????????????????????????????????

def numify(x):
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    x = x.astype(str).str.replace(",", "", regex=False).str.strip()
    x[x.isin(["", "NA", "NaN", "NULL"])] = np.nan
    return pd.to_numeric(x, errors="coerce")

def sat_vp_kpa(Tc):
    return 0.61078 * np.exp((17.2694 * np.asarray(Tc, float)) / (np.asarray(Tc, float) + 237.3))

def dewpoint_c(Tc, RH):
    es = sat_vp_kpa(Tc)
    e  = np.clip(np.asarray(RH, float) * es, 1e-6, es)
    ln_ratio = np.log(e / 0.61078)
    return (237.3 * ln_ratio) / (17.2694 - ln_ratio)

def vpd_kpa(Tc, RH):
    es = sat_vp_kpa(Tc)
    ea = np.clip(np.asarray(RH, float) * es, 0, es)
    return np.maximum(0, es - ea)

def hinge(z):
    return np.maximum(0, z)


# ?? 3. ?섍린 諛??숈뿭??紐⑤뜽 ??????????????????????????????????????????????????

def ach_model(x_vent, wind, a0, a1, a2, ACH_max=15):
    ach = a0 + a1*x_vent + a2*wind*x_vent
    return float(np.clip(ach, 0, ACH_max))

def step_dynamics_dt(state: dict, u, a: dict, theta: dict,
                     dt_sec: float, A=200, V=300, Imax=650, T0=5) -> dict:
    """?ㅼ젣 dt_sec瑜??ъ슜?섎뒗 ?숈뿭???쒕??덉씠??(?ㅼ륫 ??꾩뒪?????."""
    Tin  = state["Tin"]
    e_in = state["e_in"]
    CO2  = state["CO2"]

    ACH    = ach_model(a["x_vent"], float(u["wind"]), theta["a0"], theta["a1"], theta["a2"])
    It_eff = float(u["It"]) * (1 - a["curtain"])

    Q_trans = theta["UA"]    * (float(u["Tout"]) - Tin)
    Q_solar = theta["eta"] * A * It_eff
    Q_heat  = theta["k_heat"] * a["u_heat"]
    Q_vent  = theta["rho_cp"] * V * (ACH/3600) * (float(u["Tout"]) - Tin)
    Tin_next = Tin + (dt_sec / theta["C"]) * (Q_trans + Q_solar + Q_heat + Q_vent)

    e_out  = float(u["RHout"]) * float(sat_vp_kpa(float(u["Tout"])))
    evap   = theta["k_evap"] * (It_eff / Imax) * max(0, Tin - T0)
    e_next = max(0.05, e_in + dt_sec * ((ACH/3600)*(e_out - e_in) + evap))

    RHin = min(1, max(0, e_next / float(sat_vp_kpa(Tin_next))))
    VPD  = float(vpd_kpa(Tin_next, RHin))
    gT   = min(1, max(0, (Tin_next - 5) / 20))
    gV   = min(1, max(0, VPD / 1.2))
    uptake   = theta["k_photo"] * It_eff * gT * gV * 1e6
    inj      = 50 * a["u_co2"]
    CO2out_v = float(u["CO2out"]) if "CO2out" in u else 420
    CO2_next = max(300, CO2 + dt_sec*((ACH/3600)*(CO2out_v - CO2) + inj - uptake))

    Q_ventloss = max(0, theta["rho_cp"]*V*(ACH/3600)*max(0, Tin_next - float(u["Tout"])))

    return {"Tin": Tin_next, "e_in": e_next, "CO2": CO2_next,
            "ACH": ACH, "It_eff": It_eff,
            "Q_heat": Q_heat, "Q_ventloss": Q_ventloss}


def project_action(a: dict, prev_a: dict, ramp_max=0.15) -> dict:
    a = dict(a)
    a["u_heat"]  = 1 if a["u_heat"]  > 0.5 else 0
    a["u_co2"]   = 1 if a["u_co2"]   > 0.5 else 0
    a["x_vent"]  = float(np.clip(a["x_vent"],  0, 1))
    a["curtain"] = float(np.clip(a["curtain"], 0, 1))
    dx = a["x_vent"] - prev_a["x_vent"]
    dx = float(np.clip(dx, -ramp_max, ramp_max))
    a["x_vent"] = prev_a["x_vent"] + dx
    return a


# ?? 4. KPI 怨꾩궛 ??????????????????????????????????????????????????????????????

def max_run(b):
    if not b.any():
        return 0
    count, best = 0, 0
    for v in b:
        count = count + 1 if v else 0
        best  = max(best, count)
    return best

def calc_metrics(df: pd.DataFrame,
                 T_min=12, T_max=28, RH_max=0.90,
                 VPD_min=0.30, dTcond_min=0.8, alpha=0.25) -> pd.DataFrame:
    temp_viol = (df["Tin"] < T_min) | (df["Tin"] > T_max)
    cond_viol = df["dTcond"] < dTcond_min
    rh_viol   = df["RHin"] > RH_max
    vpd_viol  = df["VPD"]  < VPD_min
    idx_heat  = df["u_heat"] > 0.5
    hv_ineff  = (float((df.loc[idx_heat,"Q_ventloss"] >
                        alpha * df.loc[idx_heat,"Q_heat"].clip(lower=1e-6)).mean())
                 if idx_heat.any() else 0.0)
    return pd.DataFrame([{
        "temp_viol_rate":   float(temp_viol.mean()),
        "temp_viol_maxrun": max_run(temp_viol.values),
        "cond_viol_rate":   float(cond_viol.mean()),
        "cond_viol_maxrun": max_run(cond_viol.values),
        "rh_viol_rate":     float(rh_viol.mean()),
        "vpd_viol_rate":    float(vpd_viol.mean()),
        "hv_ineff_rate":    hv_ineff,
        "tv_vent":          float(df["x_vent"].diff().abs().sum()),
        "sw_heat":          int((df["u_heat"].diff().abs() > 0).sum()),
    }])


# ?? 5. ?꾩쿂由?????????????????????????????????????????????????????????????????

def prepare_timeseries(data: pd.DataFrame, tz="Asia/Seoul") -> pd.DataFrame:
    """raw DB ?곗씠?곕? ?쒕??덉씠?섏뿉 ?꾩슂???뺥깭濡??꾩쿂由ы븳??"""
    df = data.copy()
    df["reg_date"] = pd.to_datetime(df["reg_date"], utc=False).dt.tz_localize(None)
    df = df.sort_values("reg_date").reset_index(drop=True)

    numeric_cols = {
        "in_temp": "float", "in_hum": "float", "in_co2": "float",
        "out_temp": "float", "out_hum": "float", "out_light": "float",
        "out_windsp": "float", "cont_heater_run": "float",
        "cont_skyl_vol": "float", "cont_cur_vol": "float",
        "cont_co2_run": "float", "etc_blackout": "float",
        "etc_plc_abnorm": "float",
    }
    for col in numeric_cols:
        if col in df.columns:
            df[col] = numify(df[col])

    if "out_co2" not in df.columns or df["out_co2"].isna().all():
        df["out_co2"] = 420.0
    else:
        df["out_co2"] = numify(df["out_co2"]).fillna(420.0)

    # ?댁긽/?뺤쟾 ?쒖젏 ?쒓굅
    for col in ["etc_blackout", "etc_plc_abnorm"]:
        if col in df.columns:
            df = df[(df[col].isna()) | (df[col] == 0)]

    df = df.dropna(subset=["in_temp", "in_hum", "out_temp", "out_hum"])

    df["dt_sec"] = df["reg_date"].diff().dt.total_seconds()
    med_dt = df["dt_sec"].median()
    df["dt_sec"] = df["dt_sec"].apply(
        lambda x: med_dt if (pd.isna(x) or x <= 0) else x
    )

    df["RHin_obs"] = np.clip(df["in_hum"] / 100, 0, 1)
    df["RHout"]    = np.clip(df["out_hum"] / 100, 0, 1)
    df["It"]       = df["out_light"].clip(lower=0)
    df["wind"]     = df["out_windsp"].clip(lower=0.1)
    df["Tout"]     = df["out_temp"]
    df["CO2out"]   = df["out_co2"].fillna(420)

    return df.reset_index(drop=True)


def make_inputs_actions(df: pd.DataFrame,
                        vent_scale=100, curtain_scale=100):
    """?꾩쿂由щ맂 DataFrame?먯꽌 ?멸린 寃쎄퀎議곌굔怨??ㅼ륫 ?됰룞??遺꾨━?쒕떎."""
    boundary = df.assign(k=range(1, len(df)+1))[[
        "k", "reg_date", "dt_sec", "Tout", "RHout", "It", "wind", "CO2out"
    ]].copy()

    action = pd.DataFrame({
        "u_heat":  (df["cont_heater_run"] > 0).astype(float),
        "x_vent":  np.clip(df["cont_skyl_vol"] / vent_scale, 0, 1),
        "curtain": np.clip(df["cont_cur_vol"]  / curtain_scale, 0, 1),
        "u_co2":   (df["cont_co2_run"] > 0).astype(float),
    })
    return {"boundary": boundary, "action": action}


# ?? 6. Open-loop ?쒕??덉씠????????????????????????????????????????????????????

def simulate_openloop(boundary: pd.DataFrame, action: pd.DataFrame,
                      theta: dict, Tin0, RHin0, CO20,
                      A=200, V=300, ramp_max=0.15,
                      show_progress=True) -> pd.DataFrame:
    """?ㅼ륫 ?됰룞 ?쒗?ㅻ? 洹몃?濡??ъ깮(open-loop)?섏뿬 ?쒕??덉씠?섑븳??"""
    n = len(boundary)
    Tin  = np.zeros(n); Tin[0]  = Tin0
    e_in = np.zeros(n); e_in[0] = RHin0 * float(sat_vp_kpa(Tin0))
    CO2  = np.zeros(n); CO2[0]  = CO20

    u_heat = np.zeros(n); x_vent = np.zeros(n)
    curtain = np.zeros(n); u_co2 = np.zeros(n)
    ACH = np.zeros(n); Q_heat = np.zeros(n); Q_ventloss = np.zeros(n)

    prev_a = {col: float(action[col].iloc[0]) for col in ["u_heat","x_vent","curtain","u_co2"]}

    it = range(1, n) if not show_progress else tqdm(range(1, n), desc="open-loop sim")
    for t in it:
        state = {"Tin": Tin[t-1], "e_in": e_in[t-1], "CO2": CO2[t-1]}
        u     = boundary.iloc[t-1]
        a_raw = {col: float(action[col].iloc[t-1]) for col in ["u_heat","x_vent","curtain","u_co2"]}
        a     = project_action(a_raw, prev_a, ramp_max)

        pred = step_dynamics_dt(state, u, a, theta, dt_sec=float(u["dt_sec"]), A=A, V=V)

        Tin[t] = pred["Tin"]; e_in[t] = pred["e_in"]; CO2[t] = pred["CO2"]
        u_heat[t] = a["u_heat"]; x_vent[t] = a["x_vent"]
        curtain[t] = a["curtain"]; u_co2[t] = a["u_co2"]
        ACH[t] = pred["ACH"]; Q_heat[t] = pred["Q_heat"]; Q_ventloss[t] = pred["Q_ventloss"]
        prev_a = a

    RHin   = np.clip(e_in / sat_vp_kpa(Tin), 0, 1)
    VPD    = vpd_kpa(Tin, RHin)
    Td     = dewpoint_c(Tin, RHin)
    dTcond = Tin - Td

    return pd.DataFrame({
        "k": boundary["k"].values, "reg_date": boundary["reg_date"].values,
        "Tout": boundary["Tout"].values, "RHout": boundary["RHout"].values,
        "It": boundary["It"].values, "wind": boundary["wind"].values,
        "CO2out": boundary["CO2out"].values,
        "Tin": Tin, "RHin": RHin, "VPD": VPD, "dTcond": dTcond, "CO2": CO2,
        "u_heat": u_heat, "x_vent": x_vent, "curtain": curtain, "u_co2": u_co2,
        "ACH": ACH, "Q_heat": Q_heat, "Q_ventloss": Q_ventloss,
    })


# ?? 7. Monte Carlo ?뚮씪誘명꽣 ?섑뵆留???????????????????????????????????????????

def sample_theta(rng=None, *, auto_save: bool = False, farm_sn: int | None = None, stage_name: str | None = None, model_name: str = "default", valid_from=None, valid_to=None, store_path: str | None = None) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    theta = {
        "UA":      float(rng.uniform(2000, 7000)),
        "C":       float(rng.uniform(5e6,  4e7)),
        "eta":     float(rng.uniform(0.2,  0.8)),
        "k_heat":  float(rng.uniform(15000, 50000)),
        "a0":      float(rng.uniform(0.05, 0.5)),
        "a1":      float(rng.uniform(2.0,  10.0)),
        "a2":      float(rng.uniform(0.3,  2.0)),
        "a3":      0.0,
        "k_evap":  float(rng.uniform(1e-6, 5e-6)),
        "k_photo": float(rng.uniform(1e-5, 5e-5)),
        "rho_cp":  1.2 * 1005,
    }
    if auto_save:
        saved = approve_and_save_theta_case(
            theta,
            farm_sn=farm_sn,
            stage_name=stage_name,
            data_case="limited",
            model_name=model_name,
            valid_from=valid_from,
            valid_to=valid_to,
            store_path=store_path,
            source="modeling_limited_sampling",
        )
        theta["approval"] = saved["approval"]
        if "record" in saved:
            theta["store_key"] = saved["record"].key
    return theta


def robust_openloop_mc(boundary: pd.DataFrame, action: pd.DataFrame,
                       df_obs: pd.DataFrame,
                       N=30, A=200, V=300, ramp_max=0.15,
                       show_progress=True) -> pd.DataFrame:
    """
    N??Monte Carlo ?섑뵆留곸쑝濡?open-loop KPI 遺꾪룷瑜?怨꾩궛?쒕떎.
    """
    Tin0  = float(df_obs["in_temp"].iloc[0])
    RHin0 = float(np.clip(df_obs["in_hum"].iloc[0] / 100, 0, 1))
    CO20  = float(df_obs["in_co2"].iloc[0]) if df_obs["in_co2"].notna().any() else 900.0

    rng = np.random.default_rng(42)
    Ms  = []
    it  = range(N) if not show_progress else tqdm(range(N), desc="MC open-loop")

    for _ in it:
        theta = sample_theta(rng)
        sim   = simulate_openloop(boundary, action, theta,
                                  Tin0=Tin0, RHin0=RHin0, CO20=CO20,
                                  A=A, V=V, ramp_max=ramp_max,
                                  show_progress=False)
        Ms.append(calc_metrics(sim))

    return pd.concat(Ms, ignore_index=True)


# ?? 8. 硫붿씤 ?ㅽ뻾 ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    start_date = "2025-10-16"
    end_date   = "2025-11-15"

    # data = connector(start_date, end_date)
    # df_ts = prepare_timeseries(data, tz="Asia/Seoul")
    # io = make_inputs_actions(df_ts, vent_scale=100, curtain_scale=100)
    # boundary = io["boundary"]
    # action   = io["action"]

    # ?⑥씪 ?쒕??덉씠??
    # theta = sample_theta()
    # sim = simulate_openloop(boundary, action, theta,
    #                         Tin0=float(df_ts["in_temp"].iloc[0]),
    #                         RHin0=float(np.clip(df_ts["in_hum"].iloc[0]/100, 0, 1)),
    #                         CO20=float(df_ts["in_co2"].iloc[0]) if df_ts["in_co2"].notna().any() else 900,
    #                         show_progress=True)
    # print(calc_metrics(sim))

    # Monte Carlo ?됯?
    # M = robust_openloop_mc(boundary, action, df_ts, N=20, show_progress=True)
    # print(M)

    print("紐⑤뱢 濡쒕뱶 ?꾨즺.")

