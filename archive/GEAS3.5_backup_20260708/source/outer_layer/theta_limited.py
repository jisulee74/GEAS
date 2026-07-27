"""
2. ?⑥씪 援ы쉷 紐⑤뜽 ?뚭? 湲곕컲 ?앸퀎 諛⑸쾿 ?쇰컲??
?⑥떎 硫댁쟻쨌?쇰났??醫낅쪟瑜??낅젰諛쏆븘 臾쇰━ prior瑜??먮룞 怨꾩궛?섍퀬,
?ㅼ쨷 ?쒖옉??L-BFGS-B 理쒖쟻?붾줈 ?뚮씪誘명꽣瑜??앸퀎?쒕떎.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sqlalchemy
from sqlalchemy import create_engine

from physics_store_sync import save_theta_case


# ?? 1. DB ?곌껐 ???????????????????????????????????????????????????????????????

def connector(start_date, end_date=None):
    import datetime
    if end_date is None:
        end_date = datetime.datetime.now()

    start_chr = pd.Timestamp(start_date).strftime("%Y-%m-%d %H:%M:%S")
    end_chr   = pd.Timestamp(end_date).strftime("%Y-%m-%d %H:%M:%S")

    engine = create_engine(
        "mysql+pymysql://<user>:<password>@<host>:<port>/<database>"
    )
    query = f"""
        SELECT *
        FROM data_silla_enc
        WHERE iot_data_idx = 97
          AND reg_date >= '{start_chr}'
          AND reg_date <  '{end_chr}'
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    print(df.columns.tolist())
    return df


# ?? 2. ?꾩쿂由?????????????????????????????????????????????????????????????????

def preprocess_for_ident(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["reg_date"] = pd.to_datetime(df["reg_date"])
    df = df.sort_values("reg_date").reset_index(drop=True)

    df["Tin"]   = pd.to_numeric(df["in_temp"],         errors="coerce")
    df["Tout"]  = pd.to_numeric(df["out_temp"],         errors="coerce")
    df["It"]    = pd.to_numeric(df["out_light"],        errors="coerce")
    df["H_t"]   = pd.to_numeric(df["cont_heater_run"], errors="coerce")
    df["V_raw"] = pd.to_numeric(df["cont_skyl_vol"],   errors="coerce")

    Vmax = df["V_raw"].max()
    if np.isfinite(Vmax) and Vmax > 0:
        df["V_t"] = df["V_raw"] / Vmax
    else:
        df["V_t"] = 0.0

    df["dt_sec"] = df["reg_date"].diff().dt.total_seconds()
    df["dt_hr"]  = df["dt_sec"] / 3600.0

    keep_cols = ["Tin", "Tout", "It", "H_t", "V_t", "dt_hr"]
    df = df.dropna(subset=keep_cols)
    df = df[df["dt_hr"] > 0].reset_index(drop=True)

    return df


# ?? 3. Prior 怨꾩궛 ????????????????????????????????????????????????????????????

COVER_U = {
    "single_film": 7,
    "double_film": 5,
    "glass":       5,
    "panel":       2,
}

def compute_priors_from_area(area_m2: float,
                              cover_type: str = "single_film",
                              height_m: float = 4.0,
                              air_cp: float = 1005.0,
                              air_rho: float = 1.2) -> dict:
    """
    ?⑥떎 硫댁쟻쨌?쇰났???뺣낫瑜??댁슜??臾쇰━ ?뚮씪誘명꽣??珥덇린媛믨낵 ?먯깋 踰붿쐞瑜?諛섑솚?쒕떎.
    """
    U_base = COVER_U.get(cover_type, 7)

    UA_prior  = U_base * area_m2
    volume_m3 = area_m2 * height_m
    C_air     = air_cp * air_rho * volume_m3
    C_prior   = C_air * 3

    return {
        "priors": {
            "C_prior":      C_prior,
            "UA_prior":     UA_prior,
            "k_heat_prior": UA_prior * 10,
            "eta_prior":    UA_prior * 0.1,
            "Kvent_prior":  UA_prior * 5,
        },
        "bounds": {
            "C_lower":    C_prior * 0.3,
            "C_upper":    C_prior * 5.0,
            "UA_lower":   UA_prior * 0.3,
            "UA_upper":   UA_prior * 3.0,
            "k_heat_min": 0,
            "k_heat_max": UA_prior * 10 * 20,
            "eta_min":    0,
            "eta_max":    UA_prior * 0.1 * 20,
            "Kvent_min":  0,
            "Kvent_max":  UA_prior * 5 * 20,
        },
    }


# ?? 4. ?쒕??덉씠??????????????????????????????????????????????????????????????

def simulate_Tin(params, data: pd.DataFrame) -> np.ndarray:
    """?ㅼ씪???곷텇?쇰줈 ?ㅻ궡?⑤룄瑜??쒕??덉씠?섑븳??"""
    C, k_heat, eta, UA, Kvent = params

    n = len(data)
    Tin_sim = np.full(n, np.nan)
    Tin_sim[0] = data["Tin"].iloc[0]

    Tout = data["Tout"].values
    It   = data["It"].values
    H_t  = data["H_t"].values
    V_t  = data["V_t"].values
    dt   = data["dt_hr"].values

    for i in range(n - 1):
        dTin_dt = (1 / C) * (
            k_heat * H_t[i]
            + eta  * It[i]
            - UA   * (Tin_sim[i] - Tout[i])
            - Kvent * V_t[i] * (Tin_sim[i] - Tout[i])
        )
        Tin_sim[i + 1] = Tin_sim[i] + dTin_dt * dt[i]

    return Tin_sim


# ?? 5. ?먯떎 ?⑥닔 (soft prior ?ы븿) ??????????????????????????????????????????

def loss_mse_with_soft_priors(param_vec, data, priors,
                               lambda_C=0.0, lambda_UA=0.0):
    C, k_heat, eta, UA, Kvent = param_vec

    if C <= 0 or UA <= 0 or k_heat < 0 or eta < 0 or Kvent < 0:
        return 1e12

    Tin_sim = simulate_Tin(param_vec, data)
    mse = float(np.nanmean((Tin_sim - data["Tin"].values) ** 2))

    penalty_C  = lambda_C  * (C  - priors["C_prior"])  ** 2
    penalty_UA = lambda_UA * (UA - priors["UA_prior"]) ** 2

    return mse + penalty_C + penalty_UA


# ?? 6. ?ㅼ쨷 ?쒖옉??理쒖쟻??????????????????????????????????????????????????????

def _period_bounds(df: pd.DataFrame):
    if df is None or df.empty or 'reg_date' not in df.columns:
        return None, None
    ts = pd.to_datetime(df['reg_date'], errors='coerce').dropna()
    if ts.empty:
        return None, None
    return ts.iloc[0].isoformat(), ts.iloc[-1].isoformat()


def identify_physical_params_auto(df: pd.DataFrame,
                                   area_m2: float,
                                   cover_type: str = "single_film",
                                   height_m: float = 4.0,
                                   lambda_C: float = 0.0,
                                   lambda_UA: float = 0.0,
                                   n_start: int = 3,
                                   *,
                                   auto_save: bool = True,
                                   farm_sn: int | None = None,
                                   stage_name: str | None = None,
                                   model_name: str = 'default',
                                   store_path: str | None = None) -> dict:
    """
    臾쇰━ prior瑜?湲곕컲?쇰줈 n_start踰??쒕뜡 珥덇린????L-BFGS-B 理쒖쟻?붾? ?섑뻾?섍퀬
    媛????? MSE瑜?湲곕줉??寃곌낵瑜?諛섑솚?쒕떎.
    """
    prior_info = compute_priors_from_area(area_m2, cover_type, height_m)
    pr = prior_info["priors"]
    bd = prior_info["bounds"]

    init_center = np.array([
        pr["C_prior"],
        pr["k_heat_prior"],
        pr["eta_prior"],
        pr["UA_prior"],
        pr["Kvent_prior"],
    ])

    bounds = [
        (bd["C_lower"],    bd["C_upper"]),
        (bd["k_heat_min"], bd["k_heat_max"]),
        (bd["eta_min"],    bd["eta_max"]),
        (bd["UA_lower"],   bd["UA_upper"]),
        (bd["Kvent_min"],  bd["Kvent_max"]),
    ]

    priors_for_loss = {"C_prior": pr["C_prior"], "UA_prior": pr["UA_prior"]}

    best_val = np.inf
    best_par = init_center.copy()
    best_res = None

    rng = np.random.default_rng(seed=0)

    for s in range(n_start):
        # 濡쒓렇 ?뺢퇋 吏??(R??exp(rnorm(5, 0, 0.5))? ?숈씪)
        jitter = np.exp(rng.normal(0, 0.5, size=5))
        init_params = init_center * jitter

        # 寃쎄퀎 ?대━??
        for j, (lo, hi) in enumerate(bounds):
            init_params[j] = np.clip(init_params[j], lo, hi)

        res = minimize(
            fun=loss_mse_with_soft_priors,
            x0=init_params,
            args=(df, priors_for_loss, lambda_C, lambda_UA),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )

        if res.fun < best_val:
            best_val = res.fun
            best_par = res.x
            best_res = res

    theta_est = {
        "C": float(best_par[0]),
        "UA": float(best_par[3]),
        "k_heat": float(best_par[1]),
        "eta": float(best_par[2]),
        "k_vent": float(best_par[4]),
    }

    out = {
        "theta_est":   theta_est,
        "par_est":     best_par,
        "value":       best_val,
        "prior_info":  prior_info,
        "optim_best":  best_res,
        "convergence": best_res.status if best_res else None,
    }
    valid_from, valid_to = _period_bounds(df)
    out["method"] = "single_compartment_generalized"
    out["n_obs"] = int(len(df.index))
    out["valid_from"] = valid_from
    out["valid_to"] = valid_to
    if auto_save:
        record = save_theta_case(
            dict(out['theta_est']),
            farm_sn=farm_sn,
            stage_name=stage_name,
            data_case='limited',
            model_name=model_name,
            valid_from=valid_from,
            valid_to=valid_to,
            store_path=store_path,
            source='identification_limited_generalized',
            metadata={
                'method': out['method'],
                'n_obs': out['n_obs'],
                'value': out['value'],
                'convergence': out['convergence'],
                'raw_par_est': list(out['par_est']),
                'prior_info': out['prior_info'],
                'area_m2': area_m2,
                'cover_type': cover_type,
                'height_m': height_m,
                'lambda_C': lambda_C,
                'lambda_UA': lambda_UA,
                'n_start': n_start,
            },
        )
        out['store_key'] = record.key
    return out


# ?? 7. 硫붿씤 ?ㅽ뻾 ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    # ?곗씠??議고쉶 (?ㅼ젣 DB ?곌껐 ??二쇱꽍 ?댁젣)
    # df_raw  = connector("2025-11-08 00:00:00")
    # df_proc = preprocess_for_ident(df_raw)

    # ?뚮씪誘명꽣 ?앸퀎
    # res = identify_physical_params_auto(
    #     df=df_proc,
    #     area_m2=360,
    #     cover_type="glass",
    #     height_m=4,
    #     lambda_C=0,
    #     lambda_UA=0,
    #     n_start=5,
    # )

    # 異붿젙 ?뚮씪誘명꽣 異쒕젰
    # labels = ["C", "k_heat", "eta", "UA", "Kvent"]
    # for label, val in zip(labels, res["par_est"]):
    #     print(f"  {label:10s}: {val:.4f}")

    # 理쒖쥌 ?쒕??덉씠??
    # Tin_sim_final = simulate_Tin(res["par_est"], df_proc)

    print("紐⑤뱢 濡쒕뱶 ?꾨즺.")

