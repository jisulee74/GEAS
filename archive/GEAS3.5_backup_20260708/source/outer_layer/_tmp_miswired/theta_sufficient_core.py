"""
1. ?⑥씪 援ы쉷 紐⑤뜽 ?뚭? 湲곕컲 ?앸퀎 諛⑸쾿
?⑥떎 ?댄룊??紐⑤뜽??臾쇰━ ?뚮씪誘명꽣瑜??뚭? + 理쒖쟻?붾줈 ?앸퀎?섎뒗 肄붾뱶
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import huber
import sqlalchemy
from sqlalchemy import create_engine


# ?? 1. DB ?곌껐 諛??곗씠??議고쉶 ?????????????????????????????????????????????????

def connector(start_date, end_date=None):
    """MySQL DB?먯꽌 ?⑥떎 ?쇱꽌 ?곗씠?곕? 議고쉶?쒕떎."""
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

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """raw ?곗씠?곕? ?뚭? 遺꾩꽍???????덈룄濡??꾩쿂由ы븳??"""
    df = df.copy()
    df["reg_date"] = pd.to_datetime(df["reg_date"])
    df = df.sort_values("reg_date").reset_index(drop=True)

    df["Tin"]   = pd.to_numeric(df["in_temp"],         errors="coerce")
    df["Tout"]  = pd.to_numeric(df["out_temp"],         errors="coerce")
    df["It"]    = pd.to_numeric(df["out_light"],        errors="coerce")
    df["H_t"]   = pd.to_numeric(df["cont_heater_run"], errors="coerce")
    df["V_raw"] = pd.to_numeric(df["cont_skyl_vol"],   errors="coerce")

    Vmax = df["V_raw"].max()
    df["V_t"] = df["V_raw"] / Vmax if (np.isfinite(Vmax) and Vmax > 0) else 0.0

    # ?쒓컙 媛꾧꺽 (珥????쒓컙)
    df["dt_sec"] = df["reg_date"].diff().dt.total_seconds()
    df["dt_hr"]  = df["dt_sec"] / 3600.0

    # dTin/dt (?먯떆媛?
    df["dTin"]         = df["Tin"].diff()
    df["dTin_dt_raw"]  = df["dTin"] / df["dt_hr"]

    df = df.dropna(subset=["dTin_dt_raw"]).copy()
    df = df[df["dt_hr"] > 0].reset_index(drop=True)

    # 3-?ъ씤??濡ㅻ쭅 ?됯퇏?쇰줈 ?몄씠利?媛먯냼
    df["dTin_dt"] = (
        df["dTin_dt_raw"]
        .rolling(window=3, min_periods=1)
        .mean()
    )
    df = df.dropna(subset=["dTin_dt"]).reset_index(drop=True)

    return df


# ?? 3. Huber ?뚭? ????????????????????????????????????????????????????????????

def fit_huber(df: pd.DataFrame):
    """
    dTin/dt = a1*H_t + a2*It + a3*(Tin-Tout) + a4*V_t*(Tin-Tout)
    瑜?Huber 濡쒕쾭?ㅽ듃 ?뚭?濡?異붿젙?쒕떎.
    """
    from sklearn.linear_model import HuberRegressor

    df = df.copy()
    df["X1"] = df["H_t"]
    df["X2"] = df["It"]
    df["X3"] = df["Tin"] - df["Tout"]
    df["X4"] = df["V_t"] * (df["Tin"] - df["Tout"])

    mask = df[["X1", "X2", "X3", "X4", "dTin_dt"]].notna().all(axis=1)
    X = df.loc[mask, ["X1", "X2", "X3", "X4"]].values
    y = df.loc[mask, "dTin_dt"].values

    # HuberRegressor: sklearn 援ы쁽 (MASS::rlm怨??숈씪 紐⑹쟻)
    model = HuberRegressor(epsilon=1.35, max_iter=500, fit_intercept=True)
    model.fit(X, y)

    coefs = {
        "intercept": model.intercept_,
        "a1": model.coef_[0],   # H_t  ??k_heat/C
        "a2": model.coef_[1],   # It   ??eta/C
        "a3": model.coef_[2],   # ?T   ??-UA/C
        "a4": model.coef_[3],   # V_t*?T ??-Kvent/C
    }
    return coefs, df[mask].copy()


# ?? 4. ?쒕??덉씠??????????????????????????????????????????????????????????????

def simulate_Tin(params: list, data: pd.DataFrame) -> np.ndarray:
    """
    ?ㅼ씪???곷텇?쇰줈 ?ㅻ궡?⑤룄瑜??쒓컙 ?꾩쭊 ?쒕??덉씠?섑븳??

    params = [C, k_heat, UA, eta, K_vent]
    """
    C, k_heat, UA, eta, K_vent = params

    n = len(data)
    Tin_sim = np.full(n, np.nan)
    Tin_sim[0] = data["Tin"].iloc[0]

    Tout_arr = data["Tout"].values
    It_arr   = data["It"].values
    H_t_arr  = data["H_t"].values
    V_t_arr  = data["V_t"].values
    dt_arr   = data["dt_hr"].values

    for i in range(n - 1):
        Tin  = Tin_sim[i]
        Tout = Tout_arr[i]
        It   = It_arr[i]
        H_t  = H_t_arr[i]
        V_t  = V_t_arr[i]
        dt   = dt_arr[i]

        dTin_dt = (1 / C) * (
            k_heat * H_t
            + eta   * It
            - UA    * (Tin - Tout)
            - K_vent * V_t * (Tin - Tout)
        )
        Tin_sim[i + 1] = Tin + dTin_dt * dt

    return Tin_sim


def simulate_Tin_from_ratio(C_val: float, data: pd.DataFrame,
                             a1, a2, a3, a4) -> np.ndarray:
    """
    ?뚭?怨꾩닔 鍮꾩쑉?먯꽌 臾쇰━ ?뚮씪誘명꽣瑜??좊룄?????쒕??덉씠?섑븳??
    """
    k_heat = a1 * C_val
    eta    = a2 * C_val
    UA     = -a3 * C_val
    Kvent  = -a4 * C_val
    return simulate_Tin([C_val, k_heat, UA, eta, Kvent], data)


# ?? 5. 理쒖쟻??????????????????????????????????????????????????????????????????

def loss_C(C_val, data, a1, a2, a3, a4):
    """C?????MSE ?먯떎 ?⑥닔."""
    if C_val <= 0:
        return 1e12
    Tin_sim = simulate_Tin_from_ratio(C_val, data, a1, a2, a3, a4)
    residuals = Tin_sim - data["Tin"].values
    return float(np.nanmean(residuals ** 2))


def estimate_C(df_sub: pd.DataFrame, coefs: dict):
    """L-BFGS-B濡?理쒖쟻 C瑜??먯깋?쒕떎."""
    from scipy.optimize import minimize

    a1 = coefs["a1"]; a2 = coefs["a2"]
    a3 = coefs["a3"]; a4 = coefs["a4"]

    result = minimize(
        fun=loss_C,
        x0=[1e5],
        args=(df_sub, a1, a2, a3, a4),
        method="L-BFGS-B",
        bounds=[(1e3, 1e7)],
        options={"maxiter": 200},
    )
    return float(result.x[0])


# ?? 6. 寃곌낵 異쒕젰 ?????????????????????????????????????????????????????????????

def print_param_summary(C_est, k_heat_est, eta_est, UA_est, K_vent_est, coefs):
    print("\nPhysical parameter identification summary\n")
    print("Regression: dTin_dt ~ a1*H_t + a2*It + a3*(Tin-Tout) + a4*V_t*(Tin-Tout)")
    print(f"  a1 ??k_heat / C  ?? {coefs['a1']:.6f}")
    print(f"  a2 ??eta   / C  ?? {coefs['a2']:.6f}")
    print(f"  a3 ??-UA   / C  ?? {coefs['a3']:.6f}")
    print(f"  a4 ??-Kvent/ C  ?? {coefs['a4']:.6f}\n")
    print("Estimated physical parameters")
    print(f"  C      (?댁슜??          : {C_est:.3f}")
    print(f"  k_heat (?쒕갑 ?대뱷)       : {k_heat_est:.3f}")
    print(f"  eta    (?쒖뼇愿??대뱷)     : {eta_est:.3f}")
    print(f"  UA     (援ъ“泥??댁넀??   : {UA_est:.3f}")
    print(f"  K_vent (?섍린 ?댁넀??     : {K_vent_est:.3f}")
    print("\nInterpretation")
    print("  C ?댁닔濡??⑤룄 蹂???먮┝")
    print("  k_heat: ?쒖뼱 ?⑥쐞???쒕갑 ?④낵")
    print("  eta: ?쇱궗瑜??ㅻ궡 ?대줈 ?꾪솚?섎뒗 鍮꾩쑉")
    print("  UA: ?멸린濡쒖쓽 ?섎룞 ?댁넀??)
    print("  K_vent: ?섍린???섑븳 異붽? ?댁넀??)


# ?? 7. 硫붿씤 ?ㅽ뻾 ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    # ?곗씠??議고쉶 (?ㅼ젣 DB ?곌껐 ??二쇱꽍 ?댁젣)
    # df_raw = connector("2025-01-01 00:00:00")

    # 濡쒖뺄 ?뚯뒪?몄슜 ?붾? ?ㅽ뻾 ?덉떆 (DB 誘몄뿰寃???
    # df_raw = pd.read_csv("sample_data.csv")

    # ?꾩쿂由?
    # df_proc = preprocess(df_raw)

    # Huber ?뚭?
    # coefs, df_reg = fit_huber(df_proc)

    # ?쒕툕?섑뵆留?(5?ㅽ뀦留덈떎)
    # df_sub = df_proc.iloc[::5].reset_index(drop=True)

    # C 理쒖쟻??
    # C_est = estimate_C(df_sub, coefs)

    # ?뚮씪誘명꽣 ?꾩텧
    # k_heat_est = coefs["a1"] * C_est
    # eta_est    = coefs["a2"] * C_est
    # UA_est     = -coefs["a3"] * C_est
    # K_vent_est = -coefs["a4"] * C_est

    # 寃곌낵 異쒕젰
    # print_param_summary(C_est, k_heat_est, eta_est, UA_est, K_vent_est, coefs)

    # 理쒖쥌 ?쒕??덉씠??
    # Tin_sim_final = simulate_Tin([C_est, k_heat_est, UA_est, eta_est, K_vent_est], df_proc)

    print("紐⑤뱢 濡쒕뱶 ?꾨즺. connector() ?몄텧 ??workflow瑜??ㅽ뻾?섏꽭??")

