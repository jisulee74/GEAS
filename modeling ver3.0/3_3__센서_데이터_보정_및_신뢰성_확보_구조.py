"""
3.3. 센서 데이터 보정 및 신뢰성 확보 구조
슬라이딩 윈도우 ARX 모델 기반으로 온도·습도·CO2 이상치를 탐지하고 보정한다.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, chi2
import sqlalchemy
from sqlalchemy import create_engine
from tqdm import tqdm


# ── 1. DB 연결 ───────────────────────────────────────────────────────────────

def connector(start_date, end_date=None):
    import datetime
    if end_date is None:
        end_date = datetime.datetime.now()

    start_chr = pd.Timestamp(start_date).strftime("%Y-%m-%d %H:%M:%S")
    end_chr   = pd.Timestamp(end_date).strftime("%Y-%m-%d %H:%M:%S")

    engine = create_engine(
        "mysql+pymysql://root:theimc#10!@211.195.9.227:3306/farmstom"
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


# ── 2. 유틸리티 ──────────────────────────────────────────────────────────────

def numify(x):
    """문자열 컬럼을 숫자로 변환한다. 변환 불가 값은 NaN 처리."""
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    x = x.astype(str).str.replace(",", "", regex=False).str.strip()
    x[x.isin(["", "NA", "NaN", "NULL"])] = np.nan
    return pd.to_numeric(x, errors="coerce")


# ── 3. ARX 모델 적합 ─────────────────────────────────────────────────────────

def fit_arx(y: np.ndarray, U: np.ndarray) -> dict:
    """
    ARX(1) 모델: y[t] = phi*y[t-1] + (1-phi)*(a + U[t] @ b) + noise
    BFGS 최적화로 파라미터를 추정한다.
    """
    n = len(y)
    p = U.shape[1]

    def pred_all(par):
        a   = par[0]
        b   = par[1:1+p]
        phi = 1 / (1 + np.exp(-par[1+p]))   # sigmoid → (0,1)
        yhat = np.full(n, np.nan)
        yhat[0] = y[0]
        for t in range(1, n):
            yhat[t] = phi * y[t-1] + (1 - phi) * (a + float(U[t] @ b))
        return yhat

    def nll(par):
        yhat = pred_all(par)
        e = (y - yhat)[1:]
        s2 = np.nanmean(e**2)
        if s2 <= 0:
            return 1e12
        return 0.5*(n-1)*np.log(s2) + 0.5*np.nansum(e**2)/s2

    init = np.concatenate([[np.nanmean(y)], np.zeros(p), [0.0]])
    res  = minimize(nll, init, method="BFGS", options={"maxiter": 500})
    par  = res.x

    a   = par[0]
    b   = par[1:1+p]
    phi = 1 / (1 + np.exp(-par[1+p]))
    yhat = pred_all(par)
    e    = y - yhat
    sigma = float(np.sqrt(np.nanmean(e[1:]**2)))

    return {"a": a, "b": b, "phi": phi, "sigma": sigma, "fitted": yhat, "resid": e}


# ── 4. 이상치 탐지 및 클리핑 보정 ───────────────────────────────────────────

def detect_and_correct_clip(y: np.ndarray, mu: np.ndarray,
                             sigma: np.ndarray, alpha: float = 0.01) -> dict:
    """
    신뢰구간 [mu ± z*sigma] 밖의 값을 이상치로 탐지하고 경계값으로 클리핑한다.
    """
    z = norm.ppf(1 - alpha / 2)
    L = mu - z * sigma
    U = mu + z * sigma

    ok = (np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
          & np.isfinite(L) & np.isfinite(U) & (sigma > 0))

    flag  = np.zeros(len(y), dtype=bool)
    flag[ok] = (y[ok] < L[ok]) | (y[ok] > U[ok])

    ycorr = y.copy()
    idx = np.where(flag)[0]
    if len(idx) > 0:
        ycorr[idx] = np.clip(y[idx], L[idx], U[idx])

    return {"ycorr": ycorr, "flag": flag, "L": L, "U": U}


# ── 5. 온도·습도 QC (슬라이딩 윈도우 + 다변량) ──────────────────────────────

def run_qc_T_RH(df: pd.DataFrame, win: int = 1440, step: int = 30,
                alpha: float = 0.01) -> pd.DataFrame:
    """
    슬라이딩 윈도우마다 ARX 적합 → 단변량 이상치 탐지 → Mahalanobis 다변량 탐지
    순서로 Tin, RHin을 보정한다.
    """
    df = df.sort_values("time").reset_index(drop=True)
    U_cols = ["Tout", "RHout", "Rin", "vWind"]
    U = df[U_cols].values.astype(float)

    Tin_corr  = df["Tin"].values.astype(float)
    RHin_corr = df["RHin"].values.astype(float)
    n = len(df)
    QC_flag = np.zeros(n, dtype=bool)

    starts = list(range(0, n - win, step))
    if not starts:
        raise ValueError("window가 데이터보다 큽니다.")

    for s in tqdm(starts, desc="QC(T/RH)"):
        tr = slice(s, s + win)

        fitT  = fit_arx(Tin_corr[tr],  U[tr])
        detT  = detect_and_correct_clip(Tin_corr[tr],  fitT["fitted"],
                                        np.full(win, fitT["sigma"]), alpha)

        fitH  = fit_arx(RHin_corr[tr], U[tr])
        detH  = detect_and_correct_clip(RHin_corr[tr], fitH["fitted"],
                                        np.full(win, fitH["sigma"]), alpha)

        # 다변량 Mahalanobis 거리
        Y  = np.column_stack([Tin_corr[tr], RHin_corr[tr]])
        MU = np.column_stack([fitT["fitted"], fitH["fitted"]])
        E  = Y - MU
        valid = np.all(np.isfinite(E), axis=1)
        flag_mv = np.zeros(win, dtype=bool)
        if valid.sum() > 2:
            Sigma = np.cov(E[valid].T)
            try:
                invS = np.linalg.inv(Sigma)
                D2   = np.einsum("ij,jk,ik->i", E, invS, E)
                flag_mv = np.isfinite(D2) & (D2 > chi2.ppf(1 - alpha, df=2))
            except np.linalg.LinAlgError:
                pass

        flag_final = detT["flag"] | detH["flag"] | flag_mv
        Tin_corr[tr]    = detT["ycorr"]
        RHin_corr[tr]   = detH["ycorr"]
        QC_flag[s:s+win] |= flag_final

    df = df.copy()
    df["Tin_corr"]  = Tin_corr
    df["RHin_corr"] = RHin_corr
    df["QC_flag"]   = QC_flag
    return df


# ── 6. CO2 델타 모델 ─────────────────────────────────────────────────────────

def fit_co2_delta(C: np.ndarray, X: np.ndarray) -> dict | None:
    """
    dC[t] = b0 + b1*(C[t-1] - Cstar) + X[t] @ b + noise
    BFGS로 추정한다.
    """
    dC = np.diff(C, prepend=np.nan)
    C1 = np.concatenate([[np.nan], C[:-1]])

    mask = (np.isfinite(dC) & np.isfinite(C1)
            & np.all(np.isfinite(X), axis=1))
    if mask.sum() < 80:
        return None

    dC_m = dC[mask]; C1_m = C1[mask]; X_m = X[mask]
    p    = X.shape[1]
    Cstar0 = float(np.median(C1_m))

    def nll(par):
        b0, b1, Cstar = par[0], par[1], par[2]
        b   = par[3:3+p]
        sig = np.exp(par[3+p])
        mu  = b0 + b1*(C1_m - Cstar) + X_m @ b
        e   = dC_m - mu
        return float(0.5 * np.sum(np.log(2*np.pi*sig**2) + e**2/sig**2))

    sig0 = float(np.std(dC_m) + 1e-6)
    init = np.concatenate([[0, -0.01, Cstar0], np.zeros(p), [np.log(sig0)]])
    res  = minimize(nll, init, method="BFGS", options={"maxiter": 500})
    par  = res.x

    return {
        "b0": par[0], "b1": par[1], "Cstar": par[2],
        "b":  par[3:3+p], "sigma": float(np.exp(par[3+p])),
    }


def predict_co2_1step(C: np.ndarray, X: np.ndarray, fit: dict) -> np.ndarray:
    """1-step ahead CO2 예측값을 계산한다."""
    n   = len(C)
    muC = np.full(n, np.nan)
    C1  = np.concatenate([[np.nan], C[:-1]])

    for t in range(1, n):
        if not np.isfinite(C1[t]):
            continue
        if not np.all(np.isfinite(X[t])):
            continue
        mu_dC  = fit["b0"] + fit["b1"]*(C1[t] - fit["Cstar"]) + float(X[t] @ fit["b"])
        muC[t] = C1[t] + mu_dC

    return muC


# ── 7. CO2 QC ────────────────────────────────────────────────────────────────

def run_qc_CO2(df: pd.DataFrame, win: int = 1440, step: int = 30,
               alpha: float = 0.01) -> pd.DataFrame:
    """슬라이딩 윈도우로 CO2 이상치를 탐지·보정한다."""
    df = df.sort_values("time").reset_index(drop=True)
    n  = len(df)

    CO2 = df["CO2"].values.copy().astype(float)
    CO2[(~np.isfinite(CO2)) | (CO2 <= 0) | (CO2 > 5000)] = np.nan

    Tin = df["Tin_corr"].values.astype(float)
    RH  = df["RHin_corr"].values.astype(float)
    dTin = np.concatenate([[np.nan], np.diff(Tin)])
    dRH  = np.concatenate([[np.nan], np.diff(RH)])
    day  = (df["Rin"].values > 5).astype(float)

    fan  = df["fan"].fillna(0).values.astype(float)
    skyl = df["skyl"].fillna(0).values.astype(float)
    heat = df["heat"].fillna(0).values.astype(float)

    X = np.column_stack([day, df["Rin"].values, df["vWind"].values,
                         fan, skyl, heat, dTin, dRH])

    starts  = list(range(0, n - win, step))
    CO2_corr = CO2.copy()
    CO2_flag = np.zeros(n, dtype=bool)

    for s in tqdm(starts, desc="QC(CO2)"):
        tr = slice(s, s + win)
        ok = np.all(np.isfinite(np.column_stack([CO2_corr[tr], X[tr]])), axis=1)
        if ok.sum() < 200:
            continue

        fit = fit_co2_delta(CO2_corr[tr][ok], X[tr][ok])
        if fit is None:
            continue

        mu   = np.full(win, np.nan)
        mu_ok = predict_co2_1step(CO2_corr[tr][ok], X[tr][ok], fit)

        idx_ok = np.where(ok)[0]
        for ii, mu_val in zip(idx_ok, mu_ok):
            mu[ii] = mu_val

        det = detect_and_correct_clip(CO2_corr[tr], mu,
                                      np.full(win, fit["sigma"]), alpha)
        CO2_corr[s:s+win]  = det["ycorr"]
        CO2_flag[s:s+win] |= det["flag"]

    df = df.copy()
    df["CO2_corr"]  = CO2_corr
    df["CO2_QC_flag"] = CO2_flag
    return df


# ── 8. 통합 파이프라인 ───────────────────────────────────────────────────────

def make_state_df(data: pd.DataFrame, plot: bool = False,
                  win: int = 1440, step: int = 30,
                  alpha: float = 0.01) -> pd.DataFrame:
    """raw DB 데이터 → 정제된 상태 DataFrame 반환."""

    def _get(col, default=0):
        return numify(data[col]) if col in data.columns else pd.Series(default, index=data.index)

    df = pd.DataFrame({
        "time":   pd.to_datetime(data["reg_date"]),
        "Tin":    numify(data["in_temp"]),
        "RHin":   numify(data["in_hum"]),
        "CO2":    numify(data["in_co2"]),
        "Tout":   numify(data["out_temp"]),
        "RHout":  numify(data["out_hum"]),
        "Rin":    numify(data["out_light"]),
        "vWind":  numify(data["out_windsp"]),
        "wDir":   numify(data["out_winddirec"]),
        "fan":    _get("cont_fan_run"),
        "skyl":   _get("cont_skyl_vol"),
        "heat":   _get("cont_heater_run"),
    })

    df = df.sort_values("time").dropna(subset=["time"]).reset_index(drop=True)

    df1 = run_qc_T_RH(df,  win=win, step=step, alpha=alpha)
    df2 = run_qc_CO2(df1,  win=win, step=step, alpha=alpha)

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(df2["time"], df2["Tin"],      label="raw",       color="black")
        axes[0].plot(df2["time"], df2["Tin_corr"], label="corrected", color="red")
        axes[0].set_ylabel("Temp"); axes[0].legend()
        axes[0].set_title("Internal Temperature: raw vs corrected")

        axes[1].plot(df2["time"], df2["RHin"],      color="black", label="raw")
        axes[1].plot(df2["time"], df2["RHin_corr"], color="red",   label="corrected")
        axes[1].set_ylabel("RH"); axes[1].legend()
        axes[1].set_title("Internal Humidity: raw vs corrected")

        axes[2].plot(df2["time"], df2["CO2"],      color="black", label="raw")
        axes[2].plot(df2["time"], df2["CO2_corr"], color="red",   label="corrected")
        axes[2].set_ylabel("CO2"); axes[2].legend()
        axes[2].set_title("Internal CO2: raw vs corrected")

        plt.tight_layout()
        plt.show()

    return df2


# ── 9. 메인 실행 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_date = "2025-10-16"
    end_date   = "2025-11-15"

    # data = connector(start_date, end_date)
    # df_state = make_state_df(data, plot=True, win=1440, step=30, alpha=0.01)

    print("모듈 로드 완료.")
