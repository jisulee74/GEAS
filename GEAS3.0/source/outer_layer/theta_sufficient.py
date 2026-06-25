"""
1. 단일 구획 모델 회귀 기반 식별 방법
온실 열평형 모델의 물리 파라미터를 회귀 + 최적화로 식별하는 코드
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import huber
import sqlalchemy
from sqlalchemy import create_engine

from physics_store_sync import save_theta_case


# ── 1. DB 연결 및 데이터 조회 ─────────────────────────────────────────────────

def connector(start_date, end_date=None):
    """MySQL DB에서 온실 센서 데이터를 조회한다."""
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


# ── 2. 전처리 ────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """raw 데이터를 회귀 분석에 쓸 수 있도록 전처리한다."""
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

    # 시간 간격 (초 → 시간)
    df["dt_sec"] = df["reg_date"].diff().dt.total_seconds()
    df["dt_hr"]  = df["dt_sec"] / 3600.0

    # dTin/dt (원시값)
    df["dTin"]         = df["Tin"].diff()
    df["dTin_dt_raw"]  = df["dTin"] / df["dt_hr"]

    df = df.dropna(subset=["dTin_dt_raw"]).copy()
    df = df[df["dt_hr"] > 0].reset_index(drop=True)

    # 3-포인트 롤링 평균으로 노이즈 감소
    df["dTin_dt"] = (
        df["dTin_dt_raw"]
        .rolling(window=3, min_periods=1)
        .mean()
    )
    df = df.dropna(subset=["dTin_dt"]).reset_index(drop=True)

    return df


# ── 3. Huber 회귀 ────────────────────────────────────────────────────────────

def fit_huber(df: pd.DataFrame, area_m2: float = 360.0):
    """
    dTin/dt = a1*H_t + a2*(It*A) + a3*(Tin-Tout) + a4*V_t*(Tin-Tout)
    를 Huber 로버스트 회귀로 추정한다.
    """
    from sklearn.linear_model import HuberRegressor

    area = float(area_m2)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("area_m2 must be a positive finite value")

    df = df.copy()
    df["X1"] = df["H_t"]
    df["X2"] = df["It"] * area
    df["X3"] = df["Tin"] - df["Tout"]
    df["X4"] = df["V_t"] * (df["Tin"] - df["Tout"])

    mask = df[["X1", "X2", "X3", "X4", "dTin_dt"]].notna().all(axis=1)
    X = df.loc[mask, ["X1", "X2", "X3", "X4"]].values
    y = df.loc[mask, "dTin_dt"].values

    # HuberRegressor: sklearn 구현 (MASS::rlm과 동일 목적)
    model = HuberRegressor(epsilon=1.35, max_iter=500, fit_intercept=True)
    model.fit(X, y)

    coefs = {
        "intercept": model.intercept_,
        "a1": model.coef_[0],   # H_t  → k_heat/C
        "a2": model.coef_[1],   # It*A → eta/C
        "a3": model.coef_[2],   # ΔT   → -UA/C
        "a4": model.coef_[3],   # V_t*ΔT → -Kvent/C
    }
    return coefs, df[mask].copy()


# ── 4. 시뮬레이션 ────────────────────────────────────────────────────────────

def simulate_Tin(params: list, data: pd.DataFrame, area_m2: float = 360.0) -> np.ndarray:
    """
    오일러 적분으로 실내온도를 시간 전진 시뮬레이션한다.

    params = [C, k_heat, UA, eta, K_vent]
    """
    C, k_heat, UA, eta, K_vent = params
    area = float(area_m2)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("area_m2 must be a positive finite value")

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
            + eta   * It * area
            - UA    * (Tin - Tout)
            - K_vent * V_t * (Tin - Tout)
        )
        Tin_sim[i + 1] = Tin + dTin_dt * dt

    return Tin_sim


def simulate_Tin_from_ratio(C_val: float, data: pd.DataFrame,
                             a1, a2, a3, a4, area_m2: float = 360.0) -> np.ndarray:
    """
    회귀계수 비율에서 물리 파라미터를 유도한 뒤 시뮬레이션한다.
    """
    k_heat = a1 * C_val
    eta    = a2 * C_val
    UA     = -a3 * C_val
    Kvent  = -a4 * C_val
    return simulate_Tin([C_val, k_heat, UA, eta, Kvent], data, area_m2=area_m2)


# ── 5. 최적화 ────────────────────────────────────────────────────────────────

def loss_C(C_val, data, a1, a2, a3, a4, area_m2):
    """C에 대한 MSE 손실 함수."""
    if C_val <= 0:
        return 1e12
    Tin_sim = simulate_Tin_from_ratio(C_val, data, a1, a2, a3, a4, area_m2=area_m2)
    residuals = Tin_sim - data["Tin"].values
    return float(np.nanmean(residuals ** 2))


def estimate_C(df_sub: pd.DataFrame, coefs: dict, area_m2: float = 360.0):
    """L-BFGS-B로 최적 C를 탐색한다."""
    from scipy.optimize import minimize

    a1 = coefs["a1"]; a2 = coefs["a2"]
    a3 = coefs["a3"]; a4 = coefs["a4"]

    result = minimize(
        fun=loss_C,
        x0=[1e5],
        args=(df_sub, a1, a2, a3, a4, area_m2),
        method="L-BFGS-B",
        bounds=[(1e3, 1e7)],
        options={"maxiter": 200},
    )
    return float(result.x[0])


def build_identification_result(coefs: dict, c_est: float, area_m2: float = 360.0) -> dict:
    """
    문서형 물리파라미터(theta_est)를 구성한다.
    """
    k_heat_est = float(coefs["a1"] * c_est)
    eta_est = float(coefs["a2"] * c_est)
    UA_est = float(-coefs["a3"] * c_est)
    k_vent_est = float(-coefs["a4"] * c_est)

    theta_est = {
        "C": float(c_est),
        "UA": UA_est,
        "k_heat": k_heat_est,
        "eta": eta_est,
        "k_vent": k_vent_est,
        "A": float(area_m2),
    }
    return {
        "theta_est": theta_est,
        "coefs": dict(coefs),
        "c_est": float(c_est),
    }


def _period_bounds(df: pd.DataFrame):
    if df is None or df.empty or 'reg_date' not in df.columns:
        return None, None
    ts = pd.to_datetime(df['reg_date'], errors='coerce').dropna()
    if ts.empty:
        return None, None
    return ts.iloc[0].isoformat(), ts.iloc[-1].isoformat()


def identify_physical_params(
    df_raw: pd.DataFrame,
    subsample_step: int = 5,
    *,
    auto_save: bool = True,
    farm_sn: int | None = None,
    stage_name: str | None = None,
    model_name: str = 'default',
    store_path: str | None = None,
    area_m2: float = 360.0,
) -> dict:
    """
    raw 데이터를 받아 물리파라미터를 식별하고 공통 반환 구조로 돌려준다.
    필요 시 현재 저장소 형식에 맞게 자동 저장한다.
    """
    df_proc = preprocess(df_raw)
    coefs, df_reg = fit_huber(df_proc, area_m2=area_m2)
    df_sub = df_proc.iloc[::subsample_step].reset_index(drop=True)
    c_est = estimate_C(df_sub, coefs, area_m2=area_m2)

    out = build_identification_result(coefs, c_est, area_m2=area_m2)
    valid_from, valid_to = _period_bounds(df_raw)
    out.update(
        {
            "method": "single_compartment_regression",
            "n_obs": int(len(df_proc.index)),
            "n_reg": int(len(df_reg.index)),
            "n_subsample": int(len(df_sub.index)),
            "area_m2": float(area_m2),
            "valid_from": valid_from,
            "valid_to": valid_to,
        }
    )
    if auto_save:
        record = save_theta_case(
            dict(out['theta_est']),
            farm_sn=farm_sn,
            stage_name=stage_name,
            data_case='sufficient',
            model_name=model_name,
            valid_from=valid_from,
            valid_to=valid_to,
            store_path=store_path,
            source='identification_sufficient_basic',
            metadata={
                'method': out['method'],
                'n_obs': out['n_obs'],
                'n_reg': out['n_reg'],
                'n_subsample': out['n_subsample'],
                'area_m2': float(area_m2),
                'coefs': out['coefs'],
                'c_est': out['c_est'],
            },
        )
        out['store_key'] = record.key
    return out


# ── 6. 결과 출력 ─────────────────────────────────────────────────────────────

def print_param_summary(C_est, k_heat_est, eta_est, UA_est, K_vent_est, coefs):
    print("\nPhysical parameter identification summary\n")
    print("Regression: dTin_dt ~ a1*H_t + a2*It + a3*(Tin-Tout) + a4*V_t*(Tin-Tout)")
    print(f"  a1 ≈ k_heat / C  →  {coefs['a1']:.6f}")
    print(f"  a2 ≈ eta   / C  →  {coefs['a2']:.6f}")
    print(f"  a3 ≈ -UA   / C  →  {coefs['a3']:.6f}")
    print(f"  a4 ≈ -Kvent/ C  →  {coefs['a4']:.6f}\n")
    print("Estimated physical parameters")
    print(f"  C      (열용량)          : {C_est:.3f}")
    print(f"  k_heat (난방 이득)       : {k_heat_est:.3f}")
    print(f"  eta    (태양광 이득)     : {eta_est:.3f}")
    print(f"  UA     (구조체 열손실)   : {UA_est:.3f}")
    print(f"  K_vent (환기 열손실)     : {K_vent_est:.3f}")
    print("\nInterpretation")
    print("  C 클수록 온도 변화 느림")
    print("  k_heat: 제어 단위당 난방 효과")
    print("  eta: 일사를 실내 열로 전환하는 비율")
    print("  UA: 외기로의 수동 열손실")
    print("  K_vent: 환기에 의한 추가 열손실")


# ── 7. 메인 실행 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 데이터 조회 (실제 DB 연결 시 주석 해제)
    # df_raw = connector("2025-01-01 00:00:00")

    # 로컬 테스트용 더미 실행 예시 (DB 미연결 시)
    # df_raw = pd.read_csv("sample_data.csv")

    # 전처리
    # df_proc = preprocess(df_raw)

    # Huber 회귀
    # coefs, df_reg = fit_huber(df_proc)

    # 서브샘플링 (5스텝마다)
    # df_sub = df_proc.iloc[::5].reset_index(drop=True)

    # C 최적화
    # C_est = estimate_C(df_sub, coefs)

    # 파라미터 도출
    # k_heat_est = coefs["a1"] * C_est
    # eta_est    = coefs["a2"] * C_est
    # UA_est     = -coefs["a3"] * C_est
    # K_vent_est = -coefs["a4"] * C_est

    # 결과 출력
    # print_param_summary(C_est, k_heat_est, eta_est, UA_est, K_vent_est, coefs)

    # 최종 시뮬레이션
    # Tin_sim_final = simulate_Tin([C_est, k_heat_est, UA_est, eta_est, K_vent_est], df_proc)

    print("모듈 로드 완료. connector() 호출 후 workflow를 실행하세요.")
