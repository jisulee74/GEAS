"""
3.6.3. AI 자율제어기 견고성 검증 방법론 - 무데이터 환경
합성 외기 시나리오 + Monte Carlo 파라미터 샘플링으로
AI 제어 정책의 견고성을 검증한다.
"""

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm


# ── 1. 기상 관련 유틸리티 ───────────────────────────────────────────────────

def sat_vp_kpa(Tc):
    """포화 수증기압 (kPa)."""
    return 0.61078 * np.exp((17.2694 * Tc) / (Tc + 237.3))

def dewpoint_c(Tc, RH):
    """이슬점 온도 (°C)."""
    es = sat_vp_kpa(Tc)
    e  = np.clip(RH * es, 1e-6, es)
    ln_ratio = np.log(e / 0.61078)
    return (237.3 * ln_ratio) / (17.2694 - ln_ratio)

def vpd_kpa(Tc, RH):
    """수증기압 부족량 VPD (kPa)."""
    es = sat_vp_kpa(Tc)
    ea = np.clip(RH * es, 0, es)
    return np.maximum(0, es - ea)

def hinge(z):
    """ReLU / hinge 함수."""
    return np.maximum(0, z)


# ── 2. 합성 외기 조건 생성 ──────────────────────────────────────────────────

def gen_boundary_simple(n: int = 7*144, dt_min: int = 10, seed: int = 1) -> pd.DataFrame:
    """합성 외기 시계열을 생성한다 (실측 데이터 없는 경우 사용)."""
    rng = np.random.default_rng(seed)
    t_sec  = np.arange(n) * dt_min * 60
    daysec = 24 * 3600
    phase  = (t_sec % daysec) / daysec
    hour   = (t_sec / 3600) % 24

    # 외기온 (사인파 + AR(0.9) 노이즈)
    ar_noise = np.zeros(n)
    for i in range(1, n):
        ar_noise[i] = 0.9 * ar_noise[i-1] + rng.normal(0, 0.6)
    Tout = 5 + 6 * np.sin(2*np.pi*(phase - 0.25)) + ar_noise

    # 일사
    sun = np.maximum(0, np.sin(np.pi * (phase - 0.25) / 0.5))
    sun[(phase < 0.25) | (phase > 0.75)] = 0
    It = 650 * sun * (0.3 + 0.7 * rng.random(n))

    RHout  = np.clip(0.75 - 0.02*(Tout - 5) + rng.normal(0, 0.05, n), 0.3, 1.0)
    wind   = np.maximum(0.2, np.abs(2 + rng.normal(0, 0.6, n)))
    CO2out = 420 + 5*np.cos(2*np.pi*(phase - 0.1)) + rng.normal(0, 5, n)

    return pd.DataFrame({
        "k": np.arange(1, n+1),
        "t_sec": t_sec,
        "hour": hour,
        "Tout": Tout, "RHout": RHout, "It": It, "wind": wind, "CO2out": CO2out,
    })


# ── 3. Monte Carlo 파라미터 샘플링 ──────────────────────────────────────────

def sample_theta(rng=None) -> dict:
    """물리 파라미터를 균등분포로 무작위 샘플링한다."""
    if rng is None:
        rng = np.random.default_rng()
    return {
        "UA":      float(rng.uniform(2000, 7000)),
        "C":       float(rng.uniform(5e6,  4e7)),
        "eta":     float(rng.uniform(0.2,  0.8)),
        "k_heat":  float(rng.uniform(15000, 50000)),
        "a0":      float(rng.uniform(0.05, 0.5)),
        "a1":      float(rng.uniform(2.0,  10.0)),
        "a2":      float(rng.uniform(0.3,  2.0)),
        "k_evap":  float(rng.uniform(1e-6, 5e-6)),
        "k_photo": float(rng.uniform(1e-5, 5e-5)),
        "rho_cp":  1.2 * 1005,
    }


# ── 4. 환기 ACH 모델 ─────────────────────────────────────────────────────────

def ach_model(x_vent, wind, a0, a1, a2, ACH_max=15):
    """환기 개도율과 풍속으로 시간당 환기 횟수(ACH)를 계산한다."""
    ach = a0 + a1*x_vent + a2*wind*x_vent
    return float(np.clip(ach, 0, ACH_max))


# ── 5. 단계별 동역학 시뮬레이션 ─────────────────────────────────────────────

def step_dynamics(state: dict, u: dict, a: dict, theta: dict,
                  dt_min: int = 10, A: float = 200, V: float = 300,
                  Imax: float = 650, T0: float = 5) -> dict:
    """한 타임스텝의 온실 내부 상태를 계산한다."""
    dt   = dt_min * 60
    Tin  = state["Tin"]
    e_in = state["e_in"]
    CO2  = state["CO2"]

    ACH    = ach_model(a["x_vent"], u["wind"], theta["a0"], theta["a1"], theta["a2"])
    It_eff = u["It"] * (1 - a["curtain"])

    # 온도
    Q_trans = theta["UA"]     * (u["Tout"] - Tin)
    Q_solar = theta["eta"] * A * It_eff
    Q_heat  = theta["k_heat"] * a["u_heat"]
    Q_vent  = theta["rho_cp"] * V * (ACH/3600) * (u["Tout"] - Tin)
    Tin_next = Tin + (dt / theta["C"]) * (Q_trans + Q_solar + Q_heat + Q_vent)

    # 절대습도 (수증기압 형태)
    e_out  = u["RHout"] * sat_vp_kpa(u["Tout"])
    evap   = theta["k_evap"] * (It_eff / Imax) * max(0, Tin - T0)
    e_next = max(0.05, e_in + dt * ((ACH/3600)*(e_out - e_in) + evap))

    RHin = min(1, max(0, e_next / sat_vp_kpa(Tin_next)))
    VPD  = float(vpd_kpa(Tin_next, RHin))

    # CO2
    gT     = min(1, max(0, (Tin_next - 5) / 20))
    gV     = min(1, max(0, VPD / 1.2))
    uptake = theta["k_photo"] * It_eff * gT * gV * 1e6
    inj    = 50 * a["u_co2"]
    CO2_next = max(300, CO2 + dt*((ACH/3600)*(u["CO2out"] - CO2) + inj - uptake))

    Q_ventloss = max(0, theta["rho_cp"]*V*(ACH/3600)*max(0, Tin_next - u["Tout"]))

    return {
        "Tin": Tin_next, "e_in": e_next, "CO2": CO2_next,
        "ACH": ACH, "It_eff": It_eff,
        "Q_heat": Q_heat, "Q_ventloss": Q_ventloss,
    }


# ── 6. 행동 투영 (제약 적용) ─────────────────────────────────────────────────

def project_action(a: dict, prev_a: dict, ramp_max: float = 0.15) -> dict:
    """행동 벡터에 on/off 이산화, 클리핑, ramp 제한을 적용한다."""
    a = dict(a)
    a["u_heat"]  = 1 if a["u_heat"]  > 0.5 else 0
    a["u_co2"]   = 1 if a["u_co2"]   > 0.5 else 0
    a["x_vent"]  = float(np.clip(a["x_vent"],  0, 1))
    a["curtain"] = float(np.clip(a["curtain"], 0, 1))

    dx = a["x_vent"] - prev_a["x_vent"]
    dx = float(np.clip(dx, -ramp_max, ramp_max))
    a["x_vent"] = prev_a["x_vent"] + dx
    return a


# ── 7. 실현 가능성 체크 ──────────────────────────────────────────────────────

def check_feasible(pred: dict, u: dict, a: dict, theta: dict,
                   T_min=12, T_max=28, dTcond_min=0.8, alpha=0.25) -> bool:
    """온도 범위, 결로 마진, 난방-환기 비효율 조건을 모두 만족하는지 확인한다."""
    RHin   = min(1, max(0, pred["e_in"] / sat_vp_kpa(pred["Tin"])))
    Td     = float(dewpoint_c(pred["Tin"], RHin))
    dTcond = pred["Tin"] - Td

    c1 = T_min <= pred["Tin"] <= T_max
    c2 = dTcond >= dTcond_min
    c3 = True
    if a["u_heat"] > 0.5:
        c3 = pred["Q_ventloss"] <= alpha * max(1e-6, pred["Q_heat"])
    return c1 and c2 and c3


# ── 8. 행동 그리드 생성 ──────────────────────────────────────────────────────

def make_action_grid() -> list[dict]:
    """가능한 모든 행동 조합을 리스트로 반환한다."""
    return [
        {"u_heat": uh, "x_vent": xv, "curtain": cu, "u_co2": uco}
        for uh in [0, 1]
        for xv in [0, 0.1, 0.3, 0.6]
        for cu in [0, 0.6, 0.9]
        for uco in [0, 1]
    ]


# ── 9. 최적 정책 (그리드 탐색) ──────────────────────────────────────────────

def policy_optimal(state, u, prev_a, theta,
                   weights=None, Tref=18, VPD_min=0.30,
                   T_min=12, T_max=28, dTcond_min=0.8, alpha=0.25,
                   dt_min=10, A=200, V=300, ramp_max=0.15) -> dict:
    """
    행동 그리드 전체를 탐색하여 비용 함수를 최소화하는 행동을 선택한다.
    """
    if weights is None:
        weights = {"wT": 1, "wVPD": 1, "wE": 1e-8, "wDx": 0.2, "wSlack": 50}

    G = make_action_grid()
    best_J = np.inf
    best_a = prev_a
    feasible_found = False

    for g in G:
        a    = project_action(g, prev_a, ramp_max)
        pred = step_dynamics(state, u, a, theta, dt_min, A, V)

        if not check_feasible(pred, u, a, theta, T_min, T_max, dTcond_min, alpha):
            continue
        feasible_found = True

        RHin   = min(1, max(0, pred["e_in"] / sat_vp_kpa(pred["Tin"])))
        VPD    = float(vpd_kpa(pred["Tin"], RHin))
        Td     = float(dewpoint_c(pred["Tin"], RHin))
        dTcond = pred["Tin"] - Td

        J  = weights["wT"]   * (pred["Tin"] - Tref)**2
        J += weights["wVPD"] * float(hinge(VPD_min - VPD))
        J += weights["wE"]   * (pred["Q_heat"] + pred["Q_ventloss"])
        J += weights["wDx"]  * abs(a["x_vent"] - prev_a["x_vent"])
        slackT = float(hinge(T_min - pred["Tin"])) + float(hinge(pred["Tin"] - T_max))
        slackC = float(hinge(dTcond_min - dTcond))
        J += weights["wSlack"] * (slackT + slackC)

        if J < best_J:
            best_J = J; best_a = a

    if not feasible_found:
        best_a = project_action(
            {"u_heat": 1 if state["Tin"] < (Tref-1) else 0,
             "x_vent": 0, "curtain": 0.6, "u_co2": 0},
            prev_a, ramp_max
        )

    return best_a


# ── 10. 에피소드 시뮬레이션 ──────────────────────────────────────────────────

def simulate_episode(boundary: pd.DataFrame, theta: dict, weights: dict,
                     dt_min=10, A=200, V=300,
                     Tin0=15, RHin0=0.70, CO20=900,
                     show_progress=False) -> pd.DataFrame:
    """외기 시나리오와 물리 파라미터 θ에 대해 에피소드를 시뮬레이션한다."""
    n    = len(boundary)
    Tin  = np.zeros(n);  Tin[0]  = Tin0
    e_in = np.zeros(n);  e_in[0] = RHin0 * float(sat_vp_kpa(Tin0))
    CO2  = np.zeros(n);  CO2[0]  = CO20

    u_heat = np.zeros(n); x_vent = np.zeros(n)
    curtain = np.zeros(n); u_co2 = np.zeros(n)
    ACH = np.zeros(n); Q_heat = np.zeros(n); Q_ventloss = np.zeros(n)

    prev_a = {"u_heat": 0, "x_vent": 0, "curtain": 0, "u_co2": 0}

    it = range(1, n)
    if show_progress:
        it = tqdm(it, desc="simulate_episode")

    for t in it:
        state = {"Tin": Tin[t-1], "e_in": e_in[t-1], "CO2": CO2[t-1]}
        u     = boundary.iloc[t-1].to_dict()
        a     = policy_optimal(state, u, prev_a, theta, weights=weights,
                               Tref=18, dt_min=dt_min, A=A, V=V)
        pred  = step_dynamics(state, u, a, theta, dt_min, A, V)

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
        "k": boundary["k"].values, "hour": boundary["hour"].values,
        "Tout": boundary["Tout"].values, "RHout": boundary["RHout"].values,
        "It": boundary["It"].values, "wind": boundary["wind"].values,
        "CO2out": boundary["CO2out"].values,
        "Tin": Tin, "RHin": RHin, "VPD": VPD, "dTcond": dTcond, "CO2": CO2,
        "u_heat": u_heat, "x_vent": x_vent, "curtain": curtain, "u_co2": u_co2,
        "ACH": ACH, "Q_heat": Q_heat, "Q_ventloss": Q_ventloss,
    })


# ── 11. KPI 계산 ─────────────────────────────────────────────────────────────

def max_run(b: np.ndarray) -> int:
    """연속 True의 최대 길이를 반환한다."""
    if not b.any():
        return 0
    lengths = []
    count = 0
    for v in b:
        if v:
            count += 1
            lengths.append(count)
        else:
            count = 0
    return max(lengths)

def calc_metrics(df: pd.DataFrame,
                 T_min=12, T_max=28, RH_max=0.90,
                 VPD_min=0.30, dTcond_min=0.8, alpha=0.25) -> pd.DataFrame:
    """한 에피소드의 KPI를 계산한다."""
    temp_viol = (df["Tin"] < T_min) | (df["Tin"] > T_max)
    cond_viol = df["dTcond"] < dTcond_min
    rh_viol   = df["RHin"] > RH_max
    vpd_viol  = df["VPD"] < VPD_min

    idx_heat = df["u_heat"] > 0.5
    if idx_heat.any():
        hv_ineff = float((df.loc[idx_heat, "Q_ventloss"] >
                          alpha * df.loc[idx_heat, "Q_heat"].clip(lower=1e-6)).mean())
    else:
        hv_ineff = 0.0

    return pd.DataFrame([{
        "temp_viol_rate":    float(temp_viol.mean()),
        "temp_viol_maxrun":  max_run(temp_viol.values),
        "cond_viol_rate":    float(cond_viol.mean()),
        "cond_viol_maxrun":  max_run(cond_viol.values),
        "rh_viol_rate":      float(rh_viol.mean()),
        "vpd_viol_rate":     float(vpd_viol.mean()),
        "hv_ineff_rate":     hv_ineff,
        "tv_vent":           float(df["x_vent"].diff().abs().sum()),
        "sw_heat":           int((df["u_heat"].diff().abs() > 0).sum()),
    }])


def cvar(x: np.ndarray, q: float = 0.9) -> float:
    """Conditional Value-at-Risk (CVaR) 계산."""
    x = x[np.isfinite(x)]
    thr = float(np.quantile(x, q))
    return float(x[x >= thr].mean())


# ── 12. 견고성 평가 (Monte Carlo) ────────────────────────────────────────────

def robust_eval(N=50, days=7, dt_min=10, seed=1, weights=None,
                show_progress=True) -> dict:
    """
    N번 Monte Carlo 샘플링으로 AI 정책의 견고성 지표(q90, CVaR90)를 계산한다.
    """
    if weights is None:
        weights = {"wT": 1, "wVPD": 1, "wE": 1e-8, "wDx": 0.2, "wSlack": 50}

    n = days * (24 * 60 // dt_min)
    boundary = gen_boundary_simple(n=n, dt_min=dt_min, seed=seed)
    rng = np.random.default_rng(seed)

    Ms = []
    it = range(N) if not show_progress else tqdm(range(N), desc="robust_eval")
    for _ in it:
        theta = sample_theta(rng)
        ep = simulate_episode(boundary, theta, weights, dt_min=dt_min,
                              show_progress=False)
        Ms.append(calc_metrics(ep))

    M    = pd.concat(Ms, ignore_index=True)
    cols = M.columns.tolist()
    q90  = {c: float(np.quantile(M[c], 0.9)) for c in cols}
    c90  = {c: cvar(M[c].values, 0.9)        for c in cols}

    return {"metrics": M, "q90": q90, "cvar90": c90}


# ── 13. 가중치 튜닝 (그리드 탐색) ───────────────────────────────────────────

def tune_weights_grid(grid: pd.DataFrame, N=40, seed=1,
                      show_progress=True) -> dict:
    """
    가중치 조합 그리드를 탐색하여 CVaR90 기반 목적함수를 최소화하는 가중치를 찾는다.
    """
    best = None
    best_obj = np.inf

    it = grid.iterrows() if not show_progress else tqdm(grid.iterrows(), total=len(grid))
    for _, row in it:
        w = {k: float(row[k]) for k in ["wT", "wVPD", "wE", "wDx", "wSlack"]}
        res = robust_eval(N=N, seed=seed, weights=w, show_progress=False)
        obj = (res["cvar90"]["temp_viol_rate"]
               + res["cvar90"]["cond_viol_rate"]
               + 0.1 * res["cvar90"]["hv_ineff_rate"])
        if obj < best_obj:
            best_obj = obj; best = w

    return {"best_weights": best, "best_obj": best_obj}


# ── 14. 메인 실행 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 견고성 평가
    res = robust_eval(N=60, days=7, dt_min=10, seed=42, show_progress=True)
    print("q90:\n",   pd.Series(res["q90"]))
    print("cvar90:\n", pd.Series(res["cvar90"]))
    print(res["metrics"].head())

    # 단일 에피소드 시뮬레이션
    boundary = gen_boundary_simple(n=7*144, dt_min=10, seed=42)
    theta    = sample_theta(np.random.default_rng(0))
    weights  = {"wT": 1, "wVPD": 1, "wE": 1e-8, "wDx": 0.2, "wSlack": 50}
    ep = simulate_episode(boundary, theta, weights, show_progress=True)

    import matplotlib.pyplot as plt
    plt.plot(ep["Tin"]); plt.ylabel("Tin (°C)"); plt.title("Simulated Indoor Temperature")
    plt.tight_layout(); plt.show()

    # 가중치 튜닝
    import itertools
    grid_vals = list(itertools.product([0.5,1,2],[0.5,1],[1e-9,1e-8],[0.1,0.2,0.4],[30,50,80]))
    grid = pd.DataFrame(grid_vals, columns=["wT","wVPD","wE","wDx","wSlack"])
    tuned = tune_weights_grid(grid, N=30, seed=42, show_progress=True)
    print("best_weights:", tuned["best_weights"])
    print("best_obj:",     tuned["best_obj"])
