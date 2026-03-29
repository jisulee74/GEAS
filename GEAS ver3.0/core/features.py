#core/features.py: 하루치 데이터(df_day)로부터 파생변수 계산
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from core.timeutils import ensure_datetime

def _ensure_time(df: pd.DataFrame, col: str = "reg_date") -> pd.DataFrame:
    df = ensure_datetime(df, col=col)
    return df

def _step_minutes(df: pd.DataFrame, col: str = "reg_date") -> float:
    if df.empty:
        return 1.0
    t = pd.to_datetime(df[col])
    if len(t) < 2:
        return 1.0
    dt = t.diff().dt.total_seconds().dropna()
    if dt.empty:
        return 1.0
    return float(dt.median() / 60.0)

# 이슬점(Tdew) 계산
def _dewpoint_magnus(T_c: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
    T = np.asarray(T_c, dtype=float)
    RH = np.clip(np.asarray(RH_pct, dtype=float), 1e-6, 100.0)
    a, b = 17.62, 243.12
    gamma = np.log(RH / 100.0) + (a * T) / (b + T)
    Td = (b * gamma) / (a - gamma)
    return Td

# 증기압차(VPD) 계산
def _vpd_kpa(T_c: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
    T = np.asarray(T_c, dtype=float)
    RH = np.clip(np.asarray(RH_pct, dtype=float), 0.0, 100.0)
    es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
    ea = es * (RH / 100.0)
    return np.maximum(es - ea, 0.0)

# 생리 및 결로 안전 변수 계산
def compute_humidity_features(
    df_day: pd.DataFrame, # 하루치 데이터 기준
    now: Optional[pd.Timestamp] = None,
    temp_col: str = "in_temp",
    rh_col: str = "in_hum",
    risk_threshold: float = 0.8,     # °C, condensation safety margin: 결로 안전 변수(cond_risk_10m)
    ) -> Dict[str, Any]:

    res: Dict[str, Any] = {}

    if df_day.empty or temp_col not in df_day.columns:
        return {
            "VPD": np.nan,
            "Tdew": np.nan,
            "dTcond": np.nan,
            "rh90_min_1h": 0.0,
            "vpdlo_min_1h": 0.0,
            "cond_risk_10m": 0,
            "cond_dTcond_10m": np.nan,
        }

    df = _ensure_time(df_day)

    if rh_col not in df.columns and "in_rh" in df.columns:
        rh_col = "in_rh"

    T = df[temp_col].astype(float).to_numpy()
    RH = df.get(rh_col, pd.Series(np.full(len(df), np.nan), index=df.index)).astype(float).to_numpy()

    Td = _dewpoint_magnus(T, RH)
    vpd = _vpd_kpa(T, RH)
    dTcond = T - Td

    res["VPD"] = float(vpd[-1]) if len(vpd) else np.nan
    res["Tdew"] = float(Td[-1]) if len(Td) else np.nan
    res["dTcond"] = float(dTcond[-1]) if len(dTcond) else np.nan

    # ---- 1-hour window stats ----
    ## to calculate vpdlo_min_1h and rh90_min_1h
    t = pd.to_datetime(df["reg_date"])
    if now is None:
        now = t.iloc[-1]
    window_start = now - pd.Timedelta(minutes=60)
    mask = (t >= window_start) & (t <= now)
    sub = df.loc[mask].copy()

    step_min = _step_minutes(sub) if not sub.empty else _step_minutes(df)
    n_samples = len(sub)
    minutes_per_sample = step_min

    if n_samples > 0:
        RH_sub = sub.get(rh_col, pd.Series(np.nan, index=sub.index)).astype(float).to_numpy()
        T_sub = sub[temp_col].astype(float).to_numpy()
        Td_sub = _dewpoint_magnus(T_sub, RH_sub)
        vpd_sub = _vpd_kpa(T_sub, RH_sub)

        rh90_min = float(((RH_sub >= 90.0).sum()) * minutes_per_sample)
        vpdlo_min = float(((vpd_sub < 0.5).sum()) * minutes_per_sample)
    else:
        rh90_min = 0.0
        vpdlo_min = 0.0

    res["rh90_min_1h"] = rh90_min
    res["vpdlo_min_1h"] = vpdlo_min

    # ------------------------------------------------------------------------
    # <simple 10-min condensation risk (linear extrapolation)>
    # 최근 30분의 dTcond 추세를 선형으로 추정해서, 10분 뒤 dTcond가 위험 기준보다 낮아지는지 판단
    # ------------------------------------------------------------------------

    #1. 유의미한 추세 계산 조건: 하루치 전체 중 현재까지 유효한 데이터포인트가 최소 5개 이상
    if len(dTcond) >= 5: 
        t_s = (t - t.iloc[0]).dt.total_seconds().to_numpy() # 시간 → 숫자 변환

        #2. focus last 30 minutes
        t_last = now - pd.Timedelta(minutes=30) 
        mask30 = t >= t_last
        
        #3. 회귀용 데이터 추출 (입력 x: t_seg, 출력 y: dT_seg)
        t_seg = t_s[mask30] # t_seg: 최근 30분 구간의 시간 (초 단위)
        dT_seg = dTcond[mask30] # dT_seg: 그 시간 내의 이슬점여유(dTcond) 값

        #4. 시간 기준 재정렬
        if len(t_seg) >= 2:
            t0 = t_seg[0] # t0: 최근 30분 구간의 '시작 시각(초)'
            tt = t_seg - t0 # 시간을 0부터 시작하도록 평행이동 ⇒ 안정적 회귀 수행 (∵큰 절대시간 대신 작은 상대시간 사용)

            #5. 선형 회귀 (원리: 최근 30분의 t와 dT로 직선 추세 맞춤)
            X = np.column_stack([np.ones_like(tt), tt]) # 시간 1차원 배열 tt를 2차원 입력행렬로 바꾸는 역할 (절편, 기울기)
            beta, *_ = np.linalg.lstsq(X, dT_seg, rcond=None) # 위 X와 dT_seg로 '가장 잘 맞는 직선' 계수 beta=[b0, b1] 계산

            #6. 10 min ahead from now: 과거 30분 데이터 내 추세를 10분 미래로 extrapolation
            dt_pred = 10 * 60.0 # 예측할 미래 시점: 10분 → 초 단위
            t_pred = (now - t.iloc[0]).total_seconds() - t0 + dt_pred # 10분 뒤 시점을 tt 좌표계로 변환 (기준이 0인 좌표계)
            dT_future = beta[0] + beta[1] * t_pred # 구한 직선식에 t_pred 대입해서 10분 뒤 dTcond 예측값 계산

            res["cond_dTcond_10m"] = float(dT_future)
            res["cond_risk_10m"] = int(dT_future < risk_threshold)
        else:
            res["cond_dTcond_10m"] = np.nan
            res["cond_risk_10m"] = 0
    else:
        res["cond_dTcond_10m"] = np.nan
        res["cond_risk_10m"] = 0

    return res

# 광/에너지 균형 변수 계산
def compute_light_features(
    df_day: pd.DataFrame,
    sunrise: Optional[pd.Timestamp] = None,
    sunset: Optional[pd.Timestamp] = None,
    cs_sum_jcm2: Optional[np.ndarray] = None,
    target_jcm2: Optional[float] = None,
    light_eta: Optional[pd.Timestamp] = None,
    light_col: str = "out_light",
    light_sum_col: str = "out_light_sum",
    ) -> Dict[str, Any]:

    res: Dict[str, Any] = {
        "DLI": np.nan,
        "cs_sum_jcm2": np.nan,
        "SumRatio": np.nan,
        "LightETA": light_eta,
        "vent_loss_proxy": np.nan,
    }

    if df_day.empty:
        return res

    df = _ensure_time(df_day)

    # DLI based on out_light (assume μmol/m2/s if user wants real DLI units)
    if light_col in df.columns:
        t = pd.to_datetime(df["reg_date"])
        q = df[light_col].astype(float).to_numpy()
        if len(t) > 1:
            dt_sec = t.diff().dt.total_seconds().fillna(0).to_numpy()
            # DLI [mol m-2 d-1] = ∑ PAR[μmol m-2 s-1] * dt[s] / 1e6
            dli = np.sum(q * dt_sec) / 1e6
            res["DLI"] = float(dli)

    # measured cumulative J/cm2 or similar from out_light_sum
    measured_sum = np.nan
    if light_sum_col in df.columns:
        measured_sum = float(df[light_sum_col].astype(float).iloc[-1])
    else:
        measured_sum = np.nan

    # clear-sky cumulative
    cs_total = np.nan
    if cs_sum_jcm2 is not None and len(cs_sum_jcm2) > 0:
        cs_total = float(cs_sum_jcm2[-1])

    res["cs_sum_jcm2"] = cs_total

    if np.isfinite(measured_sum) and np.isfinite(cs_total) and cs_total > 0:
        res["SumRatio"] = float(measured_sum / cs_total)

    # LightETA: just pass through from caller
    res["LightETA"] = light_eta

    return res

#  물리 안전 및 기구 한계 변수 계산
def compute_thermal_features(
    df_day: pd.DataFrame,
    sunrise: Optional[pd.Timestamp],
    sunset: Optional[pd.Timestamp],
    T_day: float,
    T_night: float,
    ramp_limit_default: float = 15.0,     # %
    ach_min_day_base: float = 0.10,       # h^-1
    ach_min_night_base: float = 0.05,     # h^-1
    humidity_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    res: Dict[str, Any] = {
        "dT": np.nan,
        "ACH_min_day": ach_min_day_base,
        "ACH_min_night": ach_min_night_base,
        "RampLim": ramp_limit_default,
        "degnin_heat": 0.0,
        "degnin_cool": 0.0,
    }

    if df_day.empty or "in_temp" not in df_day.columns or "out_temp" not in df_day.columns:
        return res

    df = _ensure_time(df_day)
    t = pd.to_datetime(df["reg_date"])

    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)

    def _to_naive_kst(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
        if ts is None or pd.isna(ts):
            return None
        ts = pd.to_datetime(ts)
        if ts.tzinfo is not None:
            return ts.tz_convert("Asia/Seoul").tz_localize(None)
        return ts

    sunrise = _to_naive_kst(sunrise)
    sunset = _to_naive_kst(sunset)

    if sunrise is None or sunset is None:
        base = t.iloc[0]
        sunrise = base.normalize() + pd.Timedelta(hours=6)
        sunset = base.normalize() + pd.Timedelta(hours=18)

    Tin = df["in_temp"].astype(float).to_numpy()
    Tout = df["out_temp"].astype(float).to_numpy()
    dT = Tin - Tout
    res["dT"] = float(dT[-1])

    dt_sec = t.diff().dt.total_seconds().fillna(0).to_numpy()
    deg_heat = 0.0
    deg_cool = 0.0

    for i in range(1, len(t)):
        Ti = Tin[i]
        ti = t.iloc[i]
        dt_min = dt_sec[i] / 60.0
        if dt_min <= 0:
            continue

        if sunrise <= ti < sunset:
            Tset = T_day
        else:
            Tset = T_night

        if Ti < Tset:
            deg_heat += (Tset - Ti) * dt_min
        elif Ti > Tset:
            deg_cool += (Ti - Tset) * dt_min

    res["degnin_heat"] = float(deg_heat)
    res["degnin_cool"] = float(deg_cool)

    rh90 = 0.0
    vpdlo = 0.0
    if humidity_features is not None:
        rh90 = float(humidity_features.get("rh90_min_1h", 0.0) or 0.0)
        vpdlo = float(humidity_features.get("vpdlo_min_1h", 0.0) or 0.0)

    ach_day = ach_min_day_base
    ach_night = ach_min_night_base

    if rh90 >= 20.0 or vpdlo >= 20.0:
        ach_day *= 1.5
        ach_night *= 1.3
    elif rh90 >= 10.0 or vpdlo >= 10.0:
        ach_day *= 1.2
        ach_night *= 1.1

    res["ACH_min_day"] = float(ach_day)
    res["ACH_min_night"] = float(ach_night)

    return res

def build_features(
    df_day: pd.DataFrame,
    lat: float,
    lon: float,
    sunrise: Optional[pd.Timestamp],
    sunset: Optional[pd.Timestamp],
    cs_sum_jcm2: Optional[np.ndarray] = None,
    target_jcm2: Optional[float] = None,
    T_day: float = 22.0,
    T_night: float = 18.0,
    light_eta: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:

    return compute_all_features(
        df_day=df_day,
        sunrise=sunrise,
        sunset=sunset,
        cs_sum_jcm2=cs_sum_jcm2,
        target_jcm2=target_jcm2,
        light_eta=light_eta,
        T_day=T_day,
        T_night=T_night,
    )

# 생리/광/열 지표 각각 계산 후 하나로 합침
def compute_all_features(
    df_day: pd.DataFrame,
    sunrise: Optional[pd.Timestamp],
    sunset: Optional[pd.Timestamp],
    cs_sum_jcm2: Optional[np.ndarray] = None,
    target_jcm2: Optional[float] = None,
    light_eta: Optional[pd.Timestamp] = None,
    T_day: float = 22.0,
    T_night: float = 18.0,
    ) -> Dict[str, Any]:

    now = None
    if not df_day.empty and "reg_date" in df_day.columns:
        now = pd.to_datetime(df_day["reg_date"]).iloc[-1]

    hum = compute_humidity_features(df_day, now=now)
    light = compute_light_features(
        df_day,
        sunrise=sunrise,
        sunset=sunset,
        cs_sum_jcm2=cs_sum_jcm2,
        target_jcm2=target_jcm2,
        light_eta=light_eta,
    )
    therm = compute_thermal_features(
        df_day,
        sunrise=sunrise,
        sunset=sunset,
        T_day=T_day,
        T_night=T_night,
        humidity_features=hum,
    )

    out: Dict[str, Any] = {}
    out.update(hum)
    out.update(light)
    out.update(therm)
    return out