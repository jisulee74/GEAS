"""
3.6.5. 충분한 데이터 환경에서의 검증방법론

기술문서 3.6.5의 취지에 맞춰, 실제 운영 로그를 사용해
AI 자율제어기의 성능/안전성을 ex-post 방식으로 직접 평가한다.

핵심 특징
1. 외기/내기/제어 로그를 실측값으로 그대로 사용한다.
2. 반사실 폐루프 시뮬레이션이나 posterior 샘플링을 수행하지 않는다.
3. KPI는 실측 내부환경 궤적에서 직접 계산한다.
4. 보조 파생지표(ACH, 난방열량, 환기열손실)는 충분데이터 점추정 물리계수로 계산한다.

결과 파일 형식은 no_data_eval.py, limited_data_eval.py와 맞춘다.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

try:
    from sqlalchemy import create_engine
except Exception:  # pragma: no cover
    create_engine = None

_UPDATED_ROOT = Path(__file__).resolve().parents[1] / "source"
_INNER_ROOT = _UPDATED_ROOT / "inner_layer"
for _path in (_UPDATED_ROOT, _INNER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


_RESULTS_ROOT = Path(__file__).resolve().parent / "eval_results" / "sufficient_data"
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


def connector(
    start_date,
    end_date=None,
    *,
    table: str = "data_silla_enc",
    iot_data_idx: int = 97,
    uri: str = "mysql+pymysql://root:theimc#10!@211.195.9.227:3306/farmstom",
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


def _ensure_results_dir() -> Path:
    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = _RESULTS_ROOT / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
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


def sat_vp_kpa_arr(tc) -> np.ndarray:
    tc_arr = np.asarray(tc, dtype=float)
    return 0.61078 * np.exp((17.2694 * tc_arr) / (tc_arr + 237.3))


def dewpoint_c_arr(tc, rh_frac) -> np.ndarray:
    tc_arr = np.asarray(tc, dtype=float)
    rh_arr = np.asarray(rh_frac, dtype=float)
    es = sat_vp_kpa_arr(tc_arr)
    e = np.clip(rh_arr * es, 1e-6, es)
    ln_ratio = np.log(e / 0.61078)
    return (237.3 * ln_ratio) / (17.2694 - ln_ratio)


def vpd_kpa_arr(tc, rh_frac) -> np.ndarray:
    tc_arr = np.asarray(tc, dtype=float)
    rh_arr = np.asarray(rh_frac, dtype=float)
    es = sat_vp_kpa_arr(tc_arr)
    ea = np.clip(rh_arr * es, 0.0, es)
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
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    thr = float(np.quantile(x, q))
    return float(x[x >= thr].mean())


def normalize_01(x) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if abs(hi - lo) < 1e-9:
        return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)
    out = (arr - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


def _pick_series(df: pd.DataFrame, *cols: str, default=np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in cols:
        if col in df.columns:
            ser = numify(df[col])
            out = out.where(out.notna(), ser)
    if not pd.isna(default):
        out = out.fillna(default)
    return out


def map_columns_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    t_in = _pick_series(df, "in_temp", "in_temp2")
    rh_in = _pick_series(df, "in_hum", "in_hum2")
    co2_in = _pick_series(df, "in_co2", "in_co2_2")
    t_out = _pick_series(df, "out_temp")
    rh_out = _pick_series(df, "out_hum")
    light = _pick_series(df, "out_light", default=0.0).fillna(0.0)
    wind = _pick_series(df, "out_windsp", default=0.0).fillna(0.0)
    co2_out = _pick_series(df, "out_co2", default=420.0).fillna(420.0)

    vent_left = _pick_series(df, "cont_skyl_vol")
    vent_right = _pick_series(df, "cont_skyr_vol", "cont_skyl_vol")
    vent_raw = pd.concat([vent_left, vent_right], axis=1).mean(axis=1, skipna=True)
    vent = pd.Series(normalize_01(vent_raw.values), index=df.index)

    curtain_raw = _pick_series(df, "etc_blackout", "cont_cur_vol", default=0.0).fillna(0.0)
    curtain = pd.Series(normalize_01(curtain_raw.values), index=df.index)

    heat = (_pick_series(df, "cont_heater_run", default=0.0).fillna(0.0) > 0).astype(float)
    co2inj = (_pick_series(df, "cont_co2_run", default=0.0).fillna(0.0) > 0).astype(float)

    out = pd.DataFrame(
        {
            "reg_date": pd.to_datetime(df["reg_date"], errors="coerce"),
            "Tin": t_in,
            "RHin_raw": rh_in,
            "CO2": co2_in,
            "Tout": t_out,
            "RHout_raw": rh_out,
            "It": light.clip(lower=0.0),
            "wind": wind.clip(lower=0.0),
            "CO2out": co2_out,
            "u_heat": heat,
            "x_vent": vent,
            "curtain": curtain,
            "u_co2": co2inj,
        }
    )
    return out


def regularize_time(df_std: pd.DataFrame, step_mins: int = 5) -> pd.DataFrame:
    df = df_std.copy()
    df = df.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)
    df["reg_date"] = pd.to_datetime(df["reg_date"]).dt.tz_localize(None)
    epoch = pd.Timestamp("1970-01-01")
    step_sec = step_mins * 60
    df["time"] = pd.to_datetime(
        ((df["reg_date"] - epoch).dt.total_seconds() // step_sec) * step_sec,
        unit="s",
    )
    agg = (
        df.groupby("time")
        .agg(
            {
                "Tin": "mean",
                "RHin_raw": "mean",
                "CO2": "mean",
                "Tout": "mean",
                "RHout_raw": "mean",
                "It": "mean",
                "wind": "mean",
                "CO2out": "mean",
                "u_heat": "mean",
                "x_vent": "mean",
                "curtain": "mean",
                "u_co2": "mean",
            }
        )
        .reset_index()
        .rename(columns={"time": "reg_date"})
    )
    # 이진 구동 신호는 평균 후 > 0.5로 자르면 0/1 혼합 bin이 0으로 사라질 수 있다.
    # ex-post 평가에서는 해당 집계 bin 중 한 번이라도 구동된 경우 활성으로 본다.
    agg["u_heat"] = (agg["u_heat"] > 0.0).astype(float)
    agg["u_co2"] = (agg["u_co2"] > 0.0).astype(float)
    agg["x_vent"] = agg["x_vent"].clip(0.0, 1.0)
    agg["curtain"] = agg["curtain"].clip(0.0, 1.0)
    return agg


def convert_rh_to_fraction(rh_series: pd.Series) -> pd.Series:
    rh = pd.to_numeric(rh_series, errors="coerce")
    if rh.dropna().empty:
        return rh
    if float(rh.dropna().quantile(0.95)) > 1.5:
        rh = rh / 100.0
    return rh.clip(0.0, 1.0)


def prepare_timeseries(df_raw: pd.DataFrame, *, step_mins: int = 5) -> pd.DataFrame:
    df = map_columns_raw(df_raw)
    df = regularize_time(df, step_mins=step_mins)
    df["RHin"] = convert_rh_to_fraction(df["RHin_raw"])
    df["RHout"] = convert_rh_to_fraction(df["RHout_raw"])
    df = df.drop(columns=["RHin_raw", "RHout_raw"])
    df = df.dropna(subset=["Tin", "RHin", "CO2", "Tout", "RHout"]).reset_index(drop=True)

    df["hour"] = df["reg_date"].dt.hour + df["reg_date"].dt.minute / 60.0
    df["k"] = np.arange(1, len(df) + 1)
    df["e_in"] = df["RHin"] * sat_vp_kpa_arr(df["Tin"].values)
    df["VPD"] = vpd_kpa_arr(df["Tin"].values, df["RHin"].values)
    dew = dewpoint_c_arr(df["Tin"].values, df["RHin"].values)
    df["dTcond"] = df["Tin"].values - dew
    return df


def prepare_identification_frame(df_ts: pd.DataFrame) -> pd.DataFrame:
    df = df_ts.copy()
    df["dt_hr"] = df["reg_date"].diff().dt.total_seconds() / 3600.0
    df["dTin_dt"] = df["Tin"].diff() / df["dt_hr"]
    df["V_t"] = df["x_vent"]
    cols = ["Tin", "Tout", "It", "u_heat", "V_t", "dt_hr", "dTin_dt"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).copy()
    return df.reset_index(drop=True)


def simulate_tin_from_ratios(c_val: float, data: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    k_heat = max(float(beta[0] * c_val), 0.0)
    eta = max(float(beta[1] * c_val), 0.0)
    ua = max(float(-beta[2] * c_val), 0.0)
    k_vent = max(float(-beta[3] * c_val), 0.0)
    n = len(data)
    tin_sim = np.full(n, np.nan)
    tin_sim[0] = float(data["Tin"].iloc[0])
    for i in range(n - 1):
        tin = tin_sim[i]
        tout = float(data["Tout"].iloc[i])
        it = float(data["It"].iloc[i])
        heat = float(data["u_heat"].iloc[i])
        vent = float(data["V_t"].iloc[i])
        dt_hr = float(data["dt_hr"].iloc[i])
        d_tin_dt = (1.0 / c_val) * (
            k_heat * heat
            + eta * it
            - ua * (tin - tout)
            - k_vent * vent * (tin - tout)
        )
        tin_sim[i + 1] = tin + d_tin_dt * dt_hr
    return tin_sim


def estimate_point_theta(df_ts: pd.DataFrame) -> Dict[str, float]:
    df_id = prepare_identification_frame(df_ts)
    if len(df_id) < 20:
        return {
            "C": 1.0e6,
            "UA": 2500.0,
            "k_heat": 15000.0,
            "eta": 250.0,
            "k_vent": 2000.0,
            "rho_cp": 1200.0,
        }

    x = np.column_stack(
        [
            df_id["u_heat"].values,
            df_id["It"].values,
            (df_id["Tin"] - df_id["Tout"]).values,
            (df_id["V_t"] * (df_id["Tin"] - df_id["Tout"])).values,
        ]
    )
    y = df_id["dTin_dt"].values
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)

    def objective(log_c: float) -> float:
        c_val = float(np.exp(log_c))
        tin_sim = simulate_tin_from_ratios(c_val, df_id, beta)
        mse = np.nanmean((tin_sim - df_id["Tin"].values) ** 2)
        return float(mse if np.isfinite(mse) else 1.0e12)

    log_grid = np.linspace(math.log(1.0e3), math.log(1.0e8), 121)
    losses = np.array([objective(v) for v in log_grid], dtype=float)
    c_est = float(np.exp(log_grid[int(np.nanargmin(losses))]))
    k_heat = max(float(beta[0] * c_est), 0.0)
    eta = max(float(beta[1] * c_est), 0.0)
    ua = max(float(-beta[2] * c_est), 0.0)
    k_vent = max(float(-beta[3] * c_est), 0.0)
    return {
        "C": c_est,
        "UA": ua,
        "k_heat": k_heat,
        "eta": eta,
        "k_vent": k_vent,
        "rho_cp": 1200.0,
    }


def add_reporting_fields(
    df_ts: pd.DataFrame,
    theta: Dict[str, float],
    *,
    V: float = 300.0,
    infiltration_ach: float = 0.05,
) -> pd.DataFrame:
    df = df_ts.copy()
    ach_from_vent = 3600.0 * float(theta["k_vent"]) * df["x_vent"].values / max(float(theta["rho_cp"]) * V, 1e-6)
    df["ACH"] = np.clip(infiltration_ach + ach_from_vent, 0.0, 15.0)
    df["Q_heat"] = float(theta["k_heat"]) * df["u_heat"].values
    df["Q_ventloss"] = (
        float(theta["rho_cp"]) * V * (df["ACH"].values / 3600.0) * np.maximum(0.0, df["Tin"].values - df["Tout"].values)
    )
    cols = [
        "k",
        "hour",
        "Tout",
        "RHout",
        "It",
        "wind",
        "CO2out",
        "Tin",
        "RHin",
        "VPD",
        "dTcond",
        "CO2",
        "u_heat",
        "x_vent",
        "curtain",
        "u_co2",
        "ACH",
        "Q_heat",
        "Q_ventloss",
    ]
    return df[cols].copy()


def calc_metrics(
    df: pd.DataFrame,
    *,
    T_min=12.0,
    T_max=28.0,
    RH_max=0.90,
    VPD_min=0.30,
    dTcond_min=0.8,
    alpha=0.25,
) -> Dict[str, float]:
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

    return {
        "temp_viol_rate": float(temp_viol.mean()),
        "temp_viol_maxrun": int(max_run(temp_viol.values)),
        "cond_viol_rate": float(cond_viol.mean()),
        "cond_viol_maxrun": int(max_run(cond_viol.values)),
        "rh_viol_rate": float(rh_viol.mean()),
        "vpd_viol_rate": float(vpd_viol.mean()),
        "hv_ineff_rate": hv_ineff,
        "tv_vent": float(df["x_vent"].diff().abs().fillna(0.0).sum()),
        "sw_heat": int((df["u_heat"].diff().abs().fillna(0.0) > 0).sum()),
    }


def evaluate_full_period(
    episode_df: pd.DataFrame,
    *,
    constraints: Dict[str, float],
) -> Dict[str, object]:
    df = episode_df.copy().reset_index(drop=True)
    metrics = calc_metrics(
        df,
        T_min=float(constraints["T_min"]),
        T_max=float(constraints["T_max"]),
        RH_max=float(constraints["RH_max"]),
        VPD_min=float(constraints["VPD_min"]),
        dTcond_min=float(constraints["dTcond_min"]),
        alpha=float(constraints["alpha"]),
    ).copy()
    if "reg_date" in df.columns and not df.empty:
        metrics["period_start"] = str(pd.Timestamp(df["reg_date"].iloc[0]))
        metrics["period_end"] = str(pd.Timestamp(df["reg_date"].iloc[-1]))
    metrics["n_steps"] = int(len(df))
    metrics_df = pd.DataFrame([metrics])
    return {
        "metrics": metrics_df,
        "episode": df,
    }


def evaluate_observed_windows(
    episode_df: pd.DataFrame,
    reg_dates: pd.Series,
    *,
    episode_days: int = 1,
    episode_stride_days: int = 1,
    min_steps_per_episode: int = 72,
    constraints: Dict[str, float],
) -> Dict[str, object]:
    df = episode_df.copy()
    reg_dates = pd.to_datetime(reg_dates).reset_index(drop=True)
    dt_days = pd.to_timedelta(episode_days, unit="D")
    stride_days = pd.to_timedelta(episode_stride_days, unit="D")
    windows = []
    first_time = pd.Timestamp(reg_dates.iloc[0]).floor("D")
    last_time = pd.Timestamp(reg_dates.iloc[-1])
    win_start = first_time
    while win_start <= last_time:
        win_end = win_start + dt_days
        mask = (reg_dates >= win_start) & (reg_dates < win_end)
        windows.append((win_start, win_end, mask))
        win_start = win_start + stride_days

    rows = []
    representative_idx = None
    representative_score = -np.inf
    for idx, (win_start, win_end, mask) in enumerate(windows):
        ep = df.loc[mask].copy()
        if len(ep) < int(min_steps_per_episode):
            continue
        metrics = calc_metrics(
            ep,
            T_min=float(constraints["T_min"]),
            T_max=float(constraints["T_max"]),
            RH_max=float(constraints["RH_max"]),
            VPD_min=float(constraints["VPD_min"]),
            dTcond_min=float(constraints["dTcond_min"]),
            alpha=float(constraints["alpha"]),
        )
        metrics["episode_id"] = idx + 1
        metrics["episode_start"] = str(win_start)
        metrics["episode_end"] = str(win_end)
        metrics["n_steps"] = int(len(ep))
        rows.append(metrics)

        score = float(metrics["temp_viol_rate"]) + 0.1 * float(metrics["cond_viol_rate"])
        if score > representative_score:
            representative_score = score
            representative_idx = idx

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise RuntimeError("No valid observed episodes were formed. Check date range or min_steps_per_episode.")

    q90 = {c: float(np.quantile(metrics_df[c].values, 0.9)) for c in _DOC_METRIC_COLUMNS}
    c90 = {c: cvar(metrics_df[c].values, 0.9) for c in _DOC_METRIC_COLUMNS}

    rep_row = metrics_df.loc[metrics_df["episode_id"] == representative_idx + 1].iloc[0]
    rep_start = pd.Timestamp(rep_row["episode_start"])
    rep_end = pd.Timestamp(rep_row["episode_end"])
    rep_mask = (reg_dates >= rep_start) & (reg_dates < rep_end)
    rep_episode = df.loc[rep_mask].copy().reset_index(drop=True)

    return {
        "window_metrics": metrics_df,
        "q90": q90,
        "cvar90": c90,
        "representative_window": rep_episode,
        "representative_episode_id": int(rep_row["episode_id"]),
    }


def save_eval_outputs(
    output_dir: Path,
    *,
    main_result: Dict[str, object],
    aux_result: Dict[str, object],
    config: Dict[str, object],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    window_metrics_path = output_dir / "window_metrics.csv"
    q90_path = output_dir / "q90.json"
    cvar90_path = output_dir / "cvar90.json"
    episode_path = output_dir / "episode.csv"
    representative_window_path = output_dir / "representative_window.csv"
    config_path = output_dir / "run_config.json"

    metrics_to_save = main_result["metrics"].copy()
    float_cols = metrics_to_save.select_dtypes(include=["float", "float16", "float32", "float64"]).columns
    metrics_to_save[float_cols] = metrics_to_save[float_cols].round(3)

    window_metrics_to_save = aux_result["window_metrics"].copy()
    float_cols = window_metrics_to_save.select_dtypes(include=["float", "float16", "float32", "float64"]).columns
    window_metrics_to_save[float_cols] = window_metrics_to_save[float_cols].round(3)

    episode_to_save = main_result["episode"].copy()
    float_cols = episode_to_save.select_dtypes(include=["float", "float16", "float32", "float64"]).columns
    episode_to_save[float_cols] = episode_to_save[float_cols].round(3)

    representative_window_to_save = aux_result["representative_window"].copy()
    float_cols = representative_window_to_save.select_dtypes(include=["float", "float16", "float32", "float64"]).columns
    representative_window_to_save[float_cols] = representative_window_to_save[float_cols].round(3)

    metrics_to_save.to_csv(metrics_path, index=False)
    window_metrics_to_save.to_csv(window_metrics_path, index=False)
    episode_to_save.to_csv(episode_path, index=False)
    representative_window_to_save.to_csv(representative_window_path, index=False)
    q90_path.write_text(
        json.dumps(_round_floats({k: aux_result["q90"][k] for k in _DOC_METRIC_COLUMNS}, 3), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cvar90_path.write_text(
        json.dumps(_round_floats({k: aux_result["cvar90"][k] for k in _DOC_METRIC_COLUMNS}, 3), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(_round_floats(_to_builtin(config), 3), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "metrics_csv": str(metrics_path),
        "window_metrics_csv": str(window_metrics_path),
        "q90_json": str(q90_path),
        "cvar90_json": str(cvar90_path),
        "episode_csv": str(episode_path),
        "representative_window_csv": str(representative_window_path),
        "run_config_json": str(config_path),
    }


def run_sufficient_data_eval(
    df_raw: pd.DataFrame,
    *,
    step_mins: int = 5,
    episode_days: int = 7,
    episode_stride_days: int = 3,
    min_steps_per_episode: int | None = None,
    V: float = 300.0,
    infiltration_ach: float = 0.05,
    constraints: Dict[str, float] | None = None,
    output_dir: Path | None = None,
) -> Dict[str, object]:
    if constraints is None:
        constraints = {
            "T_min": 12.0,
            "T_max": 28.0,
            "RH_max": 0.90,
            "VPD_min": 0.30,
            "dTcond_min": 0.8,
            "alpha": 0.25,
        }

    if output_dir is None:
        output_dir = _ensure_results_dir()

    if min_steps_per_episode is None:
        # 5분(또는 전달된 step_mins) 집계 기준으로, 한 에피소드의 80% 이상이 채워졌을 때만 사용한다.
        expected_steps = int(episode_days * 24 * (60 / step_mins))
        min_steps_per_episode = max(1, int(expected_steps * 0.8))

    df_ts = prepare_timeseries(df_raw, step_mins=step_mins)
    theta_hat = estimate_point_theta(df_ts)
    episode_df = add_reporting_fields(df_ts, theta_hat, V=V, infiltration_ach=infiltration_ach)

    main_result = evaluate_full_period(
        episode_df,
        constraints=constraints,
    )

    aux_result = evaluate_observed_windows(
        episode_df,
        df_ts["reg_date"],
        episode_days=episode_days,
        episode_stride_days=episode_stride_days,
        min_steps_per_episode=min_steps_per_episode,
        constraints=constraints,
    )

    config = {
        "mode": "sufficient_data_eval",
        "doc_section": "3.6.5",
        "evaluation_structure": "main_ex_post_full_period + auxiliary_sliding_window_tail_risk",
        "input_rows": int(len(df_raw)),
        "timeseries_rows": int(len(df_ts)),
        "step_mins": int(step_mins),
        "episode_days": int(episode_days),
        "episode_stride_days": int(episode_stride_days),
        "min_steps_per_episode": int(min_steps_per_episode),
        "volume_m3": float(V),
        "infiltration_ach": float(infiltration_ach),
        "constraints": dict(constraints),
        "point_theta": dict(theta_hat),
        "representative_episode_id": int(aux_result["representative_episode_id"]),
    }
    saved = save_eval_outputs(output_dir, main_result=main_result, aux_result=aux_result, config=config)
    return {
        "saved": saved,
        "config": config,
        "main_result": main_result,
        "aux_result": aux_result,
        "episode_df": episode_df,
    }


if __name__ == "__main__":
    start_date = "2025-11-29"
    end_date = "2026-01-22"
    data = connector(start_date, end_date)
    res = run_sufficient_data_eval(data)
    print(f"saved_outputs: {res['saved']['output_dir']}")
