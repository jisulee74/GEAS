from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import types
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib import font_manager as fm
try:
    import koreanize_matplotlib  # noqa: F401
except Exception:
    koreanize_matplotlib = None


ROOT = Path("/home/ljs/jslee")
UPDATED_ROOT = ROOT / "updated" / "GEAS3.0"
INNER_ROOT = UPDATED_ROOT / "inner_layer"
EVAL_ROOT = ROOT / "updated" / "evaluation"
for _path in (UPDATED_ROOT, INNER_ROOT, EVAL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from policy.controller import (  # noqa: E402
    Controller,
    PolicyState,
    sat_vp_kpa_no_data,
    dewpoint_c_no_data,
    vpd_kpa_no_data,
    step_dynamics_no_data,
)


DB_URI = "mysql+pymysql://root:theimc#10!@211.195.9.227:3306/farmstom"
TABLE_NAME = "data_silla_enc"
IOT_DATA_IDX = 97

RESULTS_ROOT = ROOT / "geas_predictor_benchmark" / "results"
KPI_COLUMNS = [
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

KPI_TITLES_KO = {
    "temp_viol_rate": "온도 제약 위반율",
    "temp_viol_maxrun": "온도 연속 위반 최대 길이",
    "cond_viol_rate": "결로 위험 위반율",
    "cond_viol_maxrun": "결로 위험 연속 위반 최대 길이",
    "rh_viol_rate": "과습 위반율",
    "vpd_viol_rate": "저VPD 위반율",
    "hv_ineff_rate": "난방-환기 동시 사용 비효율",
    "tv_vent": "환기 개도율 총 변화량",
    "sw_heat": "난방 스위칭 횟수",
}

MODEL_COLORS = {
    "physics": "#4C78A8",
    "tiny_ttm": "#F58518",
    "physics_residual_tiny_ttm": "#54A24B",
    "random_forest": "#E45756",
}
MODEL_ORDER = ["physics", "physics_residual_tiny_ttm", "random_forest", "tiny_ttm"]
MODEL_DISPLAY_NAMES = {
    "physics": "물리식",
    "physics_residual_tiny_ttm": "물리식+경량TTM",
    "random_forest": "Random Forest",
    "tiny_ttm": "경량TTM",
}


def configure_plot_fonts() -> None:
    try:
        import koreanize_matplotlib  # noqa: F401
    except Exception:
        pass
    available = {f.name for f in fm.fontManager.ttflist}
    preferred = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",
        "AppleGothic",
    ]
    selected = next((name for name in preferred if name in available), None)
    if selected is not None:
        plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class BenchmarkConfig:
    db_uri: str = DB_URI
    table_name: str = TABLE_NAME
    iot_data_idx: int = IOT_DATA_IDX
    window_days: int = 7
    lookback: int = 6
    step_minutes: int = 5
    train_ratio: float = 0.7
    greenhouse_area_m2: float = 200.0
    greenhouse_volume_m3: float = 300.0
    day_temp: float = 22.0
    night_temp: float = 18.0
    out_light_threshold: float = 10.0
    t_min: float = 12.0
    t_max: float = 28.0
    rh_max: float = 0.90
    vpd_min: float = 0.30
    dtcond_min: float = 0.8
    alpha: float = 0.25
    physics_dt_sec: float = 600.0
    model_name: str = "benchmark_common_theta"
    max_eval_steps: int = 0
    period_start: str = "2025-03-17 00:00:00"
    period_end: str = "2025-03-24 00:00:00"


def _format_period_label(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    return f"{start_ts.strftime('%Y%m%d')}_to_{end_ts.strftime('%Y%m%d')}"


def _ensure_results_dir(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    period_label = _format_period_label(start_ts, end_ts)
    run_dir = RESULTS_ROOT / f"run_{period_label}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "csv").mkdir(parents=True, exist_ok=True)
    (run_dir / "png").mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _norm_humidity(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if float(s.dropna().quantile(0.95)) > 1.5:
        s = s / 100.0
    return s.clip(0.0, 1.0)


def fetch_full_history(cfg: BenchmarkConfig) -> pd.DataFrame:
    engine = create_engine(cfg.db_uri)
    query = f"""
        SELECT reg_date, in_temp, in_hum, in_co2,
               out_temp, out_hum, out_windsp, out_light, out_rain,
               cont_heater_run, cont_skyl_vol, cont_cur_vol, cont_co2_run, cont_fan_run
        FROM {cfg.table_name}
        WHERE iot_data_idx = {int(cfg.iot_data_idx)}
        ORDER BY reg_date ASC
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def preprocess_history(df_raw: pd.DataFrame, cfg: BenchmarkConfig) -> pd.DataFrame:
    df = df_raw.copy()
    df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
    df = df.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)

    df["Tin"] = pd.to_numeric(df["in_temp"], errors="coerce")
    df["RHin"] = _norm_humidity(df["in_hum"])
    df["CO2"] = pd.to_numeric(df["in_co2"], errors="coerce").ffill()
    df["CO2"] = df["CO2"].fillna(900.0)

    df["Tout"] = pd.to_numeric(df["out_temp"], errors="coerce")
    df["RHout"] = _norm_humidity(df["out_hum"])
    df["It"] = pd.to_numeric(df["out_light"], errors="coerce").clip(lower=0.0)
    df["wind"] = pd.to_numeric(df["out_windsp"], errors="coerce").clip(lower=0.0)
    df["rain"] = pd.to_numeric(df["out_rain"], errors="coerce").fillna(0.0)

    skyl = pd.to_numeric(df["cont_skyl_vol"], errors="coerce")
    cur = pd.to_numeric(df["cont_cur_vol"], errors="coerce")
    skyl_max = max(float(skyl.abs().quantile(0.99)), 1.0)
    cur_max = max(float(cur.abs().quantile(0.99)), 1.0)
    df["window_pct_obs"] = (100.0 * skyl / skyl_max).clip(0.0, 100.0).fillna(0.0)
    df["curtain_closed_obs"] = (cur / cur_max).clip(0.0, 1.0).fillna(0.0)
    df["u_heat_obs"] = (pd.to_numeric(df["cont_heater_run"], errors="coerce").fillna(0.0) > 0.5).astype(float)
    df["u_co2_obs"] = (pd.to_numeric(df["cont_co2_run"], errors="coerce").fillna(0.0) > 0.5).astype(float)
    df["fan_obs"] = (pd.to_numeric(df["cont_fan_run"], errors="coerce").fillna(0.0) > 0.5).astype(float)

    df["bucket"] = df["reg_date"].dt.floor(f"{cfg.step_minutes}min")
    agg = {
        "Tin": "mean",
        "RHin": "mean",
        "CO2": "mean",
        "Tout": "mean",
        "RHout": "mean",
        "It": "mean",
        "wind": "mean",
        "rain": "max",
        "window_pct_obs": "mean",
        "curtain_closed_obs": "mean",
        "u_heat_obs": "mean",
        "u_co2_obs": "mean",
        "fan_obs": "mean",
    }
    df = df.groupby("bucket", as_index=False).agg(agg).rename(columns={"bucket": "reg_date"})

    df["u_heat_obs"] = (df["u_heat_obs"] > 0.5).astype(float)
    df["u_co2_obs"] = (df["u_co2_obs"] > 0.5).astype(float)
    df["fan_obs"] = (df["fan_obs"] > 0.5).astype(float)
    df["dt_sec"] = df["reg_date"].diff().dt.total_seconds().fillna(cfg.step_minutes * 60.0)
    df = df[(df["dt_sec"] > 0.0) & (df["dt_sec"] <= 4 * cfg.step_minutes * 60.0)].copy()
    df = df.dropna(subset=["Tin", "RHin", "Tout", "RHout", "It", "wind"]).reset_index(drop=True)

    df["day"] = df["reg_date"].dt.date
    dt_hours = df["dt_sec"] / 3600.0
    df["light_jcm2_step"] = df["It"] * dt_hours * 0.36
    df["out_light_sum"] = df.groupby("day")["light_jcm2_step"].cumsum()
    df["is_day"] = (df["It"] >= cfg.out_light_threshold).astype(float)
    return df


def select_best_common_period(df: pd.DataFrame, cfg: BenchmarkConfig) -> Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    step = timedelta(days=1)
    window = timedelta(days=cfg.window_days)
    start_min = pd.Timestamp(df["reg_date"].min()).floor("D")
    start_max = pd.Timestamp(df["reg_date"].max()) - window

    best_score = -np.inf
    best_slice = None
    best_bounds = None

    current = start_min
    required_cols = ["Tout", "RHout", "It", "wind", "Tin", "RHin"]
    while current <= start_max:
        end = current + window
        seg = df[(df["reg_date"] >= current) & (df["reg_date"] < end)].copy()
        if len(seg) < int(0.85 * cfg.window_days * 24 * 60 / cfg.step_minutes):
            current += step
            continue
        completeness = float(seg[required_cols].notna().mean().mean())
        variability = float(
            seg["Tout"].std(ddof=0)
            + 0.25 * seg["RHout"].std(ddof=0)
            + 0.01 * seg["It"].std(ddof=0)
            + 0.5 * seg["wind"].std(ddof=0)
        )
        sunrise_signal = float(((seg["It"] >= cfg.out_light_threshold).astype(int).diff().abs() > 0).sum())
        score = 100.0 * completeness + variability + 0.05 * sunrise_signal
        if score > best_score:
            best_score = score
            best_slice = seg
            best_bounds = (current, end)
        current += step

    if best_slice is None or best_bounds is None:
        raise RuntimeError("Failed to select a valid common period from the historical data.")
    return best_bounds[0], best_bounds[1], best_slice.reset_index(drop=True)


def select_fixed_period(df: pd.DataFrame, cfg: BenchmarkConfig) -> Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    start_ts = pd.Timestamp(cfg.period_start)
    end_ts = pd.Timestamp(cfg.period_end)
    seg = df[(df["reg_date"] >= start_ts) & (df["reg_date"] < end_ts)].copy().reset_index(drop=True)
    if seg.empty:
        raise RuntimeError(f"No rows found for fixed period: {start_ts} -> {end_ts}")
    return start_ts, end_ts, seg


def solar_windows(df: pd.DataFrame, cfg: BenchmarkConfig) -> Dict[pd.Timestamp.date, Tuple[pd.Timestamp, pd.Timestamp]]:
    out: Dict[pd.Timestamp.date, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for day, seg in df.groupby(df["reg_date"].dt.date):
        bright = seg[seg["It"] >= cfg.out_light_threshold]
        if bright.empty:
            day_start = pd.Timestamp(day)
            out[day] = (day_start + pd.Timedelta(hours=7), day_start + pd.Timedelta(hours=18))
        else:
            out[day] = (pd.Timestamp(bright["reg_date"].iloc[0]), pd.Timestamp(bright["reg_date"].iloc[-1]))
    return out


def _e_from_t_rh(temp_c: float, rh: float) -> float:
    return float(max(0.05, rh * sat_vp_kpa_no_data(float(temp_c))))


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
        "eta": float(np.clip(eta, 1e-5, 1.0 - 1e-5)),
        "rho_cp": 1.2 * 1005.0,
    }


def _theta_init() -> np.ndarray:
    init = {
        "UA": 3500.0,
        "C": 1.5e7,
        "k_heat": 30000.0,
        "a0": 0.18,
        "a1": 4.5,
        "a2": 0.9,
        "k_evap": 2.5e-6,
        "k_photo": 2.5e-5,
        "eta": 0.5,
    }
    return np.array(
        [
            np.log(init["UA"]),
            np.log(init["C"]),
            np.log(init["k_heat"]),
            np.log(init["a0"]),
            np.log(init["a1"]),
            np.log(init["a2"]),
            np.log(init["k_evap"]),
            np.log(init["k_photo"]),
            math.log(init["eta"] / (1.0 - init["eta"])),
        ],
        dtype=float,
    )


def fit_common_theta(df_train: pd.DataFrame, cfg: BenchmarkConfig) -> Tuple[Dict[str, float], Dict[str, float]]:
    work = df_train.reset_index(drop=True).copy()
    if len(work) < 20:
        raise ValueError("Need at least 20 training rows to fit a common theta.")
    if len(work) > 360:
        sample_idx = np.linspace(0, len(work) - 1, num=360, dtype=int)
        work = work.iloc[np.unique(sample_idx)].reset_index(drop=True)

    def objective(z: np.ndarray) -> float:
        theta = _theta_from_unconstrained(z)
        loss_terms: List[float] = []
        for i in range(len(work) - 1):
            cur = work.iloc[i]
            nxt = work.iloc[i + 1]
            state = {
                "Tin": float(cur["Tin"]),
                "e_in": _e_from_t_rh(float(cur["Tin"]), float(cur["RHin"])),
                "CO2": float(cur["CO2"]),
            }
            disturbance = {
                "Tout": float(cur["Tout"]),
                "RHout": float(cur["RHout"]),
                "It": float(cur["It"]),
                "wind": float(cur["wind"]),
                "CO2out": 420.0,
                "hour": float(pd.Timestamp(cur["reg_date"]).hour + pd.Timestamp(cur["reg_date"]).minute / 60.0),
                "is_day": bool(cur["It"] >= cfg.out_light_threshold),
            }
            action = {
                "u_heat": float(cur["u_heat_obs"]),
                "x_vent": float(cur["window_pct_obs"]) / 100.0,
                "curtain": float(cur["curtain_closed_obs"]),
                "u_co2": float(cur["u_co2_obs"]),
            }
            pred = step_dynamics_no_data(
                state,
                disturbance,
                action,
                theta,
                dt_min=max(1, int(round(float(cur["dt_sec"]) / 60.0))),
                A=cfg.greenhouse_area_m2,
                V=cfg.greenhouse_volume_m3,
            )
            tin_pred = float(pred["Tin"])
            e_pred = float(pred["e_in"])
            co2_pred = float(pred["CO2"])
            if (not np.isfinite(tin_pred)) or (not np.isfinite(e_pred)) or (not np.isfinite(co2_pred)):
                return 1e9
            if tin_pred < -20.0 or tin_pred > 60.0:
                return 1e8 + abs(tin_pred) * 1e4
            sat = max(float(sat_vp_kpa_no_data(float(np.clip(tin_pred, -20.0, 60.0)))), 1e-6)
            rh_pred = float(np.clip(e_pred / sat, 0.0, 1.0))
            tin_err = (float(pred["Tin"]) - float(nxt["Tin"])) / 4.0
            rh_err = (rh_pred - float(nxt["RHin"])) / 0.15
            co2_err = (float(pred["CO2"]) - float(nxt["CO2"])) / 250.0
            loss_terms.append(tin_err * tin_err + 0.5 * rh_err * rh_err + 0.2 * co2_err * co2_err)

        reg = 1e-4 * float(np.sum(z[:8] ** 2))
        return float(np.mean(loss_terms) + reg)

    bounds = [
        (np.log(500.0), np.log(15000.0)),
        (np.log(1.0e6), np.log(8.0e7)),
        (np.log(3000.0), np.log(80000.0)),
        (np.log(0.01), np.log(1.0)),
        (np.log(0.2), np.log(15.0)),
        (np.log(0.05), np.log(4.0)),
        (np.log(1.0e-7), np.log(5.0e-5)),
        (np.log(1.0e-6), np.log(5.0e-4)),
        (-4.0, 4.0),
    ]
    res = minimize(
        objective,
        _theta_init(),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 80},
    )
    theta = _theta_from_unconstrained(res.x)
    diagnostics = {
        "objective": float(res.fun),
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "n_iter": int(getattr(res, "nit", -1)),
    }
    return theta, diagnostics


def max_run(mask: Sequence[bool]) -> int:
    best = 0
    cur = 0
    for flag in mask:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def calc_metrics(df: pd.DataFrame, cfg: BenchmarkConfig) -> pd.DataFrame:
    temp_viol = (df["Tin"] < cfg.t_min) | (df["Tin"] > cfg.t_max)
    cond_viol = df["dTcond"] < cfg.dtcond_min
    rh_viol = df["RHin"] > cfg.rh_max
    vpd_viol = df["VPD"] < cfg.vpd_min
    idx_heat = df["u_heat"] > 0.5
    if idx_heat.any():
        hv_ineff = float((df.loc[idx_heat, "Q_ventloss"] > cfg.alpha * df.loc[idx_heat, "Q_heat"].clip(lower=1e-6)).mean())
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
                "sw_heat": int((df["u_heat"].diff().abs() > 0.0).sum()),
            }
        ]
    )


def derived_result_from_theta(theta: Dict[str, float]) -> Dict[str, Any]:
    return {
        "est": {
            "UA": float(theta["UA"]),
            "C": float(theta["C"]),
            "g_solar": float(theta["eta"]),
            "ACH_model": {
                "a0": float(theta["a0"]),
                "a1": float(theta["a1"]),
                "a2": float(theta["a2"]),
                "a3": 0.0,
            },
        },
        "lambda": {"UA": 1.0, "C": 1.0, "ACH": 1.0, "g_solar": 1.0},
        "mode": "benchmark_common_theta",
        "safety": {},
    }


def physics_predict_temp(current_row: pd.Series, candidate: Any, theta: Dict[str, float], cfg: BenchmarkConfig) -> float:
    tin = float(current_row["Tin"])
    tout = float(current_row["Tout"])
    out_light = float(current_row["It"])
    wind = float(current_row["wind"])
    window_pct = int(candidate.window[0])
    shade_open = int(candidate.curtain[1])
    thermal_open = int(candidate.thermal_curtain[1])
    fcu_mode, fcu_state = candidate.fcu

    x = float(window_pct) / 100.0
    ach = max(float(theta["a0"]) + float(theta["a1"]) * x + float(theta["a2"]) * wind * x, 0.0)
    shade_fraction = (100.0 - shade_open) / 100.0
    thermal_fraction = (100.0 - thermal_open) / 100.0
    solar_transmission = np.clip(1.0 - 0.70 * shade_fraction, 0.0, 1.0)
    thermal_loss_factor = np.clip(1.0 - 0.30 * thermal_fraction, 0.1, 1.0)

    q_trans = float(theta["UA"]) * thermal_loss_factor * (tout - tin)
    q_vent = float(theta["rho_cp"]) * cfg.greenhouse_volume_m3 * (ach / 3600.0) * (tout - tin)
    q_solar = float(theta["eta"]) * cfg.greenhouse_area_m2 * max(out_light, 0.0) * solar_transmission
    q_fcu = 0.0
    if fcu_state == "on":
        q_fcu = float(theta["k_heat"]) if fcu_mode == "heat" else -float(theta["k_heat"])
    return float(tin + (cfg.physics_dt_sec / float(theta["C"])) * (q_trans + q_vent + q_solar + q_fcu))


def candidate_to_action(candidate: Any) -> Dict[str, float]:
    return {
        "u_heat": 1.0 if candidate.fcu[1] == "on" and candidate.fcu[0] == "heat" else 0.0,
        "x_vent": float(candidate.window[0]) / 100.0,
        "curtain": (100.0 - float(candidate.curtain[1])) / 100.0,
        "u_co2": 0.0,
    }


def history_temporal_features(window: pd.DataFrame) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    dyn_cols = ["Tin", "RHin", "CO2", "Tout", "RHout", "It", "wind", "u_heat", "x_vent", "curtain", "fan"]
    for col in dyn_cols:
        values = window[col].to_numpy(dtype=float)
        feats[f"{col}_last"] = float(values[-1])
        feats[f"{col}_mean"] = float(values.mean())
        feats[f"{col}_delta"] = float(values[-1] - values[0])
        feats[f"{col}_recent_mean"] = float(values[-min(3, len(values)) :].mean())

    ts = pd.Timestamp(window["reg_date"].iloc[-1])
    hour = ts.hour + ts.minute / 60.0
    feats["hour_sin"] = float(np.sin(2.0 * np.pi * hour / 24.0))
    feats["hour_cos"] = float(np.cos(2.0 * np.pi * hour / 24.0))
    feats["is_day_now"] = float(window["It"].iloc[-1] >= 10.0)
    return feats


def build_training_table(df: pd.DataFrame, theta: Dict[str, float], lookback: int, cfg: BenchmarkConfig) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    work = df.reset_index(drop=True).copy()
    work["u_heat"] = work["u_heat_obs"]
    work["x_vent"] = work["window_pct_obs"] / 100.0
    work["curtain"] = work["curtain_closed_obs"]
    work["fan"] = work["fan_obs"]
    for i in range(lookback - 1, len(work) - 1):
        window = work.iloc[i - lookback + 1 : i + 1].copy()
        cur = work.iloc[i]
        nxt = work.iloc[i + 1]
        action = {
            "u_heat": float(cur["u_heat_obs"]),
            "x_vent": float(cur["window_pct_obs"]) / 100.0,
            "curtain": float(cur["curtain_closed_obs"]),
            "u_co2": float(cur["u_co2_obs"]),
        }
        features = history_temporal_features(window)
        features["cand_u_heat"] = float(action["u_heat"])
        features["cand_x_vent"] = float(action["x_vent"])
        features["cand_curtain"] = float(action["curtain"])
        features["cand_u_co2"] = float(action["u_co2"])
        pseudo_candidate = types.SimpleNamespace(
            window=(int(round(action["x_vent"] * 100.0)), "main", "OPEN" if action["x_vent"] > 0 else "HOLD"),
            curtain=("shade", int(round((1.0 - action["curtain"]) * 100.0))),
            thermal_curtain=("thermal", 100),
            fcu=("heat", "on" if action["u_heat"] > 0.5 else "off"),
        )
        physics_pred = physics_predict_temp(cur, pseudo_candidate, theta, cfg)
        features["physics_pred"] = float(physics_pred)
        features["target_tin_next"] = float(nxt["Tin"])
        features["target_dTin"] = float(nxt["Tin"] - cur["Tin"])
        features["target_residual"] = float(nxt["Tin"] - physics_pred)
        rows.append(features)
    return pd.DataFrame(rows)


class BasePredictor:
    name: str = "base"

    def fit(self, table: pd.DataFrame) -> None:
        _ = table

    def predict_temp(self, history: pd.DataFrame, current_row: pd.Series, candidate: Any, theta: Dict[str, float], cfg: BenchmarkConfig) -> float:
        raise NotImplementedError

    def artifact_size_bytes(self) -> int:
        return len(pickle.dumps(self))


class PhysicsPredictor(BasePredictor):
    name = "physics"

    def predict_temp(self, history: pd.DataFrame, current_row: pd.Series, candidate: Any, theta: Dict[str, float], cfg: BenchmarkConfig) -> float:
        _ = history
        return physics_predict_temp(current_row, candidate, theta, cfg)


class LearnedPredictor(BasePredictor):
    feature_cols: List[str]
    target_col: str

    def __init__(self, name: str, target_col: str):
        self.name = name
        self.target_col = target_col
        self.feature_cols = []

    def _build_features(self, history: pd.DataFrame, candidate_action: Dict[str, float], physics_pred: float) -> pd.DataFrame:
        window = history.iloc[-self.lookback :].copy()
        feats = history_temporal_features(window)
        feats["cand_u_heat"] = float(candidate_action["u_heat"])
        feats["cand_x_vent"] = float(candidate_action["x_vent"])
        feats["cand_curtain"] = float(candidate_action["curtain"])
        feats["cand_u_co2"] = float(candidate_action["u_co2"])
        feats["physics_pred"] = float(physics_pred)
        return pd.DataFrame([feats])

    def _build_features_from_base(self, base_feats: Dict[str, float], candidate_action: Dict[str, float], physics_pred: float) -> pd.DataFrame:
        feats = dict(base_feats)
        feats["cand_u_heat"] = float(candidate_action["u_heat"])
        feats["cand_x_vent"] = float(candidate_action["x_vent"])
        feats["cand_curtain"] = float(candidate_action["curtain"])
        feats["cand_u_co2"] = float(candidate_action["u_co2"])
        feats["physics_pred"] = float(physics_pred)
        return pd.DataFrame([feats])


class TinyTTMPredictor(LearnedPredictor):
    def __init__(self, lookback: int):
        super().__init__(name="tiny_ttm", target_col="target_dTin")
        self.lookback = lookback
        self.model = MLPRegressor(
            hidden_layer_sizes=(24, 12),
            activation="tanh",
            solver="adam",
            max_iter=400,
            random_state=42,
        )

    def fit(self, table: pd.DataFrame) -> None:
        self.feature_cols = [c for c in table.columns if c not in {"target_tin_next", "target_dTin", "target_residual"}]
        self.model.fit(table[self.feature_cols], table[self.target_col])

    def predict_temp(self, history: pd.DataFrame, current_row: pd.Series, candidate: Any, theta: Dict[str, float], cfg: BenchmarkConfig) -> float:
        physics_pred = physics_predict_temp(current_row, candidate, theta, cfg)
        action = candidate_to_action(candidate)
        x = self._build_features(history, action, physics_pred)
        delta = float(self.model.predict(x[self.feature_cols])[0])
        return float(current_row["Tin"] + delta)

    def predict_temp_from_base(self, base_feats: Dict[str, float], current_row: pd.Series, candidate: Any, theta: Dict[str, float], cfg: BenchmarkConfig) -> float:
        physics_pred = physics_predict_temp(current_row, candidate, theta, cfg)
        action = candidate_to_action(candidate)
        x = self._build_features_from_base(base_feats, action, physics_pred)
        delta = float(self.model.predict(x[self.feature_cols])[0])
        return float(current_row["Tin"] + delta)


class PhysicsResidualTinyTTMPredictor(LearnedPredictor):
    def __init__(self, lookback: int):
        super().__init__(name="physics_residual_tiny_ttm", target_col="target_residual")
        self.lookback = lookback
        self.model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="tanh",
            solver="adam",
            max_iter=400,
            random_state=7,
        )

    def fit(self, table: pd.DataFrame) -> None:
        self.feature_cols = [c for c in table.columns if c not in {"target_tin_next", "target_dTin", "target_residual"}]
        self.model.fit(table[self.feature_cols], table[self.target_col])

    def predict_temp(self, history: pd.DataFrame, current_row: pd.Series, candidate: Any, theta: Dict[str, float], cfg: BenchmarkConfig) -> float:
        physics_pred = physics_predict_temp(current_row, candidate, theta, cfg)
        action = candidate_to_action(candidate)
        x = self._build_features(history, action, physics_pred)
        residual = float(self.model.predict(x[self.feature_cols])[0])
        return float(physics_pred + residual)

    def predict_temp_from_base(self, base_feats: Dict[str, float], current_row: pd.Series, candidate: Any, theta: Dict[str, float], cfg: BenchmarkConfig) -> float:
        physics_pred = physics_predict_temp(current_row, candidate, theta, cfg)
        action = candidate_to_action(candidate)
        x = self._build_features_from_base(base_feats, action, physics_pred)
        residual = float(self.model.predict(x[self.feature_cols])[0])
        return float(physics_pred + residual)


class RandomForestPredictor(LearnedPredictor):
    def __init__(self, lookback: int):
        super().__init__(name="random_forest", target_col="target_dTin")
        self.lookback = lookback
        self.model = RandomForestRegressor(
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        )

    def fit(self, table: pd.DataFrame) -> None:
        self.feature_cols = [c for c in table.columns if c not in {"target_tin_next", "target_dTin", "target_residual"}]
        self.model.fit(table[self.feature_cols], table[self.target_col])

    def predict_temp(self, history: pd.DataFrame, current_row: pd.Series, candidate: Any, theta: Dict[str, float], cfg: BenchmarkConfig) -> float:
        physics_pred = physics_predict_temp(current_row, candidate, theta, cfg)
        action = candidate_to_action(candidate)
        x = self._build_features(history, action, physics_pred)
        delta = float(self.model.predict(x[self.feature_cols])[0])
        return float(current_row["Tin"] + delta)

    def predict_temp_from_base(self, base_feats: Dict[str, float], current_row: pd.Series, candidate: Any, theta: Dict[str, float], cfg: BenchmarkConfig) -> float:
        physics_pred = physics_predict_temp(current_row, candidate, theta, cfg)
        action = candidate_to_action(candidate)
        x = self._build_features_from_base(base_feats, action, physics_pred)
        delta = float(self.model.predict(x[self.feature_cols])[0])
        return float(current_row["Tin"] + delta)


def build_predictors(lookback: int) -> List[BasePredictor]:
    return [
        PhysicsPredictor(),
        TinyTTMPredictor(lookback=lookback),
        PhysicsResidualTinyTTMPredictor(lookback=lookback),
        RandomForestPredictor(lookback=lookback),
    ]


def prepare_history_window(seed_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    hist = seed_df.copy().reset_index(drop=True)
    hist["u_heat"] = hist["u_heat_obs"]
    hist["x_vent"] = hist["window_pct_obs"] / 100.0
    hist["curtain"] = hist["curtain_closed_obs"]
    hist["fan"] = hist["fan_obs"]
    return hist.iloc[-lookback:].copy().reset_index(drop=True)


def make_controller_row(boundary_row: pd.Series, state: Dict[str, float], prev_window_pct: float) -> pd.DataFrame:
    rh = float(np.clip(state["e_in"] / sat_vp_kpa_no_data(float(state["Tin"])), 0.0, 1.0))
    return pd.DataFrame(
        [
            {
                "reg_date": pd.Timestamp(boundary_row["reg_date"]),
                "in_temp": float(state["Tin"]),
                "in_humidity": rh,
                "in_co2": float(state["CO2"]),
                "out_temp": float(boundary_row["Tout"]),
                "out_humidity": float(boundary_row["RHout"]),
                "out_light": float(boundary_row["It"]),
                "out_rain": float(boundary_row["rain"]),
                "wind_speed": float(boundary_row["wind"]),
                "window_pct": float(prev_window_pct),
                "out_light_sum": float(boundary_row["out_light_sum"]),
            }
        ]
    )


def replay_with_predictor(
    predictor: BasePredictor,
    seed_history: pd.DataFrame,
    boundary_df: pd.DataFrame,
    theta: Dict[str, float],
    solar_map: Dict[pd.Timestamp.date, Tuple[pd.Timestamp, pd.Timestamp]],
    cfg: BenchmarkConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    history = prepare_history_window(seed_history, cfg.lookback)
    first = boundary_df.iloc[0]
    state = {
        "Tin": float(first["Tin"]),
        "e_in": _e_from_t_rh(float(first["Tin"]), float(first["RHin"])),
        "CO2": float(first["CO2"]),
    }
    prev_window_pct = float(first["window_pct_obs"])

    rows: List[Dict[str, Any]] = []
    control_times: List[float] = []
    rollout_start = time.perf_counter()

    for i in range(len(boundary_df) - 1):
        cur = boundary_df.iloc[i]
        sunrise, sunset = solar_map[pd.Timestamp(cur["reg_date"]).date()]
        ctrl_row = make_controller_row(cur, state, prev_window_pct)
        base_feats = history_temporal_features(history.iloc[-cfg.lookback :].copy())
        controller = Controller(
            ctrl_row,
            out_light_info=(sunrise, sunset, pd.NaT, np.array([]), 100.0),
            temp=(cfg.day_temp, cfg.night_temp),
            params={
                "fcu_mode": "heat",
                "farm_sn": cfg.iot_data_idx,
                "stage_name": "benchmark",
                "physics_model_name": cfg.model_name,
                "greenhouse_area_m2": cfg.greenhouse_area_m2,
                "greenhouse_volume_m3": cfg.greenhouse_volume_m3,
                "physics_dt_sec": float(cur["dt_sec"]),
                "physics_k_heat": float(theta["k_heat"]),
                "physics_rho_cp": float(theta["rho_cp"]),
            },
            policy_state=PolicyState(fcu_mode="heat"),
            derived_result=derived_result_from_theta(theta),
            agro_kpis={},
        )
        inner = controller.inner

        def _patched_predict_temp(self: Any, candidate: Any) -> float:
            current_row = pd.Series(
                {
                    "Tin": float(state["Tin"]),
                    "Tout": float(cur["Tout"]),
                    "It": float(cur["It"]),
                    "wind": float(cur["wind"]),
                }
            )
            if hasattr(predictor, "predict_temp_from_base"):
                return predictor.predict_temp_from_base(base_feats, current_row, candidate, theta, cfg)
            return predictor.predict_temp(history, current_row, candidate, theta, cfg)

        inner._predict_temp = types.MethodType(_patched_predict_temp, inner)

        t0 = time.perf_counter()
        result = controller.run()
        control_times.append((time.perf_counter() - t0) * 1000.0)

        action = {
            "u_heat": 1.0 if result["fcu"][1] == "on" and result["fcu"][0] == "heat" else 0.0,
            "x_vent": float(result["window"][0]) / 100.0,
            "curtain": (100.0 - float(result["curtain"][1])) / 100.0,
            "u_co2": 0.0,
        }
        disturbance = {
            "Tout": float(cur["Tout"]),
            "RHout": float(cur["RHout"]),
            "It": float(cur["It"]),
            "wind": float(cur["wind"]),
            "CO2out": 420.0,
            "hour": float(pd.Timestamp(cur["reg_date"]).hour + pd.Timestamp(cur["reg_date"]).minute / 60.0),
            "is_day": bool(cur["It"] >= cfg.out_light_threshold),
        }
        pred = step_dynamics_no_data(
            state,
            disturbance,
            action,
            theta,
            dt_min=max(1, int(round(float(cur["dt_sec"]) / 60.0))),
            A=cfg.greenhouse_area_m2,
            V=cfg.greenhouse_volume_m3,
        )

        rh_now = float(np.clip(state["e_in"] / sat_vp_kpa_no_data(float(state["Tin"])), 0.0, 1.0))
        vpd_now = float(vpd_kpa_no_data(float(state["Tin"]), rh_now))
        td_now = float(dewpoint_c_no_data(float(state["Tin"]), rh_now))
        rows.append(
            {
                "reg_date": pd.Timestamp(cur["reg_date"]),
                "Tin": float(state["Tin"]),
                "RHin": rh_now,
                "VPD": vpd_now,
                "dTcond": float(state["Tin"] - td_now),
                "CO2": float(state["CO2"]),
                "u_heat": float(action["u_heat"]),
                "x_vent": float(action["x_vent"]),
                "curtain": float(action["curtain"]),
                "u_co2": float(action["u_co2"]),
                "ACH": float(pred["ACH"]),
                "Q_heat": float(pred["Q_heat"]),
                "Q_ventloss": float(pred["Q_ventloss"]),
                "model": predictor.name,
            }
        )

        next_hist = {
            "reg_date": pd.Timestamp(boundary_df.iloc[i + 1]["reg_date"]),
            "Tin": float(pred["Tin"]),
            "RHin": float(np.clip(pred["e_in"] / sat_vp_kpa_no_data(float(pred["Tin"])), 0.0, 1.0)),
            "CO2": float(pred["CO2"]),
            "Tout": float(boundary_df.iloc[i + 1]["Tout"]),
            "RHout": float(boundary_df.iloc[i + 1]["RHout"]),
            "It": float(boundary_df.iloc[i + 1]["It"]),
            "wind": float(boundary_df.iloc[i + 1]["wind"]),
            "u_heat": float(action["u_heat"]),
            "x_vent": float(action["x_vent"]),
            "curtain": float(action["curtain"]),
            "fan": 1.0 if result["fan"][0] == "on" else 0.0,
        }
        history = pd.concat([history, pd.DataFrame([next_hist])], ignore_index=True).iloc[-cfg.lookback :].reset_index(drop=True)
        state = {"Tin": float(pred["Tin"]), "e_in": float(pred["e_in"]), "CO2": float(pred["CO2"])}
        prev_window_pct = 100.0 * float(action["x_vent"])

    total_rollout_sec = time.perf_counter() - rollout_start
    episode = pd.DataFrame(rows)
    efficiency = {
        "rollout_sec": float(total_rollout_sec),
        "mean_control_step_ms": float(np.mean(control_times) if control_times else 0.0),
        "p95_control_step_ms": float(np.quantile(control_times, 0.95) if control_times else 0.0),
        "model_bytes": int(predictor.artifact_size_bytes()),
    }
    return episode, efficiency


def evaluate_prediction_accuracy(
    predictor: BasePredictor,
    hist_seed: pd.DataFrame,
    test_df: pd.DataFrame,
    theta: Dict[str, float],
    cfg: BenchmarkConfig,
) -> Dict[str, float]:
    work = pd.concat([hist_seed.tail(cfg.lookback), test_df], ignore_index=True).copy()
    work["u_heat"] = work.get("u_heat_obs", 0.0)
    work["x_vent"] = work.get("window_pct_obs", 0.0) / 100.0
    work["curtain"] = work.get("curtain_closed_obs", 0.0)
    work["fan"] = work.get("fan_obs", 0.0)

    y_true: List[float] = []
    y_pred: List[float] = []
    for i in range(cfg.lookback - 1, len(work) - 1):
        cur = work.iloc[i]
        nxt = work.iloc[i + 1]
        window = work.iloc[i - cfg.lookback + 1 : i + 1].copy()
        pseudo_candidate = types.SimpleNamespace(
            window=(int(round(float(cur.get("window_pct_obs", 0.0)))), "main", "OPEN" if float(cur.get("window_pct_obs", 0.0)) > 0 else "HOLD"),
            curtain=("shade", int(round((1.0 - float(cur.get("curtain_closed_obs", 0.0))) * 100.0))),
            thermal_curtain=("thermal", 100),
            fcu=("heat", "on" if float(cur.get("u_heat_obs", 0.0)) > 0.5 else "off"),
        )
        current_row = pd.Series({"Tin": float(cur["Tin"]), "Tout": float(cur["Tout"]), "It": float(cur["It"]), "wind": float(cur["wind"])})
        y_true.append(float(nxt["Tin"]))
        y_pred.append(float(predictor.predict_temp(window, current_row, pseudo_candidate, theta, cfg)))
    arr_true = np.asarray(y_true, dtype=float)
    arr_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(arr_pred - arr_true)))
    rmse = float(np.sqrt(np.mean((arr_pred - arr_true) ** 2)))
    return {"Tin_next_MAE": mae, "Tin_next_RMSE": rmse}


def plot_temperature_trajectory(episodes: Dict[str, pd.DataFrame], actual: pd.DataFrame, output_path: Path) -> None:
    configure_plot_fonts()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(actual["reg_date"], actual["Tin"], label="actual", color="black", linewidth=2.8, alpha=0.85)
    for name in MODEL_ORDER:
        if name not in episodes:
            continue
        ep = episodes[name]
        ax.plot(ep["reg_date"], ep["Tin"], label=MODEL_DISPLAY_NAMES.get(name, name), linewidth=2.2, color=MODEL_COLORS.get(name))
    ax.set_title("실내온도 시계열 비교", fontsize=20)
    ax.set_xlabel("시간", fontsize=17)
    ax.set_ylabel("실내온도 [C]", fontsize=17)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(summary: pd.DataFrame, output_path: Path) -> None:
    configure_plot_fonts()
    fig, ax = plt.subplots(figsize=(9, 6))
    ordered = summary.copy()
    ordered["model"] = pd.Categorical(ordered["model"], categories=MODEL_ORDER, ordered=True)
    ordered = ordered.sort_values("model").reset_index(drop=True)
    for _, row in ordered.iterrows():
        ax.scatter(row["model_bytes"], row["mean_control_step_ms"], s=120)
        label = MODEL_DISPLAY_NAMES.get(str(row["model"]), str(row["model"]))
        if row["model"] == "random_forest":
            ax.text(row["model_bytes"], row["mean_control_step_ms"], f"{label} ", va="center", ha="right", fontsize=15)
        else:
            ax.text(row["model_bytes"], row["mean_control_step_ms"], f" {label}", va="center", ha="left", fontsize=15)
    ax.set_xlabel("모델 크기 [bytes]", fontsize=17)
    ax.set_ylabel("평균 제어 계산 시간 [ms]", fontsize=17)
    ax.set_title("경량화 효율 비교", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_trajectory_errors(episodes: Dict[str, pd.DataFrame], actual: pd.DataFrame) -> pd.DataFrame:
    actual_df = actual[["reg_date", "Tin"]].copy()
    actual_df["reg_date"] = pd.to_datetime(actual_df["reg_date"])
    rows: List[Dict[str, float]] = []
    for model in MODEL_ORDER:
        if model not in episodes:
            continue
        ep = episodes[model][["reg_date", "Tin"]].copy()
        ep["reg_date"] = pd.to_datetime(ep["reg_date"])
        merged = ep.merge(actual_df, on="reg_date", how="inner", suffixes=("_pred", "_actual"))
        if merged.empty:
            continue
        err = merged["Tin_pred"] - merged["Tin_actual"]
        rows.append(
            {
                "model": model,
                "traj_mae": float(err.abs().mean()),
                "traj_rmse": float(np.sqrt(np.mean(err.values ** 2))),
                "traj_bias": float(err.mean()),
                "traj_max_abs": float(err.abs().max()),
                "traj_corr": float(np.corrcoef(merged["Tin_pred"], merged["Tin_actual"])[0, 1]) if len(merged) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_trajectory_error(summary: pd.DataFrame, output_path: Path) -> None:
    configure_plot_fonts()
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = [
        ("Tin_next_RMSE", "다음시점 온도 RMSE"),
        ("traj_rmse", "시계열 RMSE"),
        ("traj_bias", "시계열 편향"),
        ("traj_max_abs", "최대 절대오차"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        colors = [MODEL_COLORS.get(m, "#999999") for m in summary["model"]]
        ax.bar(range(len(summary)), summary[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_ylabel("오차 값", fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS.get(model, "#999999"), label=model) for model in summary["model"]]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report(
    selected_period: Tuple[pd.Timestamp, pd.Timestamp],
    eval_period: Tuple[pd.Timestamp, pd.Timestamp],
    theta: Dict[str, float],
    theta_diag: Dict[str, float],
    merged: pd.DataFrame,
) -> str:
    lines = []
    lines.append("# GEAS Predictor Benchmark Report")
    lines.append("")
    lines.append(f"- Selected period: `{selected_period[0]}` to `{selected_period[1]}`")
    lines.append(f"- Evaluation period: `{eval_period[0]}` to `{eval_period[1]}`")
    lines.append(f"- Common theta objective: `{theta_diag['objective']:.4f}`")
    lines.append(f"- Theta fit success: `{theta_diag['success']}`")
    lines.append("")
    lines.append("## Common Theta")
    lines.append("")
    for key, value in theta.items():
        lines.append(f"- `{key}`: `{value:.6g}`")
    lines.append("")
    lines.append("## Comparison Summary")
    lines.append("")
    lines.append("```text")
    lines.append(merged.round(4).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Edge Notes")
    lines.append("")
    lines.append("- `physics` is the lightest and most deterministic baseline.")
    lines.append("- `tiny_ttm` uses temporal feature mixing with a small MLP head, so sequence context is reflected while remaining edge-friendly.")
    lines.append("- `physics_residual_tiny_ttm` is usually the safest AI deployment path because it preserves physics priors and learns only correction residuals.")
    lines.append("- `random_forest` is easy to fit and interpret, but serialized size and branch-heavy inference can grow quickly on edge devices.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GEAS inner-layer predictors on a shared offline replay experiment.")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--lookback", type=int, default=6)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--period-start", type=str, default="2025-03-17 00:00:00")
    parser.add_argument("--period-end", type=str, default="2025-03-24 00:00:00")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        window_days=int(args.window_days),
        lookback=int(args.lookback),
        step_minutes=int(args.step_minutes),
        train_ratio=float(args.train_ratio),
        max_eval_steps=int(args.max_eval_steps),
        period_start=str(args.period_start),
        period_end=str(args.period_end),
        physics_dt_sec=float(args.step_minutes) * 60.0,
    )

    print("[INFO] fetching full history from DB", flush=True)
    raw = fetch_full_history(cfg)
    print(f"[INFO] fetched rows={len(raw)}", flush=True)
    hist = preprocess_history(raw, cfg)
    print(f"[INFO] preprocessed rows={len(hist)}", flush=True)

    start_ts, end_ts, period_df = select_fixed_period(hist, cfg)
    run_dir = _ensure_results_dir(start_ts, end_ts)
    csv_dir = run_dir / "csv"
    png_dir = run_dir / "png"
    print(f"[INFO] results_dir={run_dir}", flush=True)
    _json_dump(run_dir / "selected_period.json", {"start": start_ts.isoformat(), "end": end_ts.isoformat(), "rows": int(len(period_df))})
    print(f"[INFO] selected period {start_ts} -> {end_ts}, rows={len(period_df)}", flush=True)

    split_idx = max(cfg.lookback + 5, int(len(period_df) * cfg.train_ratio))
    train_df = period_df.iloc[:split_idx].reset_index(drop=True)
    test_df = period_df.iloc[cfg.lookback :].reset_index(drop=True)
    eval_df = period_df.iloc[cfg.lookback :].reset_index(drop=True)
    if cfg.max_eval_steps and len(eval_df) > cfg.max_eval_steps:
        eval_df = eval_df.iloc[: cfg.max_eval_steps].reset_index(drop=True)
    solar_map = solar_windows(period_df, cfg)
    _json_dump(
        run_dir / "evaluation_period.json",
        {
            "start": pd.Timestamp(eval_df["reg_date"].iloc[0]).isoformat(),
            "end": pd.Timestamp(eval_df["reg_date"].iloc[-1]).isoformat(),
            "rows": int(len(eval_df)),
        },
    )

    print("[INFO] fitting common theta", flush=True)
    theta, theta_diag = fit_common_theta(train_df, cfg)
    _json_dump(run_dir / "theta_common.json", {"theta": theta, "diagnostics": theta_diag})

    print("[INFO] building supervised training table", flush=True)
    train_table = build_training_table(train_df, theta, cfg.lookback, cfg)

    predictors = build_predictors(cfg.lookback)
    kpi_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    eff_rows: List[Dict[str, Any]] = []
    episodes: Dict[str, pd.DataFrame] = {}

    seed_history = period_df.iloc[: cfg.lookback].copy().reset_index(drop=True)

    for predictor in predictors:
        print(f"[INFO] fitting predictor={predictor.name}", flush=True)
        fit_start = time.perf_counter()
        predictor.fit(train_table)
        fit_sec = time.perf_counter() - fit_start

        print(f"[INFO] replaying predictor={predictor.name}", flush=True)
        episode, eff = replay_with_predictor(predictor, seed_history, eval_df, theta, solar_map, cfg)
        episodes[predictor.name] = episode
        episode.to_csv(csv_dir / f"episode_{predictor.name}.csv", index=False)

        kpi = calc_metrics(episode, cfg).iloc[0].to_dict()
        kpi["model"] = predictor.name
        kpi_rows.append(kpi)

        pred_summary = evaluate_prediction_accuracy(predictor, seed_history, eval_df, theta, cfg)
        pred_summary["model"] = predictor.name
        pred_rows.append(pred_summary)

        eff["fit_sec"] = float(fit_sec)
        eff["model"] = predictor.name
        eff_rows.append(eff)

    kpi_df = pd.DataFrame(kpi_rows).sort_values("model").reset_index(drop=True)
    pred_df = pd.DataFrame(pred_rows).sort_values("model").reset_index(drop=True)
    eff_df = pd.DataFrame(eff_rows).sort_values("model").reset_index(drop=True)
    merged = kpi_df.merge(pred_df, on="model", how="left").merge(eff_df, on="model", how="left")
    traj_df = summarize_trajectory_errors(episodes, eval_df).sort_values("model").reset_index(drop=True)

    kpi_df.to_csv(csv_dir / "kpi_summary.csv", index=False)
    pred_df.to_csv(csv_dir / "prediction_summary.csv", index=False)
    eff_df.to_csv(csv_dir / "efficiency_summary.csv", index=False)
    traj_df = traj_df.merge(pred_df[["model", "Tin_next_RMSE"]], on="model", how="left")
    traj_df.to_csv(csv_dir / "trajectory_summary.csv", index=False)
    merged.to_csv(csv_dir / "comparison_summary.csv", index=False)

    plot_temperature_trajectory(episodes, eval_df, png_dir / "temperature_trajectory.png")
    plot_efficiency(eff_df, png_dir / "efficiency_comparison.png")
    if not traj_df.empty:
        plot_trajectory_error(traj_df, png_dir / "trajectory_error_comparison.png")

    eval_period = (pd.Timestamp(eval_df["reg_date"].iloc[0]), pd.Timestamp(eval_df["reg_date"].iloc[-1]))
    report = build_report((start_ts, end_ts), eval_period, theta, theta_diag, merged)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    print("[INFO] benchmark complete", flush=True)
    print(merged.round(4).to_string(index=False), flush=True)
    print(f"[INFO] saved report to {run_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
