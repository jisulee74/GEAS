from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PolicyState:
    T_day_bias: float = 0.0
    T_night_bias: float = 0.0
    DB_day: float = 0.3
    PB_day: float = 2.0
    DB_night: float = 0.4
    PB_night: float = 1.5
    K_window_day: float = 1.0
    K_window_night: float = 1.0
    alpha_curtain: float = 1.0
    fcu_mode: str = "cool"
    last_score: Optional[float] = None
    last_daily_eval_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _compute_kpis(
    df_today: pd.DataFrame,
    sunrise: pd.Timestamp,
    sunset: pd.Timestamp,
    T_day: float,
    T_night: float,
    band_width: float = 1.0,
) -> Dict[str, float]:
    df = df_today[["reg_date", "in_temp"]].dropna().copy()
    if df.shape[0] < 2:
        return dict(cov_day=0.0, cov_night=0.0, mdev_day=0.0, mdev_night=0.0)

    df["reg_date"] = pd.to_datetime(df["reg_date"])
    if getattr(df["reg_date"].dt, "tz", None) is None:
        df["reg_date"] = df["reg_date"].dt.tz_localize("Asia/Seoul")
    else:
        df["reg_date"] = df["reg_date"].dt.tz_convert("Asia/Seoul")

    sunrise_ts = sunrise.tz_localize("Asia/Seoul") if sunrise.tzinfo is None else sunrise.tz_convert("Asia/Seoul")
    sunset_ts = sunset.tz_localize("Asia/Seoul") if sunset.tzinfo is None else sunset.tz_convert("Asia/Seoul")

    t_ns = df["reg_date"].astype("int64").to_numpy()
    x = df["in_temp"].to_numpy(dtype=float)

    t1 = t_ns[:-1]
    t2 = t_ns[1:]
    x1 = x[:-1]

    sunrise_ns = sunrise_ts.value
    sunset_ns = sunset_ts.value

    first_ts = df["reg_date"].iloc[0]
    base0_ts = first_ts.tz_localize("Asia/Seoul").normalize() if first_ts.tzinfo is None else first_ts.tz_convert("Asia/Seoul").normalize()
    base0_ns = base0_ts.value
    next0_ns = base0_ns + int(24 * 3600 * 1e9)

    def clip_ns(a_ns: np.ndarray, b_ns: np.ndarray, lo_ns: int, hi_ns: int) -> np.ndarray:
        lo_ = np.maximum(a_ns, lo_ns)
        hi_ = np.minimum(b_ns, hi_ns)
        z_ns = np.clip(hi_ - lo_, 0, None)
        return z_ns / 1e9

    dt_day = clip_ns(t1, t2, sunrise_ns, sunset_ns)
    dt_night = clip_ns(t1, t2, base0_ns, sunrise_ns) + clip_ns(t1, t2, sunset_ns, next0_ns)

    sum_day = dt_day.sum()
    sum_night = dt_night.sum()

    in_day = (x1 >= T_day - band_width) & (x1 <= T_day + band_width)
    in_nit = (x1 >= T_night - band_width) & (x1 <= T_night + band_width)

    cov_day = float((dt_day * in_day).sum() / sum_day) if sum_day > 0 else 0.0
    cov_night = float((dt_night * in_nit).sum() / sum_night) if sum_night > 0 else 0.0

    mdev_day = float((dt_day * (x1 - T_day)).sum() / sum_day) if sum_day > 0 else 0.0
    mdev_night = float((dt_night * (x1 - T_night)).sum() / sum_night) if sum_night > 0 else 0.0

    return dict(cov_day=cov_day, cov_night=cov_night, mdev_day=mdev_day, mdev_night=mdev_night)


def _score(kpi: Dict[str, float]) -> float:
    cov = 0.5 * kpi.get("cov_day", 0.0) + 0.5 * kpi.get("cov_night", 0.0)
    dev = 0.5 * abs(kpi.get("mdev_day", 0.0)) + 0.5 * abs(kpi.get("mdev_night", 0.0))
    return cov - 0.8 * dev


__all__ = ["PolicyState", "_clamp", "_compute_kpis", "_score"]
