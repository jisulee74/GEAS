from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import sys

_UPDATED_ROOT = Path(__file__).resolve().parents[1]
if str(_UPDATED_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPDATED_ROOT))

import numpy as np
import pandas as pd

from controller_layer.policy_common import PolicyState, _clamp, _compute_kpis, _score


class DailyPolicyUpdater:
    COV_TARGET = 0.90
    STEP_T = 0.2
    STEP_DB = 0.05
    STEP_PB = 0.2
    STEP_K = 0.05
    CAP_T = 1.0
    K_MIN, K_MAX = 0.5, 2.0
    DB_MIN, DB_MAX = 0.1, 0.6
    PB_MIN, PB_MAX = 1.0, 3.0

    def __init__(
        self,
        *,
        df_today: Optional[pd.DataFrame],
        out_light_info: Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | pd.NaT, np.ndarray, float],
        base_temp: Tuple[float, float],
        policy_state: PolicyState,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.df_today = df_today
        self.sunrise, self.sunset, self.eta, self.cs_sum, self.target_jcm2 = out_light_info
        self.day_temp, self.night_temp = base_temp
        self.st = policy_state
        self.params = params or {}
        self.daily_eval_hour = int(self.params.get("DAILY_EVAL_HOUR", 23))
        self.daily_eval_minute = int(self.params.get("DAILY_EVAL_MINUTE", 55))
        self.force_daily_eval = bool(self.params.get("FORCE_DAILY_EVAL", False))

    def _update(self, kpi: Dict[str, float], sunlight_ratio: float) -> PolicyState:
        st = self.st

        if st.fcu_mode == "cool":
            dT_day = _clamp(-kpi["mdev_day"], -self.STEP_T, 0.0)
            dT_night = _clamp(-kpi["mdev_night"], -self.STEP_T, 0.0)
        else:
            dT_day = _clamp(-kpi["mdev_day"], 0.0, self.STEP_T)
            dT_night = _clamp(-kpi["mdev_night"], 0.0, self.STEP_T)

        st.T_day_bias = _clamp(st.T_day_bias + dT_day, -self.CAP_T, self.CAP_T)
        st.T_night_bias = _clamp(st.T_night_bias + dT_night, -self.CAP_T, self.CAP_T)

        def tune(DB: float, PB: float, cov: float, mdev: float) -> tuple[float, float]:
            if cov < self.COV_TARGET and abs(mdev) < 0.3:
                DB = _clamp(DB + self.STEP_DB, self.DB_MIN, self.DB_MAX)
            if cov < self.COV_TARGET and abs(mdev) > 0.4:
                PB = _clamp(PB - self.STEP_PB, self.PB_MIN, self.PB_MAX)
            return DB, PB

        st.DB_day, st.PB_day = tune(st.DB_day, st.PB_day, kpi["cov_day"], kpi["mdev_day"])
        st.DB_night, st.PB_night = tune(st.DB_night, st.PB_night, kpi["cov_night"], kpi["mdev_night"])

        st.K_window_day = _clamp(
            st.K_window_day + _clamp(0.15 * kpi["mdev_day"], -self.STEP_K, self.STEP_K),
            self.K_MIN,
            self.K_MAX,
        )
        st.K_window_night = _clamp(
            st.K_window_night + _clamp(0.15 * kpi["mdev_night"], -self.STEP_K, self.STEP_K),
            self.K_MIN,
            self.K_MAX,
        )

        if (kpi["mdev_day"] > +0.6) and (sunlight_ratio > 1.1):
            st.alpha_curtain = _clamp(st.alpha_curtain + 0.1, 0.5, 2.0)
        elif (kpi["mdev_day"] < -0.6) and (sunlight_ratio < 0.9):
            st.alpha_curtain = _clamp(st.alpha_curtain - 0.1, 0.5, 2.0)

        return st

    def _current_kpi(self) -> Dict[str, float]:
        if self.df_today is None or self.df_today.empty:
            return {}
        t_day = self.day_temp + self.st.T_day_bias
        t_night = self.night_temp + self.st.T_night_bias
        return _compute_kpis(
            self.df_today,
            self.sunrise,
            self.sunset,
            t_day,
            t_night,
            band_width=1.0,
        )

    def _sunlight_ratio(self) -> float:
        if self.df_today is None or self.df_today.empty:
            return 0.0
        try:
            return float(self.df_today["out_light_sum"].iloc[-1]) / float(self.target_jcm2 or 1.0)
        except Exception:
            return 0.0

    def _eval_date(self) -> Optional[str]:
        if self.df_today is None or self.df_today.empty:
            return None
        ts = pd.Timestamp(self.df_today["reg_date"].iloc[-1])
        return ts.date().isoformat()

    def _is_batch_due(self) -> bool:
        if self.force_daily_eval:
            return True
        if self.df_today is None or self.df_today.empty:
            return False
        ts = pd.Timestamp(self.df_today["reg_date"].iloc[-1])
        cutoff = ts.normalize() + pd.Timedelta(hours=self.daily_eval_hour, minutes=self.daily_eval_minute)
        eval_date = ts.date().isoformat()
        if self.st.last_daily_eval_date == eval_date:
            return False
        return ts >= cutoff

    def run(self) -> Dict[str, Any]:
        if self.df_today is None or self.df_today.empty:
            return {
                "updated": False,
                "policy": self.st.to_dict(),
                "kpi": {},
                "intraday_kpi": {},
                "sunlight_ratio": 0.0,
                "score": self.st.last_score,
                "batch_due": False,
                "batch_executed": False,
                "evaluation_mode": "no_data",
            }

        intraday_kpi = self._current_kpi()
        sr = self._sunlight_ratio()
        eval_date = self._eval_date()

        if not self._is_batch_due():
            return {
                "updated": False,
                "policy": self.st.to_dict(),
                "kpi": intraday_kpi,
                "intraday_kpi": intraday_kpi,
                "sunlight_ratio": sr,
                "score": self.st.last_score,
                "batch_due": False,
                "batch_executed": False,
                "evaluation_mode": "intraday_monitoring",
            }

        today_score = _score(intraday_kpi)
        eps = 1e-4

        if (self.st.last_score is not None) and (today_score < self.st.last_score - eps):
            accepted = False
            self.st.last_daily_eval_date = eval_date
            final_state = self.st
            final_score = self.st.last_score
        else:
            accepted = True
            updated_state = self._update(intraday_kpi, sunlight_ratio=sr)
            updated_state.last_score = today_score
            updated_state.last_daily_eval_date = eval_date
            self.st = updated_state
            final_state = updated_state
            final_score = today_score

        return {
            "updated": accepted,
            "policy": final_state.to_dict(),
            "kpi": intraday_kpi,
            "intraday_kpi": intraday_kpi,
            "sunlight_ratio": sr,
            "score": final_score,
            "batch_due": True,
            "batch_executed": True,
            "evaluation_mode": "daily_batch",
        }


__all__ = ["DailyPolicyUpdater"]
