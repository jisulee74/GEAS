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


def _max_consecutive_true(mask: pd.Series) -> int:
    values = pd.Series(mask).fillna(False).astype(bool).to_numpy()
    best = cur = 0
    for flag in values:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


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
        df_state: Optional[pd.DataFrame] = None,
        df_actuation: Optional[pd.DataFrame] = None,
    ) -> None:
        self.df_today = df_today
        self.df_state = df_state
        self.df_actuation = df_actuation
        self.sunrise, self.sunset, self.eta, self.cs_sum, self.target_jcm2 = out_light_info
        self.day_temp, self.night_temp = base_temp
        self.st = policy_state
        self.params = params or {}
        self.daily_eval_hour = int(self.params.get("DAILY_EVAL_HOUR", 10))
        self.daily_eval_minute = int(self.params.get("DAILY_EVAL_MINUTE", 0))
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

    def _state_frame(self) -> pd.DataFrame:
        if self.df_state is None or self.df_state.empty:
            return pd.DataFrame()
        df = self.df_state.copy()
        df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
        df = df.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)
        return df

    def _raw_frame(self) -> pd.DataFrame:
        if self.df_today is None or self.df_today.empty:
            return pd.DataFrame()
        df = self.df_today.copy()
        df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
        df = df.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)
        return df

    def _actuation_frame(self) -> pd.DataFrame:
        if self.df_actuation is None or self.df_actuation.empty:
            return pd.DataFrame()
        df = self.df_actuation.copy()
        df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
        df = df.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)
        return df

    def _merge_state_with_raw(self) -> pd.DataFrame:
        state = self._state_frame()
        raw = self._raw_frame()
        if state.empty:
            return pd.DataFrame()
        if raw.empty:
            return state
        cols = [c for c in ["reg_date", "in_temp", "in_hum", "out_light_sum"] if c in raw.columns]
        raw = raw[cols].copy()
        return pd.merge_asof(
            state.sort_values("reg_date"),
            raw.sort_values("reg_date"),
            on="reg_date",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=5),
        )

    def _compute_state_metrics(self) -> Dict[str, float]:
        df = self._merge_state_with_raw()
        if df.empty or "reg_date" not in df.columns:
            return {}
        if "in_temp" not in df.columns or "T_current_eff" not in df.columns:
            return {}

        t = pd.to_datetime(df["reg_date"])
        if len(t) < 2:
            return {}

        t_ns = t.astype("int64").to_numpy()
        x = pd.to_numeric(df["in_temp"], errors="coerce").to_numpy(dtype=float)
        target = pd.to_numeric(df.get("T_current_eff"), errors="coerce").to_numpy(dtype=float)
        pb = pd.to_numeric(df.get("PB_used_now", 1.0), errors="coerce")
        if isinstance(pb, pd.Series):
            pb = pb.fillna(1.0).to_numpy(dtype=float)
        else:
            pb = np.full(len(df), 1.0)
        day_mask = pd.Series(df.get("is_daytime", 0)).fillna(0).astype(int).to_numpy(dtype=int)

        dt_sec = np.diff(t_ns) / 1e9
        x1 = x[:-1]
        target1 = target[:-1]
        pb1 = pb[:-1]
        day1 = day_mask[:-1].astype(bool)
        valid = np.isfinite(x1) & np.isfinite(target1) & np.isfinite(pb1)
        if not np.any(valid):
            return {}

        dt_sec = dt_sec[valid]
        x1 = x1[valid]
        target1 = target1[valid]
        pb1 = pb1[valid]
        day1 = day1[valid]
        night1 = ~day1

        temp_in_band = np.abs(x1 - target1) <= np.maximum(pb1, 1e-6)
        sum_day = dt_sec[day1].sum()
        sum_night = dt_sec[night1].sum()
        cov_day = float((dt_sec[day1] * temp_in_band[day1]).sum() / sum_day) if sum_day > 0 else 0.0
        cov_night = float((dt_sec[night1] * temp_in_band[night1]).sum() / sum_night) if sum_night > 0 else 0.0
        mdev_day = float((dt_sec[day1] * (x1[day1] - target1[day1])).sum() / sum_day) if sum_day > 0 else 0.0
        mdev_night = float((dt_sec[night1] * (x1[night1] - target1[night1])).sum() / sum_night) if sum_night > 0 else 0.0

        temp_viol = ~temp_in_band
        cond_viol = (pd.to_numeric(df.get("delta_cond"), errors="coerce").fillna(np.inf) < 0.8) | (pd.Series(df.get("cond_risk_10m", 0)).fillna(0).astype(int) == 1)
        rh_viol = pd.to_numeric(df.get("rh90_min_1h", 0), errors="coerce").fillna(0.0) > 0.0
        vpd_viol = (pd.to_numeric(df.get("vpd"), errors="coerce").fillna(np.inf) < 0.5) | (pd.to_numeric(df.get("vpdlo_min_1h", 0), errors="coerce").fillna(0.0) > 0.0)

        act = self._actuation_frame()
        hv_ineff_rate = 0.0
        if not act.empty:
            heater_on = (pd.to_numeric(act.get("pred_heater", 0), errors="coerce").fillna(0.0) > 0) | ((act.get("fcu_mode") == "heat") & (act.get("fcu_state") == "on"))
            vent_on = (pd.to_numeric(act.get("window_pct", 0), errors="coerce").fillna(0.0) > 0) | (pd.to_numeric(act.get("pred_ltw", 0), errors="coerce").fillna(0.0) > 0) | (pd.to_numeric(act.get("pred_rtw", 0), errors="coerce").fillna(0.0) > 0)
            hv_ineff_rate = float((heater_on & vent_on).mean())

        return {
            "cov_day": cov_day,
            "cov_night": cov_night,
            "mdev_day": mdev_day,
            "mdev_night": mdev_night,
            "temp_viol_rate": float(pd.Series(temp_viol).mean()),
            "cond_viol_rate": float(pd.Series(cond_viol).mean()),
            "rh_viol_rate": float(pd.Series(rh_viol).mean()),
            "vpd_viol_rate": float(pd.Series(vpd_viol).mean()),
            "hv_ineff_rate": hv_ineff_rate,
            "max_consec_temp_viol": _max_consecutive_true(pd.Series(temp_viol)),
            "max_consec_cond_viol": _max_consecutive_true(pd.Series(cond_viol)),
            "max_consec_rh_viol": _max_consecutive_true(pd.Series(rh_viol)),
            "max_consec_vpd_viol": _max_consecutive_true(pd.Series(vpd_viol)),
        }

    def _current_kpi(self) -> Dict[str, float]:
        state_metrics = self._compute_state_metrics()
        if state_metrics:
            return state_metrics
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
        state = self._state_frame()
        if not state.empty and "sunlight_ratio" in state.columns:
            try:
                val = pd.to_numeric(state["sunlight_ratio"], errors="coerce").dropna()
                if not val.empty:
                    return float(val.iloc[-1])
            except Exception:
                pass
        if self.df_today is None or self.df_today.empty:
            return 0.0
        try:
            return float(self.df_today["out_light_sum"].iloc[-1]) / float(self.target_jcm2 or 1.0)
        except Exception:
            return 0.0

    def _eval_date(self) -> Optional[str]:
        state = self._state_frame()
        if not state.empty:
            return pd.Timestamp(state["reg_date"].iloc[-1]).date().isoformat()
        if self.df_today is None or self.df_today.empty:
            return None
        ts = pd.Timestamp(self.df_today["reg_date"].iloc[-1])
        return ts.date().isoformat()

    def _is_batch_due(self) -> bool:
        if self.force_daily_eval:
            return True
        state = self._state_frame()
        if not state.empty:
            ts = pd.Timestamp(state["reg_date"].iloc[-1])
        elif self.df_today is None or self.df_today.empty:
            return False
        else:
            ts = pd.Timestamp(self.df_today["reg_date"].iloc[-1])
        cutoff = ts.normalize() + pd.Timedelta(hours=self.daily_eval_hour, minutes=self.daily_eval_minute)
        eval_date = ts.date().isoformat()
        if self.st.last_daily_eval_date == eval_date:
            return False
        return ts >= cutoff

    def run(self) -> Dict[str, Any]:
        if (self.df_today is None or self.df_today.empty) and (self.df_state is None or self.df_state.empty):
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
