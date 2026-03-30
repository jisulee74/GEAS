from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from physics_store import PhysicsStore

import numpy as np
import pandas as pd

from policy.controller import (
    PolicyState,
    PhysicsDefaults,
    PhysicsContext,
    SafetyState,
    _ach_table,
    _clamp,
    _compute_kpis,
    _dead_pb,
    _is_daytime,
    _mk_phys_context,
    _mk_safety,
    _score,
)


WINDOW_PERCENT_CANDIDATES: Tuple[int, ...] = tuple(range(0, 101, 10))
SHADE_PERCENT_CANDIDATES: Tuple[int, ...] = tuple(range(0, 101, 10))
THERMAL_PERCENT_CANDIDATES: Tuple[int, ...] = tuple(range(0, 101, 10))
FCU_STATE_CANDIDATES: Tuple[str, ...] = ("off", "on")
FAN_STATE_CANDIDATES: Tuple[str, ...] = ("off", "on")


@dataclass(frozen=True)
class CandidateAction:
    fcu: Tuple[str, str]
    window: Tuple[int, str, str]
    curtain: Tuple[str, int]
    thermal_curtain: Tuple[str, int]
    fan: Tuple[str]


@dataclass(frozen=True)
class ScoredCandidate:
    action: CandidateAction
    cost: float
    detail: Dict[str, float]


class _OuterDoc:
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
        df_today: Optional[pd.DataFrame],
        out_light_info: Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | pd.NaT, np.ndarray, float],
        base_temp: Tuple[float, float],
        st: PolicyState,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.df_today = df_today
        self.sunrise, self.sunset, self.eta, self.cs, self.target_jcm2 = out_light_info
        self.day_temp, self.night_temp = base_temp
        self.st = st
        self.params = params or {}
        self.daily_eval_hour = int(self.params.get('DAILY_EVAL_HOUR', 23))
        self.daily_eval_minute = int(self.params.get('DAILY_EVAL_MINUTE', 55))
        self.force_daily_eval = bool(self.params.get('FORCE_DAILY_EVAL', False))

    def _update(self, kpi: Dict[str, float], sunlight_ratio: float) -> PolicyState:
        st = self.st

        if st.fcu_mode == 'cool':
            dT_day = _clamp(-kpi['mdev_day'], -self.STEP_T, 0.0)
            dT_night = _clamp(-kpi['mdev_night'], -self.STEP_T, 0.0)
        else:
            dT_day = _clamp(-kpi['mdev_day'], 0.0, self.STEP_T)
            dT_night = _clamp(-kpi['mdev_night'], 0.0, self.STEP_T)

        st.T_day_bias = _clamp(st.T_day_bias + dT_day, -self.CAP_T, self.CAP_T)
        st.T_night_bias = _clamp(st.T_night_bias + dT_night, -self.CAP_T, self.CAP_T)

        def tune(DB, PB, cov, mdev):
            if cov < self.COV_TARGET and abs(mdev) < 0.3:
                DB = _clamp(DB + self.STEP_DB, self.DB_MIN, self.DB_MAX)
            if cov < self.COV_TARGET and abs(mdev) > 0.4:
                PB = _clamp(PB - self.STEP_PB, self.PB_MIN, self.PB_MAX)
            return DB, PB

        st.DB_day, st.PB_day = tune(st.DB_day, st.PB_day, kpi['cov_day'], kpi['mdev_day'])
        st.DB_night, st.PB_night = tune(st.DB_night, st.PB_night, kpi['cov_night'], kpi['mdev_night'])

        st.K_window_day = _clamp(
            st.K_window_day + _clamp(0.15 * kpi['mdev_day'], -self.STEP_K, self.STEP_K),
            self.K_MIN,
            self.K_MAX,
        )
        st.K_window_night = _clamp(
            st.K_window_night + _clamp(0.15 * kpi['mdev_night'], -self.STEP_K, self.STEP_K),
            self.K_MIN,
            self.K_MAX,
        )

        if (kpi['mdev_day'] > +0.6) and (sunlight_ratio > 1.1):
            st.alpha_curtain = _clamp(st.alpha_curtain + 0.1, 0.5, 2.0)
        elif (kpi['mdev_day'] < -0.6) and (sunlight_ratio < 0.9):
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
            return float(self.df_today['out_light_sum'].iloc[-1]) / float(self.target_jcm2 or 1.0)
        except Exception:
            return 0.0

    def _eval_date(self) -> Optional[str]:
        if self.df_today is None or self.df_today.empty:
            return None
        ts = pd.Timestamp(self.df_today['reg_date'].iloc[-1])
        return ts.date().isoformat()

    def _is_batch_due(self) -> bool:
        if self.force_daily_eval:
            return True
        if self.df_today is None or self.df_today.empty:
            return False
        ts = pd.Timestamp(self.df_today['reg_date'].iloc[-1])
        cutoff = ts.normalize() + pd.Timedelta(hours=self.daily_eval_hour, minutes=self.daily_eval_minute)
        eval_date = ts.date().isoformat()
        if getattr(self.st, 'last_daily_eval_date', None) == eval_date:
            return False
        return ts >= cutoff

    def run(self) -> Dict[str, Any]:
        if self.df_today is None or self.df_today.empty:
            return {
                'updated': False,
                'policy': self.st.to_dict(),
                'kpi': {},
                'intraday_kpi': {},
                'sunlight_ratio': 0.0,
                'score': self.st.last_score,
                'batch_due': False,
                'batch_executed': False,
                'evaluation_mode': 'no_data',
            }

        intraday_kpi = self._current_kpi()
        sr = self._sunlight_ratio()
        eval_date = self._eval_date()

        if not self._is_batch_due():
            return {
                'updated': False,
                'policy': self.st.to_dict(),
                'kpi': intraday_kpi,
                'intraday_kpi': intraday_kpi,
                'sunlight_ratio': sr,
                'score': self.st.last_score,
                'batch_due': False,
                'batch_executed': False,
                'evaluation_mode': 'intraday_monitoring',
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
            'updated': accepted,
            'policy': final_state.to_dict(),
            'kpi': intraday_kpi,
            'intraday_kpi': intraday_kpi,
            'sunlight_ratio': sr,
            'score': final_score,
            'batch_due': True,
            'batch_executed': True,
            'evaluation_mode': 'daily_batch',
        }


class _InnerDoc:
    WIND_CAP_TH: float = 5.0
    WIND_CAP_OPEN: int = 20
    ACH_MIN_DAY: float = 0.10
    ACH_MIN_NIGHT: float = 0.05
    RAMP_LIMIT: int = 15

    def __init__(
        self,
        latest: pd.DataFrame,
        out_light_info: Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | pd.NaT, np.ndarray, float],
        base_temp: Tuple[float, float],
        st: PolicyState,
        phys: PhysicsContext,
        safety: SafetyState,
        defaults: PhysicsDefaults,
        params: Dict[str, Any],
        agro_kpis: Optional[Dict[str, Any]] = None,
    ):
        self.row = latest.iloc[0]
        self.sunrise, self.sunset, self.eta, self.cs_sum, self.target_jcm2 = out_light_info
        self.day_temp, self.night_temp = base_temp
        self.st = st
        self.phys = phys
        self.safety = safety
        self.defaults = defaults
        self.params = params or {}
        self.kpis = agro_kpis or {}

        self.WIND_CAP_TH = float(self.params.get("WIND_CAP_TH", self.WIND_CAP_TH))
        self.WIND_CAP_OPEN = int(self.params.get("WIND_CAP_OPEN", self.WIND_CAP_OPEN))
        self.ACH_MIN_DAY = float(self.params.get("ACH_MIN_DAY", self.ACH_MIN_DAY))
        self.ACH_MIN_NIGHT = float(self.params.get("ACH_MIN_NIGHT", self.ACH_MIN_NIGHT))
        self.RAMP_LIMIT = int(self.params.get("RAMP_LIMIT", self.RAMP_LIMIT))

        try:
            ach_day_kpi = float(self.kpis.get("ACH_min_day", np.nan))
            if np.isfinite(ach_day_kpi) and ach_day_kpi > 0:
                self.ACH_MIN_DAY = ach_day_kpi
        except Exception:
            pass

        try:
            ach_night_kpi = float(self.kpis.get("ACH_min_night", np.nan))
            if np.isfinite(ach_night_kpi) and ach_night_kpi > 0:
                self.ACH_MIN_NIGHT = ach_night_kpi
        except Exception:
            pass

        self.now = pd.Timestamp(self.row["reg_date"])
        self.in_temp = float(self.row.get("in_temp", np.nan)) if pd.notna(self.row.get("in_temp", np.nan)) else np.nan
        self.out_temp = float(self.row.get("out_temp", np.nan)) if pd.notna(self.row.get("out_temp", np.nan)) else np.nan
        self.out_rain = float(self.row.get("out_rain", 0.0))
        self.out_light_sum = float(self.row.get("out_light_sum", 0.0))
        self.out_light = float(self.row.get("out_light", 0.0))
        self.wind_speed = float(self.row.get("wind_speed", np.nan)) if "wind_speed" in self.row else np.nan
        self.window_pct_fb = float(self.row.get("window_pct", np.nan)) if "window_pct" in self.row else np.nan

        self.sum_ratio = np.nan
        self.dTcond_kpi = None
        self.cond_risk_10m = 0
        try:
            self.sum_ratio = float(self.kpis.get("SumRatio", np.nan))
        except Exception:
            pass
        try:
            v = float(self.kpis.get("dTcond", np.nan))
            if np.isfinite(v):
                self.dTcond_kpi = v
        except Exception:
            pass
        try:
            self.cond_risk_10m = int(self.kpis.get("cond_risk_10m", 0) or 0)
        except Exception:
            self.cond_risk_10m = 0

        self.light_eta = self.kpis.get("LightETA", self.eta)

        self.physics_case = str(self.params.get("physics_case", "default"))
        self.physics_model_name = str(self.params.get("physics_model_name", "default"))
        self.physics_store = PhysicsStore(self.params.get("physics_store_path"))
        self.store_record = self.physics_store.get_record(
            farm_sn=self.params.get("farm_sn"),
            stage_name=self.params.get("stage_name"),
            data_case=self.physics_case,
            model_name=self.physics_model_name,
        )
        self.store_phys = self._resolve_store_physics_params()

        self._last_selected: Optional[ScoredCandidate] = None
        self._last_counts: Dict[str, int] = {"generated": 0, "feasible": 0}

    def _targets(self) -> Tuple[bool, float, float, float]:
        is_day = _is_daytime(self.now, self.sunrise, self.sunset)
        t_day_eff = self.day_temp + self.st.T_day_bias
        t_night_eff = self.night_temp + self.st.T_night_bias
        t_now = t_day_eff if is_day else t_night_eff
        return is_day, t_day_eff, t_night_eff, t_now

    def _resolve_store_physics_params(self) -> Dict[str, float]:
        ach_coef = self.phys.ACH_coef or {}
        defaults = {
            "UA": float(self.phys.UA if self.phys.UA is not None else 2500.0),
            "C": float(self.phys.C if self.phys.C is not None else 8.0e5),
            "eta": float(self.phys.g_solar if self.phys.g_solar is not None else 0.12),
            "a0": float(ach_coef.get("a0", 0.02)),
            "a1": float(ach_coef.get("a1", 2.5)),
            "a2": float(ach_coef.get("a2", 0.15)),
            "a3": float(ach_coef.get("a3", 0.50)),
            "k_heat": float(self.params.get("physics_k_heat", 15000.0)),
            "k_cool": float(self.params.get("physics_k_cool", 15000.0)),
            "rho_cp": float(self.params.get("physics_rho_cp", 1200.0)),
            "A": float(self.params.get("greenhouse_area_m2", 200.0)),
            "V": float(self.params.get("greenhouse_volume_m3", 300.0)),
            "dt_sec": float(self.params.get("physics_dt_sec", 300.0)),
            "k_shade": float(self.params.get("physics_k_shade", 0.70)),
            "k_thermal": float(self.params.get("physics_k_thermal", 0.30)),
        }
        return self.physics_store.resolve_params(
            farm_sn=self.params.get("farm_sn"),
            stage_name=self.params.get("stage_name"),
            data_case=self.physics_case,
            model_name=self.physics_model_name,
            defaults=defaults,
        )

    def _effective_params(self) -> Dict[str, Any]:
        is_day, t_day_eff, t_night_eff, t_now = self._targets()
        db, pb = _dead_pb(is_day, self.st)
        selected = self._last_selected
        return {
            "timestamp": self.now,
            "is_daytime": bool(is_day),
            "fcu_mode": self.st.fcu_mode,
            "targets": {
                "T_day_eff": float(t_day_eff),
                "T_night_eff": float(t_night_eff),
                "T_current_eff": float(t_now),
            },
            "bands": {
                "DB_day": float(self.st.DB_day),
                "PB_day": float(self.st.PB_day),
                "DB_night": float(self.st.DB_night),
                "PB_night": float(self.st.PB_night),
                "DB_used_now": float(db),
                "PB_used_now": float(pb),
            },
            "candidate_search": {
                "generated": int(self._last_counts["generated"]),
                "feasible": int(self._last_counts["feasible"]),
                "selected_cost": None if selected is None else float(selected.cost),
                "selected_detail": {} if selected is None else dict(selected.detail),
            },
            "physics": {
                "UA": self.phys.UA,
                "C": self.phys.C,
                "g_solar": self.phys.g_solar,
                "ACH_model": self.phys.ACH_coef,
                "lambda": self.phys.lam,
                "mode": self.phys.mode,
                "store_key": None if self.store_record is None else self.store_record.key,
                "store_source": None if self.store_record is None else self.store_record.source,
                "store_case": self.physics_case,
                "store_model_name": self.physics_model_name,
                "store_params": dict(self.store_phys),
            },
            "safety": {
                "delta_cond": self.safety.delta_cond,
                "vpd": self.safety.vpd,
            },
        }

    def _ventilation_open_from_ach(self, target_ach: float) -> int:
        if self.phys.ACH_coef is None:
            return int(
                round(
                    next((x for x, y in self.defaults.ACH_table if y >= target_ach), 100)
                )
            )
        a0 = self.phys.ACH_coef.get("a0", 0.0)
        a1 = self.phys.ACH_coef.get("a1", 0.0)
        a2 = self.phys.ACH_coef.get("a2", 0.0)
        a3 = self.phys.ACH_coef.get("a3", 0.0)
        w = float(self.wind_speed) if np.isfinite(self.wind_speed) else 0.5
        denom = a1 + a3 * w
        if abs(denom) < 1e-9:
            return 0
        x = (target_ach - a0 - a2 * w) / denom
        return int(_clamp(x * 100.0, 0.0, 100.0))

    def _safety_caps(self, is_day: bool) -> Dict[str, float]:
        cap_open = 100 if is_day else 20
        if self.out_rain > 0:
            return {"open_cap": 0, "ach_min": 0.0, "ramp": self.RAMP_LIMIT}

        if np.isfinite(self.wind_speed) and self.wind_speed > self.WIND_CAP_TH:
            cap_open = min(cap_open, self.WIND_CAP_OPEN)

        delta_cond = self.safety.delta_cond
        if delta_cond is None or not np.isfinite(delta_cond):
            delta_cond = self.dTcond_kpi
        if delta_cond is not None and np.isfinite(delta_cond):
            if delta_cond < 0.8 or self.cond_risk_10m == 1:
                cap_open = min(cap_open, 10)

        ach_min = self.ACH_MIN_DAY if is_day else self.ACH_MIN_NIGHT
        return {"open_cap": cap_open, "ach_min": ach_min, "ramp": self.RAMP_LIMIT}

    # Module 5-1) candidate generation
    def _generate_candidates(self) -> List[CandidateAction]:
        mode = "cool" if self.st.fcu_mode == "cool" else "heat"

        if np.isnan(self.in_temp):
            window_levels = [0]
            fcu_states = ["off"]
        else:
            window_levels = [int(v) for v in WINDOW_PERCENT_CANDIDATES]
            fcu_states = list(FCU_STATE_CANDIDATES)

        shade_levels = [int(v) for v in SHADE_PERCENT_CANDIDATES]
        thermal_levels = [int(v) for v in THERMAL_PERCENT_CANDIDATES]
        fan_states = list(FAN_STATE_CANDIDATES)

        actions: List[CandidateAction] = []
        for fcu_state, open_pct, shade_open, thermal_open, fan_state in product(
            fcu_states, window_levels, shade_levels, thermal_levels, fan_states
        ):
            actions.append(
                CandidateAction(
                    fcu=(mode, fcu_state),
                    window=(
                        int(open_pct),
                        "main",
                        "OPEN" if int(open_pct) > 0 else "HOLD",
                    ),
                    curtain=("shade", int(shade_open)),
                    thermal_curtain=("thermal", int(thermal_open)),
                    fan=(fan_state,),
                )
            )
        self._last_counts["generated"] = len(actions)
        return actions

    # Module 5-2) candidate pruning
    def _is_feasible(self, candidate: CandidateAction) -> bool:
        is_day, _, _, _ = self._targets()
        caps = self._safety_caps(is_day)

        window_pct = int(candidate.window[0])
        shade_open = int(candidate.curtain[1])
        thermal_open = int(candidate.thermal_curtain[1])
        fcu_state = candidate.fcu[1]
        fan_state = candidate.fan[0]

        if self.out_rain > 0 and window_pct > 0:
            return False
        if window_pct > caps["open_cap"]:
            return False

        min_open = 0
        if caps["ach_min"] > 0:
            min_open = int(_clamp(self._ventilation_open_from_ach(caps["ach_min"]), 0, 100))
        if window_pct > 0 and window_pct < min_open:
            return False

        prev_open = int(_clamp(self.window_pct_fb, 0, 100)) if np.isfinite(self.window_pct_fb) else None
        if prev_open is not None:
            step = max(int(caps["ramp"]), 1)
            if abs(window_pct - prev_open) > step:
                return False

        if not is_day and shade_open != 100:
            return False
        if is_day and thermal_open != 100:
            return False
        if fan_state == "off" and (fcu_state == "on" or window_pct >= 20):
            return False
        if fan_state == "on" and fcu_state == "off" and window_pct == 0:
            return False
        return True

    def _estimate_ach(self, window_pct: int) -> float:
        a0 = self.store_phys.get("a0")
        a1 = self.store_phys.get("a1")
        a2 = self.store_phys.get("a2")
        a3 = self.store_phys.get("a3")
        if any(v is None for v in (a0, a1, a2, a3)):
            if self.phys.ACH_coef is None:
                return float(_ach_table(window_pct, self.defaults.ACH_table))
            a0 = self.phys.ACH_coef.get("a0", 0.0)
            a1 = self.phys.ACH_coef.get("a1", 0.0)
            a2 = self.phys.ACH_coef.get("a2", 0.0)
            a3 = self.phys.ACH_coef.get("a3", 0.0)
        x = float(window_pct) / 100.0
        w = float(self.wind_speed) if np.isfinite(self.wind_speed) else 0.5
        return max(float(a0) + float(a1) * x + float(a2) * w + float(a3) * x * w, 0.0)

    def _predict_temp(self, candidate: CandidateAction) -> float:
        _, _, _, t_target = self._targets()
        if np.isnan(self.in_temp):
            return float(t_target)

        window_pct = int(candidate.window[0])
        shade_open = int(candidate.curtain[1])
        thermal_open = int(candidate.thermal_curtain[1])
        fcu_mode, fcu_state = candidate.fcu

        tin = float(self.in_temp)
        tout = float(self.out_temp) if np.isfinite(self.out_temp) else tin
        ach = self._estimate_ach(window_pct)

        ua = max(float(self.store_phys.get("UA", self.phys.UA or 2500.0)), 1e-6)
        cap = max(float(self.store_phys.get("C", self.phys.C or 8.0e5)), 1.0)
        eta = float(self.store_phys.get("eta", self.phys.g_solar or 0.12))
        k_heat = float(self.store_phys.get("k_heat", 15000.0))
        k_cool = float(self.store_phys.get("k_cool", 15000.0))
        rho_cp = float(self.store_phys.get("rho_cp", 1200.0))
        area_m2 = float(self.store_phys.get("A", 200.0))
        volume_m3 = float(self.store_phys.get("V", 300.0))
        dt_sec = float(self.store_phys.get("dt_sec", 300.0))
        k_shade = float(self.store_phys.get("k_shade", 0.70))
        k_thermal = float(self.store_phys.get("k_thermal", 0.30))

        shade_fraction = (100.0 - shade_open) / 100.0
        thermal_fraction = (100.0 - thermal_open) / 100.0
        solar_transmission = 1.0 - k_shade * shade_fraction
        solar_transmission = _clamp(solar_transmission, 0.0, 1.0)
        thermal_loss_factor = 1.0 - k_thermal * thermal_fraction
        thermal_loss_factor = _clamp(thermal_loss_factor, 0.1, 1.0)

        q_trans = ua * thermal_loss_factor * (tout - tin)
        q_vent = rho_cp * volume_m3 * (ach / 3600.0) * (tout - tin)
        q_solar = eta * area_m2 * max(self.out_light, 0.0) * solar_transmission

        q_fcu = 0.0
        if fcu_state == "on":
            if fcu_mode == "heat":
                q_fcu = k_heat
            else:
                q_fcu = -k_cool

        tin_next = tin + (dt_sec / cap) * (q_trans + q_vent + q_solar + q_fcu)
        return float(tin_next)

    def _predict_energy(self, candidate: CandidateAction) -> float:
        window_pct = int(candidate.window[0])
        shade_open = int(candidate.curtain[1])
        thermal_open = int(candidate.thermal_curtain[1])
        _, fcu_state = candidate.fcu
        fan_state = candidate.fan[0]

        e_fcu = 1.0 if fcu_state == "on" else 0.0
        e_fan = 0.3 if fan_state == "on" else 0.0
        e_window = 0.1 * (window_pct / 100.0)
        e_curtain = 0.05 * ((100 - shade_open) / 100.0)
        e_thermal = 0.05 * ((100 - thermal_open) / 100.0)
        return float(e_fcu + e_fan + e_window + e_curtain + e_thermal)

    def _constraint_penalty(self, candidate: CandidateAction) -> float:
        is_day, _, t_night_eff, _ = self._targets()
        window_pct = int(candidate.window[0])
        shade_open = int(candidate.curtain[1])
        thermal_open = int(candidate.thermal_curtain[1])
        fcu_state = candidate.fcu[1]
        fan_state = candidate.fan[0]

        penalty = 0.0
        if self.out_rain > 0 and window_pct > 0:
            penalty += 1.0

        caps = self._safety_caps(is_day)
        if window_pct > caps["open_cap"]:
            penalty += (window_pct - caps["open_cap"]) / 10.0

        ach_est = self._estimate_ach(window_pct)
        min_ach = self.ACH_MIN_DAY if is_day else self.ACH_MIN_NIGHT
        if ach_est < min_ach:
            penalty += 2.0 * (min_ach - ach_est)

        if fan_state == "off" and (fcu_state == "on" or window_pct >= 20):
            penalty += 1.0
        if fan_state == "on" and fcu_state == "off" and window_pct == 0:
            penalty += 0.5

        if not is_day and shade_open != 100:
            penalty += abs(shade_open - 100) / 100.0
        if is_day and thermal_open != 100:
            penalty += abs(thermal_open - 100) / 100.0

        delta_cond = self.safety.delta_cond
        if delta_cond is None or not np.isfinite(delta_cond):
            delta_cond = self.dTcond_kpi
        if delta_cond is not None and np.isfinite(delta_cond) and delta_cond < 0.8 and window_pct > 10:
            penalty += (window_pct - 10) / 10.0

        if (not is_day) and np.isfinite(self.in_temp):
            cold_gap = max(t_night_eff - float(self.in_temp), 0.0)
            if cold_gap > 0 and thermal_open > 50:
                penalty += cold_gap * (thermal_open - 50) / 100.0

        if np.isfinite(self.window_pct_fb):
            penalty += 0.1 * abs(window_pct - float(self.window_pct_fb)) / max(float(self.RAMP_LIMIT), 1.0)

        return float(max(penalty, 0.0))

    def _cost(self, candidate: CandidateAction) -> ScoredCandidate:
        _, _, _, t_target = self._targets()
        t_pred = self._predict_temp(candidate)
        e_pred = self._predict_energy(candidate)
        r = self._constraint_penalty(candidate)

        w1 = float(self.params.get("W_TEMP", 1.0))
        w2 = float(self.params.get("W_ENERGY", 0.2))
        w3 = float(self.params.get("W_PENALTY", 5.0))

        temp_error = abs(float(t_pred) - float(t_target))
        total = w1 * temp_error + w2 * e_pred + w3 * r
        detail = {
            "T_pred": float(t_pred),
            "T_target": float(t_target),
            "temp_error": float(temp_error),
            "E_pred": float(e_pred),
            "r": float(r),
            "w1": float(w1),
            "w2": float(w2),
            "w3": float(w3),
        }
        return ScoredCandidate(action=candidate, cost=float(total), detail=detail)

    def _select(self) -> CandidateAction:
        generated = self._generate_candidates()
        feasible = [cand for cand in generated if self._is_feasible(cand)]
        if not feasible:
            feasible = generated
        self._last_counts["feasible"] = len(feasible)

        scored = [self._cost(cand) for cand in feasible]
        winner = min(scored, key=lambda item: item.cost)
        self._last_selected = winner
        return winner.action

    def control_fcu(self) -> Tuple[str, str]:
        return self._select().fcu

    def control_window(self) -> Tuple[int, str, str]:
        return self._select().window

    def control_curtain(self) -> Tuple[str, int]:
        return self._select().curtain

    def control_thermal_curtain(self) -> Tuple[str, int]:
        return self._select().thermal_curtain

    def control_fan(self) -> Tuple[str]:
        return self._select().fan

    def run(self) -> Dict[str, Any]:
        action = self._select()
        return {
            "window": action.window,
            "curtain": action.curtain,
            "thermal_curtain": action.thermal_curtain,
            "fcu": action.fcu,
            "fan": action.fan,
            "effective_params": self._effective_params(),
        }


class Controller:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        out_light_info: Tuple[
            pd.Timestamp, pd.Timestamp, pd.Timestamp | pd.NaT, np.ndarray, float
        ],
        temp: Tuple[float, float],
        params: Dict[str, Any] | None = None,
        policy_state: PolicyState | None = None,
        df_today: Optional[pd.DataFrame] = None,
        derived_result: Optional[Dict[str, Any]] = None,
        phys_defaults: PhysicsDefaults | None = None,
        agro_kpis: Optional[Dict[str, Any]] = None,
    ):
        self.latest = dataframe.reset_index(drop=True)
        self.out_light_info = out_light_info
        self.base_temp = temp
        self.params = params or {}
        self.df_today = df_today
        self.defaults = phys_defaults or PhysicsDefaults()

        self.phys = _mk_phys_context(derived_result, self.defaults)
        self.safety = _mk_safety(derived_result)

        if policy_state is None:
            self.state = PolicyState()
        else:
            self.state = policy_state

        if "fcu_mode" in self.params:
            self.state.fcu_mode = "heat" if self.params["fcu_mode"] == "heat" else "cool"

        self.inner = _InnerDoc(
            self.latest,
            self.out_light_info,
            self.base_temp,
            self.state,
            self.phys,
            self.safety,
            self.defaults,
            self.params,
            agro_kpis=agro_kpis,
        )
        self.outer = _OuterDoc(self.df_today, self.out_light_info, self.base_temp, self.state, self.params)

    def control_window(self):
        return self.inner.control_window()

    def control_curtain(self):
        return self.inner.control_curtain()

    def control_fcu(self):
        return self.inner.control_fcu()

    def control_fan(self):
        return self.inner.control_fan()

    def run(self) -> Dict[str, Any]:
        inner_res = self.inner.run()
        outer_res = (
            self.outer.run()
            if (self.df_today is not None)
            else {"updated": False, "policy": self.state.to_dict(), "kpi": {}}
        )

        phys = self.phys
        lam = phys.lam or {}
        active = any(float(v) > 0 for v in lam.values()) or (phys.ACH_coef is not None)

        return {
            "window": inner_res["window"],
            "curtain": inner_res["curtain"],
            "fcu": inner_res["fcu"],
            "fan": inner_res["fan"],
            "effective_params": inner_res["effective_params"],
            "thermal_curtain": inner_res.get("thermal_curtain"),
            "outer": outer_res,
            "physics": {
                "mode": phys.mode,
                "active": bool(active),
                "UA": float(phys.UA),
                "C": float(phys.C),
                "g_solar": float(phys.g_solar),
                "ACH_model": bool(phys.ACH_coef is not None),
                "lambda": {k: float(v) for k, v in lam.items()},
            },
        }


__all__ = [
    "Controller",
    "PolicyState",
    "PhysicsDefaults",
    "PhysicsContext",
    "SafetyState",
    "_compute_kpis",
    "_score",
]
