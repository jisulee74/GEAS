from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
import sys

_UPDATED_ROOT = Path(__file__).resolve().parents[2]
if str(_UPDATED_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPDATED_ROOT))

from controller_layer.physics_store import PhysicsStore

import numpy as np
import pandas as pd

from controller_layer.policy_common import PolicyState, _clamp, _compute_kpis, _score
from policy.controller_legacy import (
    PhysicsDefaults,
    PhysicsContext,
    SafetyState,
    _ach_table,
    _dead_pb,
    _is_daytime,
    _mk_phys_context,
    _mk_safety,
)


WINDOW_PERCENT_CANDIDATES: Tuple[int, ...] = tuple(range(0, 101, 10))
SHADE_PERCENT_CANDIDATES: Tuple[int, ...] = tuple(range(0, 101, 10))
THERMAL_PERCENT_CANDIDATES: Tuple[int, ...] = tuple(range(0, 101, 10))
FCU_STATE_CANDIDATES: Tuple[str, ...] = ("off", "on")
FAN_STATE_CANDIDATES: Tuple[str, ...] = ("off", "on")

_NO_DATA_EVAL_API: Optional[Dict[str, Any]] = None


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


def sat_vp_kpa_no_data(Tc: float) -> float:
    return float(0.61078 * np.exp((17.2694 * Tc) / (Tc + 237.3)))


def dewpoint_c_no_data(Tc: float, RH: float) -> float:
    es = sat_vp_kpa_no_data(Tc)
    e = np.clip(RH * es, 1e-6, es)
    ln_ratio = np.log(e / 0.61078)
    return float((237.3 * ln_ratio) / (17.2694 - ln_ratio))


def vpd_kpa_no_data(Tc: float, RH: float) -> float:
    es = sat_vp_kpa_no_data(Tc)
    ea = np.clip(RH * es, 0, es)
    return float(np.maximum(0.0, es - ea))


def hinge_no_data(z: float) -> float:
    return float(np.maximum(0.0, z))


def sample_theta_no_data(rng=None) -> Dict[str, float]:
    if rng is None:
        rng = np.random.default_rng()
    return {
        "UA": float(rng.uniform(2000, 7000)),
        "C": float(rng.uniform(5e6, 4e7)),
        "eta": float(rng.uniform(0.2, 0.8)),
        "k_heat": float(rng.uniform(15000, 50000)),
        "a0": float(rng.uniform(0.05, 0.5)),
        "a1": float(rng.uniform(2.0, 10.0)),
        "a2": float(rng.uniform(0.3, 2.0)),
        "k_evap": float(rng.uniform(1e-6, 5e-6)),
        "k_photo": float(rng.uniform(1e-5, 5e-5)),
        "rho_cp": 1.2 * 1005,
    }


def ach_model_no_data(x_vent: float, wind: float, a0: float, a1: float, a2: float, ACH_max: float = 15.0) -> float:
    ach = a0 + a1 * x_vent + a2 * wind * x_vent
    return float(np.clip(ach, 0.0, ACH_max))


def _sat_vp_kpa_no_data_arr(Tc) -> np.ndarray:
    Tc_arr = np.asarray(Tc, dtype=float)
    return 0.61078 * np.exp((17.2694 * Tc_arr) / (Tc_arr + 237.3))


def _project_action_grid_no_data(prev_a: Dict[str, float], ramp_max: float = 0.15) -> Dict[str, np.ndarray]:
    grid = make_action_grid_no_data()
    u_heat = np.array([1.0 if float(g['u_heat']) > 0.5 else 0.0 for g in grid], dtype=float)
    u_co2 = np.array([1.0 if float(g['u_co2']) > 0.5 else 0.0 for g in grid], dtype=float)
    curtain = np.clip(np.array([float(g['curtain']) for g in grid], dtype=float), 0.0, 1.0)
    x_raw = np.clip(np.array([float(g['x_vent']) for g in grid], dtype=float), 0.0, 1.0)
    # 3.6.3 documents heat/vent mutual exclusion as a safety-rule example.
    x_raw = np.where(u_heat > 0.5, 0.0, x_raw)
    prev_x = float(prev_a['x_vent'])
    dx = np.clip(x_raw - prev_x, -ramp_max, ramp_max)
    x_vent = prev_x + dx
    return {
        'u_heat': u_heat,
        'u_co2': u_co2,
        'curtain': curtain,
        'x_vent': x_vent,
    }


def _step_dynamics_no_data_batch(
    state: Dict[str, float],
    u: Dict[str, float],
    actions: Dict[str, np.ndarray],
    theta: Dict[str, float],
    dt_min: int = 10,
    A: float = 200.0,
    V: float = 300.0,
    Imax: float = 650.0,
    T0: float = 5.0,
) -> Dict[str, np.ndarray]:
    dt = dt_min * 60.0
    Tin = float(state['Tin'])
    e_in = float(state['e_in'])
    CO2 = float(state['CO2'])

    x_vent = np.asarray(actions['x_vent'], dtype=float)
    curtain = np.asarray(actions['curtain'], dtype=float)
    u_heat = np.asarray(actions['u_heat'], dtype=float)
    u_co2 = np.asarray(actions['u_co2'], dtype=float)

    ACH = np.clip(float(theta['a0']) + float(theta['a1']) * x_vent + float(theta['a2']) * float(u['wind']) * x_vent, 0.0, 15.0)
    It_eff = float(u['It']) * (1.0 - curtain)

    Q_trans = float(theta['UA']) * (float(u['Tout']) - Tin)
    Q_solar = float(theta['eta']) * A * It_eff
    Q_heat = float(theta['k_heat']) * u_heat
    Q_vent = float(theta['rho_cp']) * V * (ACH / 3600.0) * (float(u['Tout']) - Tin)
    Tin_next = Tin + (dt / float(theta['C'])) * (Q_trans + Q_solar + Q_heat + Q_vent)

    e_out = float(u['RHout']) * sat_vp_kpa_no_data(float(u['Tout']))
    evap = float(theta['k_evap']) * (It_eff / Imax) * max(0.0, Tin - T0)
    e_next = np.maximum(0.05, e_in + dt * ((ACH / 3600.0) * (e_out - e_in) + evap))

    RHin = np.clip(e_next / _sat_vp_kpa_no_data_arr(Tin_next), 0.0, 1.0)
    VPD = np.maximum(0.0, _sat_vp_kpa_no_data_arr(Tin_next) - np.clip(RHin * _sat_vp_kpa_no_data_arr(Tin_next), 0.0, _sat_vp_kpa_no_data_arr(Tin_next)))

    gT = np.clip((Tin_next - 5.0) / 20.0, 0.0, 1.0)
    gV = np.clip(VPD / 1.2, 0.0, 1.0)
    uptake = float(theta['k_photo']) * It_eff * gT * gV * 1e6
    inj = 50.0 * u_co2
    CO2_next = np.maximum(300.0, CO2 + dt * ((ACH / 3600.0) * (float(u['CO2out']) - CO2) + inj - uptake))

    Q_ventloss = np.maximum(0.0, float(theta['rho_cp']) * V * (ACH / 3600.0) * np.maximum(0.0, Tin_next - float(u['Tout'])))
    return {
        'Tin': Tin_next,
        'e_in': e_next,
        'CO2': CO2_next,
        'ACH': ACH,
        'It_eff': It_eff,
        'Q_heat': Q_heat,
        'Q_ventloss': Q_ventloss,
        'RHin': RHin,
        'VPD': VPD,
    }


def step_dynamics_no_data(
    state: Dict[str, float],
    u: Dict[str, float],
    a: Dict[str, float],
    theta: Dict[str, float],
    dt_min: int = 10,
    A: float = 200.0,
    V: float = 300.0,
    Imax: float = 650.0,
    T0: float = 5.0,
) -> Dict[str, float]:
    batch = _step_dynamics_no_data_batch(
        state,
        u,
        {k: np.array([float(v)], dtype=float) for k, v in a.items()},
        theta,
        dt_min=dt_min,
        A=A,
        V=V,
        Imax=Imax,
        T0=T0,
    )
    return {key: float(np.asarray(value)[0]) for key, value in batch.items() if key not in ('RHin', 'VPD')}


def project_action_no_data(a: Dict[str, float], prev_a: Dict[str, float], ramp_max: float = 0.15) -> Dict[str, float]:
    projected = dict(a)
    projected['u_heat'] = 1.0 if float(projected['u_heat']) > 0.5 else 0.0
    projected['u_co2'] = 1.0 if float(projected['u_co2']) > 0.5 else 0.0
    projected['x_vent'] = float(np.clip(projected['x_vent'], 0.0, 1.0))
    projected['curtain'] = float(np.clip(projected['curtain'], 0.0, 1.0))
    if projected['u_heat'] > 0.5:
        projected['x_vent'] = 0.0
    dx = float(projected['x_vent']) - float(prev_a['x_vent'])
    dx = float(np.clip(dx, -ramp_max, ramp_max))
    projected['x_vent'] = float(prev_a['x_vent']) + dx
    return projected


def check_feasible_no_data(
    pred: Dict[str, float],
    u: Dict[str, float],
    a: Dict[str, float],
    theta: Dict[str, float],
    T_min: float = 12.0,
    T_max: float = 28.0,
    dTcond_min: float = 0.8,
    alpha: float = 0.25,
) -> bool:
    RHin = min(1.0, max(0.0, float(pred['e_in']) / sat_vp_kpa_no_data(float(pred['Tin']))))
    Td = dewpoint_c_no_data(float(pred['Tin']), RHin)
    dTcond = float(pred['Tin']) - Td
    c1 = T_min <= float(pred['Tin']) <= T_max
    c2 = dTcond >= dTcond_min
    c3 = float(a['x_vent']) <= 1e-9 if float(a['u_heat']) > 0.5 else True
    if float(a['u_heat']) > 0.5:
        c3 = c3 and float(pred['Q_ventloss']) <= alpha * max(1e-6, float(pred['Q_heat']))
    return bool(c1 and c2 and c3)


def make_action_grid_no_data() -> List[Dict[str, float]]:
    return [
        {'u_heat': uh, 'x_vent': xv, 'curtain': cu, 'u_co2': uco}
        for uh in [0.0, 1.0]
        for xv in [0.0, 0.1, 0.3, 0.6]
        for cu in [0.0, 0.6, 0.9]
        for uco in [0.0, 1.0]
    ]


def policy_optimal_no_data(
    state: Dict[str, float],
    u: Dict[str, float],
    prev_a: Dict[str, float],
    theta: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    Tref: float = 18.0,
    VPD_min: float = 0.30,
    T_min: float = 12.0,
    T_max: float = 28.0,
    dTcond_min: float = 0.8,
    alpha: float = 0.25,
    dt_min: int = 10,
    A: float = 200.0,
    V: float = 300.0,
    ramp_max: float = 0.15,
) -> Dict[str, float]:
    if weights is None:
        weights = {'wT': 1.0, 'wVPD': 1.0, 'wE': 1e-8, 'wDx': 0.2, 'wSlack': 50.0}

    actions = _project_action_grid_no_data(prev_a, ramp_max)
    pred = _step_dynamics_no_data_batch(state, u, actions, theta, dt_min=dt_min, A=A, V=V)

    RHin = pred['RHin']
    Tin_next = pred['Tin']
    es = _sat_vp_kpa_no_data_arr(Tin_next)
    e = np.clip(RHin * es, 1e-6, es)
    ln_ratio = np.log(e / 0.61078)
    Td = (237.3 * ln_ratio) / (17.2694 - ln_ratio)
    dTcond = Tin_next - Td

    feasible = (Tin_next >= T_min) & (Tin_next <= T_max) & (dTcond >= dTcond_min)
    heat_mask = actions['u_heat'] > 0.5
    feasible &= (~heat_mask) | (actions['x_vent'] <= 1e-9)
    heat_feasible = pred['Q_ventloss'] <= alpha * np.maximum(1e-6, pred['Q_heat'])
    feasible &= (~heat_mask) | heat_feasible

    J = np.full(Tin_next.shape, np.inf, dtype=float)
    if np.any(feasible):
        Jf = np.zeros(Tin_next.shape, dtype=float)
        Jf += float(weights['wT']) * (Tin_next - Tref) ** 2
        Jf += float(weights['wVPD']) * np.maximum(0.0, VPD_min - pred['VPD'])
        Jf += float(weights['wE']) * (pred['Q_heat'] + pred['Q_ventloss'])
        Jf += float(weights['wDx']) * np.abs(actions['x_vent'] - float(prev_a['x_vent']))
        slackT = np.maximum(0.0, T_min - Tin_next) + np.maximum(0.0, Tin_next - T_max)
        slackC = np.maximum(0.0, dTcond_min - dTcond)
        Jf += float(weights['wSlack']) * (slackT + slackC)
        J[feasible] = Jf[feasible]
        best_idx = int(np.argmin(J))
        return {
            'u_heat': float(actions['u_heat'][best_idx]),
            'x_vent': float(actions['x_vent'][best_idx]),
            'curtain': float(actions['curtain'][best_idx]),
            'u_co2': float(actions['u_co2'][best_idx]),
        }

    return project_action_no_data(
        {
            'u_heat': 1.0 if float(state['Tin']) < (Tref - 1.0) else 0.0,
            'x_vent': 0.0,
            'curtain': 0.6,
            'u_co2': 0.0,
        },
        prev_a,
        ramp_max,
    )


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
            model_name=self.physics_model_name,
            target_ts=self.now,
        )
        self.store_phys = self._resolve_store_physics_params()
        self.store_lambda = self._resolve_store_lambda_map()

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
            model_name=self.physics_model_name,
            target_ts=self.now,
            defaults=defaults,
        )

    def _resolve_store_lambda_map(self) -> Dict[str, float]:
        metadata = dict(self.store_record.metadata) if self.store_record is not None else {}
        approval = metadata.get("approval") if isinstance(metadata.get("approval"), dict) else {}
        record_params = dict(self.store_record.params) if self.store_record is not None else {}
        legacy_lambda = dict(self.phys.lam or {})

        def _lookup(*aliases: str, default: float = 0.0) -> float:
            for container_key in ("lambda", "lambdas"):
                container = metadata.get(container_key)
                if isinstance(container, dict):
                    for alias in aliases:
                        try:
                            return max(0.0, min(1.0, float(container.get(alias))))
                        except Exception:
                            pass
                elif container is not None:
                    try:
                        return max(0.0, min(1.0, float(container)))
                    except Exception:
                        pass

                container = approval.get(container_key) if isinstance(approval, dict) else None
                if isinstance(container, dict):
                    for alias in aliases:
                        try:
                            return max(0.0, min(1.0, float(container.get(alias))))
                        except Exception:
                            pass
                elif container is not None:
                    try:
                        return max(0.0, min(1.0, float(container)))
                    except Exception:
                        pass

            for alias in aliases:
                for key in (f"lambda_{alias}", f"lam_{alias}"):
                    for source in (metadata, approval):
                        if not isinstance(source, dict):
                            continue
                        try:
                            return max(0.0, min(1.0, float(source.get(key))))
                        except Exception:
                            pass
            return float(default)

        ach_default = legacy_lambda.get("ACH", 1.0 if any(k in record_params for k in ("a0", "a1", "a2", "a3")) else 0.0)
        return {
            "UA": _lookup("UA", default=legacy_lambda.get("UA", 1.0 if "UA" in record_params else 0.0)),
            "C": _lookup("C", default=legacy_lambda.get("C", 1.0 if "C" in record_params else 0.0)),
            "ACH": _lookup("ACH", "a0", "a1", "a2", "a3", default=ach_default),
            "g_solar": _lookup("g_solar", "eta", default=legacy_lambda.get("g_solar", 1.0 if ("eta" in record_params or "g_solar" in record_params) else 0.0)),
        }

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
            "window_gain": {
                "K_window_day": float(self.st.K_window_day),
                "K_window_night": float(self.st.K_window_night),
                "K_used_now": float(self.st.K_window_day if is_day else self.st.K_window_night),
            },
            "curtain": {
                "alpha_curtain": float(self.st.alpha_curtain),
                "target_jcm2": float(self.target_jcm2),
                "measured_sum_jcm2": float(self.out_light_sum),
                "sunlight_ratio": None if not np.isfinite(self.sum_ratio) else float(self.sum_ratio),
            },
            "candidate_search": {
                "generated": int(self._last_counts["generated"]),
                "feasible": int(self._last_counts["feasible"]),
                "selected_cost": None if selected is None else float(selected.cost),
                "selected_detail": {} if selected is None else dict(selected.detail),
            },
            "physics": {
                "UA": float(self.store_phys.get("UA", self.phys.UA)),
                "C": float(self.store_phys.get("C", self.phys.C)),
                "g_solar": float(self.store_phys.get("eta", self.phys.g_solar)),
                "ACH_model": {
                    "a0": float(self.store_phys.get("a0")),
                    "a1": float(self.store_phys.get("a1")),
                    "a2": float(self.store_phys.get("a2")),
                    "a3": float(self.store_phys.get("a3")),
                } if all(self.store_phys.get(k) is not None for k in ("a0", "a1", "a2", "a3")) else self.phys.ACH_coef,
                "lambda": dict(self.store_lambda),
                "mode": "store_calibrated" if self.store_record is not None else self.phys.mode,
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

    # Module 4) define a reduced candidate space from time-of-day / current state,
    # then generate grid candidates only within that reduced space.
    def _candidate_axes(self) -> Dict[str, List[Any]]:
        is_day, _, _, _ = self._targets()
        mode = "cool" if self.st.fcu_mode == "cool" else "heat"

        if np.isnan(self.in_temp):
            window_levels = [0]
            fcu_states = ["off"]
            fan_states = ["off"]
        else:
            max_window = 100 if is_day else 20
            window_levels = [int(v) for v in WINDOW_PERCENT_CANDIDATES if int(v) <= max_window]
            if 0 not in window_levels:
                window_levels.insert(0, 0)
            fcu_states = list(FCU_STATE_CANDIDATES)
            fan_states = list(FAN_STATE_CANDIDATES)

        if is_day:
            shade_levels = [int(v) for v in SHADE_PERCENT_CANDIDATES]
            thermal_levels = [100]
        else:
            shade_levels = [100]
            thermal_levels = [int(v) for v in THERMAL_PERCENT_CANDIDATES]

        if self.target_jcm2 <= 0:
            shade_levels = [100]

        return {
            "mode": [mode],
            "fcu_states": fcu_states,
            "window_levels": window_levels,
            "shade_levels": shade_levels,
            "thermal_levels": thermal_levels,
            "fan_states": fan_states,
        }

    def _generate_candidates(self) -> List[CandidateAction]:
        axes = self._candidate_axes()
        mode = str(axes["mode"][0])
        actions: List[CandidateAction] = []

        for fcu_state, open_pct, shade_open, thermal_open, fan_state in product(
            axes["fcu_states"],
            axes["window_levels"],
            axes["shade_levels"],
            axes["thermal_levels"],
            axes["fan_states"],
        ):
            actions.append(
                CandidateAction(
                    fcu=(mode, str(fcu_state)),
                    window=(
                        int(open_pct),
                        "main",
                        "OPEN" if int(open_pct) > 0 else "HOLD",
                    ),
                    curtain=("shade", int(shade_open)),
                    thermal_curtain=("thermal", int(thermal_open)),
                    fan=(str(fan_state),),
                )
            )
        self._last_counts["generated"] = len(actions)
        return actions

    # Module 5-1) hard filtering by safety / physics constraints.
    def _is_feasible(self, candidate: CandidateAction) -> bool:
        is_day, _, _, t_target = self._targets()
        _, pb = _dead_pb(is_day, self.st)
        caps = self._safety_caps(is_day)

        window_pct = int(candidate.window[0])
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

        if fan_state == "off" and (fcu_state == "on" or window_pct >= 20):
            return False
        if fan_state == "on" and fcu_state == "off" and window_pct == 0:
            return False

        t_pred = self._predict_temp(candidate)
        if np.isfinite(t_pred) and np.isfinite(t_target):
            t_min = float(t_target) - float(pb)
            t_max = float(t_target) + float(pb)
            if t_pred < t_min or t_pred > t_max:
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
        # Placeholder for future energy-cost modeling. Keep the hook in the
        # cost function so later work can populate this without changing the
        # candidate-evaluation flow.
        _ = candidate
        return 0.0

    def _constraint_penalty(self, candidate: CandidateAction) -> float:
        is_day, _, _, t_target = self._targets()
        _, pb = _dead_pb(is_day, self.st)
        window_pct = int(candidate.window[0])
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

        prev_open = int(_clamp(self.window_pct_fb, 0, 100)) if np.isfinite(self.window_pct_fb) else None
        if prev_open is not None:
            step = max(int(caps["ramp"]), 1)
            excess = abs(window_pct - prev_open) - step
            if excess > 0:
                penalty += excess / float(step)

        if fan_state == "off" and (fcu_state == "on" or window_pct >= 20):
            penalty += 1.0
        if fan_state == "on" and fcu_state == "off" and window_pct == 0:
            penalty += 1.0

        t_pred = self._predict_temp(candidate)
        if np.isfinite(t_pred) and np.isfinite(t_target):
            t_min = float(t_target) - float(pb)
            t_max = float(t_target) + float(pb)
            if t_pred < t_min:
                penalty += float(t_min - t_pred)
            elif t_pred > t_max:
                penalty += float(t_pred - t_max)

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


class _NoDataEvalInner:
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
        self.now = pd.Timestamp(self.row["reg_date"])
        self.physics_case = str(self.params.get("physics_case", "default"))
        self.physics_model_name = str(self.params.get("physics_model_name", "default"))
        self.physics_store = PhysicsStore(self.params.get("physics_store_path"))
        self.store_record = None
        self.store_phys: Dict[str, float] = {}
        self.store_lambda: Dict[str, float] = {}

        self._last_action: Optional[Dict[str, Any]] = None
        self._last_effective_params: Dict[str, Any] = {}
        self._theta = self._build_theta()

    def _coalesce_float(self, *values: Any, default: float) -> float:
        for value in values:
            try:
                fval = float(value)
            except Exception:
                continue
            if np.isfinite(fval):
                return fval
        return float(default)

    def _sample_seed(self) -> int:
        explicit_seed = self.params.get("no_data_eval_seed")
        if explicit_seed is not None:
            try:
                return int(explicit_seed)
            except Exception:
                pass

        farm_sn = self.params.get("farm_sn", 0)
        try:
            farm_val = int(farm_sn)
        except Exception:
            farm_val = sum(ord(ch) for ch in str(farm_sn))

        bucket = int(self.now.floor("5min").timestamp())
        return int((bucket + farm_val) % (2**32 - 1))

    def _build_theta(self) -> Dict[str, float]:
        theta = sample_theta_no_data(np.random.default_rng(self._sample_seed()))

        store_record = self.physics_store.get_record(
            farm_sn=self.params.get("farm_sn"),
            stage_name=self.params.get("stage_name"),
            model_name=self.physics_model_name,
            target_ts=self.now,
        )
        self.store_record = store_record
        store_params = dict(store_record.params) if store_record is not None else {}
        self.store_phys = dict(store_params)

        ach_coef = self.phys.ACH_coef or {}
        theta["UA"] = self._coalesce_float(
            store_params.get("UA"),
            self.phys.UA,
            theta.get("UA"),
            default=3000.0,
        )
        theta["C"] = self._coalesce_float(
            store_params.get("C"),
            self.phys.C,
            theta.get("C"),
            default=8.0e5,
        )
        theta["eta"] = self._coalesce_float(
            store_params.get("eta"),
            self.phys.g_solar,
            theta.get("eta"),
            default=0.12,
        )
        theta["a0"] = self._coalesce_float(
            store_params.get("a0"),
            ach_coef.get("a0"),
            theta.get("a0"),
            default=0.05,
        )
        theta["a1"] = self._coalesce_float(
            store_params.get("a1"),
            ach_coef.get("a1"),
            theta.get("a1"),
            default=2.5,
        )
        theta["a2"] = self._coalesce_float(
            store_params.get("a2"),
            ach_coef.get("a2"),
            theta.get("a2"),
            default=0.15,
        )
        theta["rho_cp"] = self._coalesce_float(
            store_params.get("rho_cp"),
            theta.get("rho_cp"),
            default=1.2 * 1005,
        )
        theta["k_heat"] = self._coalesce_float(
            store_params.get("k_heat"),
            self.params.get("physics_k_heat"),
            theta.get("k_heat"),
            default=15000.0,
        )
        return theta

    def _state(self) -> Dict[str, float]:
        tin = self._coalesce_float(self.row.get("in_temp"), default=15.0)
        rh = self._coalesce_float(
            self.row.get("in_humidity"),
            self.row.get("in_hum"),
            self.row.get("humidity"),
            default=70.0,
        )
        if rh > 1.5:
            rh = rh / 100.0
        rh = float(np.clip(rh, 0.0, 1.0))
        co2 = self._coalesce_float(
            self.row.get("in_co2"),
            self.row.get("co2"),
            default=900.0,
        )
        return {
            "Tin": float(tin),
            "e_in": float(rh * sat_vp_kpa_no_data(float(tin))),
            "CO2": float(co2),
        }

    def _disturbance(self) -> Dict[str, float]:
        is_day = bool(self.sunrise <= self.now <= self.sunset)
        tout = self._coalesce_float(self.row.get("out_temp"), default=5.0)
        rh_out = self._coalesce_float(
            self.row.get("out_humidity"),
            self.row.get("out_hum"),
            default=75.0,
        )
        if rh_out > 1.5:
            rh_out = rh_out / 100.0
        out_light = self._coalesce_float(self.row.get("out_light"), default=0.0)
        wind = self._coalesce_float(self.row.get("wind_speed"), default=0.5)
        co2_out = self._coalesce_float(self.row.get("out_co2"), default=420.0)
        return {
            "Tout": float(tout),
            "RHout": float(np.clip(rh_out, 0.0, 1.0)),
            "It": float(max(out_light, 0.0)),
            "wind": float(max(wind, 0.0)),
            "CO2out": float(co2_out),
            "hour": float(self.now.hour + self.now.minute / 60.0),
            "is_day": is_day,
        }

    def _prev_action(self) -> Dict[str, float]:
        window_pct = self._coalesce_float(self.row.get("window_pct"), default=0.0)
        return {"u_heat": 0.0, "x_vent": float(np.clip(window_pct / 100.0, 0.0, 1.0)), "curtain": 0.0, "u_co2": 0.0}

    def _weights(self) -> Dict[str, float]:
        return {
            "wT": float(self.params.get("NO_DATA_WT", 1.0)),
            "wVPD": float(self.params.get("NO_DATA_WVPD", 1.0)),
            "wE": float(self.params.get("NO_DATA_WE", 1.0e-8)),
            "wDx": float(self.params.get("NO_DATA_WDX", 0.2)),
            "wSlack": float(self.params.get("NO_DATA_WSLACK", 50.0)),
        }

    def _select_action(self) -> Dict[str, Any]:
        if self._last_action is not None:
            return self._last_action

        state = self._state()
        disturbance = self._disturbance()
        prev_action = self._prev_action()
        chosen = policy_optimal_no_data(
            state,
            disturbance,
            prev_action,
            self._theta,
            weights=self._weights(),
            Tref=float(self.day_temp if disturbance["is_day"] else self.night_temp),
            dt_min=int(self.params.get("physics_dt_min", 10)),
            A=float(self.params.get("greenhouse_area_m2", 200.0)),
            V=float(self.params.get("greenhouse_volume_m3", 300.0)),
            ramp_max=float(self.params.get("NO_DATA_RAMP_MAX", 0.15)),
        )
        pred = step_dynamics_no_data(
            state,
            disturbance,
            chosen,
            self._theta,
            dt_min=int(self.params.get("physics_dt_min", 10)),
            A=float(self.params.get("greenhouse_area_m2", 200.0)),
            V=float(self.params.get("greenhouse_volume_m3", 300.0)),
        )

        curtain_open_pct = int(np.clip(round((1.0 - float(chosen["curtain"])) * 100.0), 0, 100))
        thermal_pct = 100 if disturbance["is_day"] else max(curtain_open_pct, 10)
        fan_state = "on" if (float(chosen["x_vent"]) > 0 or float(chosen["u_heat"]) > 0.5) else "off"

        self._last_action = {
            "window": (
                int(np.clip(round(float(chosen["x_vent"]) * 100.0), 0, 100)),
                "main",
                "OPEN" if float(chosen["x_vent"]) > 0 else "HOLD",
            ),
            "curtain": ("shade", curtain_open_pct),
            "thermal_curtain": ("thermal", int(np.clip(thermal_pct, 0, 100))),
            "fcu": (self.st.fcu_mode, "on" if float(chosen["u_heat"]) > 0.5 else "off"),
            "fan": (fan_state,),
        }
        self._last_effective_params = {
            "timestamp": self.now,
            "mode": "no_data_eval_3_6_3",
            "physics_case": self.physics_case,
            "theta": {k: float(v) for k, v in self._theta.items() if isinstance(v, (int, float, np.floating))},
            "selected_action": {k: float(v) for k, v in chosen.items()},
            "predicted_next": {k: float(v) for k, v in pred.items() if isinstance(v, (int, float, np.floating))},
            "weights": self._weights(),
        }
        return self._last_action

    def control_fcu(self) -> Tuple[str, str]:
        return self._select_action()["fcu"]

    def control_window(self) -> Tuple[int, str, str]:
        return self._select_action()["window"]

    def control_curtain(self) -> Tuple[str, int]:
        return self._select_action()["curtain"]

    def control_thermal_curtain(self) -> Tuple[str, int]:
        return self._select_action()["thermal_curtain"]

    def control_fan(self) -> Tuple[str]:
        return self._select_action()["fan"]

    def run(self) -> Dict[str, Any]:
        action = self._select_action()
        return {
            "window": action["window"],
            "curtain": action["curtain"],
            "thermal_curtain": action["thermal_curtain"],
            "fcu": action["fcu"],
            "fan": action["fan"],
            "effective_params": dict(self._last_effective_params),
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

        physics_case = str(self.params.get("physics_case", "default")).lower()
        if physics_case == "no_data":
            self.inner = _NoDataEvalInner(
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
        else:
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

        phys = self.phys
        lam = self.inner.store_lambda or {}
        store_phys = self.inner.store_phys or {}
        active = any(float(v) > 0 for v in lam.values()) or all(store_phys.get(k) is not None for k in ("a0", "a1", "a2", "a3"))

        return {
            "window": inner_res["window"],
            "curtain": inner_res["curtain"],
            "fcu": inner_res["fcu"],
            "fan": inner_res["fan"],
            "effective_params": inner_res["effective_params"],
            "thermal_curtain": inner_res.get("thermal_curtain"),
            "outer": {
                "updated": False,
                "policy": self.state.to_dict(),
                "kpi": {},
                "intraday_kpi": {},
                "sunlight_ratio": 0.0,
                "score": self.state.last_score,
                "batch_due": False,
                "batch_executed": False,
                "evaluation_mode": "delegated_to_outer_layer",
            },
            "physics": {
                "mode": "store_calibrated" if self.inner.store_record is not None else phys.mode,
                "active": bool(active),
                "UA": float(store_phys.get("UA", phys.UA)),
                "C": float(store_phys.get("C", phys.C)),
                "g_solar": float(store_phys.get("eta", phys.g_solar)),
                "ACH_model": bool(all(store_phys.get(k) is not None for k in ("a0", "a1", "a2", "a3")) or (phys.ACH_coef is not None)),
                "lambda": {k: float(v) for k, v in lam.items()},
            },
        }


__all__ = [
    "Controller",
    "PolicyState",
    "PhysicsDefaults",
    "PhysicsContext",
    "SafetyState",
    "sat_vp_kpa_no_data",
    "dewpoint_c_no_data",
    "vpd_kpa_no_data",
    "hinge_no_data",
    "sample_theta_no_data",
    "ach_model_no_data",
    "step_dynamics_no_data",
    "project_action_no_data",
    "check_feasible_no_data",
    "make_action_grid_no_data",
    "policy_optimal_no_data",
    "_compute_kpis",
    "_score",
]
