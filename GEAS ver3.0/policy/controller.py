from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
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
    fcu_mode: str = "cool"  # "cool" or "heat"
    last_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PhysicsDefaults:
    UA_default: float = 1.0
    C_default: float = 1.0
    g_solar_default: float = 0.0
    # fallback ACH(open) lookup if no model:
    ACH_table: Tuple[Tuple[float, float], ...] = ((0, 0.1), (20, 0.3), (50, 1.0), (100, 3.0))


@dataclass
class PhysicsContext:
    UA: float
    C: float
    g_solar: float
    ACH_coef: Optional[Dict[str, float]]
    lam: Dict[str, float]
    mode: str

@dataclass
class SafetyState:
    delta_cond: Optional[float] = None   # last delta to dewpoint [°C]
    vpd: Optional[float] = None          # last VPD [kPa]


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _is_daytime(now: pd.Timestamp, sunrise: pd.Timestamp, sunset: pd.Timestamp) -> bool:
    t = now
    if t.tzinfo is None and sunrise.tzinfo is not None:
        t = t.tz_localize(sunrise.tzinfo)
    return (t >= sunrise) and (t < sunset)


def _dead_pb(is_day: bool, st: PolicyState) -> Tuple[float, float]:
    return (st.DB_day, st.PB_day) if is_day else (st.DB_night, st.PB_night)


def _scale_01(err_abs: float, DB: float, PB: float) -> float:
    if err_abs <= DB:
        return 0.0
    if err_abs >= PB:
        return 1.0
    return (err_abs - DB) / (PB - DB + 1e-9)

def _blend(val_est: Optional[float], val_def: float, lam: float,
           lo: float = -np.inf, hi: float = np.inf) -> float:
    if val_est is None or (isinstance(val_est, float) and not np.isfinite(val_est)) or lam <= 0.0:
        return _clamp(val_def, lo, hi)
    return _clamp((1.0 - lam) * val_def + lam * float(val_est), lo, hi)


def _ach_table(open_pct: float, table: Tuple[Tuple[float, float], ...]) -> float:
    x = float(_clamp(open_pct, 0.0, 100.0))
    xs = [p for p, _ in table]
    ys = [q for _, q in table]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def _mk_phys_context(derived: Optional[Dict[str, Any]], defaults: PhysicsDefaults) -> PhysicsContext:
    if not derived:
        return PhysicsContext(
            UA=defaults.UA_default,
            C=defaults.C_default,
            g_solar=defaults.g_solar_default,
            ACH_coef=None,
            lam={"UA": 0.0, "C": 0.0, "ACH": 0.0, "g_solar": 0.0},
            mode="baseline",
        )

    est = derived.get("est", {}) or {}
    lam = derived.get("lambda", {}) or {}
    mode = derived.get("mode", "baseline")

    UA = _blend(est.get("UA"), defaults.UA_default, float(lam.get("UA", 0.0)))
    C = _blend(est.get("C"), defaults.C_default, float(lam.get("C", 0.0)))
    gs = _blend(est.get("g_solar"), defaults.g_solar_default, float(lam.get("g_solar", 0.0)))

    ach_model = est.get("ACH_model")
    ach_lam = float(lam.get("ACH", 0.0))
    ach_coef = ach_model if (ach_lam > 0.4 and isinstance(ach_model, dict)) else None

    return PhysicsContext(UA=UA, C=C, g_solar=gs, ACH_coef=ach_coef, lam=lam, mode=mode)


def _mk_safety(derived: Optional[Dict[str, Any]]) -> SafetyState:
    if not derived:
        return SafetyState()
    saf = derived.get("safety", {}) or {}
    dc = saf.get("delta_cond")
    vpd = saf.get("vpd")

    dc_val = None
    if isinstance(dc, np.ndarray) and dc.size > 0 and np.isfinite(dc[-1]):
        dc_val = float(dc[-1])

    vpd_val = None
    if isinstance(vpd, np.ndarray) and vpd.size > 0 and np.isfinite(vpd[-1]):
        vpd_val = float(vpd[-1])

    return SafetyState(delta_cond=dc_val, vpd=vpd_val)

class _Inner:
    # Default policies for new safety features (can be overridden via params)
    WIND_CAP_TH: float = 5.0      # m/s
    WIND_CAP_OPEN: int = 20       # %
    ACH_MIN_DAY: float = 0.10     # h^-1
    ACH_MIN_NIGHT: float = 0.05   # h^-1
    RAMP_LIMIT: int = 15          # % per step

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

        # feature toggles (safe defaults)
        self.USE_WIND_CAP = bool(self.params.get("USE_WIND_CAP", True))
        self.USE_MIN_ACH = bool(self.params.get("USE_MIN_ACH", True))
        self.USE_RAMP = bool(self.params.get("USE_RAMP_LIMIT", True))
        self.USE_VPD_FOR_MIN_ACH = bool(self.params.get("USE_VPD_FOR_MIN_ACH", False))
        self.USE_ETA_FOR_CURTAIN = bool(self.params.get("USE_ETA_FOR_CURTAIN", False))

        # thresholds (override if provided)
        self.WIND_CAP_TH = float(self.params.get("WIND_CAP_TH", self.WIND_CAP_TH))
        self.WIND_CAP_OPEN = int(self.params.get("WIND_CAP_OPEN", self.WIND_CAP_OPEN))

        # ACH_min defaults; may be overridden by features
        self.ACH_MIN_DAY = float(self.params.get("ACH_MIN_DAY", self.ACH_MIN_DAY))
        self.ACH_MIN_NIGHT = float(self.params.get("ACH_MIN_NIGHT", self.ACH_MIN_NIGHT))

        # Try to override from features (ACH_min_day/night)
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

        # Ramp limit from params or features
        ramp_from_params = self.params.get("RAMP_LIMIT", None)
        ramp_from_kpi = self.kpis.get("RampLim", None)
        if ramp_from_params is not None:
            self.RAMP_LIMIT = int(ramp_from_params)
        elif ramp_from_kpi is not None:
            try:
                self.RAMP_LIMIT = int(ramp_from_kpi)
            except Exception:
                pass

        # telemetry
        self.now = pd.Timestamp(self.row["reg_date"])
        self.in_temp = float(self.row.get("in_temp", np.nan)) if pd.notna(self.row.get("in_temp", np.nan)) else np.nan
        self.out_rain = float(self.row.get("out_rain", 0.0))
        self.out_light_sum = float(self.row.get("out_light_sum", 0.0))
        self.out_light = float(self.row.get("out_light", 0.0))
        self.wind_speed = float(self.row.get("wind_speed", np.nan)) if "wind_speed" in self.row else np.nan
        self.window_pct_fb = float(self.row.get("window_pct", np.nan)) if "window_pct" in self.row else np.nan

        self.dTcond_kpi = None
        try:
            v = float(self.kpis.get("dTcond", np.nan))
            if np.isfinite(v):
                self.dTcond_kpi = v
        except Exception:
            pass

        self.cond_risk_10m = 0
        try:
            self.cond_risk_10m = int(self.kpis.get("cond_risk_10m", 0) or 0)
        except Exception:
            self.cond_risk_10m = 0

        # light-related KPIs
        self.sum_ratio = np.nan
        self.dli = np.nan

        try:
            self.sum_ratio = float(self.kpis.get("SumRatio", np.nan))
        except Exception:
            pass

        try:
            self.dli = float(self.kpis.get("DLI", np.nan))
        except Exception:
            pass

        # Light ETA: if provided in KPIs, prefer it over compute_eta's eta
        self.light_eta = self.kpis.get("LightETA", self.eta)

    def _targets(self) -> Tuple[bool, float, float, float]:
        is_day = _is_daytime(self.now, self.sunrise, self.sunset)
        T_day_eff = self.day_temp + self.st.T_day_bias
        T_night_eff = self.night_temp + self.st.T_night_bias
        T_now = T_day_eff if is_day else T_night_eff
        return is_day, T_day_eff, T_night_eff, T_now

    def _effective_params(self) -> Dict[str, Any]:
        is_day, T_day_eff, T_night_eff, T_now = self._targets()
        DB, PB = _dead_pb(is_day, self.st)
        return {
            "timestamp": self.now,
            "is_daytime": bool(is_day),
            "fcu_mode": self.st.fcu_mode,
            "targets": {
                "T_day_eff": float(T_day_eff),
                "T_night_eff": float(T_night_eff),
                "T_current_eff": float(T_now),
            },
            "bands": {
                "DB_day": float(self.st.DB_day),
                "PB_day": float(self.st.PB_day),
                "DB_night": float(self.st.DB_night),
                "PB_night": float(self.st.PB_night),
                "DB_used_now": float(DB),
                "PB_used_now": float(PB),
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
                "sunlight_ratio": float(
                    self.out_light_sum / max(self.target_jcm2, 1e-6)
                )
                if self.target_jcm2 > 0
                else 0.0,
            },
            "physics": {
                "UA": self.phys.UA,
                "C": self.phys.C,
                "g_solar": self.phys.g_solar,
                "ACH_model": self.phys.ACH_coef,
                "lambda": self.phys.lam,
                "mode": self.phys.mode,
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
                    next(
                        (x for x, y in self.defaults.ACH_table if y >= target_ach),
                        100,
                    )
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

        if (
            self.USE_WIND_CAP
            and np.isfinite(self.wind_speed)
            and (self.wind_speed > self.WIND_CAP_TH)
        ):
            cap_open = min(cap_open, self.WIND_CAP_OPEN)

        delta_cond = self.safety.delta_cond
        if delta_cond is None or not np.isfinite(delta_cond):
            delta_cond = self.dTcond_kpi

        if (delta_cond is not None) and np.isfinite(delta_cond):
            if delta_cond < 0.8 or self.cond_risk_10m == 1:
                cap_open = min(cap_open, 10)

        ach_min = 0.0
        if self.USE_MIN_ACH:
            ach_min = self.ACH_MIN_DAY if is_day else self.ACH_MIN_NIGHT

        vpd_val = self.safety.vpd
        if vpd_val is None or not np.isfinite(vpd_val):
            try:
                vpd_val = float(self.kpis.get("VPD", np.nan))
            except Exception:
                vpd_val = None

        if self.USE_VPD_FOR_MIN_ACH and (vpd_val is not None) and np.isfinite(vpd_val):
            if float(vpd_val) < 0.5:
                ach_min *= 1.5

        return {"open_cap": cap_open, "ach_min": ach_min, "ramp": self.RAMP_LIMIT}

    def control_fcu(self) -> Tuple[str, str]:
        mode = "cool" if self.st.fcu_mode == "cool" else "heat"
        if np.isnan(self.in_temp):
            return (mode, "off")
        is_day = _is_daytime(self.now, self.sunrise, self.sunset)
        DB, _ = _dead_pb(is_day, self.st)
        _, _, _, T_now = self._targets()
        err = self.in_temp - T_now
        if mode == "cool":
            return (mode, "on" if err > DB else "off")
        else:
            return (mode, "on" if err < -DB else "off")

    def control_window(self) -> Tuple[int, str, str]:
        if np.isnan(self.in_temp):
            return (0, "main", "HOLD")
        if self.out_rain > 0.0:
            return (0, "main", "CLOSE")

        is_day, _, _, T_now = self._targets()
        DB, PB = _dead_pb(is_day, self.st)
        err = self.in_temp - T_now
        caps = self._safety_caps(is_day)
        lam_ach = float(self.phys.lam.get("ACH", 0.0))
        gain_base = self.st.K_window_day if is_day else self.st.K_window_night

        if err <= DB:
            pct_raw = 0
        elif float(lam_ach) >= 0.7:
            target_ach = 0.5 if is_day else 0.2
            pct_raw = self._ventilation_open_from_ach(target_ach)
        else:
            u = _scale_01(abs(err), DB, PB) * 100.0 * (0.5 + 0.5 * lam_ach) * gain_base
            pct_raw = int(_clamp(u, 0.0, 100.0))

        if caps["open_cap"] > 0 and caps["ach_min"] > 0:
            min_pct = self._ventilation_open_from_ach(caps["ach_min"])
            pct_raw = max(pct_raw, min_pct)

        pct_cap = int(_clamp(pct_raw, 0, caps["open_cap"]))

        if self.USE_RAMP and np.isfinite(self.window_pct_fb):
            prev = int(_clamp(self.window_pct_fb, 0, 100))
            step = self.RAMP_LIMIT
            if pct_cap > prev:
                pct_final = min(prev + step, pct_cap)
            elif pct_cap < prev:
                pct_final = max(prev - step, pct_cap)
            else:
                pct_final = prev
        else:
            pct_final = pct_cap

        return (pct_final, "main", "OPEN" if pct_final > 0 else "HOLD")

    def control_curtain(self) -> Tuple[str, int]:
        if self.target_jcm2 <= 0:
            return ("shade", 100)

        ratio = self.out_light_sum / max(self.target_jcm2, 1e-6)
        base = 20 if ratio <= 0.8 else 40 if ratio <= 1.0 else 60 if ratio <= 1.2 else 80
        size = int(_clamp(base * self.st.alpha_curtain, 0, 100))

        eta_ref = self.light_eta if self.light_eta is not None else self.eta
        if self.USE_ETA_FOR_CURTAIN and pd.notna(eta_ref):
            if (ratio < 1.0) and (self.now < eta_ref):
                size = max(0, size - 20)

        if _is_daytime(self.now, self.sunrise, self.sunset):
            if np.isfinite(self.sum_ratio):
                if self.sum_ratio < 0.6:
                    size = max(0, size - 10)
                elif self.sum_ratio > 1.2:
                    size = min(100, size + 10)

        if not _is_daytime(self.now, self.sunrise, self.sunset):
            size = 0

        return ("shade", int(100 - size))

    def control_thermal_curtain(self) -> Tuple[str, int]:
        if np.isnan(self.in_temp):
            return ("thermal", 100)

        if _is_daytime(self.now, self.sunrise, self.sunset):
            return ("thermal", 100)

        _, _, T_night_eff, _ = self._targets()
        DB, PB = _dead_pb(False, self.st)

        cold_err = max(T_night_eff - self.in_temp, 0.0)
        close_pct = int(
            _clamp(
                _scale_01(cold_err, DB, PB) * 100.0,
                0.0,
                100.0,
            )
        )

        delta_cond = self.safety.delta_cond
        if delta_cond is None or not np.isfinite(delta_cond):
            delta_cond = self.dTcond_kpi

        if (delta_cond is not None) and np.isfinite(delta_cond):
            if (delta_cond < 0.5) or (self.cond_risk_10m == 1):
                close_pct = min(close_pct, 70)

        open_pct = int(_clamp(100 - close_pct, 0, 100))
        return ("thermal", open_pct)
    
    def control_fan(self) -> Tuple[str]:
        mode, state = self.control_fcu()
        win_pct, _, _ = self.control_window()
        return ("on",) if (state == "on" or win_pct >= 20) else ("off",)

    def run(self) -> Dict[str, Any]:
        return {
            "window": self.control_window(),
            "curtain": self.control_curtain(),
            "thermal_curtain": self.control_thermal_curtain(),
            "fcu": self.control_fcu(),
            "fan": self.control_fan(),
            "effective_params": self._effective_params(),
        }

class _Outer:
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
    ):
        self.df_today = df_today
        self.sunrise, self.sunset, self.eta, self.cs, self.target_jcm2 = out_light_info
        self.day_temp, self.night_temp = base_temp
        self.st = st

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

        def tune(DB, PB, cov, mdev):
            if cov < self.COV_TARGET and abs(mdev) < 0.3:
                DB = _clamp(DB + self.STEP_DB, self.DB_MIN, self.DB_MAX)
            if cov < self.COV_TARGET and abs(mdev) > 0.4:
                PB = _clamp(PB - self.STEP_PB, self.PB_MIN, self.PB_MAX)
            return DB, PB

        st.DB_day, st.PB_day = tune(
            st.DB_day, st.PB_day, kpi["cov_day"], kpi["mdev_day"]
        )

        st.DB_night, st.PB_night = tune(
            st.DB_night, st.PB_night, kpi["cov_night"], kpi["mdev_night"]
        )

        st.K_window_day = _clamp(
            st.K_window_day
            + _clamp(0.15 * kpi["mdev_day"], -self.STEP_K, self.STEP_K),
            self.K_MIN,
            self.K_MAX,
        )
        st.K_window_night = _clamp(
            st.K_window_night
            + _clamp(0.15 * kpi["mdev_night"], -self.STEP_K, self.STEP_K),
            self.K_MIN,
            self.K_MAX,
        )

        if (kpi["mdev_day"] > +0.6) and (sunlight_ratio > 1.1):
            st.alpha_curtain = _clamp(st.alpha_curtain + 0.1, 0.5, 2.0)

        elif (kpi["mdev_day"] < -0.6) and (sunlight_ratio < 0.9):
            st.alpha_curtain = _clamp(st.alpha_curtain - 0.1, 0.5, 2.0)

        return st

    def run(self) -> Dict[str, Any]:
        if self.df_today is None or self.df_today.empty:
            return {"updated": False, "policy": self.st.to_dict(), "kpi": {}}

        T_day = self.day_temp + self.st.T_day_bias
        T_night = self.night_temp + self.st.T_night_bias

        kpi = _compute_kpis(
            self.df_today,
            self.sunrise,
            self.sunset,
            T_day,
            T_night,
            band_width=1.0,
        )

        try:
            sr = float(self.df_today["out_light_sum"].iloc[-1]) / float(
                self.target_jcm2 or 1.0
            )
        except Exception:
            sr = 0.0

        today_score = _score(kpi)
        eps = 1e-4

        if (self.st.last_score is not None) and (
            today_score < self.st.last_score - eps
        ):
            accepted = False
            final_state = self.st
            final_score = self.st.last_score
        else:
            accepted = True
            updated_state = self._update(kpi, sunlight_ratio=sr)
            updated_state.last_score = today_score
            self.st = updated_state
            final_state = updated_state
            final_score = today_score

        return {
            "updated": accepted,
            "policy": final_state.to_dict(),
            "kpi": kpi,
            "sunlight_ratio": sr,
            "score": final_score,
        }


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

    sunrise_ts = sunrise
    sunset_ts = sunset
    if sunrise_ts.tzinfo is None:
        sunrise_ts = sunrise_ts.tz_localize("Asia/Seoul")

    else:
        sunrise_ts = sunrise_ts.tz_convert("Asia/Seoul")

    if sunset_ts.tzinfo is None:
        sunset_ts = sunset_ts.tz_localize("Asia/Seoul")

    else:
        sunset_ts = sunset_ts.tz_convert("Asia/Seoul")

    t_ns = df["reg_date"].astype("int64").to_numpy()
    x = df["in_temp"].to_numpy(dtype=float)

    t1 = t_ns[:-1]
    t2 = t_ns[1:]
    dt = (t2 - t1) / 1e9
    x1 = x[:-1]

    sunrise_ns = sunrise_ts.value
    sunset_ns = sunset_ts.value

    first_ts = df["reg_date"].iloc[0]
    if first_ts.tzinfo is None:
        base0_ts = first_ts.tz_localize("Asia/Seoul").normalize()
    else:
        base0_ts = first_ts.tz_convert("Asia/Seoul").normalize()
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

    return dict(
        cov_day=cov_day,
        cov_night=cov_night,
        mdev_day=mdev_day,
        mdev_night=mdev_night,
    )


def _score(kpi: Dict[str, float]) -> float:
    cov = 0.5 * kpi.get("cov_day", 0.0) + 0.5 * kpi.get("cov_night", 0.0)
    dev = 0.5 * abs(kpi.get("mdev_day", 0.0)) + 0.5 * abs(kpi.get("mdev_night", 0.0))
    return cov - 0.8 * dev


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
            self.state.fcu_mode = (
                "heat" if self.params["fcu_mode"] == "heat" else "cool"
            )

        self.inner = _Inner(
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
        self.outer = _Outer(self.df_today, self.out_light_info, self.base_temp, self.state)

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
        active = (any(float(v) > 0 for v in lam.values()) or (phys.ACH_coef is not None))

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


if __name__ == "__main__":
    from core.solar_eta import get_sun_times

    lat, lon = 36.46, 128.22
    now = pd.Timestamp.now(tz="Asia/Seoul")

    latest = pd.DataFrame(
        [
            {
                "reg_date": now,
                "in_temp": 24.0,
                "out_temp": 15.0,
                "out_rain": 0.0,
                "out_light": 300.0,
                "out_light_sum": 2500.0,
                "wind_speed": 2.0,
                "window_pct": 10.0,
            }
        ]
    )

    sunrise, sunset = get_sun_times(lat, lon, now, tz="Asia/Seoul")
    eta = sunset - pd.Timedelta(hours=1)
    cs_sum = np.linspace(0, 8000, 100)
    target_jcm2 = 5000.0
    out_light_info = (sunrise, sunset, eta, cs_sum, target_jcm2)

    idx = pd.date_range(start=now.normalize(), periods=24, freq="1h", tz="Asia/Seoul")
    df_today = pd.DataFrame(
        {
            "reg_date": idx,
            "in_temp": np.linspace(20, 26, len(idx)),
            "out_light_sum": np.linspace(0, target_jcm2, len(idx)),
        }
    )

    agro_kpis = {
        "ACH_min_day": 0.12,
        "ACH_min_night": 0.06,
        "dTcond": 1.2,
        "cond_risk_10m": 0,
        "SumRatio": 0.7,
        "DLI": 10.0,
        "LightETA": eta,
        "RampLim": 15.0,
        "VPD": 0.6,
    }

    ctrl = Controller(
        dataframe=latest,
        out_light_info=out_light_info,
        temp=(25.0, 18.0),
        params={"fcu_mode": "heat"},
        policy_state=None,
        df_today=df_today,
        derived_result=None,
        phys_defaults=None,
        agro_kpis=agro_kpis,
    )

    res = ctrl.run()
    print("window:", res["window"])
    print("curtain:", res["curtain"])
    print("fcu:", res["fcu"])
    print("fan:", res["fan"])
    print("physics:", res["physics"])
    print("outer:", res["outer"])
