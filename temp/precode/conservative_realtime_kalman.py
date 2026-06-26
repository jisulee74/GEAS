"""Conservative real-time Kalman imputation for GEAS continuous signals.

The class in this module is intended as a precode candidate for GEAS3.5.
It keeps valid observations unchanged and uses a 1D random-walk Kalman
estimate only while a missing run remains inside a data-driven safe gap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


IDENTIFIER_TIME_COLUMNS = ["idx", "iot_data_idx", "reg_date"]

CONTINUOUS_KALMAN_COLUMNS = [
    "in_medium_temp1",
    "in_medium_temp2",
    "in_temp",
    "in_temp2",
    "in_water_hot",
    "in_water_cold",
    "in_hum",
    "in_hum2",
    "in_medium_hum1",
    "in_medium_hum2",
    "in_co2",
    "in_co2_2",
    "in_medium_ec1",
    "in_medium_ec2",
    "out_temp",
    "out_hum",
    "out_windsp",
    "out_light",
    "out_light_sum",
    "out_rainfall",
    "out_airpress",
    "cont_skyl_vol",
    "cont_skyr_vol",
    "cont_cur_vol",
    "cont_kwcur_vol",
    "cont_3way1_vol",
    "cont_3way2_vol",
]

DISCRETE_OR_HELD_COLUMNS = [
    "out_rain",
    "etc_blackout",
    "etc_plc_abnorm",
    "etc_plc_norm",
    "cont_heater_run",
    "cont_cooler_run",
    "cont_co2_run",
    "cont_pump1_run",
    "cont_pump2_run",
    "cont_fan_run",
]

CIRCULAR_CONTINUOUS_SKIP_COLUMNS = ["out_winddirec"]


# These are engineering starting bounds for precode experiments. For deployment,
# pass controller-owned tolerances through ``error_limits`` when fitting.
DEFAULT_CONTROL_ERROR_LIMITS = {
    "in_medium_temp1": 1.0,
    "in_medium_temp2": 1.0,
    "in_temp": 1.0,
    "in_temp2": 1.0,
    "in_water_hot": 2.0,
    "in_water_cold": 2.0,
    "in_hum": 3.0,
    "in_hum2": 3.0,
    "in_medium_hum1": 5.0,
    "in_medium_hum2": 5.0,
    "in_co2": 100.0,
    "in_co2_2": 100.0,
    "in_medium_ec1": 0.25,
    "in_medium_ec2": 0.25,
    "out_temp": 1.5,
    "out_hum": 5.0,
    "out_windsp": 1.0,
    "out_light": 250.0,
    "out_light_sum": 250.0,
    "out_rainfall": 5.0,
    "out_airpress": 3.0,
    "cont_skyl_vol": 5.0,
    "cont_skyr_vol": 5.0,
    "cont_cur_vol": 5.0,
    "cont_kwcur_vol": 5.0,
    "cont_3way1_vol": 5.0,
    "cont_3way2_vol": 5.0,
}


@dataclass(frozen=True)
class SensorKalmanConfig:
    """Per-column Kalman and control-safety settings."""

    process_var: Optional[float] = None
    initial_var: Optional[float] = None
    default_observation_var: Optional[float] = None
    r_floor: float = 1e-6
    r_ceiling: Optional[float] = None
    rolling_window: int = 288
    min_periods: int = 36
    max_gap_cap_steps: int = 12
    min_safe_gap_steps: int = 0
    control_error_limit: Optional[float] = None
    confidence_z: float = 2.0
    delta_quantile: float = 0.995
    delta_stat: str = "max"
    hard_min: Optional[float] = None
    hard_max: Optional[float] = None


@dataclass(frozen=True)
class FittedSensorPolicy:
    """Fitted safety policy for one sensor."""

    column: str
    process_var: float
    initial_var: float
    default_observation_var: float
    max_gap_steps: int
    control_error_limit: Optional[float]
    selected_reason: str


def _as_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    present = [col for col in columns if col in df.columns]
    return df[present].apply(pd.to_numeric, errors="coerce")


def _finite_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values)


class ConservativeRealtimeKalmanImputer:
    """Kalman imputer designed for real-time control input stability.

    Model:
        x_t = x_{t-1} + w_t, w_t ~ N(0, Q)
        z_t = x_t + v_t,     v_t ~ N(0, R_t)

    Output policy:
        - finite observations are passed through unchanged;
        - missing values are filled with the predicted state only when the
          sensor-specific gap policy and dynamic uncertainty bound are safe;
        - long or unsafe gaps remain missing so the controller can enter a
          timeout/safe mode instead of consuming overconfident pseudo-data.
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        *,
        configs: Optional[Mapping[str, SensorKalmanConfig]] = None,
        default_config: Optional[SensorKalmanConfig] = None,
        default_max_gap_steps: int = 3,
        candidate_gap_steps: Optional[Iterable[int]] = None,
        use_default_control_limits: bool = True,
    ) -> None:
        self.columns = list(columns or CONTINUOUS_KALMAN_COLUMNS)
        self.configs = dict(configs or {})
        self.default_config = default_config or SensorKalmanConfig()
        self.default_max_gap_steps = int(default_max_gap_steps)
        self.candidate_gap_steps = tuple(candidate_gap_steps or range(1, 13))
        self.use_default_control_limits = use_default_control_limits

        self.policies_: Dict[str, FittedSensorPolicy] = {}
        self.gap_report_: pd.DataFrame = pd.DataFrame()
        self.global_r_: Dict[str, float] = {}
        self.fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        *,
        normal_mask: Optional[pd.Series] = None,
        error_limits: Optional[Mapping[str, float]] = None,
    ) -> "ConservativeRealtimeKalmanImputer":
        """Fit per-sensor R/Q defaults and data-driven max-gap guidance."""

        numeric = _as_numeric_frame(df, self.columns)
        if normal_mask is not None:
            normal_mask = normal_mask.reindex(df.index).fillna(False).astype(bool)
            numeric_for_stats = numeric.loc[normal_mask]
        else:
            numeric_for_stats = numeric

        policies: Dict[str, FittedSensorPolicy] = {}
        gap_rows = []

        for column in self.columns:
            cfg = self._config_for(column)
            if column not in numeric.columns:
                policies[column] = self._missing_column_policy(column, cfg)
                gap_rows.append(
                    {
                        "column": column,
                        "gap_steps": np.nan,
                        "gap_minutes": np.nan,
                        "max_abs_delta": np.nan,
                        "quantile_abs_delta": np.nan,
                        "uncertainty_bound": np.nan,
                        "risk_bound": np.nan,
                        "control_error_limit": self._control_limit(column, cfg, error_limits),
                        "safe_for_control": False,
                        "selected_max_gap_steps": 0,
                        "reason": "column_missing_in_training_data",
                    }
                )
                continue

            series = numeric_for_stats[column].dropna()
            process_var, initial_var, default_r = self._estimate_variances(series, cfg)
            self.global_r_[column] = default_r
            limit = self._control_limit(column, cfg, error_limits)

            selected_gap, reason, rows = self._fit_gap_for_column(
                column=column,
                series=series,
                cfg=cfg,
                process_var=process_var,
                initial_var=initial_var,
                control_limit=limit,
            )
            gap_rows.extend(rows)

            policies[column] = FittedSensorPolicy(
                column=column,
                process_var=process_var,
                initial_var=initial_var,
                default_observation_var=default_r,
                max_gap_steps=selected_gap,
                control_error_limit=limit,
                selected_reason=reason,
            )

        self.policies_ = policies
        self.gap_report_ = pd.DataFrame(gap_rows)
        self.fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        sort_by_time: Optional[str] = "reg_date",
        output_suffix: str = "_kf",
        keep_diagnostics: bool = True,
    ) -> pd.DataFrame:
        """Return a copy with Kalman-filled value and diagnostic columns."""

        if not self.fitted_:
            self.fit(df)

        if sort_by_time and sort_by_time in df.columns:
            out = df.sort_values(sort_by_time).copy()
        else:
            out = df.copy()

        numeric = _as_numeric_frame(out, self.columns)
        for column in self.columns:
            if column not in numeric.columns:
                continue

            result = self._transform_one(column, numeric[column])
            out[f"{column}{output_suffix}"] = result["value"]
            out[f"{column}_imputed_flag"] = result["imputed_flag"]
            out[f"{column}_quality"] = result["quality"]
            if keep_diagnostics:
                out[f"{column}_gap_steps"] = result["gap_steps"]
                out[f"{column}_state_var"] = result["state_var"]
                out[f"{column}_r_t"] = result["r_t"]
                out[f"{column}_control_safe"] = result["control_safe"]
                out[f"{column}_uncertainty_bound"] = result["uncertainty_bound"]

        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        normal_mask: Optional[pd.Series] = None,
        error_limits: Optional[Mapping[str, float]] = None,
        sort_by_time: Optional[str] = "reg_date",
        output_suffix: str = "_kf",
        keep_diagnostics: bool = True,
    ) -> pd.DataFrame:
        self.fit(df, normal_mask=normal_mask, error_limits=error_limits)
        return self.transform(
            df,
            sort_by_time=sort_by_time,
            output_suffix=output_suffix,
            keep_diagnostics=keep_diagnostics,
        )

    def gap_policy_report(self) -> pd.DataFrame:
        """Per-column evidence used to justify dynamic ``max_gap_steps``."""

        if self.gap_report_.empty:
            return self.gap_report_.copy()
        return self.gap_report_.sort_values(["column", "gap_steps"]).reset_index(drop=True)

    def policy_summary(self) -> pd.DataFrame:
        """One-row-per-column fitted policy summary."""

        rows = [
            {
                "column": policy.column,
                "process_var_q": policy.process_var,
                "initial_var_p0": policy.initial_var,
                "default_observation_var_r": policy.default_observation_var,
                "selected_max_gap_steps": policy.max_gap_steps,
                "selected_max_gap_minutes": policy.max_gap_steps * 5,
                "control_error_limit": policy.control_error_limit,
                "reason": policy.selected_reason,
            }
            for policy in self.policies_.values()
        ]
        return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)

    def result_summary(self, transformed: pd.DataFrame, *, output_suffix: str = "_kf") -> pd.DataFrame:
        """Summarize observed, imputed, and timeout counts by column."""

        rows = []
        for column in self.columns:
            value_col = f"{column}{output_suffix}"
            quality_col = f"{column}_quality"
            flag_col = f"{column}_imputed_flag"
            if value_col not in transformed.columns:
                continue
            quality_counts = transformed[quality_col].value_counts(dropna=False).to_dict()
            rows.append(
                {
                    "column": column,
                    "observed_raw_count": int(quality_counts.get("observed_raw", 0)),
                    "kalman_imputed_count": int(transformed[flag_col].sum()),
                    "missing_timeout_count": int(quality_counts.get("missing_timeout", 0)),
                    "missing_no_state_count": int(quality_counts.get("missing_no_state", 0)),
                    "remaining_missing_after_kalman": int(transformed[value_col].isna().sum()),
                }
            )
        return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)

    def _config_for(self, column: str) -> SensorKalmanConfig:
        return self.configs.get(column, self.default_config)

    def _control_limit(
        self,
        column: str,
        cfg: SensorKalmanConfig,
        error_limits: Optional[Mapping[str, float]],
    ) -> Optional[float]:
        if cfg.control_error_limit is not None:
            return float(cfg.control_error_limit)
        if error_limits and column in error_limits:
            return float(error_limits[column])
        if self.use_default_control_limits:
            return DEFAULT_CONTROL_ERROR_LIMITS.get(column)
        return None

    def _estimate_variances(
        self,
        series: pd.Series,
        cfg: SensorKalmanConfig,
    ) -> tuple[float, float, float]:
        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        diffs = clean.diff().dropna()
        diff_var = float(diffs.var(ddof=1)) if len(diffs) >= 2 else np.nan
        value_var = float(clean.var(ddof=1)) if len(clean) >= 2 else np.nan

        base_var = diff_var if np.isfinite(diff_var) and diff_var > 0 else value_var
        if not np.isfinite(base_var) or base_var <= 0:
            base_var = max(cfg.r_floor, 1.0)

        default_r = cfg.default_observation_var
        if default_r is None:
            default_r = base_var
        default_r = self._clip_r(float(default_r), cfg)

        process_var = cfg.process_var
        if process_var is None:
            process_var = max(cfg.r_floor, base_var * 0.1)

        initial_var = cfg.initial_var
        if initial_var is None:
            initial_var = max(default_r, float(process_var))

        return float(process_var), float(initial_var), float(default_r)

    def _fit_gap_for_column(
        self,
        *,
        column: str,
        series: pd.Series,
        cfg: SensorKalmanConfig,
        process_var: float,
        initial_var: float,
        control_limit: Optional[float],
    ) -> tuple[int, str, list[dict]]:
        cap = int(max(cfg.min_safe_gap_steps, cfg.max_gap_cap_steps))
        candidate_steps = [step for step in self.candidate_gap_steps if 1 <= int(step) <= cap]

        if len(series) < 2:
            selected = min(self.default_max_gap_steps, cap)
            return selected, "insufficient_history_default", []

        rows = []
        safe_steps = []
        for step in candidate_steps:
            delta = (series - series.shift(step)).abs().dropna()
            max_delta = float(delta.max()) if not delta.empty else np.nan
            quantile_delta = (
                float(delta.quantile(cfg.delta_quantile)) if not delta.empty else np.nan
            )
            empirical_delta = max_delta if cfg.delta_stat == "max" else quantile_delta
            uncertainty = float(cfg.confidence_z * np.sqrt(initial_var + step * process_var))
            risk_bound = float(empirical_delta + uncertainty) if np.isfinite(empirical_delta) else np.nan

            if control_limit is None:
                safe = step <= min(self.default_max_gap_steps, cap)
            else:
                safe = bool(np.isfinite(risk_bound) and risk_bound <= control_limit)

            if safe:
                safe_steps.append(step)

            rows.append(
                {
                    "column": column,
                    "gap_steps": int(step),
                    "gap_minutes": int(step) * 5,
                    "max_abs_delta": max_delta,
                    "quantile_abs_delta": quantile_delta,
                    "delta_stat_used": cfg.delta_stat,
                    "uncertainty_bound": uncertainty,
                    "risk_bound": risk_bound,
                    "control_error_limit": control_limit,
                    "safe_for_control": safe,
                    "selected_max_gap_steps": np.nan,
                    "reason": "",
                }
            )

        if safe_steps:
            selected = int(max(safe_steps))
            reason = "data_driven_control_safe"
        elif control_limit is None:
            selected = int(min(self.default_max_gap_steps, cap))
            reason = "missing_control_limit_default"
        else:
            selected = int(cfg.min_safe_gap_steps)
            reason = "no_gap_satisfies_control_limit"

        for row in rows:
            row["selected_max_gap_steps"] = selected
            row["reason"] = reason
        return selected, reason, rows

    def _missing_column_policy(
        self,
        column: str,
        cfg: SensorKalmanConfig,
    ) -> FittedSensorPolicy:
        limit = cfg.control_error_limit
        return FittedSensorPolicy(
            column=column,
            process_var=max(cfg.r_floor, cfg.process_var or cfg.r_floor),
            initial_var=max(cfg.r_floor, cfg.initial_var or cfg.r_floor),
            default_observation_var=max(cfg.r_floor, cfg.default_observation_var or cfg.r_floor),
            max_gap_steps=0,
            control_error_limit=limit,
            selected_reason="column_missing_in_training_data",
        )

    def _rolling_observation_variance(
        self,
        series: pd.Series,
        cfg: SensorKalmanConfig,
        default_r: float,
    ) -> np.ndarray:
        diffs = series.diff()
        rolling_r = diffs.rolling(window=cfg.rolling_window, min_periods=cfg.min_periods).var(ddof=1)
        rolling_r = rolling_r.shift(1).fillna(default_r)
        clipped = rolling_r.map(lambda value: self._clip_r(value, cfg))
        return clipped.to_numpy(dtype=float)

    def _clip_r(self, value: float, cfg: SensorKalmanConfig) -> float:
        if not np.isfinite(value) or value <= 0:
            value = cfg.r_floor
        value = max(float(value), cfg.r_floor)
        if cfg.r_ceiling is not None:
            value = min(value, float(cfg.r_ceiling))
        return value

    def _observed_is_valid(self, value: float, cfg: SensorKalmanConfig) -> bool:
        if not np.isfinite(value):
            return False
        if cfg.hard_min is not None and value < cfg.hard_min:
            return False
        if cfg.hard_max is not None and value > cfg.hard_max:
            return False
        return True

    def _transform_one(self, column: str, series: pd.Series) -> dict[str, np.ndarray]:
        cfg = self._config_for(column)
        policy = self.policies_.get(column)
        if policy is None:
            raise RuntimeError(f"Column {column!r} has no fitted Kalman policy.")

        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        n = len(values)
        output = np.full(n, np.nan, dtype=float)
        imputed = np.zeros(n, dtype=bool)
        quality = np.full(n, "missing_no_state", dtype=object)
        gap_steps_out = np.full(n, np.nan, dtype=float)
        state_var_out = np.full(n, np.nan, dtype=float)
        r_out = self._rolling_observation_variance(series, cfg, policy.default_observation_var)
        control_safe = np.zeros(n, dtype=bool)
        uncertainty_out = np.full(n, np.nan, dtype=float)

        x = np.nan
        p = policy.initial_var
        has_state = False
        last_observed_idx: Optional[int] = None

        for idx, raw_value in enumerate(values):
            observed = self._observed_is_valid(raw_value, cfg)

            if has_state:
                x_pred = x
                p_pred = p + policy.process_var
            else:
                x_pred = np.nan
                p_pred = policy.initial_var

            if observed:
                if has_state:
                    r_t = self._clip_r(r_out[idx], cfg)
                    k_gain = p_pred / (p_pred + r_t)
                    x = x_pred + k_gain * (raw_value - x_pred)
                    p = (1.0 - k_gain) * p_pred
                else:
                    x = raw_value
                    p = policy.initial_var
                    has_state = True
                output[idx] = raw_value
                quality[idx] = "observed_raw"
                gap_steps_out[idx] = 0
                control_safe[idx] = True
                uncertainty_out[idx] = cfg.confidence_z * np.sqrt(max(p, 0.0))
                last_observed_idx = idx
            else:
                if not has_state or last_observed_idx is None:
                    output[idx] = np.nan
                    quality[idx] = "missing_no_state"
                    gap_steps_out[idx] = np.nan
                    uncertainty_out[idx] = np.nan
                    control_safe[idx] = False
                else:
                    gap_steps = idx - last_observed_idx
                    uncertainty = cfg.confidence_z * np.sqrt(max(p_pred, 0.0))
                    within_gap = gap_steps <= policy.max_gap_steps
                    within_limit = (
                        True
                        if policy.control_error_limit is None
                        else uncertainty <= policy.control_error_limit
                    )
                    safe = bool(within_gap and within_limit)

                    x = x_pred
                    p = p_pred
                    gap_steps_out[idx] = gap_steps
                    uncertainty_out[idx] = uncertainty
                    control_safe[idx] = safe
                    if safe:
                        output[idx] = x_pred
                        imputed[idx] = True
                        quality[idx] = "kalman_imputed_safe"
                    else:
                        output[idx] = np.nan
                        quality[idx] = "missing_timeout"

            state_var_out[idx] = p if has_state else np.nan

        return {
            "value": output,
            "imputed_flag": imputed,
            "quality": quality,
            "gap_steps": gap_steps_out,
            "state_var": state_var_out,
            "r_t": r_out,
            "control_safe": control_safe,
            "uncertainty_bound": uncertainty_out,
        }


def continuous_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness table for GEAS continuous Kalman candidate columns."""

    numeric = _as_numeric_frame(df, CONTINUOUS_KALMAN_COLUMNS)
    rows = []
    total = len(df)
    for column in CONTINUOUS_KALMAN_COLUMNS:
        if column not in numeric.columns:
            rows.append(
                {
                    "column": column,
                    "missing_count": total,
                    "total_rows": total,
                    "missing_ratio": 1.0 if total else np.nan,
                    "present_in_input": False,
                }
            )
            continue
        missing = int((~_finite_mask(numeric[column].to_numpy(dtype=float))).sum())
        rows.append(
            {
                "column": column,
                "missing_count": missing,
                "total_rows": total,
                "missing_ratio": float(missing / total) if total else np.nan,
                "present_in_input": True,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["missing_ratio", "missing_count", "column"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


__all__ = [
    "CIRCULAR_CONTINUOUS_SKIP_COLUMNS",
    "CONTINUOUS_KALMAN_COLUMNS",
    "DEFAULT_CONTROL_ERROR_LIMITS",
    "DISCRETE_OR_HELD_COLUMNS",
    "IDENTIFIER_TIME_COLUMNS",
    "ConservativeRealtimeKalmanImputer",
    "FittedSensorPolicy",
    "SensorKalmanConfig",
    "continuous_missingness_table",
]
