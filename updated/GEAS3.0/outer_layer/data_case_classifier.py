from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class CaseDecision:
    data_case: str
    metrics: Dict[str, Any]
    reasons: list[str]


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _has_any_data(df: pd.DataFrame, cols: list[str]) -> bool:
    present = [c for c in cols if c in df.columns]
    if not present:
        return False
    return bool(df[present].notna().any().any())


def classify_data_availability(
    df: pd.DataFrame,
    *,
    time_col: str = 'reg_date',
    sufficient_days: int = 30,
    limited_days: int = 3,
) -> CaseDecision:
    if df is None or df.empty or time_col not in df.columns:
        return CaseDecision('no_data', {'row_count': 0}, ['empty_dataframe'])

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors='coerce')
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if work.empty:
        return CaseDecision('no_data', {'row_count': 0}, ['no_valid_timestamps'])

    row_count = int(len(work.index))
    start_ts = work[time_col].iloc[0]
    end_ts = work[time_col].iloc[-1]
    coverage_days = max(1.0, float((end_ts - start_ts).total_seconds() / 86400.0))
    unique_days = int(work[time_col].dt.floor('D').nunique())

    internal_cols = ['in_temp', 'in_hum', 'in_co2']
    control_cols = ['cont_heater_run', 'cont_skyl_vol', 'cont_cur_vol', 'cont_co2_run', 'cont_fan_run']
    weather_cols = ['out_temp', 'out_hum', 'out_light', 'out_windsp']

    has_internal_logs = _has_any_data(work, internal_cols)
    has_control_logs = _has_any_data(work, control_cols)
    has_weather_logs = _has_any_data(work, weather_cols)

    metrics = {
        'row_count': row_count,
        'coverage_days': coverage_days,
        'unique_days': unique_days,
        'has_internal_logs': has_internal_logs,
        'has_control_logs': has_control_logs,
        'has_weather_logs': has_weather_logs,
        'rows_per_day': _safe_ratio(row_count, coverage_days),
    }

    reasons: list[str] = []
    if not has_internal_logs:
        reasons.append('missing_internal_environment_logs')
    if not has_control_logs:
        reasons.append('missing_control_logs')

    if not has_internal_logs or not has_control_logs:
        return CaseDecision('no_data', metrics, reasons)

    if (
        coverage_days >= float(sufficient_days)
        and unique_days >= int(sufficient_days)
        and has_weather_logs
    ):
        return CaseDecision('sufficient', metrics, reasons)

    if coverage_days >= float(limited_days) or unique_days >= int(limited_days):
        reasons.append('limited_coverage_window')
        if not has_weather_logs:
            reasons.append('weather_logs_not_synchronized')
        return CaseDecision('limited', metrics, reasons)

    reasons.append('coverage_shorter_than_limited_window')
    return CaseDecision('no_data', metrics, reasons)
