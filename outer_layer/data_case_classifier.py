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


def classify_data_availability(
    df: pd.DataFrame,
    *,
    time_col: str = 'reg_date',
    sufficient_days: int = 30,
    limited_days: int = 3,
    max_missing_ratio_for_sufficient: float = 0.20,
    min_control_events_for_sufficient: int = 50,
    min_rows_for_limited: int = 24,
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

    tracked_cols = [c for c in ['in_temp', 'in_hum', 'in_co2', 'out_temp', 'out_hum', 'out_light', 'out_windsp', 'cont_heater_run', 'cont_skyl_vol'] if c in work.columns]
    missing_ratio = 0.0
    if tracked_cols:
        missing_ratio = float(work[tracked_cols].isna().mean().mean())

    control_cols = [c for c in ['cont_heater_run', 'cont_skyl_vol', 'cont_cur_vol', 'cont_co2_run', 'cont_fan_run'] if c in work.columns]
    control_events = 0
    for col in control_cols:
        s = pd.to_numeric(work[col], errors='coerce').fillna(0.0)
        control_events += int((s.diff().abs() > 1e-9).sum())

    metrics = {
        'row_count': row_count,
        'coverage_days': coverage_days,
        'unique_days': unique_days,
        'missing_ratio': missing_ratio,
        'control_events': control_events,
        'rows_per_day': _safe_ratio(row_count, coverage_days),
    }

    reasons: list[str] = []
    if row_count < min_rows_for_limited:
        reasons.append('too_few_rows_for_limited')
        return CaseDecision('no_data', metrics, reasons)

    if (
        coverage_days >= float(sufficient_days)
        and unique_days >= int(sufficient_days)
        and missing_ratio <= float(max_missing_ratio_for_sufficient)
        and control_events >= int(min_control_events_for_sufficient)
    ):
        return CaseDecision('sufficient', metrics, reasons)

    if coverage_days >= float(limited_days) or unique_days >= int(limited_days):
        if missing_ratio > float(max_missing_ratio_for_sufficient):
            reasons.append('high_missing_ratio')
        if control_events < int(min_control_events_for_sufficient):
            reasons.append('limited_control_events')
        return CaseDecision('limited', metrics, reasons)

    reasons.append('short_coverage_window')
    return CaseDecision('no_data', metrics, reasons)
