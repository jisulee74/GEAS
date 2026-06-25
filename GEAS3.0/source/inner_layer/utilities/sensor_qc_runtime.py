from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_INNER_ROOT = Path(__file__).resolve().parents[1]
_SENSOR_QC_API: Optional[Dict[str, Any]] = None


def _load_sensor_qc_api() -> Optional[Dict[str, Any]]:
    global _SENSOR_QC_API
    if _SENSOR_QC_API is not None:
        return _SENSOR_QC_API

    model_path = _INNER_ROOT / "core" / "sensor_QC.py"
    if not model_path.exists():
        _SENSOR_QC_API = None
        return None

    try:
        _SENSOR_QC_API = runpy.run_path(str(model_path))
    except Exception as exc:
        print(f"[SENSOR QC] load failed: {exc}")
        _SENSOR_QC_API = None
    return _SENSOR_QC_API


def apply_sensor_calibration(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw

    enabled = os.getenv("ENABLE_SENSOR_QC", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return df_raw

    min_rows = int(os.getenv("SENSOR_QC_MIN_ROWS", "160"))
    if len(df_raw.index) < min_rows:
        return df_raw

    api = _load_sensor_qc_api()
    if not api or "make_state_df" not in api:
        return df_raw

    make_state_df = api["make_state_df"]
    max_win = int(os.getenv("SENSOR_QC_WINDOW", "288"))
    step = int(os.getenv("SENSOR_QC_STEP", "60"))
    alpha = float(os.getenv("SENSOR_QC_ALPHA", "0.01"))

    win = min(max_win, len(df_raw.index) - 1)
    if win <= 1:
        return df_raw
    step = max(1, min(step, max(1, win // 4)))

    try:
        df_state = make_state_df(df_raw, plot=False, win=win, step=step, alpha=alpha)
    except Exception as exc:
        print(f"[SENSOR QC] calibration skipped: {exc}")
        return df_raw

    if df_state.empty or "time" not in df_state.columns:
        return df_raw

    calibrated = df_raw.copy()
    calibrated["reg_date"] = pd.to_datetime(calibrated["reg_date"], errors="coerce")

    qc_cols = [c for c in ["time", "Tin_corr", "RHin_corr", "CO2_corr", "QC_flag", "CO2_QC_flag"] if c in df_state.columns]
    merge_df = df_state[qc_cols].copy().rename(columns={"time": "reg_date"})
    calibrated = calibrated.merge(merge_df, on="reg_date", how="left")

    if "Tin_corr" in calibrated.columns:
        calibrated["in_temp"] = calibrated["Tin_corr"].where(calibrated["Tin_corr"].notna(), calibrated.get("in_temp"))
    if "RHin_corr" in calibrated.columns:
        calibrated["in_hum"] = calibrated["RHin_corr"].where(calibrated["RHin_corr"].notna(), calibrated.get("in_hum"))
    if "CO2_corr" in calibrated.columns:
        calibrated["in_co2"] = calibrated["CO2_corr"].where(calibrated["CO2_corr"].notna(), calibrated.get("in_co2"))

    if "QC_flag" in calibrated.columns:
        calibrated["sensor_qc_flag"] = calibrated["QC_flag"].fillna(False).astype(bool)
    if "CO2_QC_flag" in calibrated.columns:
        calibrated["sensor_qc_co2_flag"] = calibrated["CO2_QC_flag"].fillna(False).astype(bool)

    temp_changed = 0
    hum_changed = 0
    co2_changed = 0
    if "Tin_corr" in calibrated.columns and "in_temp" in df_raw.columns:
        raw_temp = pd.to_numeric(df_raw["in_temp"], errors="coerce")
        cal_temp = pd.to_numeric(calibrated["in_temp"], errors="coerce")
        temp_changed = int(((raw_temp - cal_temp).abs() > 1e-9).fillna(False).sum())
    if "RHin_corr" in calibrated.columns and "in_hum" in df_raw.columns:
        raw_hum = pd.to_numeric(df_raw["in_hum"], errors="coerce")
        cal_hum = pd.to_numeric(calibrated["in_hum"], errors="coerce")
        hum_changed = int(((raw_hum - cal_hum).abs() > 1e-9).fillna(False).sum())
    if "CO2_corr" in calibrated.columns and "in_co2" in df_raw.columns:
        raw_co2 = pd.to_numeric(df_raw["in_co2"], errors="coerce")
        cal_co2 = pd.to_numeric(calibrated["in_co2"], errors="coerce")
        co2_changed = int(((raw_co2 - cal_co2).abs() > 1e-9).fillna(False).sum())

    total_rows = max(int(len(calibrated.index)), 1)
    qc_rows = int(calibrated.get("sensor_qc_flag", pd.Series(False, index=calibrated.index)).fillna(False).sum())
    qc_co2_rows = int(calibrated.get("sensor_qc_co2_flag", pd.Series(False, index=calibrated.index)).fillna(False).sum())
    print(
        "[SENSOR QC] applied "
        f"rows={total_rows}, temp_changed={temp_changed}, hum_changed={hum_changed}, co2_changed={co2_changed}, "
        f"qc_rows={qc_rows} ({qc_rows/total_rows:.1%}), co2_qc_rows={qc_co2_rows} ({qc_co2_rows/total_rows:.1%})"
    )

    return calibrated


__all__ = ["apply_sensor_calibration"]
