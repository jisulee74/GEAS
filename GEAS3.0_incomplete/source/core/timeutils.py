from __future__ import annotations
import pandas as pd
from typing import Tuple

KST = "Asia/Seoul"

def ensure_datetime(df: pd.DataFrame, col: str = "reg_date") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is required.")

    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)

def kst_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=KST)

def today_window_kst(now: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:

    if now.tzinfo is None:
        now = now.tz_localize(KST)
        
    sod = now.normalize().tz_convert(KST).tz_localize(None)
    eod = sod + pd.Timedelta(days=1)
    return sod, eod

def slice_today(df: pd.DataFrame,
                now_kst: pd.Timestamp,
                col: str = "reg_date") -> pd.DataFrame:

    sod, eod = today_window_kst(now_kst)

    reg = df[col]
    if getattr(reg.dt, "tz", None) is not None:
        reg = reg.dt.tz_convert(KST).dt.tz_localize(None)

    mask = (reg >= sod) & (reg < eod)
    return df.loc[mask].copy()

def latest_row(df: pd.DataFrame, col: str = "reg_date") -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame([{
            col: pd.Timestamp.now(tz=KST).normalize().tz_localize(None)
        }])
    return df.tail(1).reset_index(drop=True)

def normalize_timestamp_kst(ts):
    if isinstance(ts, str):
        ts = pd.to_datetime(ts, errors="coerce")

    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            return ts.tz_convert(KST).tz_localize(None).to_pydatetime()
        return ts.to_pydatetime()

    if isinstance(ts, (pd.Timestamp, )):
        return ts

    if isinstance(ts, (int, float)):
        ts = pd.Timestamp(ts, unit="s")
        return ts.tz_localize(KST).tz_localize(None).to_pydatetime()

    if hasattr(ts, "tzinfo"):
        ts = pd.Timestamp(ts).tz_convert(KST)
        return ts.tz_localize(None).to_pydatetime()

    raise ValueError(f"Unsupported timestamp type: {ts}")
