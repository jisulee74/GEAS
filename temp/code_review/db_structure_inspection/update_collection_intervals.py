from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import pymysql


ROOT = Path(__file__).resolve().parent
COLUMN_INFO_PATH = ROOT / "column_info.csv"
MISSING_TOKENS = frozenset({"", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "none"})


def quote_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return "`" + value.replace("`", "``") + "`"


def connect() -> pymysql.connections.Connection:
    conn = pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "farmstom"),
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    with conn.cursor() as cur:
        cur.execute("SET time_zone = %s", ("+09:00",))
        cur.execute("SET NAMES utf8mb4")
    return conn


def semantic_not_missing(series: pd.Series, missing_tokens: Iterable[str] = MISSING_TOKENS) -> pd.Series:
    mask = series.notna()
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        stripped = series.astype("string").str.strip()
        mask = mask & ~stripped.isin(set(missing_tokens))
    return mask


def format_interval(minutes: float | None) -> str:
    if minutes is None or not pd.notna(minutes):
        return ""
    rounded = round(float(minutes))
    if abs(float(minutes) - rounded) < 1e-6:
        return f"{rounded}min"
    return f"{float(minutes):.2f}min"


def infer_interval_minutes(df: pd.DataFrame, column: str) -> float | None:
    if "reg_date" not in df.columns or column not in df.columns:
        return None

    time = pd.to_datetime(df["reg_date"], errors="coerce")
    valid_time = time.notna()
    if column != "reg_date":
        valid_time = valid_time & semantic_not_missing(df[column])

    ts = time[valid_time].drop_duplicates().sort_values()
    if len(ts.index) < 2:
        return None

    diffs = ts.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None

    return float(diffs.median() / 60.0)


def fetch_columns(columns: list[str]) -> pd.DataFrame:
    table = os.getenv("MAIN_TABLE_NAME", "data_silla_enc")
    id_col = os.getenv("ID_IDX", "iot_data_idx")
    farm_sn = int(os.getenv("FARM_SN", os.getenv("DB_FARM_SN", "97")))
    select_cols = ", ".join(quote_identifier(col) for col in columns)
    sql = f"""
        SELECT {select_cols}
        FROM {quote_identifier(table)}
        WHERE {quote_identifier(id_col)} = %s
        ORDER BY {quote_identifier("reg_date")} ASC
    """

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (farm_sn,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return pd.DataFrame(rows)


def main() -> None:
    info = pd.read_csv(COLUMN_INFO_PATH)
    columns = [str(col) for col in info["컬럼명"].tolist()]
    df = fetch_columns(columns)

    intervals = {
        column: format_interval(infer_interval_minutes(df, column))
        for column in columns
    }
    info["수집 주기(GPT)"] = info["컬럼명"].map(intervals).fillna("")

    desired = list(info.columns)
    desired.remove("수집 주기(GPT)")
    insert_at = desired.index("min") if "min" in desired else len(desired)
    desired.insert(insert_at, "수집 주기(GPT)")
    info = info[desired]
    info.to_csv(COLUMN_INFO_PATH, index=False, encoding="utf-8-sig")

    print(f"[COLLECTION INTERVAL] rows={len(df.index)}")
    print(info[["컬럼명", "수집 주기(GPT)"]].to_string(index=False))


if __name__ == "__main__":
    main()
