from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import pymysql


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


def fetch_timestamps(
    *,
    start: str,
    end: str,
    table: str,
    id_column: str,
    farm_sn: int,
) -> pd.Series:
    sql = f"""
        SELECT reg_date
        FROM {quote_identifier(table)}
        WHERE {quote_identifier(id_column)} = %s
          AND reg_date >= %s
          AND reg_date < %s
        ORDER BY reg_date ASC
    """
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (farm_sn, pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()))
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.to_datetime(pd.Series([row["reg_date"] for row in rows]), errors="coerce").dropna()


def build_gap_report(
    ts: pd.Series,
    *,
    start: str,
    end: str,
    expected_minutes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    expected_delta = pd.Timedelta(minutes=expected_minutes)
    expected_rows = int((end_ts - start_ts) / expected_delta)
    actual_rows = int(len(ts.index))

    gap_rows: List[dict[str, Any]] = []
    if actual_rows:
        first_ts = ts.iloc[0]
        last_ts = ts.iloc[-1]
        if first_ts > start_ts:
            missing_slots = int((first_ts - start_ts) / expected_delta)
            gap_rows.append(
                {
                    "gap_type": "before_first_row",
                    "gap_start": start_ts,
                    "gap_end": first_ts,
                    "gap_minutes": (first_ts - start_ts).total_seconds() / 60.0,
                    "missing_5min_slots": missing_slots,
                }
            )
        if last_ts + expected_delta < end_ts:
            missing_slots = int((end_ts - (last_ts + expected_delta)) / expected_delta)
            gap_rows.append(
                {
                    "gap_type": "after_last_row",
                    "gap_start": last_ts + expected_delta,
                    "gap_end": end_ts,
                    "gap_minutes": (end_ts - (last_ts + expected_delta)).total_seconds() / 60.0,
                    "missing_5min_slots": missing_slots,
                }
            )

    diffs = ts.diff()
    for idx, diff in diffs[diffs > expected_delta].items():
        prev_ts = ts.iloc[idx - 1]
        curr_ts = ts.iloc[idx]
        missing_slots = max(int(diff / expected_delta) - 1, 0)
        gap_rows.append(
            {
                "gap_type": "between_rows",
                "gap_start": prev_ts + expected_delta,
                "gap_end": curr_ts,
                "gap_minutes": (curr_ts - (prev_ts + expected_delta)).total_seconds() / 60.0,
                "missing_5min_slots": missing_slots,
                "previous_row_time": prev_ts,
                "next_row_time": curr_ts,
                "observed_interval_minutes": diff.total_seconds() / 60.0,
            }
        )

    gaps = pd.DataFrame(gap_rows)
    if not gaps.empty:
        gaps = gaps.sort_values(["missing_5min_slots", "gap_minutes"], ascending=False).reset_index(drop=True)

    summary = {
        "analysis_start": start_ts,
        "analysis_end": end_ts,
        "expected_interval_minutes": expected_minutes,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "missing_rows_vs_full_grid": expected_rows - actual_rows,
        "first_data_time": None if actual_rows == 0 else ts.iloc[0],
        "last_data_time": None if actual_rows == 0 else ts.iloc[-1],
        "gap_count": int(len(gaps.index)),
        "missing_5min_slots_in_gaps": 0 if gaps.empty else int(gaps["missing_5min_slots"].sum()),
    }
    return gaps, summary


def build_coverage_intervals(
    ts: pd.Series,
    *,
    expected_minutes: int,
) -> pd.DataFrame:
    if ts.empty:
        return pd.DataFrame(
            columns=[
                "interval_start",
                "interval_end",
                "rows",
                "duration_days",
                "duration_months_approx",
            ]
        )

    expected_delta = pd.Timedelta(minutes=expected_minutes)
    rows = []
    block_start = ts.iloc[0]
    block_end = ts.iloc[0]
    block_rows = 1

    for current in ts.iloc[1:]:
        if current - block_end <= expected_delta:
            block_end = current
            block_rows += 1
            continue

        duration_days = (block_end - block_start).total_seconds() / 86400.0
        rows.append(
            {
                "interval_start": block_start,
                "interval_end": block_end,
                "rows": block_rows,
                "duration_days": duration_days,
                "duration_months_approx": duration_days / 30.4375,
            }
        )
        block_start = current
        block_end = current
        block_rows = 1

    duration_days = (block_end - block_start).total_seconds() / 86400.0
    rows.append(
        {
            "interval_start": block_start,
            "interval_end": block_end,
            "rows": block_rows,
            "duration_days": duration_days,
            "duration_months_approx": duration_days / 30.4375,
        }
    )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find missing timestamp gaps in DB rows.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--expected-minutes", type=int, default=5)
    parser.add_argument("--table", default=os.getenv("MAIN_TABLE_NAME", "data_silla_enc"))
    parser.add_argument("--id-column", default=os.getenv("ID_IDX", "iot_data_idx"))
    parser.add_argument("--farm-sn", type=int, default=int(os.getenv("FARM_SN", os.getenv("DB_FARM_SN", "97"))))
    parser.add_argument("--gaps-csv", default="code_review/missingness_audit/data_gaps.csv")
    parser.add_argument("--summary-csv", default="code_review/missingness_audit/data_gap_summary.csv")
    parser.add_argument("--intervals-csv", default="code_review/missingness_audit/data_coverage_intervals.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = fetch_timestamps(
        start=args.start,
        end=args.end,
        table=args.table,
        id_column=args.id_column,
        farm_sn=args.farm_sn,
    )
    gaps, summary = build_gap_report(
        ts,
        start=args.start,
        end=args.end,
        expected_minutes=args.expected_minutes,
    )
    intervals = build_coverage_intervals(ts, expected_minutes=args.expected_minutes)

    gaps_path = Path(args.gaps_csv)
    summary_path = Path(args.summary_csv)
    intervals_path = Path(args.intervals_csv)
    gaps_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    intervals_path.parent.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(gaps_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    intervals.to_csv(intervals_path, index=False, encoding="utf-8-sig")

    print("[DATA GAP SUMMARY]")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"[DATA GAP] wrote {gaps_path}")
    print(f"[DATA GAP] wrote {summary_path}")
    print(f"[DATA GAP] wrote {intervals_path}")
    if not gaps.empty:
        print(gaps.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
