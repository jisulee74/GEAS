from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import pymysql


REPO_ROOT = Path(__file__).resolve().parents[2]
UPDATED_ROOT = REPO_ROOT / "GEAS3.0" / "source"
INNER_ROOT = UPDATED_ROOT / "inner_layer"

for path in (str(UPDATED_ROOT), str(INNER_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.config import DbConfig
from core.preprocessing import latest_row_for_controller, normalize_for_derived
from core.solar_eta import integrate_measured_to_jcm2
from core.timeutils import kst_now, slice_today
from utilities.sensor_qc_runtime import apply_sensor_calibration


MISSING_TOKENS = frozenset({"", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "none"})


def semantic_missing_mask(
    df: pd.DataFrame,
    *,
    missing_tokens: Iterable[str] = MISSING_TOKENS,
) -> pd.DataFrame:
    mask = df.isna()
    tokens = set(missing_tokens)
    object_cols = list(df.select_dtypes(include=["object", "string"]).columns)

    for col in object_cols:
        stripped = df[col].astype("string").str.strip()
        mask[col] = mask[col] | stripped.isin(tokens)

    return mask


def summarize_missingness(
    df: pd.DataFrame,
    stage: str,
    *,
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = int(len(df.index))
    cols = int(len(df.columns))
    total_cells = rows * cols
    mask = semantic_missing_mask(df)
    missing_cells = int(mask.to_numpy().sum()) if total_cells else 0
    missing_ratio = missing_cells / total_cells if total_cells else 0.0

    per_column = []
    for col in df.columns:
        missing = int(mask[col].sum())
        per_column.append(
            {
                "column": str(col),
                "missing": missing,
                "total": rows,
                "ratio": missing / rows if rows else 0.0,
            }
        )
    per_column.sort(key=lambda item: (item["ratio"], item["missing"], item["column"]), reverse=True)

    summary: Dict[str, Any] = {
        "stage": stage,
        "rows": rows,
        "columns": cols,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_ratio": missing_ratio,
        "per_column": per_column,
    }
    if "reg_date" in df.columns and rows:
        ts = pd.to_datetime(df["reg_date"], errors="coerce").dropna()
        if not ts.empty:
            data_start = ts.min()
            data_end = ts.max()
            summary["data_start"] = data_start
            summary["data_end"] = data_end
            summary["period_days"] = (data_end - data_start).total_seconds() / 86400.0

    if previous is not None:
        summary["missing_cells_delta"] = missing_cells - int(previous.get("missing_cells", 0))
        summary["missing_ratio_delta"] = missing_ratio - float(previous.get("missing_ratio", 0.0))
        summary["rows_delta"] = rows - int(previous.get("rows", 0))

    return summary


def append_stage(report: List[Dict[str, Any]], stage: str, df: pd.DataFrame) -> None:
    previous = report[-1] if report else None
    report.append(summarize_missingness(df, stage, previous=previous))


def format_report(report: List[Dict[str, Any]], *, top_n: int) -> str:
    lines = ["[MISSINGNESS] stage summary"]
    for item in report:
        ratio_pct = 100.0 * float(item["missing_ratio"])
        delta = ""
        if "missing_ratio_delta" in item:
            delta = f", delta={100.0 * float(item['missing_ratio_delta']):+.2f}%p"
        lines.append(
            f"- {item['stage']}: rows={item['rows']}, cols={item['columns']}, "
            f"missing={item['missing_cells']}/{item['total_cells']} ({ratio_pct:.2f}%{delta})"
        )

        top_cols = [
            f"{col['column']}={100.0 * float(col['ratio']):.1f}%"
            for col in item["per_column"][:max(0, top_n)]
            if int(col["missing"]) > 0
        ]
        if top_cols:
            lines.append(f"  top_missing_columns: {', '.join(top_cols)}")

    return "\n".join(lines)


def report_to_frame(report: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in report:
        rows.append(
            {
                "stage": item["stage"],
                "analysis_start": item.get("analysis_start"),
                "analysis_end": item.get("analysis_end"),
                "rows": item["rows"],
                "columns": item["columns"],
                "total_cells": item["total_cells"],
                "missing_cells": item["missing_cells"],
                "missing_ratio": item["missing_ratio"],
                "missing_ratio_pct": 100.0 * float(item["missing_ratio"]),
                "data_start": item.get("data_start"),
                "data_end": item.get("data_end"),
                "period_days": item.get("period_days"),
                "missing_cells_delta": item.get("missing_cells_delta"),
                "missing_ratio_delta_pct": (
                    None
                    if "missing_ratio_delta" not in item
                    else 100.0 * float(item["missing_ratio_delta"])
                ),
                "rows_delta": item.get("rows_delta"),
            }
        )
    return pd.DataFrame(rows)


def report_columns_to_frame(report: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in report:
        for col in item.get("per_column", []):
            rows.append(
                {
                    "stage": item["stage"],
                    "analysis_start": item.get("analysis_start"),
                    "analysis_end": item.get("analysis_end"),
                    "column": col["column"],
                    "rows": item["rows"],
                    "missing_count": col["missing"],
                    "non_missing_count": int(col["total"]) - int(col["missing"]),
                    "missing_ratio": col["ratio"],
                    "missing_ratio_pct": 100.0 * float(col["ratio"]),
                    "data_start": item.get("data_start"),
                    "data_end": item.get("data_end"),
                    "period_days": item.get("period_days"),
                }
            )
    return pd.DataFrame(rows)


def _connect(cfg: DbConfig) -> pymysql.connections.Connection:
    conn = pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=cfg.name,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    with conn.cursor() as cur:
        cur.execute("SET time_zone = %s", ("+09:00",))
        cur.execute("SET NAMES utf8mb4")
    return conn


def fetch_dataframe(
    cfg: DbConfig,
    *,
    start: Optional[str],
    end: Optional[str],
    limit: Optional[int],
) -> pd.DataFrame:
    where = [f"`{cfg.id_idx}` = %s"]
    params: List[Any] = [cfg.farm_sn]

    if start:
        where.append("reg_date >= %s")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end:
        where.append("reg_date < %s")
        params.append(pd.Timestamp(end).to_pydatetime())

    sql = f"""
        SELECT {cfg.main_cols}
        FROM `{cfg.main_table}`
        WHERE {" AND ".join(where)}
        ORDER BY reg_date ASC
    """
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))

    conn = _connect(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
    finally:
        conn.close()

    return pd.DataFrame(rows)


def build_pipeline_report(
    df_raw: pd.DataFrame,
    *,
    include_today_stages: bool,
    skip_sensor_qc: bool,
) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    append_stage(report, "raw_db_fetch", df_raw)

    if skip_sensor_qc:
        df_calibrated = df_raw.copy()
        append_stage(report, "after_sensor_qc_skipped", df_calibrated)
    else:
        df_calibrated = apply_sensor_calibration(df_raw)
        append_stage(report, "after_sensor_qc", df_calibrated)

    df_norm = normalize_for_derived(df_calibrated)
    append_stage(report, "after_normalize_for_derived", df_norm)

    if include_today_stages:
        df_today = slice_today(df_norm, kst_now(), col="reg_date")
        append_stage(report, "after_slice_today", df_today)

        if not df_today.empty:
            df_today = df_today.copy()
            df_today["out_light_sum"] = integrate_measured_to_jcm2(
                df_today, col="out_light", time_col="reg_date"
            )
            append_stage(report, "after_out_light_sum", df_today)

            latest = latest_row_for_controller(df_today)
            append_stage(report, "latest_row_for_controller", latest)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute missing-value ratios through selected GEAS3.0/source pipeline stages without modifying source code."
    )
    parser.add_argument("--start", help="Inclusive start timestamp, e.g. 2025-11-01 00:00:00")
    parser.add_argument("--end", help="Exclusive end timestamp, e.g. 2025-12-01 00:00:00")
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("MISSINGNESS_LIMIT", "50000")),
        help="Maximum rows to fetch unless --all is set.",
    )
    parser.add_argument("--all", action="store_true", help="Fetch all matching DB rows without LIMIT.")
    parser.add_argument("--today", action="store_true", help="Also run today-only runtime stages.")
    parser.add_argument("--skip-sensor-qc", action="store_true", help="Skip the sensor QC stage if it is too slow.")
    parser.add_argument("--top-n", type=int, default=8, help="Number of missing columns to display per stage.")
    parser.add_argument("--csv", help="Optional CSV path for the stage-level summary.")
    parser.add_argument("--columns-csv", help="Optional CSV path for column-level missing counts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DbConfig.from_env()
    limit = None if args.all else args.limit

    df_raw = fetch_dataframe(cfg, start=args.start, end=args.end, limit=limit)
    if df_raw.empty:
        print("[MISSINGNESS] no rows fetched")
        return

    report = build_pipeline_report(
        df_raw,
        include_today_stages=bool(args.today),
        skip_sensor_qc=bool(args.skip_sensor_qc),
    )
    for item in report:
        item["analysis_start"] = args.start
        item["analysis_end"] = args.end

    print(format_report(report, top_n=args.top_n))

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_to_frame(report).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[MISSINGNESS] wrote {out_path}")

    if args.columns_csv:
        out_path = Path(args.columns_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_columns_to_frame(report).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[MISSINGNESS] wrote {out_path}")


if __name__ == "__main__":
    main()
