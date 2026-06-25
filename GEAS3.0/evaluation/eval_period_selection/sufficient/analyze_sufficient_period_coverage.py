from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql


DB_CONFIG = {
    "host": "211.195.9.227",
    "user": "root",
    "password": "theimc#10!",
    "database": "farmstom",
    "port": 3306,
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

TABLE = "data_silla_enc"
IOT_DATA_IDX = 97
SUFFICIENT_START = "2025-11-29 00:00:00"
SUFFICIENT_END = "2026-01-22 00:00:00"
PERCENTILES = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]

VARIABLE_SPECS = [
    {
        "name": "out_temp",
        "label": "Outdoor Temperature",
        "unit": "C",
        "tail": "both",
        "thresholds": [
            {"q": 0.01, "label": "recommended lower q01", "color": "#55A868"},
        ],
    },
    {
        "name": "out_hum",
        "label": "Outdoor Humidity",
        "unit": "%",
        "tail": "both",
        "thresholds": [
            {"q": 0.95, "label": "recommended upper q95", "color": "#C44E52"},
        ],
    },
    {
        "name": "out_light",
        "label": "Outdoor Light",
        "unit": "",
        "tail": "upper",
        "thresholds": [
            {"q": 0.98, "label": "recommended upper q98", "color": "#8172B2"},
        ],
    },
    {
        "name": "out_windsp",
        "label": "Outdoor Wind Speed",
        "unit": "",
        "tail": "upper",
        "thresholds": [
            {"q": 0.97, "label": "recommended upper q97", "color": "#937860"},
        ],
    },
]


@dataclass
class VariableCoverage:
    variable: str
    full_min: float
    full_max: float
    period_min: float
    period_max: float
    full_count: int
    period_count: int
    period_share_pct: float
    min_pct_in_full: float
    max_pct_in_full: float
    abs_max_pct_in_full: float | None
    abs_min_pct_in_full: float | None
    q90_value: float
    q95_value: float
    q98_value: float
    q99_value: float
    covered_q90: bool
    covered_q95: bool
    covered_q98: bool
    covered_q99: bool


def _ensure_output_dir() -> Path:
    base = Path(__file__).resolve().parent / "results"
    base.mkdir(parents=True, exist_ok=True)
    out = base / pd.Timestamp.now(tz="UTC").strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    return out


def _fetch_dataframe() -> pd.DataFrame:
    query = f"""
        SELECT
            reg_date,
            out_temp,
            out_hum,
            out_light,
            out_windsp,
            etc_blackout,
            etc_plc_abnorm
        FROM {TABLE}
        WHERE iot_data_idx = %s
          AND reg_date IS NOT NULL
        ORDER BY reg_date
    """
    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, [IOT_DATA_IDX])
            rows = cur.fetchall()
    return pd.DataFrame(rows)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["reg_date"] = pd.to_datetime(out["reg_date"], errors="coerce")
    for col in [spec["name"] for spec in VARIABLE_SPECS] + ["etc_blackout", "etc_plc_abnorm"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["etc_blackout", "etc_plc_abnorm"]:
        if col in out.columns:
            out = out[(out[col].isna()) | (out[col] == 0)]
    out = out.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)
    return out


def _percentile_rank(series: pd.Series, value: float) -> float:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float((arr <= value).mean())


def _build_coverage_row(full_s: pd.Series, period_s: pd.Series, name: str) -> VariableCoverage:
    full_v = pd.to_numeric(full_s, errors="coerce").dropna()
    period_v = pd.to_numeric(period_s, errors="coerce").dropna()
    if full_v.empty or period_v.empty:
        raise ValueError(f"No usable data for {name}")

    min_pct = _percentile_rank(full_v, float(period_v.min()))
    max_pct = _percentile_rank(full_v, float(period_v.max()))
    abs_series = full_v.abs()
    abs_period = period_v.abs()
    abs_max_pct = _percentile_rank(abs_series, float(abs_period.max())) if name == "out_temp" else None
    abs_min_pct = _percentile_rank(abs_series, float(abs_period.min())) if name == "out_temp" else None

    q90 = float(full_v.quantile(0.90))
    q95 = float(full_v.quantile(0.95))
    q98 = float(full_v.quantile(0.98))
    q99 = float(full_v.quantile(0.99))

    return VariableCoverage(
        variable=name,
        full_min=float(full_v.min()),
        full_max=float(full_v.max()),
        period_min=float(period_v.min()),
        period_max=float(period_v.max()),
        full_count=int(len(full_v)),
        period_count=int(len(period_v)),
        period_share_pct=float(100.0 * len(period_v) / len(full_v)),
        min_pct_in_full=min_pct,
        max_pct_in_full=max_pct,
        abs_max_pct_in_full=abs_max_pct,
        abs_min_pct_in_full=abs_min_pct,
        q90_value=q90,
        q95_value=q95,
        q98_value=q98,
        q99_value=q99,
        covered_q90=bool(float(period_v.max()) >= q90),
        covered_q95=bool(float(period_v.max()) >= q95),
        covered_q98=bool(float(period_v.max()) >= q98),
        covered_q99=bool(float(period_v.max()) >= q99),
    )


def _summarize_percentiles(full_df: pd.DataFrame, period_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for spec in VARIABLE_SPECS:
        name = spec["name"]
        full_v = pd.to_numeric(full_df[name], errors="coerce").dropna()
        period_v = pd.to_numeric(period_df[name], errors="coerce").dropna()
        for q in PERCENTILES:
            value = float(full_v.quantile(q))
            rows.append(
                {
                    "variable": name,
                    "percentile": q,
                    "full_value": value,
                    "period_reaches_upper_tail": bool(float(period_v.max()) >= value),
                    "period_max": float(period_v.max()),
                    "period_min": float(period_v.min()),
                }
            )
    return pd.DataFrame(rows)


def _make_histogram(full_s: pd.Series, period_s: pd.Series, spec: Dict[str, str], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    full_v = pd.to_numeric(full_s, errors="coerce").dropna()
    period_v = pd.to_numeric(period_s, errors="coerce").dropna()
    bins = np.histogram_bin_edges(full_v, bins=80)
    ax.hist(full_v, bins=bins, alpha=0.45, label="DB full history", color="#4C72B0")
    ax2.hist(period_v, bins=bins, alpha=0.55, label="Sufficient period", color="#DD8452")
    for thr in spec.get("thresholds", []):
        q = float(thr["q"])
        label = str(thr["label"])
        color = str(thr["color"])
        thr_value = float(full_v.quantile(q))
        ax.axvline(
            thr_value,
            color=color,
            linestyle="--",
            linewidth=1.6,
            label=label,
        )
        ax2.axvline(
            thr_value,
            color=color,
            linestyle="--",
            linewidth=1.6,
        )
    xlabel = spec["label"] if not spec["unit"] else f"{spec['label']} ({spec['unit']})"
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("DB full history count", fontsize=14, color="#4C72B0")
    ax2.set_ylabel("Sufficient period count", fontsize=14, color="#DD8452")
    ax.set_title(f"{spec['label']} distribution", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.tick_params(axis="y", colors="#4C72B0")
    ax2.tick_params(axis="y", labelsize=12, colors="#DD8452")
    ax.grid(alpha=0.25)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / f"{spec['name']}_hist.png", dpi=160)
    plt.close(fig)


def _make_ecdf(full_s: pd.Series, period_s: pd.Series, spec: Dict[str, str], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    full_v = np.sort(pd.to_numeric(full_s, errors="coerce").dropna().to_numpy(dtype=float))
    period_v = np.sort(pd.to_numeric(period_s, errors="coerce").dropna().to_numpy(dtype=float))
    full_p = np.arange(1, len(full_v) + 1) / len(full_v)
    period_p = np.arange(1, len(period_v) + 1) / len(period_v)
    ax.plot(full_v, full_p, label="DB full history", color="#4C72B0", linewidth=2)
    ax.plot(period_v, period_p, label="Sufficient period", color="#DD8452", linewidth=2)
    for thr in spec.get("thresholds", []):
        q = float(thr["q"])
        label = str(thr["label"])
        color = str(thr["color"])
        ax.axvline(
            float(np.quantile(full_v, q)),
            color=color,
            linestyle="--",
            linewidth=1.6,
            label=label,
        )
    ax.set_title(f"{spec['label']} ECDF", fontsize=16)
    xlabel = spec["label"] if not spec["unit"] else f"{spec['label']} ({spec['unit']})"
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Cumulative probability", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / f"{spec['name']}_ecdf.png", dpi=160)
    plt.close(fig)


def _write_summary(
    output_dir: Path,
    full_df: pd.DataFrame,
    period_df: pd.DataFrame,
    coverage_rows: Iterable[VariableCoverage],
) -> None:
    coverage_df = pd.DataFrame([vars(row) for row in coverage_rows])
    percentiles_df = _summarize_percentiles(full_df, period_df)
    coverage_df.to_csv(output_dir / "coverage_summary.csv", index=False)
    percentiles_df.to_csv(output_dir / "percentile_thresholds.csv", index=False)

    metadata = {
        "db_table": TABLE,
        "iot_data_idx": IOT_DATA_IDX,
        "sufficient_period_start": SUFFICIENT_START,
        "sufficient_period_end_exclusive": SUFFICIENT_END,
        "full_history_start": str(full_df["reg_date"].min()),
        "full_history_end": str(full_df["reg_date"].max()),
        "full_history_rows": int(len(full_df)),
        "period_rows": int(len(period_df)),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    output_dir = _ensure_output_dir()
    raw_df = _fetch_dataframe()
    full_df = _clean_dataframe(raw_df)

    period_mask = (
        (full_df["reg_date"] >= pd.Timestamp(SUFFICIENT_START))
        & (full_df["reg_date"] < pd.Timestamp(SUFFICIENT_END))
    )
    period_df = full_df.loc[period_mask].copy().reset_index(drop=True)
    if period_df.empty:
        raise RuntimeError("The requested sufficient period contains no rows after cleaning.")

    coverage_rows: List[VariableCoverage] = []
    for spec in VARIABLE_SPECS:
        name = spec["name"]
        coverage_rows.append(_build_coverage_row(full_df[name], period_df[name], name))
        _make_histogram(full_df[name], period_df[name], spec, output_dir)
        _make_ecdf(full_df[name], period_df[name], spec, output_dir)

    _write_summary(output_dir, full_df, period_df, coverage_rows)
    print(f"saved_outputs: {output_dir}")


if __name__ == "__main__":
    main()
