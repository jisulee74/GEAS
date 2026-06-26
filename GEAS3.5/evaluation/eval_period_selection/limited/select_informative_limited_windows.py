from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
from tqdm import tqdm


EVAL_ROOT = Path(__file__).resolve().parents[2]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

import limited_data_eval as lde  # noqa: E402


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

# User-agreed first-stage thresholds from the sufficient benchmark slide.
SUFFICIENT_THRESHOLDS = {
    "temp_low_q01": {"column": "out_temp", "op": "<=", "value": -8.0, "label": "out_temp <= -8.0C"},
    "hum_high_q95": {"column": "out_hum", "op": ">=", "value": 96.0, "label": "out_hum >= 96.0%"},
    "light_high_q98": {"column": "out_light", "op": ">=", "value": 1259.0, "label": "out_light >= 1259 W/m2"},
    "wind_high_q97": {"column": "out_windsp", "op": ">=", "value": 10.0, "label": "out_windsp >= 10 m/s"},
}

CORE_KPI_COLUMNS = [
    "temp_viol_rate",
    "cond_viol_rate",
    "rh_viol_rate",
    "vpd_viol_rate",
]


@dataclass
class WindowAssessment:
    start_date: str
    end_date_exclusive: str
    window_days: int
    n_rows: int
    data_case_stage1: str
    n_failed_thresholds: int
    failed_thresholds: str
    informative_stage2: bool
    active_kpi_count: int
    tail_gap_score: float
    kpi_dispersion_score: float
    posterior_spread_score: float
    stage2_signal_score: float
    notes: str


def _ensure_output_dir() -> Path:
    base = Path(__file__).resolve().parent / "results"
    base.mkdir(parents=True, exist_ok=True)
    out = base / pd.Timestamp.now(tz="UTC").strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    return out


def _fetch_full_history() -> pd.DataFrame:
    query = f"""
        SELECT
            *
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


def _clean_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["reg_date"] = pd.to_datetime(out["reg_date"], errors="coerce")
    numeric_candidates = [
        "out_temp",
        "out_hum",
        "out_light",
        "out_windsp",
        "etc_blackout",
        "etc_plc_abnorm",
    ]
    for col in numeric_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["etc_blackout", "etc_plc_abnorm"]:
        if col in out.columns:
            out = out[(out[col].isna()) | (out[col] == 0)]

    out = out.dropna(subset=["reg_date"]).sort_values("reg_date").reset_index(drop=True)
    return out


def _compute_threshold_share(series: pd.Series, op: str, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    if op == "<=":
        return float(100.0 * (s <= value).mean())
    if op == ">=":
        return float(100.0 * (s >= value).mean())
    raise ValueError(f"Unsupported operator: {op}")


def _classify_stage1(df_window: pd.DataFrame, min_required_days: int) -> Dict[str, object]:
    if df_window.empty:
        return {"data_case": "no_data", "failed_thresholds": ["all"], "shares": {}}

    span_days = (
        (pd.Timestamp(df_window["reg_date"].max()) - pd.Timestamp(df_window["reg_date"].min())).total_seconds()
        / 86400.0
    )
    if span_days < float(min_required_days):
        return {"data_case": "no_data", "failed_thresholds": ["too_short"], "shares": {}}

    shares: Dict[str, float] = {}
    failed: List[str] = []
    for name, spec in SUFFICIENT_THRESHOLDS.items():
        share = _compute_threshold_share(df_window[spec["column"]], str(spec["op"]), float(spec["value"]))
        shares[name] = share
        if (not np.isfinite(share)) or share < 1.0:
            failed.append(name)

    data_case = "sufficient" if not failed else "limited"
    return {"data_case": data_case, "failed_thresholds": failed, "shares": shares}


def _run_stage2_eval(
    df_window: pd.DataFrame,
    *,
    pmmh_iters: int,
    burn_in: int,
    thin: int,
    n_particles: int,
    max_events: int,
    window_steps: int,
) -> Dict[str, object]:
    result = lde.run_limited_data_eval(
        df_window,
        pmmh_iters=int(pmmh_iters),
        burn_in=int(burn_in),
        thin=int(thin),
        n_particles=int(n_particles),
        max_events=int(max_events),
        window_steps=int(window_steps),
        posterior_seed=42,
        show_progress=False,
    )
    metrics_df = result["robust"]["metrics"].copy()
    posterior_df = result["posterior"]["posterior_samples"].copy()

    kpi_rows: List[Dict[str, float]] = []
    active_kpi_count = 0
    tail_gap_score = 0.0
    kpi_dispersion_score = 0.0

    for col in CORE_KPI_COLUMNS:
        s = pd.to_numeric(metrics_df[col], errors="coerce").dropna()
        q90 = float(np.quantile(s.values, 0.9))
        c90 = float(lde.cvar(s.values, 0.9))
        gap = float(abs(c90 - q90))
        std = float(s.std()) if len(s) > 1 else 0.0
        n_unique = int(s.nunique())
        is_active = bool((gap > 1e-4) or (std > 1e-4) or (n_unique >= 3))
        if is_active:
            active_kpi_count += 1
        tail_gap_score += gap
        kpi_dispersion_score += std
        kpi_rows.append(
            {
                "metric": col,
                "q90": q90,
                "cvar90": c90,
                "tail_gap": gap,
                "std": std,
                "n_unique": n_unique,
                "active": int(is_active),
            }
        )

    theta_spreads: List[Dict[str, float]] = []
    posterior_spread_score = 0.0
    for col in lde._THETA_COLUMNS:
        if col not in posterior_df.columns:
            continue
        s = pd.to_numeric(posterior_df[col], errors="coerce").dropna()
        if s.empty:
            continue
        std = float(s.std()) if len(s) > 1 else 0.0
        rel_std = float(std / max(abs(float(s.mean())), 1e-9))
        posterior_spread_score += rel_std
        theta_spreads.append({"theta": col, "std": std, "rel_std": rel_std, "n_unique": int(s.nunique())})

    informative = bool(active_kpi_count >= 2 and (tail_gap_score > 1e-4 or kpi_dispersion_score > 1e-4))
    stage2_signal_score = float(active_kpi_count + 10.0 * tail_gap_score + kpi_dispersion_score + posterior_spread_score)

    return {
        "informative": informative,
        "active_kpi_count": active_kpi_count,
        "tail_gap_score": tail_gap_score,
        "kpi_dispersion_score": kpi_dispersion_score,
        "posterior_spread_score": posterior_spread_score,
        "stage2_signal_score": stage2_signal_score,
        "kpi_diagnostics": pd.DataFrame(kpi_rows),
        "theta_spreads": pd.DataFrame(theta_spreads),
    }


def _save_scatter(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    limited = summary_df[summary_df["data_case_stage1"] == "limited"]
    colors = np.where(limited["informative_stage2"], "#C44E52", "#4C72B0")
    ax.scatter(
        limited["window_days"],
        limited["stage2_signal_score"],
        c=colors,
        alpha=0.8,
        s=55,
    )
    ax.set_title("Limited Window Screening: Stage-2 Signal by Window Length", fontsize=16)
    ax.set_xlabel("Window length (days)", fontsize=14)
    ax.set_ylabel("Stage-2 signal score", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "stage2_signal_scatter.png", dpi=160)
    plt.close(fig)


def run_selection(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _ensure_output_dir()
    full_df = _clean_history(_fetch_full_history())

    results: List[WindowAssessment] = []
    selected_stage2_outputs: Dict[str, Dict[str, str]] = {}

    start_day = pd.Timestamp(args.search_start) if args.search_start else pd.Timestamp(full_df["reg_date"].min()).floor("D")
    last_day = pd.Timestamp(args.search_end) if args.search_end else pd.Timestamp(full_df["reg_date"].max()).floor("D")

    window_days_list = [int(x) for x in str(args.window_days_list).split(",")]
    tasks: List[Dict[str, object]] = []
    for window_days in window_days_list:
        cur = start_day
        max_start = last_day - pd.Timedelta(days=window_days)
        while cur <= max_start:
            tasks.append({"start": cur, "end": cur + pd.Timedelta(days=window_days), "window_days": window_days})
            cur += pd.Timedelta(days=int(args.step_days))

    progress = tqdm(tasks, desc="limited_window_selection", unit="window")
    for task in progress:
        win_df = full_df[(full_df["reg_date"] >= task["start"]) & (full_df["reg_date"] < task["end"])].copy()
        stage1 = _classify_stage1(win_df, min_required_days=int(args.min_required_days))

        progress.set_postfix(
            start=str(task["start"].date()),
            days=int(task["window_days"]),
            stage1=stage1["data_case"],
        )

        assessment = WindowAssessment(
            start_date=str(pd.Timestamp(task["start"]).date()),
            end_date_exclusive=str(pd.Timestamp(task["end"]).date()),
            window_days=int(task["window_days"]),
            n_rows=int(len(win_df)),
            data_case_stage1=str(stage1["data_case"]),
            n_failed_thresholds=int(len(stage1["failed_thresholds"])),
            failed_thresholds=",".join(stage1["failed_thresholds"]),
            informative_stage2=False,
            active_kpi_count=0,
            tail_gap_score=0.0,
            kpi_dispersion_score=0.0,
            posterior_spread_score=0.0,
            stage2_signal_score=0.0,
            notes="",
        )

        if stage1["data_case"] != "limited":
            results.append(assessment)
            continue

        try:
            stage2 = _run_stage2_eval(
                win_df,
                pmmh_iters=int(args.pmmh_iters),
                burn_in=int(args.burn_in),
                thin=int(args.thin),
                n_particles=int(args.n_particles),
                max_events=int(args.max_events),
                window_steps=int(args.window_steps),
            )
            assessment.informative_stage2 = bool(stage2["informative"])
            assessment.active_kpi_count = int(stage2["active_kpi_count"])
            assessment.tail_gap_score = float(stage2["tail_gap_score"])
            assessment.kpi_dispersion_score = float(stage2["kpi_dispersion_score"])
            assessment.posterior_spread_score = float(stage2["posterior_spread_score"])
            assessment.stage2_signal_score = float(stage2["stage2_signal_score"])
            assessment.notes = "stage2_ok" if stage2["informative"] else "stage2_degenerate"

            key = f"{assessment.start_date}_to_{assessment.end_date_exclusive}"
            kpi_path = output_dir / f"{key}_kpi_diagnostics.csv"
            theta_path = output_dir / f"{key}_theta_spreads.csv"
            stage2["kpi_diagnostics"].to_csv(kpi_path, index=False)
            stage2["theta_spreads"].to_csv(theta_path, index=False)
            selected_stage2_outputs[key] = {
                "kpi_diagnostics_csv": str(kpi_path),
                "theta_spreads_csv": str(theta_path),
            }
        except Exception as exc:
            assessment.notes = f"stage2_error:{type(exc).__name__}"

        results.append(assessment)

    summary_df = pd.DataFrame([asdict(r) for r in results])
    summary_df.to_csv(output_dir / "window_selection_summary.csv", index=False)
    _save_scatter(summary_df, output_dir)

    informative = summary_df[
        (summary_df["data_case_stage1"] == "limited") & (summary_df["informative_stage2"])
    ].copy()
    informative = informative.sort_values(
        ["active_kpi_count", "stage2_signal_score", "window_days"],
        ascending=[False, False, True],
    )

    recommendation = informative.head(int(args.top_k)).to_dict(orient="records")
    metadata = {
        "thresholds_stage1": SUFFICIENT_THRESHOLDS,
        "window_days_list": window_days_list,
        "step_days": int(args.step_days),
        "min_required_days": int(args.min_required_days),
        "stage2_heuristics": {
            "core_kpi_columns": CORE_KPI_COLUMNS,
            "informative_condition": "active_kpi_count >= 2 and (tail_gap_score > 1e-4 or kpi_dispersion_score > 1e-4)",
        },
        "selected_stage2_outputs": selected_stage2_outputs,
        "recommendation_top_k": recommendation,
    }
    (output_dir / "selection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "window_selection_summary.csv"),
        "metadata_json": str(output_dir / "selection_metadata.json"),
        "recommended_windows": recommendation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select informative limited-data windows using a two-stage filter.",
    )
    parser.add_argument("--search-start", type=str, default="", help="Optional search start date YYYY-MM-DD.")
    parser.add_argument("--search-end", type=str, default="", help="Optional search end date YYYY-MM-DD.")
    parser.add_argument(
        "--window-days-list",
        type=str,
        default="14,21,28,35,42,49",
        help="Comma-separated candidate window lengths in days.",
    )
    parser.add_argument("--step-days", type=int, default=7, help="Sliding step in days.")
    parser.add_argument("--min-required-days", type=int, default=7, help="Minimum span to avoid no-data label.")
    parser.add_argument("--pmmh-iters", type=int, default=120, help="Stage-2 PMMH iterations for screening.")
    parser.add_argument("--burn-in", type=int, default=40, help="Stage-2 PMMH burn-in.")
    parser.add_argument("--thin", type=int, default=2, help="Stage-2 PMMH thinning.")
    parser.add_argument("--n-particles", type=int, default=48, help="Stage-2 particle count.")
    parser.add_argument("--max-events", type=int, default=12, help="Stage-2 max event windows.")
    parser.add_argument("--window-steps", type=int, default=12, help="Stage-2 event window steps.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommended windows to keep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_selection(args)
    print(f"saved_outputs: {result['output_dir']}")
    print(json.dumps(result["recommended_windows"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
