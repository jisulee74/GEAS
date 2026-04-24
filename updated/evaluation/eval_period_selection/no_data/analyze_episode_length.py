from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


EVAL_ROOT = Path(__file__).resolve().parents[2]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from no_data_eval import robust_eval  # noqa: E402


TAIL_COLUMNS = [
    "temp_viol_rate",
    "cond_viol_rate",
    "rh_viol_rate",
    "vpd_viol_rate",
]


def _ensure_output_dir() -> Path:
    base = Path(__file__).resolve().parent / "results"
    base.mkdir(parents=True, exist_ok=True)
    out = base / pd.Timestamp.now(tz="UTC").strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    return out


def _normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    vmin = float(s.min())
    vmax = float(s.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - vmin) / (vmax - vmin)


def _build_summary_row(days: int, result: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    row: Dict[str, float] = {"days": int(days)}
    for metric_name in TAIL_COLUMNS:
        row[f"q90_{metric_name}"] = float(result["q90"][metric_name])
        row[f"cvar90_{metric_name}"] = float(result["cvar90"][metric_name])
        values = pd.to_numeric(result["metrics"][metric_name], errors="coerce").dropna().values
        row[f"std_{metric_name}"] = float(np.std(values))
    return row


def _run_single_day(task: Dict[str, int | float]) -> Dict[str, float]:
    days = int(task["days"])
    result = robust_eval(
        N=int(task["N"]),
        days=days,
        dt_min=int(task["dt_min"]),
        seed=int(task["seed"]),
        weights={"wT": 1, "wVPD": 1, "wE": 1e-8, "wDx": 0.2, "wSlack": 50},
        T_min=12.0,
        T_max=28.0,
        RH_max=0.90,
        VPD_min=0.30,
        dTcond_min=0.8,
        alpha=0.25,
        show_progress=False,
        n_jobs=int(task["n_jobs"]),
    )
    return _build_summary_row(days, result)


def _derive_stability_features(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    tracked = [
        "q90_temp_viol_rate",
        "cvar90_temp_viol_rate",
        "q90_cond_viol_rate",
        "cvar90_cond_viol_rate",
    ]

    for col in tracked:
        df[f"delta_{col}"] = df[col].diff().abs()
        denom = df[col].shift(1).abs().clip(lower=1e-9)
        df[f"rel_delta_{col}"] = df[f"delta_{col}"] / denom

    rel_cols = [f"rel_delta_{col}" for col in tracked]
    abs_cols = [f"delta_{col}" for col in tracked]
    df["mean_rel_tail_delta"] = df[rel_cols].mean(axis=1, skipna=True)
    df["max_rel_tail_delta"] = df[rel_cols].max(axis=1, skipna=True)
    df["mean_abs_tail_delta"] = df[abs_cols].mean(axis=1, skipna=True)

    combined_components = [
        _normalize(df["q90_temp_viol_rate"]),
        _normalize(df["cvar90_temp_viol_rate"]),
        _normalize(df["q90_cond_viol_rate"]),
        _normalize(df["cvar90_cond_viol_rate"]),
    ]
    df["combined_tail_index"] = pd.concat(combined_components, axis=1).mean(axis=1)
    df["combined_tail_gain"] = df["combined_tail_index"].diff().abs()
    return df


def _recommend_day(
    df: pd.DataFrame,
    *,
    settle_days: int = 3,
    max_rel_threshold: float = 0.05,
) -> Dict[str, float | int | str]:
    work = df.copy().reset_index(drop=True)
    recommended = None
    for idx in range(1, len(work) - settle_days + 1):
        window = work.loc[idx : idx + settle_days - 1, "max_rel_tail_delta"]
        if window.notna().all() and bool((window <= max_rel_threshold).all()):
            recommended = work.loc[idx]
            break

    if recommended is None:
        # Fallback: choose the minimum-stability point after day 3.
        candidate = work.loc[work["days"] >= 3].copy()
        min_idx = candidate["max_rel_tail_delta"].idxmin()
        recommended = work.loc[min_idx]
        reason = (
            f"No day satisfied max_rel_tail_delta <= {max_rel_threshold:.3f} "
            f"for {settle_days} consecutive days, so the minimum observed point was selected."
        )
    else:
        reason = (
            f"Selected the first day where max_rel_tail_delta stayed <= {max_rel_threshold:.3f} "
            f"for {settle_days} consecutive days."
        )

    return {
        "recommended_days": int(recommended["days"]),
        "reason": reason,
        "max_rel_tail_delta": float(recommended["max_rel_tail_delta"]),
        "mean_rel_tail_delta": float(recommended["mean_rel_tail_delta"]),
        "combined_tail_index": float(recommended["combined_tail_index"]),
    }


def _save_plots(df: pd.DataFrame, output_dir: Path, recommendation: Dict[str, float | int | str]) -> None:
    rec_day = int(recommendation["recommended_days"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df["days"], df["q90_temp_viol_rate"], label="Q90 temp_viol_rate", linewidth=2.0)
    ax.plot(df["days"], df["cvar90_temp_viol_rate"], label="CVaR90 temp_viol_rate", linewidth=2.0)
    ax.plot(df["days"], df["q90_cond_viol_rate"], label="Q90 cond_viol_rate", linewidth=2.0)
    ax.plot(df["days"], df["cvar90_cond_viol_rate"], label="CVaR90 cond_viol_rate", linewidth=2.0)
    ax.axvline(rec_day, color="#C44E52", linestyle="--", linewidth=2.0, label=f"recommended day={rec_day}")
    ax.set_title("Tail Metrics by Episode Length", fontsize=16)
    ax.set_xlabel("Episode length (days)", fontsize=14)
    ax.set_ylabel("Metric value", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "tail_metrics_by_days.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df["days"], df["combined_tail_index"], label="combined_tail_index", linewidth=2.0)
    ax.plot(df["days"], df["max_rel_tail_delta"], label="max_rel_tail_delta", linewidth=2.0)
    ax.plot(df["days"], df["mean_rel_tail_delta"], label="mean_rel_tail_delta", linewidth=2.0)
    ax.axvline(rec_day, color="#C44E52", linestyle="--", linewidth=2.0, label=f"recommended day={rec_day}")
    ax.set_title("Stability Diagnostics by Episode Length", fontsize=16)
    ax.set_xlabel("Episode length (days)", fontsize=14)
    ax.set_ylabel("Normalized / delta scale", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "stability_diagnostics.png", dpi=160)
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _ensure_output_dir()
    rows: List[Dict[str, float]] = []
    day_values = list(range(int(args.min_days), int(args.max_days) + 1))

    if int(args.parallel_days) <= 1:
        iterator = tqdm(day_values, desc="episode_length_sweep", unit="day")
        for days in iterator:
            row = _run_single_day(
                {
                    "days": int(days),
                    "N": int(args.N),
                    "dt_min": int(args.dt_min),
                    "seed": int(args.seed),
                    "n_jobs": int(args.n_jobs),
                }
            )
            rows.append(row)
            iterator.set_postfix(
                day=int(days),
                cvar90_temp=f"{row['cvar90_temp_viol_rate']:.4f}",
                cvar90_cond=f"{row['cvar90_cond_viol_rate']:.4f}",
            )
    else:
        # When parallelizing across day horizons, keep each horizon internally single-worker
        # to avoid nested parallel oversubscription.
        tasks = [
            {
                "days": int(days),
                "N": int(args.N),
                "dt_min": int(args.dt_min),
                "seed": int(args.seed),
                "n_jobs": 1,
            }
            for days in day_values
        ]
        with ProcessPoolExecutor(max_workers=int(args.parallel_days)) as ex:
            futures = {ex.submit(_run_single_day, task): int(task["days"]) for task in tasks}
            progress = tqdm(total=len(tasks), desc="episode_length_sweep", unit="day")
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                progress.update(1)
                progress.set_postfix(
                    day=int(row["days"]),
                    cvar90_temp=f"{row['cvar90_temp_viol_rate']:.4f}",
                    cvar90_cond=f"{row['cvar90_cond_viol_rate']:.4f}",
                )
            progress.close()

    summary = pd.DataFrame(rows).sort_values("days").reset_index(drop=True)
    summary = _derive_stability_features(summary)
    recommendation = _recommend_day(
        summary,
        settle_days=int(args.settle_days),
        max_rel_threshold=float(args.max_rel_threshold),
    )

    summary.to_csv(output_dir / "episode_length_sweep.csv", index=False)
    _save_plots(summary, output_dir, recommendation)

    config = {
        "N": int(args.N),
        "dt_min": int(args.dt_min),
        "seed": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "parallel_days": int(args.parallel_days),
        "days_range": [int(args.min_days), int(args.max_days)],
        "settle_days": int(args.settle_days),
        "max_rel_threshold": float(args.max_rel_threshold),
        "tail_metrics_used": [
            "q90_temp_viol_rate",
            "cvar90_temp_viol_rate",
            "q90_cond_viol_rate",
            "cvar90_cond_viol_rate",
        ],
    }

    (output_dir / "recommendation.json").write_text(
        json.dumps({"recommendation": recommendation, "config": config}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "episode_length_sweep.csv"),
        "recommendation_path": str(output_dir / "recommendation.json"),
        "recommendation": recommendation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze no-data episode length using tail-metric stability.")
    parser.add_argument("--N", type=int, default=500, help="Monte Carlo repetitions per day horizon.")
    parser.add_argument("--dt-min", type=int, default=5, help="Control interval in minutes.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel workers passed to robust_eval.")
    parser.add_argument(
        "--parallel-days",
        type=int,
        default=6,
        help="Parallel workers across different day horizons. If >1, each horizon runs with n_jobs=1.",
    )
    parser.add_argument("--min-days", type=int, default=1, help="Minimum episode length to test.")
    parser.add_argument("--max-days", type=int, default=60, help="Maximum episode length to test.")
    parser.add_argument("--settle-days", type=int, default=3, help="Consecutive days needed for stabilization.")
    parser.add_argument(
        "--max-rel-threshold",
        type=float,
        default=0.05,
        help="Maximum allowed relative tail-metric change to treat a horizon as stabilized.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(args)
    print(f"saved_outputs: {result['output_dir']}")
    print(json.dumps(result["recommendation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
