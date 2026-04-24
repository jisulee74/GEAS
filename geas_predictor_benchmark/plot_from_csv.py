from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from sqlalchemy import create_engine
try:
    import koreanize_matplotlib  # noqa: F401
except Exception:
    koreanize_matplotlib = None


KPI_COLUMNS = [
    "temp_viol_rate",
    "temp_viol_maxrun",
    "cond_viol_rate",
    "cond_viol_maxrun",
    "rh_viol_rate",
    "vpd_viol_rate",
    "hv_ineff_rate",
    "tv_vent",
    "sw_heat",
]

KPI_TITLES_KO = {
    "temp_viol_rate": "온도 제약 위반율",
    "temp_viol_maxrun": "온도 연속 위반 최대 길이",
    "cond_viol_rate": "결로 위험 위반율",
    "cond_viol_maxrun": "결로 위험 연속 위반 최대 길이",
    "rh_viol_rate": "과습 위반율",
    "vpd_viol_rate": "저VPD 위반율",
    "hv_ineff_rate": "난방-환기 동시 사용 비효율",
    "tv_vent": "환기 개도율 총 변화량",
    "sw_heat": "난방 스위칭 횟수",
}

EPISODE_TITLES_KO = {
    "Tin": "실내온도",
    "RHin": "실내상대습도",
    "VPD": "VPD",
    "dTcond": "결로여유온도",
    "ACH": "환기횟수",
    "Q_heat": "난방열량",
    "Q_ventloss": "환기 열손실",
    "x_vent": "환기 개도율",
}

MODEL_COLORS = {
    "physics": "#4C78A8",
    "tiny_ttm": "#F58518",
    "physics_residual_tiny_ttm": "#54A24B",
    "random_forest": "#E45756",
}
MODEL_FACE_COLORS = {
    "physics": "#C9DDF2",
    "tiny_ttm": "#F9D9B3",
    "physics_residual_tiny_ttm": "#CFE9C9",
    "random_forest": "#F5C6CB",
}
MODEL_ORDER = ["physics", "physics_residual_tiny_ttm", "random_forest", "tiny_ttm"]
MODEL_DISPLAY_NAMES = {
    "physics": "물리식",
    "physics_residual_tiny_ttm": "물리식+경량TTM",
    "random_forest": "Random Forest",
    "tiny_ttm": "경량TTM",
}
DB_URI = "mysql+pymysql://root:theimc#10!@211.195.9.227:3306/farmstom"
TABLE_NAME = "data_silla_enc"
IOT_DATA_IDX = 97


def configure_plot_fonts() -> None:
    try:
        import koreanize_matplotlib  # noqa: F401
    except Exception:
        pass
    available = {f.name for f in fm.fontManager.ttflist}
    preferred = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",
        "AppleGothic",
    ]
    selected = next((name for name in preferred if name in available), None)
    if selected is not None:
        plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False


def _legend_handles(models: List[str]) -> List[plt.Rectangle]:
    return [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=MODEL_FACE_COLORS.get(model, "#DDDDDD"),
            edgecolor=MODEL_COLORS.get(model, "#666666"),
            linewidth=2.5,
            label=MODEL_DISPLAY_NAMES.get(model, model),
        )
        for model in models
    ]


def save_model_legend(run_dir: Path, models: List[str]) -> Path:
    configure_plot_fonts()
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.axis("off")
    handles = _legend_handles(models)
    fig.legend(handles=handles, loc="center", ncol=len(models), frameon=False, fontsize=16)
    out = run_dir / "png" / "legend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _load_kpi(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "csv" / "kpi_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return pd.read_csv(path)


def _load_efficiency(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "csv" / "efficiency_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return pd.read_csv(path)


def _load_prediction(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "csv" / "prediction_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return pd.read_csv(path)


def _load_episodes(run_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    paths = {path.stem.replace("episode_", ""): path for path in (run_dir / "csv").glob("episode_*.csv")}
    for model in MODEL_ORDER:
        if model in paths:
            out[model] = pd.read_csv(paths[model], parse_dates=["reg_date"])
    if not out:
        raise FileNotFoundError(f"no episode csv files found in {run_dir}")
    return out


def _selected_period(run_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    info_path = run_dir / "evaluation_period.json"
    if not info_path.exists():
        info_path = run_dir / "selected_period.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return pd.Timestamp(info["start"]), pd.Timestamp(info["end"])


def _infer_step_minutes(episodes: Dict[str, pd.DataFrame]) -> int:
    for model in MODEL_ORDER:
        if model not in episodes:
            continue
        s = pd.to_datetime(episodes[model]["reg_date"]).sort_values().diff().dropna().dt.total_seconds() / 60.0
        if not s.empty:
            return int(round(float(s.median())))
    return 5


def fetch_actual_timeseries(run_dir: Path, episodes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    start_ts, end_ts = _selected_period(run_dir)
    step_minutes = _infer_step_minutes(episodes)
    engine = create_engine(DB_URI)
    query = f"""
        SELECT reg_date, in_temp
        FROM {TABLE_NAME}
        WHERE iot_data_idx = {int(IOT_DATA_IDX)}
          AND reg_date >= '{start_ts.strftime("%Y-%m-%d %H:%M:%S")}'
          AND reg_date <  '{end_ts.strftime("%Y-%m-%d %H:%M:%S")}'
        ORDER BY reg_date ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df["reg_date"] = pd.to_datetime(df["reg_date"])
    df["Tin"] = pd.to_numeric(df["in_temp"], errors="coerce")
    df = df.dropna(subset=["Tin"]).copy()
    df["bucket"] = df["reg_date"].dt.floor(f"{step_minutes}min")
    return df.groupby("bucket", as_index=False)["Tin"].mean().rename(columns={"bucket": "reg_date"})


def plot_temperature_trajectory(run_dir: Path, output_dir: Path, episodes: Dict[str, pd.DataFrame]) -> Path:
    configure_plot_fonts()
    actual = fetch_actual_timeseries(run_dir, episodes)
    episode_ranges = []
    for model in MODEL_ORDER:
        if model not in episodes:
            continue
        ep = episodes[model]
        if ep.empty:
            continue
        episode_ranges.append((pd.to_datetime(ep["reg_date"]).min(), pd.to_datetime(ep["reg_date"]).max()))
    if episode_ranges:
        min_ts = min(s for s, _ in episode_ranges)
        max_ts = max(e for _, e in episode_ranges)
        actual = actual[(actual["reg_date"] >= min_ts) & (actual["reg_date"] <= max_ts)].copy()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(actual["reg_date"], actual["Tin"], label="actual", color="black", linewidth=2.8, alpha=0.85)
    for model in MODEL_ORDER:
        if model not in episodes:
            continue
        ep = episodes[model]
        ax.plot(ep["reg_date"], ep["Tin"], label=MODEL_DISPLAY_NAMES.get(model, model), linewidth=2.2, color=MODEL_COLORS.get(model))
    ax.set_title("실내온도 시계열 비교", fontsize=20)
    ax.set_xlabel("시간", fontsize=17)
    ax.set_ylabel("실내온도 [C]", fontsize=17)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    out = output_dir / "temperature_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_efficiency_scatter(run_dir: Path, output_dir: Path) -> Path:
    configure_plot_fonts()
    eff = _load_efficiency(run_dir)
    eff["model"] = pd.Categorical(eff["model"], categories=MODEL_ORDER, ordered=True)
    eff = eff.sort_values("model").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in eff.iterrows():
        model = row["model"]
        label = MODEL_DISPLAY_NAMES.get(str(model), str(model))
        ax.scatter(
            row["model_bytes"],
            row["mean_control_step_ms"],
            s=130,
            color=MODEL_FACE_COLORS.get(model, "#DDDDDD"),
            edgecolors=MODEL_COLORS.get(model, "#666666"),
            linewidths=2.5,
        )
        if model == "random_forest":
            ax.text(row["model_bytes"], row["mean_control_step_ms"], f"{label} ", va="center", ha="right", fontsize=15)
        else:
            ax.text(row["model_bytes"], row["mean_control_step_ms"], f" {label}", va="center", ha="left", fontsize=15)
    ax.set_xlabel("모델 크기 [bytes]", fontsize=17)
    ax.set_ylabel("평균 제어 계산 시간 [ms]", fontsize=17)
    ax.set_title("경량화 효율 비교", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    out = output_dir / "efficiency_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def summarize_trajectory_errors(run_dir: Path, episodes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    actual_df = fetch_actual_timeseries(run_dir, episodes)
    rows: List[Dict[str, float]] = []
    for model in MODEL_ORDER:
        if model not in episodes:
            continue
        ep = episodes[model][["reg_date", "Tin"]].copy()
        ep["reg_date"] = pd.to_datetime(ep["reg_date"])
        merged = ep.merge(actual_df, on="reg_date", how="inner", suffixes=("_pred", "_actual"))
        if merged.empty:
            continue
        err = merged["Tin_pred"] - merged["Tin_actual"]
        rows.append(
            {
                "model": model,
                "traj_mae": float(err.abs().mean()),
                "traj_rmse": float(np.sqrt(np.mean(err.values ** 2))),
                "traj_bias": float(err.mean()),
                "traj_max_abs": float(err.abs().max()),
                "traj_corr": float(np.corrcoef(merged["Tin_pred"], merged["Tin_actual"])[0, 1]) if len(merged) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_kpi_hist_style(run_dir: Path, output_dir: Path) -> List[Path]:
    configure_plot_fonts()
    df = _load_kpi(run_dir).sort_values("model").reset_index(drop=True)
    legacy = output_dir / "kpi_hist_style_from_csv.png"
    if legacy.exists():
        legacy.unlink()

    outputs: List[Path] = []
    for metric in KPI_COLUMNS:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        if bool((values.abs() <= 1e-12).all()):
            continue
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        for i, (_, row) in enumerate(df.iterrows()):
            model = row["model"]
            ax.bar(
                i,
                row[metric],
                width=0.65,
                facecolor=MODEL_FACE_COLORS.get(model, "#DDDDDD"),
                edgecolor=MODEL_COLORS.get(model, "#666666"),
                linewidth=2.5,
            )
        ax.set_title(KPI_TITLES_KO.get(metric, metric), fontsize=18)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = output_dir / f"kpi_{metric}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def plot_trajectory_error(summary: pd.DataFrame, output_dir: Path) -> Path:
    configure_plot_fonts()
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = [
        ("Tin_next_RMSE", "다음시점 온도 RMSE"),
        ("traj_rmse", "시계열 RMSE"),
        ("traj_bias", "시계열 편향"),
        ("traj_max_abs", "최대 절대오차"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        for i, (_, row) in enumerate(summary.iterrows()):
            model = row["model"]
            ax.bar(
                i,
                row[col],
                width=0.65,
                facecolor=MODEL_FACE_COLORS.get(model, "#DDDDDD"),
                edgecolor=MODEL_COLORS.get(model, "#666666"),
                linewidth=2.5,
            )
        ax.set_title(title, fontsize=16)
        ax.set_xticks([])
        ax.set_ylabel("오차 값", fontsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("시계열 및 다음시점 오차 비교", fontsize=20)
    fig.tight_layout()
    out = output_dir / "trajectory_error_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw histogram-style plots from existing GEAS benchmark CSV outputs.")
    parser.add_argument("--run-dir", required=True, help="Benchmark result directory, e.g. /home/ljs/jslee/geas_predictor_benchmark/results/run_20260423_055014")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    output_dir = run_dir / "png"
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_eff_hist = output_dir / "efficiency_hist_style_from_csv.png"
    if legacy_eff_hist.exists():
        legacy_eff_hist.unlink()

    episodes = _load_episodes(run_dir)
    models = [m for m in MODEL_ORDER if (run_dir / "csv" / f"episode_{m}.csv").exists()]
    legend_path = save_model_legend(run_dir, models)
    out1 = plot_kpi_hist_style(run_dir, output_dir)
    out_eff_scatter = plot_efficiency_scatter(run_dir, output_dir)
    out_temp = plot_temperature_trajectory(run_dir, output_dir, episodes)
    traj = summarize_trajectory_errors(run_dir, episodes).sort_values("model").reset_index(drop=True)
    pred = _load_prediction(run_dir).sort_values("model").reset_index(drop=True)
    traj = traj.merge(pred[["model", "Tin_next_RMSE"]], on="model", how="left")
    if not traj.empty:
        traj.to_csv(run_dir / "csv" / "trajectory_summary.csv", index=False)
        out_traj = plot_trajectory_error(traj, output_dir)
    else:
        out_traj = None

    print(f"[INFO] saved: {legend_path}")
    for path in out1:
        print(f"[INFO] saved: {path}")
    print(f"[INFO] saved: {out_eff_scatter}")
    print(f"[INFO] saved: {out_temp}")
    if out_traj is not None:
        print(f"[INFO] saved: {out_traj}")


if __name__ == "__main__":
    main()
