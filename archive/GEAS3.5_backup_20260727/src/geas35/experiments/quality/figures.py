"""Publication-oriented figures for quality-model experiment artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


FIGURE_STEMS = (
    "hpo_progress",
    "rmse_comparison",
    "mae_comparison",
    "f1_comparison",
    "pr_auc_comparison",
    "threshold_curve",
    "training_loss",
    "online_benchmark_summary",
    "reconstruction_error_heatmap",
    "model_comparison_summary",
)


def generate_quality_experiment_figures(output_root: str | Path) -> list[Path]:
    """Generate PNG and PDF figures from existing experiment artifacts."""

    root = Path(output_root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    _configure_matplotlib(plt)

    hpo = _read_json(root / "hpo_results.json")
    thresholds = _read_json(root / "threshold_calibration.json")
    comparison = _read_comparison_csv(root / "model_comparison.csv")
    history = _read_csv(root / "training_history.csv")
    reconstruction = _read_csv(root / "reconstruction_metric_table.csv")
    benchmark = _read_csv(root / "online_benchmark_table.csv")

    outputs: list[Path] = []
    outputs.extend(_plot_hpo_progress(plt, figures_dir, hpo))
    outputs.extend(
        _plot_metric_bar(
            plt,
            figures_dir,
            comparison,
            stem="rmse_comparison",
            column="validation_rmse",
            ylabel="Validation RMSE",
            title="Validation Reconstruction RMSE",
        )
    )
    outputs.extend(
        _plot_metric_bar(
            plt,
            figures_dir,
            comparison,
            stem="mae_comparison",
            column="validation_mae",
            ylabel="Validation MAE",
            title="Validation Reconstruction MAE",
        )
    )
    outputs.extend(
        _plot_metric_bar(
            plt,
            figures_dir,
            comparison,
            stem="f1_comparison",
            column="validation_f1",
            ylabel="Validation F1-score",
            title="Validation Anomaly Detection F1-score",
        )
    )
    outputs.extend(
        _plot_metric_bar(
            plt,
            figures_dir,
            comparison,
            stem="pr_auc_comparison",
            column="validation_pr_auc",
            ylabel="Validation PR-AUC",
            title="Validation Anomaly Detection PR-AUC",
        )
    )
    outputs.extend(_plot_threshold_curve(plt, figures_dir, thresholds))
    outputs.extend(_plot_training_loss(plt, figures_dir, history))
    outputs.extend(_plot_online_benchmark_summary(plt, figures_dir, benchmark, comparison))
    outputs.extend(_plot_reconstruction_heatmap(plt, figures_dir, reconstruction))
    outputs.extend(_plot_model_summary(plt, figures_dir, comparison))
    return outputs


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _configure_matplotlib(plt) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_hpo_progress(plt, figures_dir: Path, hpo: dict[str, Any]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plotted = False
    for model_name, model_payload in hpo.get("models", {}).items():
        candidates = model_payload.get("candidate_results", [])
        y_values = [
            _metric_value(
                candidate.get("validation_metrics", {}),
                "validation_synthetic_masking_rmse",
            )
            for candidate in candidates
        ]
        y_values = [value for value in y_values if value is not None]
        if y_values:
            ax.plot(
                range(1, len(y_values) + 1),
                y_values,
                marker="o",
                linewidth=1.8,
                label=str(model_name),
            )
            plotted = True
    if plotted:
        ax.set_xlabel("Random Search Candidate")
        ax.set_ylabel("Validation RMSE")
        ax.set_title("HPO Search Progress")
        ax.legend(frameon=False)
    else:
        _empty_axes(ax, "HPO Search Progress", "No HPO candidate metrics available")
    return _save_figure(fig, figures_dir, "hpo_progress")


def _plot_metric_bar(
    plt,
    figures_dir: Path,
    frame: pd.DataFrame,
    *,
    stem: str,
    column: str,
    ylabel: str,
    title: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if frame.empty or column not in frame.columns:
        _empty_axes(ax, title, f"No {column} values available")
        return _save_figure(fig, figures_dir, stem)
    plot_frame = frame[["model_name", column]].copy()
    plot_frame[column] = pd.to_numeric(plot_frame[column], errors="coerce")
    plot_frame = plot_frame.dropna(subset=[column])
    if plot_frame.empty:
        _empty_axes(ax, title, f"No finite {column} values available")
        return _save_figure(fig, figures_dir, stem)
    ax.bar(plot_frame["model_name"], plot_frame[column], color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    return _save_figure(fig, figures_dir, stem)


def _plot_threshold_curve(plt, figures_dir: Path, thresholds: dict[str, Any]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plotted = False
    for model_name, payload in thresholds.get("models", {}).items():
        candidates = payload.get("candidate_results", [])
        x_values = [_safe_float(row.get("threshold")) for row in candidates]
        y_values = [_safe_float(row.get("f1_score")) for row in candidates]
        points = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
        if points:
            points.sort(key=lambda item: item[0])
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.8,
                label=str(model_name),
            )
            plotted = True
    if plotted:
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Validation F1-score")
        ax.set_title("Threshold Calibration Curve")
        ax.legend(frameon=False)
    else:
        _empty_axes(ax, "Threshold Calibration Curve", "No threshold metrics available")
    return _save_figure(fig, figures_dir, "threshold_curve")


def _plot_training_loss(plt, figures_dir: Path, history: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if history.empty:
        _empty_axes(ax, "Training Loss Curve", "No training history available")
        return _save_figure(fig, figures_dir, "training_loss")
    loss_col = _first_existing(
        history.columns,
        ("validation_reconstruction_loss", "train_reconstruction_loss", "loss"),
    )
    if loss_col is None:
        _empty_axes(ax, "Training Loss Curve", "No loss column available")
        return _save_figure(fig, figures_dir, "training_loss")
    epoch_col = "epoch" if "epoch" in history.columns else None
    for model_name, group in history.groupby("model_name"):
        x_values = (
            pd.to_numeric(group[epoch_col], errors="coerce")
            if epoch_col is not None
            else pd.Series(range(1, len(group) + 1))
        )
        y_values = pd.to_numeric(group[loss_col], errors="coerce")
        mask = x_values.notna() & y_values.notna()
        if mask.any():
            ax.plot(
                x_values[mask],
                y_values[mask],
                marker="o",
                linewidth=1.8,
                label=str(model_name),
            )
    if ax.lines:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(loss_col.replace("_", " ").title())
        ax.set_title("Training Loss Curve")
        ax.legend(frameon=False)
    else:
        _empty_axes(ax, "Training Loss Curve", "No finite loss values available")
    return _save_figure(fig, figures_dir, "training_loss")


def _plot_model_summary(plt, figures_dir: Path, frame: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    metrics = [
        ("validation_rmse", "RMSE", "#4C78A8"),
        ("validation_f1", "F1-score", "#F58518"),
        ("validation_latency_ms_per_row", "Latency ms/row", "#54A24B"),
    ]
    for ax, (column, label, color) in zip(axes, metrics):
        if frame.empty or column not in frame.columns:
            _empty_axes(ax, label, "No data")
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        plot_frame = pd.DataFrame(
            {"model_name": frame["model_name"], column: values}
        ).dropna()
        if plot_frame.empty:
            _empty_axes(ax, label, "No finite data")
            continue
        ax.bar(plot_frame["model_name"], plot_frame[column], color=color)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Model Comparison Summary", y=1.02)
    return _save_figure(fig, figures_dir, "model_comparison_summary")


def _plot_online_benchmark_summary(
    plt,
    figures_dir: Path,
    benchmark: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    if benchmark.empty:
        benchmark = _benchmark_from_comparison(comparison)
    metrics = [
        ("inference_latency_ms_per_row", "Latency ms/row", "#4C78A8"),
        ("peak_memory_bytes", "Peak Memory bytes", "#F58518"),
        ("model_size_bytes", "Model Size bytes", "#54A24B"),
    ]
    for ax, (column, label, color) in zip(axes, metrics):
        if benchmark.empty or column not in benchmark.columns:
            _empty_axes(ax, label, "No data")
            continue
        frame = benchmark.copy()
        if "split" in frame.columns:
            frame = frame[frame["split"].astype(str) == "validation"]
        values = pd.to_numeric(frame[column], errors="coerce")
        plot_frame = pd.DataFrame(
            {"model_name": frame["model_name"], column: values}
        ).dropna()
        if plot_frame.empty:
            _empty_axes(ax, label, "No finite data")
            continue
        ax.bar(plot_frame["model_name"], plot_frame[column], color=color)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Online Benchmark Summary", y=1.02)
    return _save_figure(fig, figures_dir, "online_benchmark_summary")


def _plot_reconstruction_heatmap(
    plt,
    figures_dir: Path,
    reconstruction: pd.DataFrame,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if reconstruction.empty:
        _empty_axes(ax, "Per-column Reconstruction Error", "No reconstruction table available")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    frame = reconstruction.copy()
    if "column" not in frame.columns or "model_name" not in frame.columns:
        _empty_axes(ax, "Per-column Reconstruction Error", "Missing heatmap columns")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == "validation"]
    frame = frame[frame["column"].astype(str) != "__all__"]
    metric_col = _first_existing(frame.columns, ("rmse", "mae"))
    if metric_col is None:
        _empty_axes(ax, "Per-column Reconstruction Error", "No error metric available")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce")
    frame = frame.dropna(subset=[metric_col])
    if frame.empty:
        _empty_axes(ax, "Per-column Reconstruction Error", "No finite error values available")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    pivot = frame.pivot_table(
        index="model_name",
        columns="column",
        values=metric_col,
        aggfunc="mean",
    )
    if pivot.empty:
        _empty_axes(ax, "Per-column Reconstruction Error", "No heatmap values available")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_title(f"Validation Per-column Reconstruction {metric_col.upper()}")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_comparison_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _benchmark_from_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "model_name" not in frame.columns:
        return pd.DataFrame()
    rename_map = {
        "validation_latency_ms_per_row": "inference_latency_ms_per_row",
        "validation_peak_memory_bytes": "peak_memory_bytes",
        "validation_model_size_bytes": "model_size_bytes",
    }
    available = ["model_name", *[col for col in rename_map if col in frame.columns]]
    if len(available) == 1:
        return pd.DataFrame()
    output = frame.loc[:, available].rename(columns=rename_map)
    output["split"] = "validation"
    return output


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = _safe_float(metrics.get(key))
    if value is not None:
        return value
    for fallback_key in ("validation_synthetic_masking_mae", "validation_reconstruction_loss"):
        value = _safe_float(metrics.get(fallback_key))
        if value is not None:
            return value
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _first_existing(columns: Iterable[str], names: Iterable[str]) -> str | None:
    column_set = set(columns)
    for name in names:
        if name in column_set:
            return name
    return None


def _empty_axes(ax, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_figure(fig, figures_dir: Path, stem: str) -> list[Path]:
    outputs = [figures_dir / f"{stem}.png", figures_dir / f"{stem}.pdf"]
    fig.tight_layout()
    for path in outputs:
        fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return outputs


__all__ = ["FIGURE_STEMS", "generate_quality_experiment_figures"]
