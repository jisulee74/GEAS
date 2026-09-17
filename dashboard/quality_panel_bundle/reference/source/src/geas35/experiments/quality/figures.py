"""Publication-oriented figures for quality-model experiment artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


MODEL_ORDER = ("modern_tcn", "timesnet", "patch_tst")
MODEL_NAMES = {
    "modern_tcn": "ModernTCN",
    "timesnet": "TimesNet",
    "patch_tst": "PatchTST",
}
MODEL_COLORS = {
    "modern_tcn": "#4C78A8",
    "timesnet": "#F58518",
    "patch_tst": "#54A24B",
}


FIGURE_STEMS = (
    "hpo_progress",
    "threshold_curve",
    "threshold_curve_selected_range",
    "roc_pr_curves",
    "training_loss",
    "reconstruction_error_heatmap",
)



def generate_quality_experiment_figures(output_root: str | Path) -> list[Path]:
    """Generate PNG figures from existing experiment artifacts."""

    root = Path(output_root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    _configure_matplotlib(plt)

    hpo = _read_json(root / "hpo_results.json")
    thresholds = _read_json(root / "threshold_calibration.json")
    history = _read_csv(root / "training_history.csv")
    reconstruction = _read_csv(root / "reconstruction_metric_table.csv")
    combined_figures_dir = root.parent / "figures"
    combined_figures_dir.mkdir(parents=True, exist_ok=True)
    summary = _combined_validation_summary(root.parent)

    outputs: list[Path] = []
    outputs.extend(_plot_hpo_progress(plt, figures_dir, hpo))
    outputs.extend(_plot_performance_categories(plt, combined_figures_dir, summary))
    outputs.extend(_plot_threshold_curve(plt, figures_dir, thresholds))
    outputs.extend(_plot_threshold_curve_selected_range(plt, figures_dir, thresholds))
    outputs.extend(_plot_roc_pr_curves(plt, figures_dir, thresholds))
    outputs.extend(_plot_training_loss(plt, figures_dir, history))
    outputs.extend(_plot_reconstruction_heatmap(plt, figures_dir, reconstruction))
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
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_hpo_progress(plt, figures_dir: Path, hpo: dict[str, Any]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    crop_name = _crop_display_name(figures_dir)
    title = f"HPO Search Progress — {crop_name}"
    models = hpo.get("models", {})
    plotted = False
    for model_name in _ordered_model_names(models):
        model_payload = models[model_name]
        candidates = model_payload.get("candidate_results", [])
        y_values = [
            _metric_value(
                candidate.get("validation_metrics", {}),
                "validation_synthetic_masking_rmse",
            )
            for candidate in candidates
        ]
        y_values = [value for value in y_values if value is not None]
        if not y_values:
            continue
        color = MODEL_COLORS.get(model_name)
        ax.plot(
            range(1, len(y_values) + 1),
            y_values,
            color=color,
            marker="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgewidth=0.0,
            linewidth=1.8,
            label=MODEL_NAMES.get(model_name, model_name),
        )
        plotted = True
    if plotted:
        ax.set_xlabel("Random Search Candidate", fontsize=13)
        ax.set_ylabel("Validation RMSE", fontsize=13)
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(frameon=False, fontsize=11)
    else:
        _empty_axes(ax, title, "No HPO candidate metrics available")
    return _save_figure(fig, figures_dir, "hpo_progress")

def _validation_summary(root: Path) -> pd.DataFrame:
    reconstruction = _read_csv(root / "reconstruction_metric_table.csv")
    anomaly = _read_csv(root / "anomaly_detection_metric_table.csv")
    benchmark = _read_csv(root / "online_benchmark_table.csv")
    rec = reconstruction[
        (reconstruction["split"].astype(str) == "validation")
        & (reconstruction["column"].astype(str) == "__all__")
    ].set_index("model_name")
    det = anomaly[
        (anomaly["split"].astype(str) == "validation")
        & (anomaly["column"].astype(str) == "__all__")
    ].set_index("model_name")
    bench = benchmark[
        benchmark["split"].astype(str) == "validation"
    ].set_index("model_name")
    rows = []
    for model_name in MODEL_ORDER:
        if model_name not in rec.index or model_name not in det.index:
            continue
        r, d = rec.loc[model_name], det.loc[model_name]
        b = bench.loc[model_name] if model_name in bench.index else {}
        tp, fp = float(d["true_positive"]), float(d["false_positive"])
        tn, fn = float(d["true_negative"]), float(d["false_negative"])
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        rows.append({
            "crop": root.name,
            "model_name": model_name,
            "rmse": float(r["rmse"]),
            "mae": float(r["mae"]),
            "f1_score": float(d["f1_score"]),
            "precision": float(d["precision"]),
            "recall": float(d["recall"]),
            "mcc": 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator,
            "fpr": 0.0 if fp + tn == 0 else fp / (fp + tn),
            "roc_auc": float(d["roc_auc"]),
            "pr_auc": float(d["pr_auc"]),
            "inference_latency_ms_per_row": float(b["inference_latency_ms_per_row"]),
            "peak_memory_mb": float(b["peak_memory_bytes"]) / (1024 ** 2),
            "model_size_mb": float(b["model_size_bytes"]) / (1024 ** 2),
        })
    return pd.DataFrame(rows)


def _combined_validation_summary(artifacts_root: Path) -> pd.DataFrame:
    frames = []
    for crop_root in artifacts_root.iterdir():
        if not crop_root.is_dir() or crop_root.name == "figures":
            continue
        required = (
            crop_root / "reconstruction_metric_table.csv",
            crop_root / "anomaly_detection_metric_table.csv",
            crop_root / "online_benchmark_table.csv",
        )
        if all(path.exists() for path in required):
            frames.append(_validation_summary(crop_root))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _shared_metric_limits(
    frame: pd.DataFrame, columns: Iterable[str]
) -> dict[str, tuple[float, float]]:
    limits = {}
    bounded = {"f1_score", "precision", "recall", "mcc", "fpr", "roc_auc", "pr_auc"}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            limits[column] = (0.0, 1.0)
        elif column in bounded:
            limits[column] = (min(0.0, float(values.min()) * 1.1), 1.0)
        else:
            upper = float(values.max())
            limits[column] = (0.0, upper * 1.12 if upper > 0 else 1.0)
    return limits


def _plot_performance_categories(
    plt, figures_dir: Path, frame: pd.DataFrame
) -> list[Path]:
    categories = (
        ("reconstruction_performance", "Reconstruction Performance",
         (("rmse", "RMSE ↓", "Value"), ("mae", "MAE ↓", "Value"))),
        ("anomaly_detection_classification",
         "Anomaly Detection Performance (Classification)",
         (("f1_score", "F1 ↑", "Score"), ("precision", "Precision ↑", "Score"),
          ("recall", "Recall ↑", "Score"), ("mcc", "MCC ↑", "Score"),
          ("fpr", "FPR ↑", "Rate"))),
        ("anomaly_detection_ranking", "Anomaly Detection Performance (Ranking)",
         (("roc_auc", "ROC-AUC ↑", "Score"), ("pr_auc", "PR-AUC ↑", "Score"))),
        ("computational_efficiency", "Computational Efficiency",
         (("inference_latency_ms_per_row", "Inference Latency ↓", "ms/row"),
          ("peak_memory_mb", "Peak Memory ↓", "MB"),
          ("model_size_mb", "Model Size ↓", "MB"))),
    )
    columns = [metric[0] for _, _, metrics in categories for metric in metrics]
    limits = _shared_metric_limits(frame, columns)
    preferred = ("strawberry", "melon", "cucumber")
    available = set(frame["crop"])
    crops = [crop for crop in preferred if crop in available]
    crops.extend(sorted(available - set(crops)))
    labels = {"cucumber": "Cucumber", "melon": "Melon", "strawberry": "Strawberry"}
    markers = ("o", "s", "^")
    outputs = []
    for stem, category_title, metrics in categories:
        panel_width = 3.05
        panel_spacing = 0.24
        figure_spacing = 0.34
        horizontal_margin = 0.70
        panel_count = len(metrics)
        figure_width = horizontal_margin + panel_width * (
            panel_count + figure_spacing * (panel_count - 1)
        )
        fig, axes = plt.subplots(
            1,
            panel_count,
            figsize=(figure_width, 4.5),
        )
        axes = [axes] if len(metrics) == 1 else list(axes)
        for ax, (column, title, unit) in zip(axes, metrics):
            for index, model_name in enumerate(MODEL_ORDER):
                values = []
                for crop in crops:
                    match = frame[
                        (frame["crop"] == crop)
                        & (frame["model_name"] == model_name)
                    ]
                    values.append(float(match.iloc[0][column]) if not match.empty else math.nan)
                x_offset = (-0.16, 0.0, 0.16)[index]
                x_values = [position + x_offset for position in range(len(crops))]
                ax.scatter(
                    x_values,
                    values,
                    color=MODEL_COLORS[model_name],
                    marker=markers[index],
                    s=46,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                    label=MODEL_NAMES[model_name],
                )
            ax.set_title(title, fontsize=15)
            ax.set_ylabel(unit, fontsize=13)
            ax.set_xticks(range(len(crops)), [labels.get(c, c.title()) for c in crops])
            ax.set_xlim(-0.45, len(crops) - 0.55)
            ax.set_ylim(*limits[column])
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", visible=True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=12)
        handles, legend_labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="lower center",
            bbox_to_anchor=(0.5, 0.025), ncol=3,
            frameon=False, fontsize=12, borderaxespad=0.0,
        )
        fig.suptitle(category_title, fontsize=17, y=0.965)
        fig.subplots_adjust(
            left=0.55 / figure_width,
            right=1.0 - 0.15 / figure_width,
            bottom=0.16,
            top=0.79,
            wspace=panel_spacing,
        )
        outputs.extend(_save_fixed_size_figure(fig, figures_dir, stem))
    return outputs



def _save_fixed_size_figure(fig, figures_dir: Path, stem: str) -> list[Path]:
    """Save a category figure without recomputing its reserved legend margins."""
    path = figures_dir / f"{stem}.png"
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return [path]


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


def _automatic_selected_x_max(
    selected_thresholds: list[float],
    candidate_thresholds: list[float],
) -> float | None:
    """Return the legacy selected-range limit, capped by observed candidates."""

    if not selected_thresholds or not candidate_thresholds:
        return None
    candidate_max = max(candidate_thresholds)
    if candidate_max <= 0.0:
        return None
    raw_limit = max(selected_thresholds) * 4.0
    legacy_limit = max(10.0, math.ceil(raw_limit / 10.0) * 10.0)
    return min(candidate_max, legacy_limit)


def _plot_threshold_curve_panels(
    plt,
    figures_dir: Path,
    thresholds: dict[str, Any],
    *,
    selected_range: bool,
) -> list[Path]:
    """Plot model curves for each variable, optionally limiting the x range."""

    models = thresholds.get("models", {})
    columns: list[str] = []
    for payload in models.values():
        for col in payload.get("per_column_calibration", {}):
            if col not in columns:
                columns.append(col)
    stem = "threshold_curve_selected_range" if selected_range else "threshold_curve"
    if not columns:
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        _empty_axes(ax, "Threshold Calibration", "No per-variable threshold metrics available")
        return _save_figure(fig, figures_dir, stem)

    ncols = 2
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.8, 4.2 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, columns):
        model_details: list[tuple[int, str, dict[str, Any], list[tuple[float, float]]]] = []
        selected_thresholds: list[float] = []
        candidate_thresholds: list[float] = []
        for index, model_name in enumerate(_ordered_model_names(models)):
            detail = models[model_name].get("per_column_calibration", {}).get(col, {})
            points: list[tuple[float, float]] = []
            for candidate in detail.get("candidate_results", []):
                x = _positive_finite(candidate.get("threshold"))
                y = _positive_finite(candidate.get("f1_score"))
                if x is not None and y is not None:
                    points.append((x, y))
                    candidate_thresholds.append(x)
            points.sort()
            selected_x = _positive_finite(detail.get("best_threshold"))
            if selected_x is not None:
                selected_thresholds.append(selected_x)
            model_details.append((index, model_name, detail, points))

        x_max = (
            _automatic_selected_x_max(selected_thresholds, candidate_thresholds)
            if selected_range
            else None
        )
        for index, model_name, detail, points in model_details:
            if x_max is not None:
                points = [(x, y) for x, y in points if x <= x_max]
            if not points:
                continue
            color = MODEL_COLORS.get(model_name, f"C{index}")
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                color=color,
                marker="o",
                markersize=4.5,
                markeredgewidth=0.0,
                linewidth=1.6,
                label=MODEL_NAMES.get(model_name, model_name),
            )
            selected_x = _positive_finite(detail.get("best_threshold"))
            selected_y = _positive_finite(detail.get("best_objective_value"))
            if (
                selected_x is not None
                and selected_y is not None
                and (x_max is None or selected_x <= x_max)
            ):
                ax.scatter(
                    selected_x,
                    selected_y,
                    marker="*",
                    s=105,
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=5,
                )
        if x_max is not None:
            ax.set_xlim(0.0, x_max)
        ax.set_title(col, fontsize=15)
        ax.set_xlabel("Threshold", fontsize=13)
        ax.set_ylabel("Validation F1-score", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_ylim(0.0, 1.0)
    for ax in axes.flat[len(columns):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        handles.append(
            plt.Line2D(
                [], [], marker="*", linestyle="None", markersize=12,
                markerfacecolor="#B0B0B0", markeredgecolor="black",
            )
        )
        labels.append("Selected threshold")
        axes.flat[0].legend(
            handles,
            labels,
            loc="best",
            frameon=True,
            framealpha=0.9,
            edgecolor="none",
            fontsize=11,
        )

    crop_name = figures_dir.parent.name.replace("_", " ").title()
    range_label = " — Selected Range" if selected_range else ""
    fig.suptitle(
        f"Per-variable Threshold Calibration{range_label} — {crop_name}",
        fontsize=17,
        y=0.995,
    )
    fig.tight_layout()
    return _save_figure(fig, figures_dir, stem)


def _plot_threshold_curve(plt, figures_dir: Path, thresholds: dict[str, Any]) -> list[Path]:
    return _plot_threshold_curve_panels(
        plt, figures_dir, thresholds, selected_range=False
    )


def _plot_threshold_curve_selected_range(
    plt,
    figures_dir: Path,
    thresholds: dict[str, Any],
) -> list[Path]:
    return _plot_threshold_curve_panels(
        plt, figures_dir, thresholds, selected_range=True
    )


def _positive_finite(value: Any) -> float | None:
    number = _safe_float(value)
    return number if number is not None and number >= 0.0 else None



def _plot_roc_pr_curves(
    plt,
    figures_dir: Path,
    thresholds: dict[str, Any],
) -> list[Path]:
    model_order = ("modern_tcn", "timesnet", "patch_tst")
    display_names = {
        "modern_tcn": "ModernTCN",
        "timesnet": "TimesNet",
        "patch_tst": "PatchTST",
    }
    colors = {
        "modern_tcn": "#4C78A8",
        "timesnet": "#F58518",
        "patch_tst": "#54A24B",
    }
    models = thresholds.get("models", {})
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    prevalences: list[float] = []
    plotted = False

    for model_name in model_order:
        model = models.get(model_name)
        if not isinstance(model, dict):
            continue
        roc_points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        pr_points: list[tuple[float, float]] = [(0.0, 1.0)]
        selected_roc: tuple[float, float] | None = None
        selected_pr: tuple[float, float] | None = None
        best_threshold = _safe_float(model.get("best_threshold"))

        for candidate in model.get("candidate_results", []):
            if not isinstance(candidate, dict):
                continue
            rates = _confusion_rates(candidate)
            threshold = _safe_float(candidate.get("threshold"))
            if rates is None:
                continue
            fpr, recall, precision, prevalence = rates
            roc_points.append((fpr, recall))
            pr_points.append((recall, precision))
            prevalences.append(prevalence)
            if (
                best_threshold is not None
                and threshold is not None
                and math.isclose(
                    threshold,
                    best_threshold,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
            ):
                selected_roc = (fpr, recall)
                selected_pr = (recall, precision)

        roc_points = _deduplicate_curve_x(roc_points)
        pr_points = _deduplicate_curve_x(pr_points)
        if len(roc_points) <= 2 and len(pr_points) <= 1:
            continue
        plotted = True
        color = colors[model_name]
        roc_x, roc_y = zip(*roc_points)
        pr_x, pr_y = zip(*pr_points)
        roc_ax.plot(
            roc_x,
            roc_y,
            color=color,
            linewidth=2.2,
            label=display_names[model_name],
        )
        roc_ax.fill_between(roc_x, roc_y, 0.0, color=color, alpha=0.09)
        pr_ax.step(
            pr_x,
            pr_y,
            where="post",
            color=color,
            linewidth=2.2,
            label=display_names[model_name],
        )
        pr_ax.fill_between(
            pr_x,
            pr_y,
            0.0,
            step="post",
            color=color,
            alpha=0.09,
        )
        if selected_roc is not None:
            roc_ax.scatter(
                *selected_roc,
                marker="*",
                s=95,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )
        if selected_pr is not None:
            pr_ax.scatter(
                *selected_pr,
                marker="*",
                s=95,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )

    if not plotted:
        _empty_axes(
            roc_ax,
            "ROC Curve",
            "No threshold confusion matrices available",
        )
        _empty_axes(
            pr_ax,
            "Precision–Recall Curve",
            "No threshold confusion matrices available",
        )
        return _save_figure(fig, figures_dir, "roc_pr_curves")

    for ax in (roc_ax, pr_ax):
        ax.scatter(
            [],
            [],
            marker="*",
            s=105,
            color="#B0B0B0",
            edgecolor="black",
            linewidth=0.8,
            label="Selected threshold",
        )

    prevalence = sum(prevalences) / len(prevalences) if prevalences else 0.1
    roc_ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="#777777", linewidth=1.1)
    pr_ax.axhline(prevalence, linestyle="--", color="#777777", linewidth=1.1)
    roc_ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    pr_ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall Curve",
    )
    roc_ax.legend(frameon=False, loc="lower right")
    pr_ax.legend(frameon=False, loc="upper right")
    crop_name = figures_dir.parent.name.replace("_", " ").title()
    fig.suptitle(f"Validation Anomaly Detection — {crop_name}", fontsize=17)
    return _save_figure(fig, figures_dir, "roc_pr_curves")


def _confusion_rates(
    row: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    confusion = row.get("confusion_matrix")
    if not isinstance(confusion, dict):
        return None
    try:
        tp = int(confusion["true_positive"])
        fp = int(confusion["false_positive"])
        tn = int(confusion["true_negative"])
        fn = int(confusion["false_negative"])
    except (KeyError, TypeError, ValueError):
        return None
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    total = tp + fp + tn + fn
    prevalence = (tp + fn) / total if total else 0.0
    return fpr, tpr, precision, prevalence


def _deduplicate_curve_x(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    by_x: dict[float, float] = {}
    for x_value, y_value in points:
        by_x[x_value] = max(y_value, by_x.get(x_value, float("-inf")))
    return sorted(by_x.items())

def _plot_training_loss(plt, figures_dir: Path, history: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    crop_name = _crop_display_name(figures_dir)
    title = f"Training Loss Curve — {crop_name}"
    if history.empty:
        _empty_axes(ax, title, "No training history available")
        return _save_figure(fig, figures_dir, "training_loss")
    loss_col = _first_existing(
        history.columns,
        ("validation_reconstruction_loss", "train_reconstruction_loss", "loss"),
    )
    if loss_col is None:
        _empty_axes(ax, title, "No loss column available")
        return _save_figure(fig, figures_dir, "training_loss")
    epoch_col = "epoch" if "epoch" in history.columns else None
    grouped = {str(name): group for name, group in history.groupby("model_name")}
    for model_name in _ordered_model_names(grouped):
        group = grouped[model_name]
        x_values = (
            pd.to_numeric(group[epoch_col], errors="coerce")
            if epoch_col is not None
            else pd.Series(range(1, len(group) + 1))
        )
        y_values = pd.to_numeric(group[loss_col], errors="coerce")
        mask = x_values.notna() & y_values.notna()
        if not mask.any():
            continue
        color = MODEL_COLORS.get(model_name)
        ax.plot(
            x_values[mask],
            y_values[mask],
            color=color,
            marker="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgewidth=0.0,
            linewidth=1.8,
            label=MODEL_NAMES.get(model_name, model_name),
        )
    if ax.lines:
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel(loss_col.replace("_", " ").title(), fontsize=13)
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(frameon=False, fontsize=11)
    else:
        _empty_axes(ax, title, "No finite loss values available")
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
    crop_name = _crop_display_name(figures_dir)
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
    fig.suptitle(f"Online Benchmark Summary — {crop_name}", y=1.02)
    return _save_figure(fig, figures_dir, "online_benchmark_summary")


def _plot_reconstruction_heatmap(
    plt,
    figures_dir: Path,
    reconstruction: pd.DataFrame,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    crop_name = _crop_display_name(figures_dir)
    title = f"Per-column Reconstruction Error — {crop_name}"
    if reconstruction.empty:
        _empty_axes(ax, title, "No reconstruction table available")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    frame = reconstruction.copy()
    if "column" not in frame.columns or "model_name" not in frame.columns:
        _empty_axes(ax, title, "Missing heatmap columns")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == "validation"]
    frame = frame[frame["column"].astype(str) != "__all__"]
    metric_col = _first_existing(frame.columns, ("rmse", "mae"))
    if metric_col is None:
        _empty_axes(ax, title, "No error metric available")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce")
    frame = frame.dropna(subset=[metric_col])
    if frame.empty:
        _empty_axes(ax, title, "No finite error values available")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    pivot = frame.pivot_table(
        index="model_name",
        columns="column",
        values=metric_col,
        aggfunc="mean",
    )
    if pivot.empty:
        _empty_axes(ax, title, "No heatmap values available")
        return _save_figure(fig, figures_dir, "reconstruction_error_heatmap")
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_title(f"Validation Per-column Reconstruction {metric_col.upper()} — {crop_name}")
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


def _crop_display_name(figures_dir: Path) -> str:
    return figures_dir.parent.name.replace("_", " ").title()


def _ordered_model_names(models: Mapping[str, Any]) -> list[str]:
    ordered = [name for name in MODEL_ORDER if name in models]
    ordered.extend(name for name in models if name not in MODEL_ORDER)
    return ordered


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
    outputs = [figures_dir / f"{stem}.png"]
    fig.tight_layout()
    for path in outputs:
        fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return outputs


__all__ = ["FIGURE_STEMS", "generate_quality_experiment_figures"]
