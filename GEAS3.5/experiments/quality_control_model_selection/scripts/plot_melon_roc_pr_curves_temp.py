"""Draw temporary melon Validation ROC/PR curves from completed artifacts.

The displayed curves are reconstructed from the 100 saved threshold candidates.
Legend AUC values are the exact metrics already stored by the evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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


def _experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rates(row: dict[str, object]) -> tuple[float, float, float, float] | None:
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
    prevalence = (tp + fn) / (tp + fp + tn + fn) if tp + fp + tn + fn else 0.0
    return fpr, tpr, precision, prevalence


def _deduplicate_x(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Retain the largest y for duplicated x coordinates."""

    by_x: dict[float, float] = {}
    for x_value, y_value in points:
        by_x[x_value] = max(y_value, by_x.get(x_value, float("-inf")))
    return sorted(by_x.items())


def draw_curves(calibration_path: Path, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No model calibration results in {calibration_path}")

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    prevalences: list[float] = []

    for model_name in MODEL_ORDER:
        model = models.get(model_name)
        if not isinstance(model, dict):
            continue
        candidates = model.get("candidate_results", [])
        roc_points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        pr_points: list[tuple[float, float]] = [(0.0, 1.0)]
        selected_roc: tuple[float, float] | None = None
        selected_pr: tuple[float, float] | None = None
        best_threshold = _finite_float(model.get("best_threshold"))

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            rates = _rates(candidate)
            threshold = _finite_float(candidate.get("threshold"))
            if rates is None:
                continue
            fpr, recall, precision, prevalence = rates
            roc_points.append((fpr, recall))
            pr_points.append((recall, precision))
            prevalences.append(prevalence)
            if (
                best_threshold is not None
                and threshold is not None
                and math.isclose(threshold, best_threshold, rel_tol=1e-10, abs_tol=1e-12)
            ):
                selected_roc = (fpr, recall)
                selected_pr = (recall, precision)

        roc_points = _deduplicate_x(roc_points)
        pr_points = _deduplicate_x(pr_points)
        color = MODEL_COLORS[model_name]
        display_name = MODEL_NAMES[model_name]
        roc_x, roc_y = zip(*roc_points)
        pr_x, pr_y = zip(*pr_points)
        roc_ax.plot(roc_x, roc_y, color=color, linewidth=2.2, label=display_name)
        roc_ax.fill_between(roc_x, roc_y, 0.0, color=color, alpha=0.09)
        pr_ax.step(pr_x, pr_y, where="post", color=color, linewidth=2.2, label=display_name)
        pr_ax.fill_between(pr_x, pr_y, 0.0, step="post", color=color, alpha=0.09)

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
    fig.suptitle("Validation Anomaly Detection — Melon", fontsize=17)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    # Previous cucumber source retained for reference; its existing output is untouched.
    # cucumber_artifact_root = _experiment_root() / "artifacts" / "cucumber"
    artifact_root = _experiment_root() / "artifacts" / "melon"
    parser = argparse.ArgumentParser(
        description="Draw temporary melon Validation ROC and PR curves."
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=artifact_root / "threshold_calibration.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_root / "figures" / "roc_pr_curves_temp.png",
    )
    args = parser.parse_args()
    output = draw_curves(args.calibration.resolve(), args.output.resolve())
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
