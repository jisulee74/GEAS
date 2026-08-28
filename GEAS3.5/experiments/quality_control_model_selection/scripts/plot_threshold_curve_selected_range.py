"""Draw a crop threshold curve from completed artifacts.

This script is intentionally independent from the shared figure pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _positive_finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _automatic_x_max(selected_thresholds: list[float]) -> float:
    if not selected_thresholds:
        raise ValueError("No finite selected thresholds were found.")
    raw_limit = max(selected_thresholds) * 4.0
    return max(10.0, math.ceil(raw_limit / 10.0) * 10.0)


def draw_threshold_curve(
    calibration_path: Path,
    output_path: Path,
    *,
    crop: str,
    x_max: float | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    if not models:
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
            "grid.alpha": 0.22,
        }
    )

    selected_thresholds = [
        threshold
        for model in models.values()
        if (threshold := _positive_finite(model.get("best_threshold"))) is not None
    ]
    display_x_max = (
        _automatic_x_max(selected_thresholds)
        if x_max is None
        else float(x_max)
    )
    if not math.isfinite(display_x_max) or display_x_max <= 0.0:
        raise ValueError("x_max must be a positive finite number.")

    colors = {
        "modern_tcn": "#4C78A8",
        "timesnet": "#F58518",
        "patch_tst": "#54A24B",
    }
    display_names = {
        "modern_tcn": "ModernTCN",
        "timesnet": "TimesNet",
        "patch_tst": "PatchTST",
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    visible_f1: list[float] = []

    for index, (model_name, model) in enumerate(models.items()):
        points: list[tuple[float, float]] = []
        for candidate in model.get("candidate_results", []):
            threshold = _positive_finite(candidate.get("threshold"))
            f1_score = _positive_finite(candidate.get("f1_score"))
            if (
                threshold is not None
                and f1_score is not None
                and threshold <= display_x_max
            ):
                points.append((threshold, f1_score))
        points.sort()
        if not points:
            continue

        color = colors.get(model_name, f"C{index}")
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        visible_f1.extend(y_values)
        ax.plot(
            x_values,
            y_values,
            color=color,
            marker="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgewidth=0.0,
            linewidth=1.6,
            label=display_names.get(model_name, model_name),
            zorder=2,
        )

        selected_threshold = _positive_finite(model.get("best_threshold"))
        selected_f1 = _positive_finite(model.get("f1_score"))
        if (
            selected_threshold is None
            or selected_f1 is None
            or selected_threshold > display_x_max
        ):
            continue
        ax.scatter(
            [selected_threshold],
            [selected_f1],
            marker="*",
            s=105,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

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

    ax.set_xlim(0.0, display_x_max)
    if visible_f1:
        ax.set_ylim(0.0, min(1.0, max(visible_f1) * 1.18))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Validation F1-score")
    ax.set_title(f"Threshold Calibration — {crop.replace(chr(95), chr(32)).title()}")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    experiment_root = _experiment_root()
    parser = argparse.ArgumentParser(
        description="Redraw one completed crop threshold curve."
    )
    parser.add_argument(
        "--crop",
        default="strawberry",
        help="Artifact crop directory and plot title (default: strawberry).",
    )
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional explicit threshold-axis maximum (automatic if omitted).",
    )
    args = parser.parse_args()
    artifact_root = experiment_root / "artifacts" / args.crop
    calibration_path = args.calibration or artifact_root / "threshold_calibration.json"
    output_path = (
        args.output
        or artifact_root / "figures" / "threshold_curve_selected_range.png"
    )
    output = draw_threshold_curve(
        calibration_path.resolve(),
        output_path.resolve(),
        crop=args.crop,
        x_max=args.x_max,
    )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
