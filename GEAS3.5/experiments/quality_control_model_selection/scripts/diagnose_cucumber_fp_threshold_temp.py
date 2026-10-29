"""Temporary diagnostic for cucumber false positives and threshold range.

This script reuses the fitted cucumber model artifacts. It does not retrain or
overwrite the formal experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.quality.config import (  # noqa: E402
    load_quality_experiment_config,
    read_configured_datasets,
)
from geas35.models.quality import load_quality_model_application  # noqa: E402
from geas35.models.quality.evaluation import (  # noqa: E402
    _binary_metrics,
    _normal_candidate_mask,
    make_synthetic_anomaly_injection,
)


MODEL_NAMES = ("modern_tcn", "timesnet", "patch_tst")
DISPLAY_NAMES = {
    "modern_tcn": "ModernTCN",
    "timesnet": "TimesNet",
    "patch_tst": "PatchTST",
}
DISTANCE_WINDOWS = (0, 1, 3, 6, 12, 24, 48, 288)


def _nearest_distance(mask: np.ndarray) -> np.ndarray:
    """Return row distance to the nearest True value; -1 means none."""

    n = len(mask)
    distance = np.full(n, n + 1, dtype=np.int64)
    last = -(n + 1)
    for idx in range(n):
        if mask[idx]:
            last = idx
        distance[idx] = idx - last
    last = 2 * n + 1
    for idx in range(n - 1, -1, -1):
        if mask[idx]:
            last = idx
        distance[idx] = min(distance[idx], last - idx)
    distance[distance > n] = -1
    return distance


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    result = _binary_metrics(y_true, y_score, threshold=float(threshold))
    cm = result.confusion_matrix
    return {
        "threshold": float(threshold),
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1_score": float(result.f1_score),
        "true_positive": cm.true_positive,
        "false_positive": cm.false_positive,
        "true_negative": cm.true_negative,
        "false_negative": cm.false_negative,
    }


def _best_expanded_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    candidate_count: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    if not len(y_score):
        raise ValueError("No finite injected-validation anomaly scores.")
    candidates = np.linspace(float(y_score.min()), float(y_score.max()), candidate_count)
    rows = [_metrics(y_true, y_score, candidate) for candidate in candidates]
    # Explicit tie-break: F1, then precision, then the higher threshold.
    best = max(rows, key=lambda row: (row["f1_score"], row["precision"], row["threshold"]))
    return best, pd.DataFrame(rows)


def _timestamp_series(df: pd.DataFrame) -> pd.Series:
    for name in ("reg_date", "timestamp", "datetime"):
        if name in df.columns:
            return pd.to_datetime(df[name], errors="coerce")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def diagnose_model(
    *,
    model_name: str,
    artifact_root: Path,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    columns: tuple[str, ...],
    anomaly_fraction: float,
    anomaly_scale: float,
    random_state: int,
    candidate_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    model_dir = artifact_root / model_name
    application = load_quality_model_application(model_dir / "quality_model_application.json")
    model = application.model
    thresholds = {str(k): float(v) for k, v in dict(application.thresholds).items()}

    injection = make_synthetic_anomaly_injection(
        validation_df,
        columns,
        anomaly_fraction=anomaly_fraction,
        anomaly_scale=anomaly_scale,
        random_state=random_state,
        reference_df=train_df,
    )
    clean_prediction = model.reconstruct(validation_df, columns)
    clean_scores = model.anomaly_score(validation_df, clean_prediction, columns)
    injected_prediction = model.reconstruct(injection.frame, columns)
    injected_scores = model.anomaly_score(injection.frame, injected_prediction, columns)
    eval_mask = _normal_candidate_mask(validation_df, columns)
    timestamps = _timestamp_series(validation_df)
    any_injected = injection.anomaly_mask.any(axis=1).to_numpy(bool)
    distance_any = _nearest_distance(any_injected)

    comparison_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    cell_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for column in columns:
        selected = eval_mask[column].to_numpy(bool)
        y_true = injection.anomaly_mask[column].to_numpy(bool)[selected]
        y_score = injected_scores[column].to_numpy(float)[selected]
        original = _metrics(y_true, y_score, thresholds[column])
        expanded, curve = _best_expanded_threshold(
            y_true,
            y_score,
            candidate_count=candidate_count,
        )
        curve.insert(0, "column", column)
        curve.insert(0, "model", model_name)
        curve_frames.append(curve)

        clean_col = pd.to_numeric(clean_scores[column], errors="coerce")
        injected_col = pd.to_numeric(injected_scores[column], errors="coerce")
        clean_max = float(clean_col[eval_mask[column]].max())
        predicted = injected_col.ge(thresholds[column]) & eval_mask[column]
        truth = injection.anomaly_mask[column].astype(bool)
        false_positive = predicted & ~truth
        normal = eval_mask[column] & ~truth
        distance_same = _nearest_distance(truth.to_numpy(bool))
        normal_above_clean_max = normal & injected_col.gt(clean_max)

        comparison_rows.append({
            "model": model_name,
            "model_display": DISPLAY_NAMES[model_name],
            "column": column,
            "clean_candidate_min": float(clean_col[eval_mask[column]].min()),
            "clean_candidate_max": clean_max,
            "injected_score_min": float(injected_col[eval_mask[column]].min()),
            "injected_score_max": float(injected_col[eval_mask[column]].max()),
            "normal_cells_above_clean_max": int(normal_above_clean_max.sum()),
            "normal_cells_above_clean_max_rate": float(normal_above_clean_max.sum() / max(1, int(normal.sum()))),
            **{f"original_{key}": value for key, value in original.items()},
            **{f"expanded_{key}": value for key, value in expanded.items()},
            "precision_change": float(expanded["precision"] - original["precision"]),
            "recall_change": float(expanded["recall"] - original["recall"]),
            "f1_change": float(expanded["f1_score"] - original["f1_score"]),
        })

        fp_total = int(false_positive.sum())
        normal_values = normal.to_numpy(bool)
        normal_total = int(normal_values.sum())
        for window in DISTANCE_WINDOWS:
            near_same = false_positive.to_numpy(bool) & (distance_same >= 0) & (distance_same <= window)
            near_any = false_positive.to_numpy(bool) & (distance_any >= 0) & (distance_any <= window)
            normal_near_same = normal_values & (distance_same >= 0) & (distance_same <= window)
            normal_near_any = normal_values & (distance_any >= 0) & (distance_any <= window)
            fp_same_rate = float(near_same.sum() / max(1, fp_total))
            fp_any_rate = float(near_any.sum() / max(1, fp_total))
            baseline_same_rate = float(normal_near_same.sum() / max(1, normal_total))
            baseline_any_rate = float(normal_near_any.sum() / max(1, normal_total))
            distance_rows.append({
                "model": model_name,
                "column": column,
                "window_rows": window,
                "window_minutes": window * 5,
                "false_positive_cells": fp_total,
                "fp_near_same_column_injection": int(near_same.sum()),
                "fp_near_same_column_rate": fp_same_rate,
                "normal_baseline_near_same_column_rate": baseline_same_rate,
                "same_column_fp_enrichment": (
                    float(fp_same_rate / baseline_same_rate) if baseline_same_rate else None
                ),
                "fp_near_any_column_injection": int(near_any.sum()),
                "fp_near_any_column_rate": fp_any_rate,
                "normal_baseline_near_any_column_rate": baseline_any_rate,
                "any_column_fp_enrichment": (
                    float(fp_any_rate / baseline_any_rate) if baseline_any_rate else None
                ),
            })

        diagnostic_mask = truth | false_positive | normal_above_clean_max
        cell_frames.append(pd.DataFrame({
            "model": model_name,
            "row_position": np.arange(len(validation_df), dtype=int),
            "timestamp": timestamps,
            "column": column,
            "is_injected": truth,
            "is_false_positive": false_positive,
            "normal_error_above_clean_max": normal_above_clean_max,
            "clean_score": clean_col,
            "injected_score": injected_col,
            "original_threshold": thresholds[column],
            "clean_candidate_max": clean_max,
            "distance_to_same_column_injection_rows": distance_same,
            "distance_to_any_injection_row_rows": distance_any,
        }).loc[diagnostic_mask.to_numpy(bool)])

    return (
        comparison_rows,
        distance_rows,
        pd.concat(cell_frames, ignore_index=True),
        pd.concat(curve_frames, ignore_index=True),
    )


def _write_plot(comparison: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ("precision", "recall", "f1_score")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    x = np.arange(len(comparison))
    labels = [f"{DISPLAY_NAMES[m]}\n{c}" for m, c in zip(comparison["model"], comparison["column"])]
    for axis, metric in zip(axes, metrics):
        axis.plot(x, comparison[f"original_{metric}"], "o-", label="Original clean-error range")
        axis.plot(x, comparison[f"expanded_{metric}"], "s-", label="Expanded injected-score range")
        axis.set_title(metric.replace("_", " ").title(), fontsize=16)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=90, fontsize=8)
    axes[0].set_ylabel("Score", fontsize=14)
    axes[-1].legend(fontsize=11, loc="lower right")
    fig.suptitle("Cucumber Threshold-Range Diagnostic", fontsize=20)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "configs" / "cucumber.yaml"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_ROOT / "artifacts" / "cucumber" / "diagnostics" / "fp_threshold_temp"),
    )
    args = parser.parse_args()

    loaded = load_quality_experiment_config(args.config)
    train_df, validation_df, _, columns = read_configured_datasets(loaded)
    config = loaded.experiment_config
    artifact_root = config.output_root
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons: list[dict[str, object]] = []
    distances: list[dict[str, object]] = []
    cells: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    for model_name in args.models:
        print(f"[cucumber][{model_name}] FP/threshold diagnostic 시작", flush=True)
        result = diagnose_model(
            model_name=model_name,
            artifact_root=artifact_root,
            train_df=train_df,
            validation_df=validation_df,
            columns=columns,
            anomaly_fraction=config.anomaly_fraction,
            anomaly_scale=config.anomaly_scale,
            random_state=config.evaluation_random_seed,
            candidate_count=args.candidate_count,
        )
        comparison_rows, distance_rows, cell_frame, curve_frame = result
        comparisons.extend(comparison_rows)
        distances.extend(distance_rows)
        cells.append(cell_frame)
        curves.append(curve_frame)
        print(f"[cucumber][{model_name}] FP/threshold diagnostic 완료", flush=True)

    comparison_df = pd.DataFrame(comparisons)
    distance_df = pd.DataFrame(distances)
    cell_df = pd.concat(cells, ignore_index=True)
    curve_df = pd.concat(curves, ignore_index=True)
    comparison_df.to_csv(output_dir / "threshold_expansion_comparison.csv", index=False, encoding="utf-8-sig")
    distance_df.to_csv(output_dir / "fp_distance_distribution.csv", index=False, encoding="utf-8-sig")
    cell_df.to_parquet(output_dir / "diagnostic_cells.parquet", index=False)
    curve_df.to_parquet(output_dir / "expanded_threshold_curves.parquet", index=False)
    _write_plot(comparison_df, output_dir / "threshold_expansion_comparison.png")

    summary = {
        "crop": "cucumber",
        "models": list(args.models),
        "candidate_count": args.candidate_count,
        "anomaly_scale": config.anomaly_scale,
        "anomaly_fraction": config.anomaly_fraction,
        "random_state": config.evaluation_random_seed,
        "expanded_candidate_range": "injected_validation_score_min_to_max",
        "formal_artifacts_overwritten": False,
        "output_dir": str(output_dir),
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"완료: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
