"""Run the quality-control model-selection experiment from this folder."""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path


def _experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_src_on_path() -> None:
    src_path = _project_root() / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))



_RESULT_COLUMNS = (
    "crop", "model", "masked_cells", "rmse", "mae", "injected_cells",
    "f1_score", "precision", "recall", "mcc", "fpr", "roc_auc", "pr_auc",
    "inference_latency_ms_per_row", "peak_memory_mb", "model_size_mb",
)


def build_consolidated_result_csvs(artifacts_root: Path) -> tuple[Path, Path]:
    """Build validation/test summary tables from detailed, non-hardcoded artifacts."""
    rows = {"validation": [], "test": []}
    for crop_dir in sorted(path for path in artifacts_root.iterdir() if path.is_dir()):
        paths = {
            "rec": crop_dir / "reconstruction_metric_table.csv",
            "det": crop_dir / "anomaly_detection_metric_table.csv",
            "bench": crop_dir / "online_benchmark_table.csv",
        }
        if not all(path.exists() for path in paths.values()):
            continue
        tables = {}
        for name, path in paths.items():
            with path.open(encoding="utf-8", newline="") as fp:
                tables[name] = list(csv.DictReader(fp))
        for split in rows:
            rec = {r["model_name"]: r for r in tables["rec"]
                   if r["split"] == split and r["column"] == "__all__"}
            det = {r["model_name"]: r for r in tables["det"]
                   if r["split"] == split and r["column"] == "__all__"}
            bench = {r["model_name"]: r for r in tables["bench"]
                     if r["split"] == split}
            model_order = {"modern_tcn": 0, "timesnet": 1, "patch_tst": 2}
            models = sorted(
                set(rec) & set(det),
                key=lambda name: (model_order.get(name, len(model_order)), name),
            )
            for model in models:
                r, d, b = rec[model], det[model], bench.get(model, {})
                tp, fp = float(d["true_positive"]), float(d["false_positive"])
                tn, fn = float(d["true_negative"]), float(d["false_negative"])
                denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                rows[split].append({
                    "crop": crop_dir.name,
                    "model": {"modern_tcn": "ModernTCN", "timesnet": "TimesNet",
                              "patch_tst": "PatchTST"}.get(model, model),
                    "masked_cells": int(float(r["masked_cells"])),
                    "rmse": float(r["rmse"]),
                    "mae": float(r["mae"]),
                    "injected_cells": int(float(d["injected_cells"])),
                    "f1_score": float(d["f1_score"]),
                    "precision": float(d["precision"]),
                    "recall": float(d["recall"]),
                    "mcc": 0.0 if denom == 0 else (tp * tn - fp * fn) / denom,
                    "fpr": 0.0 if fp + tn == 0 else fp / (fp + tn),
                    "roc_auc": float(d["roc_auc"]),
                    "pr_auc": float(d["pr_auc"]),
                    "inference_latency_ms_per_row":
                        float(b["inference_latency_ms_per_row"]) if b else "",
                    "peak_memory_mb":
                        float(b["peak_memory_bytes"]) / (1024 ** 2) if b else "",
                    "model_size_mb":
                        float(b["model_size_bytes"]) / (1024 ** 2) if b else "",
                })
    output_paths = []
    for split in ("validation", "test"):
        path = artifacts_root / f"{split}_results.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=_RESULT_COLUMNS)
            writer.writeheader()
            for row in rows[split]:
                writer.writerow({
                    key: (format(value, ".4g") if isinstance(value, float) else value)
                    for key, value in row.items()
                })
        output_paths.append(path)
    return output_paths[0], output_paths[1]


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    args = [] if argv is None else list(argv)
    if "--config" not in args:
        args = ["--config", str(_experiment_root() / "configs" / "cucumber.yaml"), *args]
    os.chdir(_experiment_root())

    from geas35.experiments.quality.cli import main as quality_main

    status = quality_main(args)
    if status == 0:
        artifacts_root = _experiment_root() / "artifacts"
        build_consolidated_result_csvs(artifacts_root)
        for crop_dir in (path for path in artifacts_root.iterdir() if path.is_dir()):
            for redundant_name in (
                "model_comparison.csv",
                "model_comparison.json",
                "markdown_summary.md",
            ):
                (crop_dir / redundant_name).unlink(missing_ok=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
