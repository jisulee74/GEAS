"""Regenerate consistently styled training-diagnostic figures for completed crops."""

from __future__ import annotations

import sys
from pathlib import Path


CROPS = ("cucumber", "melon", "strawberry")


def _experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_src_on_path() -> None:
    src_path = _project_root() / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> int:
    _ensure_src_on_path()
    from geas35.experiments.quality.figures import (
        _configure_matplotlib,
        _plot_hpo_progress,
        _plot_online_benchmark_summary,
        _plot_reconstruction_heatmap,
        _plot_training_loss,
        _pyplot,
        _read_csv,
        _read_json,
    )

    plt = _pyplot()
    _configure_matplotlib(plt)
    artifacts_root = _experiment_root() / "artifacts"

    for crop in CROPS:
        artifact_root = artifacts_root / crop
        figures_dir = artifact_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        outputs = [
            *_plot_hpo_progress(
                plt,
                figures_dir,
                _read_json(artifact_root / "hpo_results.json"),
            ),
            *_plot_training_loss(
                plt,
                figures_dir,
                _read_csv(artifact_root / "training_history.csv"),
            ),
            *_plot_online_benchmark_summary(
                plt,
                figures_dir,
                _read_csv(artifact_root / "online_benchmark_table.csv"),
                _read_csv(artifact_root / "model_comparison.csv"),
            ),
            *_plot_reconstruction_heatmap(
                plt,
                figures_dir,
                _read_csv(artifact_root / "reconstruction_metric_table.csv"),
            ),
        ]
        for output in outputs:
            print(f"Saved: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
