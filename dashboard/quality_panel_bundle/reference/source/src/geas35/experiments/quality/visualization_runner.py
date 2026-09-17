"""Visualization runner for quality-model experiments.

This module implements Experiment Plan v1.1 Step 7 only. It generates
publication-oriented PNG figures from existing experiment artifacts.
Artifact integrity checks and end-to-end CLI orchestration are intentionally
left to later experiment steps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from geas35.experiments.quality.figures import generate_quality_experiment_figures


@dataclass(frozen=True)
class QualityVisualizationConfig:
    """Configuration required by the Step 7 visualization generator."""

    output_root: Path


@dataclass(frozen=True)
class QualityVisualizationResult:
    """Figure artifact paths produced by Step 7."""

    output_root: str
    figures_dir: str
    figure_paths: list[str]
    visualization_manifest_path: str


def run_quality_visualization_generation(
    *,
    config: QualityVisualizationConfig,
) -> QualityVisualizationResult:
    """Generate visualization figures and write a manifest."""

    output_root = Path(config.output_root)
    figure_paths = generate_quality_experiment_figures(output_root)
    manifest_path = output_root / "visualization_manifest.json"
    result = QualityVisualizationResult(
        output_root=str(output_root),
        figures_dir=str(output_root / "figures"),
        figure_paths=[str(path) for path in figure_paths],
        visualization_manifest_path=str(manifest_path),
    )
    _write_json(
        manifest_path,
        {
            "stage": "quality_model_visualization",
            "automatic_best_model_selection": False,
            "test_used_for_selection": False,
            "figure_count": len(result.figure_paths),
            "figures_dir": result.figures_dir,
            "figure_paths": result.figure_paths,
        },
    )
    return result


def quality_visualization_config_from_experiment_config(
    config,
) -> QualityVisualizationConfig:
    """Create a Step 7 visualization config from the broader experiment config."""

    return QualityVisualizationConfig(output_root=Path(config.output_root))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


__all__ = [
    "QualityVisualizationConfig",
    "QualityVisualizationResult",
    "quality_visualization_config_from_experiment_config",
    "run_quality_visualization_generation",
]
