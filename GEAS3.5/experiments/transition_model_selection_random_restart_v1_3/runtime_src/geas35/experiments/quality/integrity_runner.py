"""Artifact integrity checker for quality-model experiments.

This module implements Experiment Plan v1.1 Step 8 only. It validates that
expected experiment artifacts exist, are parseable, and preserve no-selection
reporting semantics. End-to-end CLI orchestration is intentionally left to
Step 9.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from geas35.experiments.quality.figures import FIGURE_STEMS


ROOT_REQUIRED_FILES = (
    "hpo_results.json",
    "best_config.json",
    "training_history.csv",
    "threshold_calibration.json",
    "evaluation.json",
    "online_benchmark.json",
    "model_comparison.json",
    "model_comparison.csv",
    "markdown_summary.md",
    "reconstruction_metric_table.csv",
    "anomaly_detection_metric_table.csv",
    "online_benchmark_table.csv",
)

FIGURE_REQUIRED_FILES = (
    "visualization_manifest.json",
)

MODEL_REQUIRED_FILES = (
    "hpo_results.json",
    "best_config.json",
    "training_history.csv",
    "threshold_calibration.json",
    "quality_model_application.json",
    "quality_model.pkl",
    "evaluation.json",
    "online_benchmark.json",
)


@dataclass(frozen=True)
class QualityArtifactIntegrityConfig:
    """Configuration required by the Step 8 artifact integrity checker."""

    output_root: Path
    require_figures: bool = True


@dataclass(frozen=True)
class QualityArtifactIntegrityResult:
    """Artifact integrity check result."""

    output_root: str
    passed: bool
    errors: list[str]
    warnings: list[str]
    checked_files: list[str]
    artifact_integrity_path: str


def run_quality_artifact_integrity_check(
    *,
    config: QualityArtifactIntegrityConfig,
) -> QualityArtifactIntegrityResult:
    """Check expected experiment artifacts and write an integrity report."""

    root = Path(config.output_root)
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[Path] = []

    _check_root_files(root, checked, errors)
    if config.require_figures:
        _check_figure_root_files(root, checked, errors)
    payloads = _load_known_json_payloads(root, checked, errors)
    _check_no_selection_policy(payloads, errors)
    model_names = _model_names(payloads)
    if not model_names:
        errors.append("Could not infer model names from hpo/model comparison artifacts.")
    _check_model_files(root, model_names, checked, errors)
    _check_csv_files(root, checked, errors)
    _check_markdown(root, checked, errors)
    if config.require_figures:
        _check_figures(root, payloads, checked, errors)
    else:
        _check_optional_figures(root, warnings)

    integrity_path = root / "artifact_integrity.json"
    result = QualityArtifactIntegrityResult(
        output_root=str(root),
        passed=not errors,
        errors=errors,
        warnings=warnings,
        checked_files=[str(path) for path in sorted(set(checked))],
        artifact_integrity_path=str(integrity_path),
    )
    _write_json(
        integrity_path,
        {
            "stage": "quality_model_artifact_integrity",
            "passed": result.passed,
            "errors": result.errors,
            "warnings": result.warnings,
            "checked_files": result.checked_files,
            "automatic_best_model_selection": False,
            "test_used_for_selection": False,
        },
    )
    return result


def quality_artifact_integrity_config_from_experiment_config(
    config,
) -> QualityArtifactIntegrityConfig:
    """Create a Step 8 integrity config from the broader experiment config."""

    return QualityArtifactIntegrityConfig(output_root=Path(config.output_root))


def _check_root_files(root: Path, checked: list[Path], errors: list[str]) -> None:
    for name in ROOT_REQUIRED_FILES:
        path = root / name
        checked.append(path)
        if not path.exists():
            errors.append(f"Missing root artifact: {name}")


def _check_figure_root_files(root: Path, checked: list[Path], errors: list[str]) -> None:
    for name in FIGURE_REQUIRED_FILES:
        path = root / name
        checked.append(path)
        if not path.exists():
            errors.append(f"Missing root artifact: {name}")


def _load_known_json_payloads(
    root: Path,
    checked: list[Path],
    errors: list[str],
) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    for name in (
        "hpo_results.json",
        "best_config.json",
        "threshold_calibration.json",
        "evaluation.json",
        "online_benchmark.json",
        "model_comparison.json",
        "visualization_manifest.json",
    ):
        path = root / name
        checked.append(path)
        if not path.exists():
            continue
        try:
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON artifact {name}: {exc}")
    return payloads


def _check_no_selection_policy(payloads: dict[str, Any], errors: list[str]) -> None:
    for name, payload in payloads.items():
        if _contains_key(payload, "selected_model"):
            errors.append(f"Artifact must not contain selected_model: {name}")
        flag = _get_nested_bool(payload, "automatic_best_model_selection")
        if flag is True:
            errors.append(f"Artifact enables automatic model selection: {name}")
        test_used = _get_nested_bool(payload, "test_used_for_selection")
        if test_used is True:
            errors.append(f"Artifact marks test as selection input: {name}")


def _model_names(payloads: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for name in ("hpo_results.json", "model_comparison.json", "evaluation.json"):
        payload = payloads.get(name, {})
        models = payload.get("models") if isinstance(payload, dict) else None
        if isinstance(models, dict):
            names.update(str(model) for model in models)
        elif isinstance(models, list):
            for row in models:
                if isinstance(row, dict) and "model_name" in row:
                    names.add(str(row["model_name"]))
    return sorted(names)


def _check_model_files(
    root: Path,
    model_names: Iterable[str],
    checked: list[Path],
    errors: list[str],
) -> None:
    for model_name in model_names:
        model_dir = root / model_name
        checked.append(model_dir)
        if not model_dir.exists():
            errors.append(f"Missing model artifact directory: {model_name}")
            continue
        for filename in MODEL_REQUIRED_FILES:
            path = model_dir / filename
            checked.append(path)
            if not path.exists():
                errors.append(f"Missing model artifact: {model_name}/{filename}")
                continue
            if filename.endswith(".json"):
                _check_json_file(path, errors)
            elif filename.endswith(".csv"):
                _check_csv_file(path, errors)


def _check_csv_files(root: Path, checked: list[Path], errors: list[str]) -> None:
    for name in (
        "model_comparison.csv",
        "training_history.csv",
        "reconstruction_metric_table.csv",
        "anomaly_detection_metric_table.csv",
        "online_benchmark_table.csv",
    ):
        path = root / name
        checked.append(path)
        if path.exists():
            _check_csv_file(path, errors)


def _check_markdown(root: Path, checked: list[Path], errors: list[str]) -> None:
    path = root / "markdown_summary.md"
    checked.append(path)
    if path.exists() and not path.read_text(encoding="utf-8").strip():
        errors.append("Markdown summary is empty: markdown_summary.md")


def _check_figures(
    root: Path,
    payloads: dict[str, Any],
    checked: list[Path],
    errors: list[str],
) -> None:
    figures_dir = root / "figures"
    checked.append(figures_dir)
    if not figures_dir.exists():
        errors.append("Missing figures directory.")
        return
    for stem in FIGURE_STEMS:
        path = figures_dir / f"{stem}.png"
        checked.append(path)
        if not path.exists():
            errors.append(f"Missing figure artifact: figures/{stem}.png")
    manifest = payloads.get("visualization_manifest.json", {})
    for figure_path in manifest.get("figure_paths", []) if isinstance(manifest, dict) else []:
        path = Path(str(figure_path))
        checked.append(path)
        if not path.exists():
            errors.append(f"Visualization manifest references missing figure: {path}")


def _check_optional_figures(root: Path, warnings: list[str]) -> None:
    if not (root / "figures").exists():
        warnings.append("Figure checks skipped and figures directory is absent.")


def _check_json_file(path: Path, errors: list[str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON artifact {path}: {exc}")
        return
    if _contains_key(payload, "selected_model"):
        errors.append(f"Artifact must not contain selected_model: {path}")


def _check_csv_file(path: Path, errors: list[str]) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            next(reader, None)
    except csv.Error as exc:
        errors.append(f"Invalid CSV artifact {path}: {exc}")


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _get_nested_bool(value: Any, key: str) -> bool | None:
    if isinstance(value, dict):
        if key in value and isinstance(value[key], bool):
            return value[key]
        for item in value.values():
            found = _get_nested_bool(item, key)
            if found is True:
                return True
    elif isinstance(value, list):
        for item in value:
            found = _get_nested_bool(item, key)
            if found is True:
                return True
    return None


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
    "QualityArtifactIntegrityConfig",
    "QualityArtifactIntegrityResult",
    "quality_artifact_integrity_config_from_experiment_config",
    "run_quality_artifact_integrity_check",
]
