"""YAML-backed configuration loading for quality-model experiments."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from geas35.experiments.quality.runner import (
    ModelExperimentSpec,
    QualityExperimentConfig,
)
from geas35.models.quality import EarlyStoppingConfig
from geas35.preprocessing import STATE_COLUMNS


@dataclass(frozen=True)
class LoadedQualityExperimentConfig:
    """Fully resolved experiment configuration loaded from a YAML file."""

    source_path: Path
    raw_config: Mapping[str, Any]
    experiment_config: QualityExperimentConfig
    train_path: Path
    validation_path: Path
    test_path: Path
    observation_columns: tuple[str, ...] | None
    generate_figures: bool


def load_quality_experiment_config(path: str | Path) -> LoadedQualityExperimentConfig:
    """Load a YAML quality experiment config without reading dataset frames."""

    source_path = Path(path).expanduser().resolve()
    payload = _load_yaml_mapping(source_path)
    dataset = _mapping(payload.get("dataset", {}), "dataset")
    experiment = _mapping(payload.get("experiment", {}), "experiment")
    hpo = _mapping(payload.get("hpo", {}), "hpo")
    threshold = _mapping(payload.get("threshold", {}), "threshold")
    evaluation = _mapping(payload.get("evaluation", {}), "evaluation")
    report = _mapping(payload.get("report", {}), "report")

    train_path = _resolve_path(dataset.get("train"), source_path)
    validation_path = _resolve_path(dataset.get("validation"), source_path)
    test_path = _resolve_path(dataset.get("test"), source_path)
    output_root = _resolve_output_path(experiment.get("output_dir"), source_path)
    random_seed = int(experiment.get("random_seed", 0))

    models = _model_specs(hpo, random_seed=random_seed)
    threshold_candidates = _threshold_candidates(threshold)
    early_stopping = _early_stopping_config(hpo)
    observation_columns = _optional_observation_columns(dataset)

    experiment_config = QualityExperimentConfig(
        output_root=output_root,
        models=models,
        threshold_candidates=threshold_candidates,
        early_stopping=early_stopping,
        mask_fraction=float(evaluation.get("mask_fraction", 0.1)),
        anomaly_fraction=float(evaluation.get("anomaly_fraction", 0.1)),
        anomaly_scale=float(evaluation.get("anomaly_scale", 8.0)),
        evaluation_random_seed=int(evaluation.get("random_seed", random_seed)),
        efficiency_repeats=int(evaluation.get("efficiency_repeats", 3)),
    )
    return LoadedQualityExperimentConfig(
        source_path=source_path,
        raw_config=payload,
        experiment_config=experiment_config,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        observation_columns=observation_columns,
        generate_figures=bool(report.get("generate_figures", False)),
    )


def read_configured_datasets(
    loaded: LoadedQualityExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Read train/validation/test frames and resolve observation columns."""

    train_df = _read_frame(loaded.train_path)
    validation_df = _read_frame(loaded.validation_path)
    test_df = _read_frame(loaded.test_path)
    columns = loaded.observation_columns or _infer_observation_columns(
        train_df,
        validation_df,
        test_df,
    )
    return train_df, validation_df, test_df, columns


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
    except ModuleNotFoundError:
        payload = _parse_yaml_subset(text)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return payload


def _model_specs(
    hpo: Mapping[str, Any],
    *,
    random_seed: int,
) -> tuple[ModelExperimentSpec, ...]:
    budget = int(hpo.get("budget", 1))
    model_names = tuple(str(name) for name in _sequence(hpo.get("models", ())))
    if not model_names:
        raise ValueError("hpo.models must define at least one model name.")
    search_spaces = _mapping(hpo.get("search_spaces", {}), "hpo.search_spaces")
    common_space = _mapping(search_spaces.get("common", {}), "hpo.search_spaces.common")
    model_spaces = _mapping(
        search_spaces.get("model_specific", {}),
        "hpo.search_spaces.model_specific",
    )
    model_budgets = _mapping(hpo.get("model_budgets", {}), "hpo.model_budgets")
    model_seeds = _mapping(hpo.get("model_random_seeds", {}), "hpo.model_random_seeds")
    specs = []
    for offset, model_name in enumerate(model_names):
        model_specific = _mapping(model_spaces.get(model_name, {}), model_name)
        specs.append(
            ModelExperimentSpec(
                model_name=model_name,
                search_space={
                    "common": dict(common_space),
                    "model_specific": dict(model_specific),
                },
                hpo_budget=int(model_budgets.get(model_name, budget)),
                random_seed=int(model_seeds.get(model_name, random_seed + offset)),
            )
        )
    return tuple(specs)


def _threshold_candidates(threshold: Mapping[str, Any]) -> tuple[float, ...]:
    if not bool(threshold.get("enabled", True)):
        raise ValueError("threshold.enabled must remain true for v1.4 experiments.")
    candidates = threshold.get("candidates", ())
    values = tuple(float(value) for value in _sequence(candidates))
    if not values:
        raise ValueError("threshold.candidates must define at least one value.")
    return values


def _early_stopping_config(hpo: Mapping[str, Any]) -> EarlyStoppingConfig:
    early = _mapping(hpo.get("early_stopping", {}), "hpo.early_stopping")
    return EarlyStoppingConfig(
        monitor=str(early.get("monitor", "validation_reconstruction_loss")),
        mode=str(early.get("mode", "min")),
        patience=int(early.get("patience", 5)),
        min_delta=float(early.get("min_delta", 0.0)),
    )


def _optional_observation_columns(dataset: Mapping[str, Any]) -> tuple[str, ...] | None:
    value = dataset.get("observation_columns")
    if value is None:
        return None
    columns = tuple(str(col) for col in _sequence(value))
    if not columns:
        raise ValueError("dataset.observation_columns must not be empty when provided.")
    return columns


def _infer_observation_columns(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[str, ...]:
    columns = tuple(
        col
        for col in STATE_COLUMNS
        if col in train_df.columns and col in validation_df.columns and col in test_df.columns
    )
    if not columns:
        raise ValueError(
            "Could not infer observation columns from GEAS STATE_COLUMNS. "
            "Set dataset.observation_columns in the experiment YAML."
        )
    return columns


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        lines = suffix == ".jsonl"
        return pd.read_json(path, lines=lines)
    raise ValueError(f"Unsupported dataset file type: {path}")


def _resolve_path(value: Any, source_path: Path) -> Path:
    if value is None:
        raise ValueError("Dataset path is required.")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (source_path.parent / path).resolve()


def _resolve_output_path(value: Any, source_path: Path) -> Path:
    if value is None:
        raise ValueError("experiment.output_dir is required.")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (_experiment_root_for_config(source_path) / path).resolve()


def _experiment_root_for_config(source_path: Path) -> Path:
    config_dir = source_path.parent
    if config_dir.name == "configs":
        return config_dir.parent
    return config_dir


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    lines = []
    for raw_line in text.splitlines():
        clean = raw_line.split("#", 1)[0].rstrip()
        if clean.strip():
            lines.append((len(clean) - len(clean.lstrip(" ")), clean.lstrip(" ")))
    if not lines:
        return {}
    parsed, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError("Could not parse complete YAML config.")
    if not isinstance(parsed, dict):
        raise ValueError("YAML root must be a mapping.")
    return parsed


def _parse_block(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if lines[index][0] < indent:
        return {}, index
    if lines[index][1].startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_dict(lines, index, indent)


def _parse_dict(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation near: {content}")
        if content.startswith("- ") or ":" not in content:
            break
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            result[key] = _parse_scalar(raw_value)
        elif index < len(lines) and lines[index][0] > indent:
            result[key], index = _parse_block(lines, index, lines[index][0])
        else:
            result[key] = {}
    return result, index


def _parse_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent or not content.startswith("- "):
            break
        item = content[2:].strip()
        index += 1
        if item:
            result.append(_parse_scalar(item))
        elif index < len(lines) and lines[index][0] > indent:
            nested, index = _parse_block(lines, index, lines[index][0])
            result.append(nested)
        else:
            result.append(None)
    return result, index


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith("[") or value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if value.startswith("[") and value.endswith("]"):
                body = value[1:-1].strip()
                if not body:
                    return []
                return [_parse_scalar(item.strip()) for item in body.split(",")]
            raise
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("\"'")


__all__ = [
    "LoadedQualityExperimentConfig",
    "load_quality_experiment_config",
    "read_configured_datasets",
]
