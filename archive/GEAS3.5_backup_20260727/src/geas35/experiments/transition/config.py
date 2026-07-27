"""YAML-backed configuration loading for transition experiments."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from geas35.experiments.transition.runner import (
    TransitionExperimentConfig,
    TransitionModelExperimentSpec,
)


@dataclass(frozen=True)
class LoadedTransitionExperimentConfig:
    """Fully resolved transition experiment configuration."""

    source_path: Path
    raw_config: Mapping[str, Any]
    experiment_config: TransitionExperimentConfig
    train_path: Path
    validation_path: Path
    test_path: Path
    rollout_path: Path | None


def load_transition_experiment_config(
    path: str | Path,
) -> LoadedTransitionExperimentConfig:
    """Load a transition experiment YAML config without reading datasets."""

    source_path = Path(path).expanduser().resolve()
    payload = _load_yaml_mapping(source_path)
    dataset = _mapping(payload.get("dataset", {}), "dataset")
    experiment = _mapping(payload.get("experiment", {}), "experiment")
    models = _mapping(payload.get("models", {}), "models")
    selection = _mapping(payload.get("selection", {}), "selection")
    rollout = _mapping(payload.get("rollout", {}), "rollout")

    train_path = _resolve_path(dataset.get("train"), source_path, required=True)
    validation_path = _resolve_path(
        dataset.get("validation"),
        source_path,
        required=True,
    )
    test_path = _resolve_path(dataset.get("test"), source_path, required=True)
    rollout_path = _resolve_path(dataset.get("rollout"), source_path, required=False)
    output_root = _resolve_output_path(experiment.get("output_dir"), source_path)
    crop = str(dataset.get("crop", experiment.get("crop", ""))).strip()
    if not crop:
        raise ValueError("dataset.crop or experiment.crop is required.")

    experiment_config = TransitionExperimentConfig(
        crop=crop,
        output_root=output_root,
        models=_model_specs(models),
        selection_strategy_name=str(selection.get("strategy", "mean_rmse")),
        selection_strategy_params=dict(
            _mapping(selection.get("params", {}), "selection.params")
        ),
        rollout_enabled=bool(rollout.get("enabled", False)),
        rollout_horizon_steps=tuple(
            int(value)
            for value in _sequence(rollout.get("horizon_steps", (3, 6, 12)))
        ),
        rollout_step_minutes=int(rollout.get("step_minutes", 5)),
        random_seed=int(experiment.get("random_seed", 0)),
    )
    return LoadedTransitionExperimentConfig(
        source_path=source_path,
        raw_config=payload,
        experiment_config=experiment_config,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        rollout_path=rollout_path,
    )


def read_configured_transition_frames(
    loaded: LoadedTransitionExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Read train/validation/test and optional rollout frames from config."""

    train_df = _read_frame(loaded.train_path)
    validation_df = _read_frame(loaded.validation_path)
    test_df = _read_frame(loaded.test_path)
    rollout_df = (
        None if loaded.rollout_path is None else _read_frame(loaded.rollout_path)
    )
    return train_df, validation_df, test_df, rollout_df


def _model_specs(models: Mapping[str, Any]) -> tuple[TransitionModelExperimentSpec, ...]:
    raw_candidates = models.get("candidates", models.get("models", ()))
    candidates = []
    for item in _sequence(raw_candidates):
        mapping = _mapping(item, "models.candidates item")
        name = str(mapping.get("name", mapping.get("model_name", ""))).strip()
        if not name:
            raise ValueError("Each transition model candidate must define name.")
        candidates.append(
            TransitionModelExperimentSpec(
                model_name=name,
                params=dict(_mapping(mapping.get("params", {}), f"{name}.params")),
                metadata=dict(
                    _mapping(mapping.get("metadata", {}), f"{name}.metadata")
                ),
            )
        )
    if not candidates:
        raise ValueError("models.candidates must define at least one model.")
    return tuple(candidates)


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


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported dataset file type: {path}")


def _resolve_path(value: Any, source_path: Path, *, required: bool) -> Path | None:
    if value is None:
        if required:
            raise ValueError("Dataset path is required.")
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    from_config = (source_path.parent / path).resolve()
    if from_config.exists():
        return from_config
    return (Path.cwd() / path).resolve()


def _resolve_output_path(value: Any, source_path: Path) -> Path:
    if value is None:
        raise ValueError("experiment.output_dir is required.")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


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
            parsed = _parse_scalar_or_inline_mapping(item)
        else:
            parsed = None
        if index < len(lines) and lines[index][0] > indent:
            nested, index = _parse_block(lines, index, lines[index][0])
            if isinstance(parsed, dict) and isinstance(nested, dict):
                parsed = {**parsed, **nested}
            elif parsed is None:
                parsed = nested
            else:
                raise ValueError(f"Unexpected nested YAML block after: {item}")
        result.append(parsed)
    return result, index


def _parse_scalar_or_inline_mapping(value: str) -> Any:
    if ":" in value and not value.startswith(("{", "[")):
        key, raw_value = value.split(":", 1)
        raw_value = raw_value.strip()
        return {
            key.strip(): _parse_scalar(raw_value) if raw_value else {}
        }
    return _parse_scalar(value)


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
    "LoadedTransitionExperimentConfig",
    "load_transition_experiment_config",
    "read_configured_transition_frames",
]
