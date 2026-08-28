"""YAML-backed configuration loading for transition experiments."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from geas35.models.transition.features import (
    NEXT_OBSERVATION_PREFIX,
    resolve_transition_feature_schema,
)
from geas35.experiments.transition.runner import (
    TransitionExperimentConfig,
    TransitionModelExperimentSpec,
)
from geas35.models.transition.hyperparameters import (
    DEFAULT_HPO_BUDGET,
    DEFAULT_HPO_RANDOM_SEED,
    TransitionHPOConfig,
)
from geas35.rl import MDP_V1_ACTION_COLUMNS


RL_VALID_TRANSITION_COLUMN = "rl_valid_transition"
RL_DATASET_PREPARATION_HINT = (
    "Transition experiments require RL-ready parquet files from "
    "offline_dataset_preparation/datasets/5_rl_dataset. Run the prerequisite "
    "steps first, for example:\n"
    "  python offline_dataset_preparation/scripts/03_control_quality.py "
    "--quality-model-artifact <quality_model_application.json>\n"
    "  python offline_dataset_preparation/scripts/04_prepare_rl_dataset.py"
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
    ranking = _mapping(
        payload.get("ranking", payload.get("selection", {})),
        "ranking",
    )
    hpo = _mapping(payload.get("hpo", {}), "hpo")
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
        ranking_strategy_name=str(ranking.get("strategy", "mean_rmse")),
        ranking_strategy_params=dict(
            _mapping(ranking.get("params", {}), "ranking.params")
        ),
        hpo_config=_hpo_config(hpo),
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

    train_df = _read_rl_ready_frame(loaded.train_path, split="train")
    validation_df = _read_rl_ready_frame(loaded.validation_path, split="validation")
    test_df = _read_rl_ready_frame(loaded.test_path, split="test")
    rollout_df = (
        None
        if loaded.rollout_path is None
        else _read_rl_ready_frame(loaded.rollout_path, split="rollout")
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


def _hpo_config(hpo: Mapping[str, Any]) -> TransitionHPOConfig:
    objective = _mapping(hpo.get("objective", {}), "hpo.objective")
    weights = _mapping(objective.get("weights", {}), "hpo.objective.weights")
    kwargs: dict[str, Any] = {
        "enabled": bool(hpo.get("enabled", True)),
        "budget": int(hpo.get("budget", DEFAULT_HPO_BUDGET)),
        "random_seed": int(hpo.get("random_seed", DEFAULT_HPO_RANDOM_SEED)),
        "max_train_rows": (None if hpo.get("max_train_rows") is None else int(hpo["max_train_rows"])),
        "max_validation_rows": (None if hpo.get("max_validation_rows") is None else int(hpo["max_validation_rows"])),
        "search_spaces": dict(
            _mapping(hpo.get("search_spaces", {}), "hpo.search_spaces")
        ),
        "target_search_spaces": dict(
            _mapping(hpo.get("target_search_spaces", {}), "hpo.target_search_spaces")
        ),
    }
    if weights:
        kwargs["objective_weights"] = dict(weights)
    return TransitionHPOConfig(**kwargs)


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
    if not path.exists():
        raise FileNotFoundError(
            f"Transition {path.name!r} input does not exist: {path}\n"
            f"{RL_DATASET_PREPARATION_HINT}"
        )
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported dataset file type: {path}")


def _read_rl_ready_frame(path: Path, *, split: str) -> pd.DataFrame:
    frame = _read_frame(path)
    _validate_rl_ready_frame(frame, path=path, split=split)
    return frame


def _validate_rl_ready_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    split: str,
) -> None:
    if frame is None or frame.empty:
        raise ValueError(_rl_ready_error(path, split, "frame is empty"))

    required_columns = (RL_VALID_TRANSITION_COLUMN, *MDP_V1_ACTION_COLUMNS)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            _rl_ready_error(
                path,
                split,
                f"missing required RL-ready columns: {', '.join(missing[:8])}",
            )
        )

    observation_columns = [column for column in frame.columns if column.startswith("obs_")]
    next_columns = [
        column for column in frame.columns if column.startswith(NEXT_OBSERVATION_PREFIX)
    ]
    if not observation_columns:
        raise ValueError(_rl_ready_error(path, split, "no obs_* columns are present"))
    if not next_columns:
        raise ValueError(_rl_ready_error(path, split, "no next_* target columns are present"))

    schema = resolve_transition_feature_schema(frame)
    if not schema.dynamic_target_columns:
        raise ValueError(
            _rl_ready_error(
                path,
                split,
                "no dynamic transition targets could be resolved from obs_* and next_* columns",
            )
        )


def _rl_ready_error(path: Path, split: str, reason: str) -> str:
    return (
        f"Transition {split} input is not RL-ready: {path}\n"
        f"Reason: {reason}.\n"
        f"{RL_DATASET_PREPARATION_HINT}"
    )


def _resolve_path(value: Any, source_path: Path, *, required: bool) -> Path | None:
    if value is None:
        if required:
            raise ValueError("Dataset path is required.")
        return None
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
