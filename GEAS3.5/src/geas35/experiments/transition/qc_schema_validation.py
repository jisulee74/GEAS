"""Step 11.2 validation for existing QC train/validation datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geas35.io_utils import flag_cell_count, flag_sum, write_json
from geas35.preprocessing import ACTION_COLUMNS, STATE_COLUMNS
from geas35.experiments.transition.qc_contract_freeze import (
    FREEZE_MANIFEST_SCHEMA_VERSION,
    OFFICIAL_QC_MODEL_BY_CROP,
    REQUIRED_QC_SPLITS,
)


SCHEMA_VALIDATION_VERSION = "geas35.transition.step11.2.v1"
REQUIRED_METADATA_COLUMNS = (
    "reg_date", "crop", "split", "series_id", "segment_id",
    "episode_date", "episode_id",
)
REPRESENTATIVE_COLUMNS = (
    "in_temp_representative", "in_hum_representative", "in_co2_representative",
)
SOLAR_COLUMNS = ("out_light", "out_light_sum")
GROWTH_STAGE_COLUMNS = (
    "growth_stage_dat", "growth_stage_order", "growth_stage_name",
    "growth_stage_unmatched_flag",
)
QUALITY_AGGREGATE_FLAGS = (
    "rule_outlier_flag", "ai_outlier_flag", "invalid_flag", "imputed_flag",
    "quality_unconfirmed_flag",
)


class QCSchemaValidationError(ValueError):
    """Raised when a Step 11.2 schema or provenance contract fails."""


def validate_existing_qc_schema_and_provenance(
    *,
    project_root: str | Path,
    dataset_root: str | Path | None = None,
    freeze_manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Validate frozen QC Train/Validation without modifying input files."""

    root = Path(project_root).expanduser().resolve()
    datasets = _resolved(dataset_root, root / "offline_dataset_preparation/datasets/03_quality_controlled")
    freeze_path = _resolved(freeze_manifest_path, datasets / "step11_1_qc_contract_freeze_manifest.json")
    destination = _resolved(output_path, datasets / "step11_2_qc_schema_provenance_manifest.json")
    qc_manifest_path = datasets / "quality_controlled_manifest.json"
    freeze = _read_object(freeze_path, "Step 11.1 freeze manifest")
    manifest = _read_object(qc_manifest_path, "QC manifest")

    _require(freeze.get("schema_version") == FREEZE_MANIFEST_SCHEMA_VERSION, "Invalid Step 11.1 freeze schema")
    _require(freeze.get("status") == "passed", "Step 11.1 freeze status must be passed")
    _require(freeze.get("selected_models") == OFFICIAL_QC_MODEL_BY_CROP, "Step 11.1 selected model mapping changed")
    _verify_hash(qc_manifest_path, freeze["qc_manifest"], "QC manifest")

    split_entries = {
        (str(item.get("crop")), str(item.get("split"))): item
        for item in _list_of_objects(manifest.get("splits"), "manifest.splits")
    }
    results: dict[str, Any] = {}
    common_schema: set[str] | None = None
    for crop in OFFICIAL_QC_MODEL_BY_CROP:
        crop_result: dict[str, Any] = {}
        for split in REQUIRED_QC_SPLITS:
            label = f"{crop}/{split}"
            _require((crop, split) in split_entries, f"{label}: missing manifest statistics")
            path = datasets / crop / f"{split}.parquet"
            frozen_record = freeze["crops"][crop]["datasets"][split]
            _verify_hash(path, frozen_record, label)
            frame = pd.read_parquet(path)
            _require(not frame.empty, f"{label}: dataset is empty")
            required = (*REQUIRED_METADATA_COLUMNS, *STATE_COLUMNS, *ACTION_COLUMNS,
                        *REPRESENTATIVE_COLUMNS, *SOLAR_COLUMNS, *GROWTH_STAGE_COLUMNS,
                        *QUALITY_AGGREGATE_FLAGS)
            missing = [column for column in required if column not in frame.columns]
            _require(not missing, f"{label}: missing required source columns: {missing}")
            _validate_dtypes(frame, label)
            temporal = _validate_temporal_and_episode(frame, label)
            contracts = _validate_contract_columns(frame, manifest, label)
            statistics = _actual_statistics(frame, manifest)
            expected_statistics = split_entries[(crop, split)]
            for key, actual in statistics.items():
                _require(expected_statistics.get(key) == actual,
                         f"{label}: manifest statistic {key} mismatch; "
                         f"manifest={expected_statistics.get(key)}, parquet={actual}")
            schema = set(frame.columns)
            common_schema = schema if common_schema is None else common_schema & schema
            crop_result[split] = {
                "path": _display(path, root),
                "sha256": _sha256(path),
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "dtypes": {column: str(frame[column].dtype) for column in required},
                "finite_coverage": {
                    column: _finite_coverage(frame[column])
                    for column in (*REPRESENTATIVE_COLUMNS, *SOLAR_COLUMNS)
                },
                "temporal_episode_validation": temporal,
                "contracts": contracts,
                "manifest_statistics": statistics,
                "immutable_input": True,
            }
        results[crop] = crop_result

    provenance = _provenance_inventory(manifest, root)
    payload = {
        "schema_version": SCHEMA_VALIDATION_VERSION,
        "step": "11.2",
        "status": "passed",
        "step11_1_freeze_manifest": {
            "path": _display(freeze_path, root), "sha256": _sha256(freeze_path),
        },
        "input_policy": "immutable_existing_train_validation",
        "legacy_provenance_policy": "record_legacy_unavailable_do_not_infer",
        "common_column_count": len(common_schema or ()),
        "provenance": provenance,
        "crops": results,
    }
    write_json(destination, payload)
    return destination


def _validate_dtypes(frame: pd.DataFrame, label: str) -> None:
    _require(pd.api.types.is_datetime64_any_dtype(frame["reg_date"]), f"{label}: reg_date must be datetime")
    for column in (*STATE_COLUMNS, *ACTION_COLUMNS, *REPRESENTATIVE_COLUMNS, *SOLAR_COLUMNS):
        _require(pd.api.types.is_numeric_dtype(frame[column]), f"{label}: {column} must be numeric")
    for column in (*QUALITY_AGGREGATE_FLAGS, "growth_stage_unmatched_flag"):
        values = pd.to_numeric(frame[column], errors="coerce")
        _require(values.notna().all() and values.isin((0, 1)).all(), f"{label}: {column} must be binary")
    for column in ("growth_stage_dat", "growth_stage_order"):
        _require(pd.api.types.is_numeric_dtype(frame[column]), f"{label}: {column} must be numeric")


def _validate_temporal_and_episode(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    time = pd.to_datetime(frame["reg_date"], errors="coerce")
    _require(time.notna().all(), f"{label}: reg_date contains invalid timestamps")
    _require(time.is_monotonic_increasing, f"{label}: timestamps are not globally ordered")
    episode = frame["episode_id"]
    _require(episode.notna().all(), f"{label}: episode_id contains null")
    previous_time = time.shift(1)
    same_episode = episode.eq(episode.shift(1))
    step = time - previous_time
    _require((step.loc[same_episode] == pd.Timedelta(minutes=5)).all(),
             f"{label}: an episode contains a non-5-minute transition")
    episode_date = pd.to_datetime(frame["episode_date"], errors="coerce")
    _require(episode_date.notna().all(), f"{label}: episode_date contains invalid values")
    _require((episode_date.dt.date == time.dt.date).all(), f"{label}: episode crosses a date boundary")
    boundary_required = (
        step.ne(pd.Timedelta(minutes=5))
        | frame["series_id"].ne(frame["series_id"].shift(1))
        | frame["segment_id"].ne(frame["segment_id"].shift(1))
        | time.dt.date.ne(time.shift(1).dt.date)
    )
    boundary_required.iloc[0] = True
    _require((~same_episode.loc[boundary_required]).iloc[1:].all(),
             f"{label}: required temporal/series/segment boundary does not start a new episode")
    return {
        "ordered": True,
        "exact_step_minutes_within_episode": 5,
        "episode_count": int(episode.nunique()),
        "first_timestamp": time.iloc[0].isoformat(),
        "last_timestamp": time.iloc[-1].isoformat(),
    }


def _validate_contract_columns(frame: pd.DataFrame, manifest: dict[str, Any], label: str) -> dict[str, Any]:
    ai_columns = tuple(str(x) for x in manifest.get("ai_imputation_columns", ()))
    expected_quality = [f"{column}_missing_flag" for column in STATE_COLUMNS]
    expected_imputation = [f"{column}_imputed_flag" for column in ai_columns]
    expected_restoration = [f"{column}_restored_flag" for column in ACTION_COLUMNS]
    for name, columns in (
        ("quality_flags", expected_quality),
        ("ai_imputation", expected_imputation),
        ("action_restoration", expected_restoration),
    ):
        missing = [column for column in columns if column not in frame.columns]
        _require(not missing, f"{label}: {name} contract columns missing: {missing}")
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            _require(values.notna().all() and values.isin((0, 1)).all(),
                     f"{label}: {column} must be a binary provenance flag")
    for representative in REPRESENTATIVE_COLUMNS:
        source = f"{representative}_source"
        _require(source in frame.columns, f"{label}: missing representative provenance {source}")
        valid = {"sensor1", "sensor2", "ai_imputed_sensor1", "unavailable"}
        _require(set(frame[source].dropna().astype(str).unique()) <= valid,
                 f"{label}: {source} contains unsupported provenance")
    return {
        "growth_stage": "available",
        "quality_flags": "available",
        "imputation_flags": "available",
        "action_restoration_flags": "available",
        "representative_source": "available",
    }


def _actual_statistics(frame: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, int]:
    ai_columns = set(str(x) for x in manifest.get("ai_imputation_columns", ()))
    return {
        "output_rows": len(frame),
        "output_columns": len(frame.columns),
        "missing_cells": flag_cell_count(frame, "_missing_flag"),
        "restored_action_cells": flag_cell_count(frame, "_restored_flag"),
        "rule_outlier_rows": flag_sum(frame, "rule_outlier_flag"),
        "rule_outlier_cells": flag_cell_count(frame, "_rule_outlier_flag", exclude={"rule_outlier_flag"}),
        "ai_outlier_rows": flag_sum(frame, "ai_outlier_flag"),
        "invalid_rows": flag_sum(frame, "invalid_flag"),
        "invalid_cells": flag_cell_count(frame, "_invalid_flag", exclude={"invalid_flag"}),
        "imputed_rows": flag_sum(frame, "imputed_flag"),
        "imputed_cells": flag_cell_count(frame, "_imputed_flag", exclude={"imputed_flag"}),
        "provisional_imputed_cells": sum(
            flag_sum(frame, f"{column}_imputed_flag") for column in STATE_COLUMNS
            if column not in ai_columns
        ),
        "unconfirmed_rows": flag_sum(frame, "quality_unconfirmed_flag"),
        "unconfirmed_cells": flag_cell_count(
            frame, "_unconfirmed_flag", exclude={"quality_unconfirmed_flag"}
        ),
    }


def _provenance_inventory(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    crop_cycle = Path(str(manifest.get("crop_cycle_manifest_path", ""))).expanduser()
    if not crop_cycle.is_absolute():
        crop_cycle = root / crop_cycle
    growth = ({"status": "available", "path": _display(crop_cycle, root), "sha256": _sha256(crop_cycle)}
              if crop_cycle.is_file() else {"status": "legacy_unavailable"})
    action_sources = {}
    applications = manifest.get("crop_applications", {})
    for crop in OFFICIAL_QC_MODEL_BY_CROP:
        configured = applications.get(crop, {}).get("control_log_path")
        if configured:
            path = Path(str(configured)).expanduser()
            if not path.is_absolute():
                path = root / path
            action_sources[crop] = ({"status": "available", "path": _display(path, root), "sha256": _sha256(path)}
                                    if path.is_file() else {"status": "legacy_unavailable"})
        else:
            action_sources[crop] = {
                "status": "legacy_unavailable",
                "reason": "no separate control_log_path; existing 02_split action values were used",
            }
    return {"growth_stage_crop_cycle": growth, "action_restoration_source": action_sources}


def _finite_coverage(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = int(np.isfinite(numeric).sum())
    total = len(series)
    return {"finite_rows": finite, "total_rows": total,
            "finite_fraction": finite / total if total else 0.0}


def _verify_hash(path: Path, record: object, label: str) -> None:
    _require(path.is_file(), f"{label}: frozen input is missing: {path}")
    _require(isinstance(record, dict) and record.get("sha256") == _sha256(path),
             f"{label}: SHA-256 differs from Step 11.1 freeze manifest")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QCSchemaValidationError(f"Cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _list_of_objects(value: object, label: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list), f"{label} must be an array")
    _require(all(isinstance(item, dict) for item in value), f"{label} items must be objects")
    return list(value)


def _resolved(value: str | Path | None, default: Path) -> Path:
    return default.resolve() if value is None else Path(value).expanduser().resolve()


def _display(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QCSchemaValidationError(message)


__all__ = [
    "QCSchemaValidationError", "SCHEMA_VALIDATION_VERSION",
    "validate_existing_qc_schema_and_provenance",
]
