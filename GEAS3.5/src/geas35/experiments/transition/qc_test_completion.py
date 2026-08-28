"""Step 11.3: generate only the missing quality-controlled Test splits."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geas35.io_utils import write_json
from geas35.experiments.transition.qc_contract_freeze import OFFICIAL_QC_MODEL_BY_CROP
from geas35.experiments.transition.qc_schema_validation import SCHEMA_VALIDATION_VERSION


TEST_COMPLETION_VERSION = "geas35.transition.step11.3.v1"
REPRESENTATIVE_COLUMNS = (
    "in_temp_representative", "in_hum_representative", "in_co2_representative",
)
GROWTH_STAGE_COLUMNS = (
    "growth_stage_dat", "growth_stage_order", "growth_stage_name",
    "growth_stage_unmatched_flag",
)
FORBIDDEN_LEAKAGE_PREFIXES = ("target_", "next_", "future_", "pred_", "metric_")


class QCTestCompletionError(ValueError):
    """Raised when Step 11.3 cannot safely generate all Test splits."""


def complete_missing_qc_test_only(
    *,
    project_root: str | Path,
    dataset_root: str | Path | None = None,
    step11_2_manifest_path: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    processor: Callable[[Path, str], object] | None = None,
) -> Path:
    """Generate all three Test splits atomically without touching Train/Validation."""

    root = Path(project_root).expanduser().resolve()
    datasets = _resolved(dataset_root, root / "offline_dataset_preparation/datasets")
    source_root = datasets / "02_split"
    qc_root = datasets / "03_quality_controlled"
    validation_path = _resolved(
        step11_2_manifest_path, qc_root / "step11_2_qc_schema_provenance_manifest.json"
    )
    destination = _resolved(
        output_manifest_path, qc_root / "step11_3_qc_test_completion_manifest.json"
    )
    validation = _read_object(validation_path, "Step 11.2 manifest")
    _require(validation.get("schema_version") == SCHEMA_VALIDATION_VERSION,
             "Step 11.2 schema version is invalid")
    _require(validation.get("status") == "passed", "Step 11.2 status must be passed")
    _validate_frozen_applications(root, validation)

    protected = _verify_immutable_train_validation(qc_root, validation)
    input_records: dict[str, Any] = {}
    for crop in OFFICIAL_QC_MODEL_BY_CROP:
        input_path = source_root / crop / "test.parquet"
        _require(input_path.is_file(), f"{crop}: 02_split Test is missing: {input_path}")
        frame = pd.read_parquet(input_path)
        _require(not frame.empty, f"{crop}: 02_split Test is empty")
        forbidden = _forbidden_columns(frame.columns)
        _require(not forbidden, f"{crop}: Test source contains leakage-prone columns: {forbidden}")
        final_path = qc_root / crop / "test.parquet"
        _require(not final_path.exists(), f"{crop}: QC Test already exists; Step 11.3 is missing-Test-only")
        input_records[crop] = _file_record(input_path, root, row_count=len(frame))

    stage_dir = Path(tempfile.mkdtemp(prefix="step11_3_staging_", dir=datasets))
    stage_name = stage_dir.name
    try:
        (processor or _run_default_processor)(datasets, stage_name)
        stage_manifest_path = stage_dir / "quality_controlled_manifest.json"
        stage_manifest = _read_object(stage_manifest_path, "staged Test QC manifest")
        _validate_stage_manifest(stage_manifest)
        split_entries = {
            str(item.get("crop")): item
            for item in _list_of_objects(stage_manifest.get("splits"), "staged manifest.splits")
        }
        test_records: dict[str, Any] = {}
        for crop in OFFICIAL_QC_MODEL_BY_CROP:
            staged_path = stage_dir / crop / "test.parquet"
            _require(staged_path.is_file(), f"{crop}: staged QC Test is missing")
            frame = pd.read_parquet(staged_path)
            _validate_test_frame(
                frame, crop=crop, source_record=input_records[crop],
                qc_root=qc_root, validation=validation,
            )
            entry = split_entries.get(crop)
            _require(entry is not None, f"{crop}: staged Test statistics are missing")
            _require(entry.get("output_rows") == len(frame), f"{crop}: staged Test row count mismatch")
            test_records[crop] = {
                **_file_record(staged_path, root, row_count=len(frame)),
                "column_count": len(frame.columns),
                "finite_coverage": {
                    column: _finite_coverage(frame[column]) for column in REPRESENTATIVE_COLUMNS
                },
                "growth_stage_unmatched_rows": int(
                    pd.to_numeric(frame["growth_stage_unmatched_flag"], errors="coerce")
                    .fillna(1).astype(bool).sum()
                ),
                "source_02_split_test": input_records[crop],
                "selected_qc_model": OFFICIAL_QC_MODEL_BY_CROP[crop],
                "generation_status": "generated_test_only",
            }

        _require(_current_hashes(protected) == protected,
                 "Existing Train/Validation changed during Step 11.3 staging")
        for crop in OFFICIAL_QC_MODEL_BY_CROP:
            final_path = qc_root / crop / "test.parquet"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            (stage_dir / crop / "test.parquet").replace(final_path)
            test_records[crop].update(_file_record(final_path, root, row_count=test_records[crop]["row_count"]))

        payload = {
            "schema_version": TEST_COMPLETION_VERSION,
            "step": "11.3",
            "status": "passed",
            "parent_step11_2_manifest": {
                "path": _display(validation_path, root), "sha256": _sha256(validation_path),
            },
            "selected_models": OFFICIAL_QC_MODEL_BY_CROP,
            "input_policy": "02_split_test_only",
            "existing_train_validation_policy": "read_only_hash_verified_unchanged",
            "test_used_for_qc_selection_or_calibration": False,
            "leakage_audit": {
                "future_target_metric_prediction_columns_consumed": [],
                "growth_stage_inputs": ["current row timestamp", "crop-cycle manifest snapshot"],
                "train_validation_frames_consumed_by_test_processor": False,
                "status": "passed",
            },
            "tests": test_records,
        }
        write_json(destination, payload)
        return destination
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _run_default_processor(dataset_root: Path, output_dir_name: str) -> object:
    project_root = Path(__file__).resolve().parents[4]
    script_path = project_root / "offline_dataset_preparation/scripts/03_control_quality.py"
    name = "geas35_step11_3_control_quality"
    spec = importlib.util.spec_from_file_location(name, script_path)
    _require(spec is not None and spec.loader is not None, f"Cannot load QC processor: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.prepare_quality_controlled_splits(
        dataset_root=dataset_root,
        output_dir_name=output_dir_name,
        crops=tuple(OFFICIAL_QC_MODEL_BY_CROP),
        splits=("test",),
        growth_stage_splits=("test",),
    )


def _validate_frozen_applications(root: Path, validation: dict[str, Any]) -> None:
    parent = validation.get("step11_1_freeze_manifest")
    _require(isinstance(parent, dict), "Step 11.2 must reference the Step 11.1 freeze manifest")
    freeze_path = Path(str(parent.get("path", ""))).expanduser()
    if not freeze_path.is_absolute():
        freeze_path = root / freeze_path
    _require(freeze_path.is_file() and _sha256(freeze_path) == parent.get("sha256"),
             "Step 11.1 freeze manifest hash does not match Step 11.2")
    freeze = _read_object(freeze_path, "Step 11.1 freeze manifest")
    _require(freeze.get("selected_models") == OFFICIAL_QC_MODEL_BY_CROP,
             "Frozen QC model mapping changed")
    for crop, model in OFFICIAL_QC_MODEL_BY_CROP.items():
        record = freeze["crops"][crop]["application_json"]
        application_path = Path(str(record.get("path", ""))).expanduser()
        if not application_path.is_absolute():
            application_path = root / application_path
        _require(application_path.is_file() and _sha256(application_path) == record.get("sha256"),
                 f"{crop}: selected application differs from Step 11.1 freeze")
        application = _read_object(application_path, f"{crop} selected application")
        _require(application.get("model_name") == model, f"{crop}: selected model changed")
        _require(application.get("test_used_for_selection") is False,
                 f"{crop}: application must declare test_used_for_selection=false")
        _require(application.get("automatic_best_model_selection") is False,
                 f"{crop}: application must declare automatic_best_model_selection=false")


def _verify_immutable_train_validation(qc_root: Path, validation: dict[str, Any]) -> dict[Path, str]:
    protected: dict[Path, str] = {}
    for crop in OFFICIAL_QC_MODEL_BY_CROP:
        for split in ("train", "validation"):
            path = qc_root / crop / f"{split}.parquet"
            expected = validation["crops"][crop][split]["sha256"]
            actual = _sha256(path)
            _require(actual == expected, f"{crop}/{split}: differs from Step 11.2 immutable hash")
            protected[path] = actual
    return protected


def _validate_stage_manifest(manifest: dict[str, Any]) -> None:
    _require(manifest.get("processed_splits") == ["test"], "Staged QC must process Test only")
    _require(manifest.get("test_processed") is True, "Staged manifest must mark Test processed")
    _require(manifest.get("growth_stage_splits") == ["test"], "Test growth-stage contract was not applied")
    _require(manifest.get("selected_models") == OFFICIAL_QC_MODEL_BY_CROP,
             "Staged selected QC mapping changed")
    for crop, application in manifest.get("crop_applications", {}).items():
        _require(application.get("quality_model") == OFFICIAL_QC_MODEL_BY_CROP.get(crop),
                 f"{crop}: staged QC model mismatch")


def _validate_test_frame(
    frame: pd.DataFrame, *, crop: str, source_record: dict[str, Any],
    qc_root: Path, validation: dict[str, Any],
) -> None:
    _require(not frame.empty, f"{crop}: generated QC Test is empty")
    _require(len(frame) == source_record["row_count"], f"{crop}: QC changed Test row count")
    _require("split" in frame and set(frame["split"].astype(str).unique()) == {"test"},
             f"{crop}: generated split column is not Test-only")
    required = {*REPRESENTATIVE_COLUMNS, *GROWTH_STAGE_COLUMNS}
    train_columns = set(pd.read_parquet(qc_root / crop / "train.parquet").columns)
    validation_columns = set(pd.read_parquet(qc_root / crop / "validation.parquet").columns)
    required |= train_columns & validation_columns
    missing = sorted(required - set(frame.columns))
    _require(not missing, f"{crop}: generated Test misses common QC schema columns: {missing}")
    _require(not _forbidden_columns(frame.columns), f"{crop}: generated Test contains leakage columns")
    for column in REPRESENTATIVE_COLUMNS:
        _require(pd.api.types.is_numeric_dtype(frame[column]), f"{crop}: {column} must be numeric")
    _require(pd.api.types.is_datetime64_any_dtype(frame["reg_date"]), f"{crop}: reg_date must be datetime")
    time = pd.to_datetime(frame["reg_date"], errors="coerce")
    _require(time.notna().all() and time.is_monotonic_increasing, f"{crop}: Test time ordering invalid")
    same_episode = frame["episode_id"].eq(frame["episode_id"].shift(1))
    _require(((time - time.shift(1)).loc[same_episode] == pd.Timedelta(minutes=5)).all(),
             f"{crop}: Test episode contains non-5-minute transition")


def _forbidden_columns(columns: object) -> list[str]:
    return sorted(str(column) for column in columns
                  if str(column).lower().startswith(FORBIDDEN_LEAKAGE_PREFIXES))


def _finite_coverage(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = int(np.isfinite(values).sum())
    return {"finite_rows": finite, "total_rows": len(series),
            "finite_fraction": finite / len(series) if len(series) else 0.0}


def _current_hashes(records: dict[Path, str]) -> dict[Path, str]:
    return {path: _sha256(path) for path in records}


def _file_record(path: Path, root: Path, *, row_count: int) -> dict[str, Any]:
    return {"path": _display(path, root), "sha256": _sha256(path),
            "size_bytes": path.stat().st_size, "row_count": int(row_count)}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _list_of_objects(value: object, label: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and all(isinstance(x, dict) for x in value),
             f"{label} must be an array of objects")
    return list(value)


def _resolved(value: str | Path | None, default: Path) -> Path:
    return default.resolve() if value is None else Path(value).expanduser().resolve()


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QCTestCompletionError(message)


__all__ = ["QCTestCompletionError", "TEST_COMPLETION_VERSION", "complete_missing_qc_test_only"]
