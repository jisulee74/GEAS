"""Step 11.1: validate and freeze existing QC datasets and selections."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from geas35.io_utils import write_json


OFFICIAL_QC_MODEL_BY_CROP = {
    "strawberry": "modern_tcn",
    "melon": "modern_tcn",
    "cucumber": "patch_tst",
}
OFFICIAL_QC_OBSERVATION_COLUMNS = (
    "in_medium_temp1",
    "in_temp",
    "in_temp2",
    "in_hum",
    "in_hum2",
    "in_medium_hum1",
    "in_co2",
    "in_co2_2",
)
REQUIRED_QC_SPLITS = ("train", "validation")
FREEZE_MANIFEST_SCHEMA_VERSION = "geas35.transition.step11.1.v1"


class QCContractFreezeError(ValueError):
    """Raised when an existing QC dataset or selected artifact is inconsistent."""


def freeze_existing_qc_contract(
    *,
    project_root: str | Path,
    dataset_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    output_path: str | Path | None = None,
    selected_models: Mapping[str, str] = OFFICIAL_QC_MODEL_BY_CROP,
) -> Path:
    """Validate Step 11.1 inputs and write a deterministic freeze manifest.

    Existing train/validation parquet files and QC artifacts are read-only. The
    output is written only after every validation has passed.
    """

    root = Path(project_root).expanduser().resolve()
    datasets = _resolve_or_default(
        dataset_root,
        root / "offline_dataset_preparation" / "datasets" / "03_quality_controlled",
    )
    artifacts = _resolve_or_default(
        artifact_root,
        root / "experiments" / "quality_control_model_selection" / "artifacts",
    )
    destination = _resolve_or_default(
        output_path,
        datasets / "step11_1_qc_contract_freeze_manifest.json",
    )
    manifest_path = datasets / "quality_controlled_manifest.json"
    manifest = _read_object(manifest_path, label="QC manifest")

    expected_mapping = {str(crop): str(model) for crop, model in selected_models.items()}
    actual_mapping = _string_mapping(manifest.get("selected_models"), "manifest.selected_models")
    _require(
        actual_mapping == expected_mapping,
        f"QC model mapping mismatch: expected {expected_mapping}, found {actual_mapping}",
    )
    _require(
        tuple(manifest.get("processed_splits", ())) == REQUIRED_QC_SPLITS,
        "QC manifest must contain processed_splits=['train', 'validation'] in that order",
    )
    _require(manifest.get("test_processed") is False, "Step 11.1 requires test_processed=false")

    split_entries = _index_split_entries(manifest.get("splits"))
    crop_records: dict[str, Any] = {}
    for crop, model_name in expected_mapping.items():
        crop_application = _mapping(
            _mapping(manifest.get("crop_applications"), "manifest.crop_applications").get(crop),
            f"manifest.crop_applications.{crop}",
        )
        _require(
            crop_application.get("quality_model") == model_name,
            f"{crop}: crop application model does not match selected model {model_name!r}",
        )

        application_path = artifacts / crop / model_name / "quality_model_application.json"
        application = _read_object(application_path, label=f"{crop} application JSON")
        configured_application = Path(str(crop_application.get("selected_quality_model_artifact", ""))).expanduser()
        if not configured_application.is_absolute():
            configured_application = root / configured_application
        _require(
            configured_application.resolve() == application_path.resolve(),
            f"{crop}: manifest application path does not resolve to selected artifact",
        )
        _require(
            str(application.get("model_name", "")) == model_name,
            f"{crop}: application model_name mismatch",
        )
        observation_columns = tuple(str(value) for value in application.get("observation_columns", ()))
        _require(
            observation_columns == OFFICIAL_QC_OBSERVATION_COLUMNS,
            f"{crop}: expected exactly 8 ordered QC observations {OFFICIAL_QC_OBSERVATION_COLUMNS}, "
            f"found {observation_columns}",
        )
        thresholds = _finite_thresholds(application.get("threshold"), crop=crop)
        _require(
            tuple(thresholds) == OFFICIAL_QC_OBSERVATION_COLUMNS,
            f"{crop}: threshold keys/order must match the 8 QC observations",
        )
        manifest_thresholds = _finite_thresholds(crop_application.get("thresholds"), crop=crop)
        _require(
            manifest_thresholds == thresholds,
            f"{crop}: manifest thresholds differ from selected application thresholds",
        )

        pickle_value = application.get("model_pickle_path")
        _require(isinstance(pickle_value, str) and bool(pickle_value.strip()), f"{crop}: model_pickle_path is missing")
        pickle_path = Path(pickle_value).expanduser()
        if not pickle_path.is_absolute():
            pickle_path = application_path.parent / pickle_path
        _require(pickle_path.is_file(), f"{crop}: selected model pickle is missing: {pickle_path}")
        _require(pickle_path.stat().st_size > 0, f"{crop}: selected model pickle is empty: {pickle_path}")

        datasets_by_split: dict[str, Any] = {}
        for split in REQUIRED_QC_SPLITS:
            key = (crop, split)
            _require(key in split_entries, f"{crop}/{split}: split entry is missing from QC manifest")
            entry = split_entries[key]
            parquet_path = datasets / crop / f"{split}.parquet"
            configured_output = Path(str(entry.get("output_path", ""))).expanduser()
            if not configured_output.is_absolute():
                configured_output = root / configured_output
            _require(
                configured_output.resolve() == parquet_path.resolve(),
                f"{crop}/{split}: manifest output_path does not match canonical QC dataset path",
            )
            actual_rows = _parquet_row_count(parquet_path)
            expected_rows = _positive_int(entry.get("output_rows"), f"{crop}/{split}.output_rows")
            _require(
                actual_rows == expected_rows,
                f"{crop}/{split}: row count mismatch; manifest={expected_rows}, parquet={actual_rows}",
            )
            datasets_by_split[split] = _file_record(parquet_path, root, row_count=actual_rows)

        crop_records[crop] = {
            "selected_model": model_name,
            "observation_columns": list(observation_columns),
            "calibrated_thresholds": thresholds,
            "application_json": _file_record(application_path, root),
            "model_pickle": _file_record(pickle_path, root),
            "datasets": datasets_by_split,
        }

    payload = {
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "step": "11.1",
        "status": "passed",
        "immutable_input_policy": "validate_and_hash_only_do_not_modify_train_validation",
        "selected_models": expected_mapping,
        "required_splits": list(REQUIRED_QC_SPLITS),
        "qc_observation_columns": list(OFFICIAL_QC_OBSERVATION_COLUMNS),
        "qc_manifest": _file_record(manifest_path, root),
        "crops": crop_records,
    }
    write_json(destination, payload)
    return destination


def _resolve_or_default(value: str | Path | None, default: Path) -> Path:
    return default.resolve() if value is None else Path(value).expanduser().resolve()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QCContractFreezeError(f"Cannot read {label}: {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _mapping(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return dict(value)


def _string_mapping(value: object, label: str) -> dict[str, str]:
    return {str(key): str(item) for key, item in _mapping(value, label).items()}


def _index_split_entries(value: object) -> dict[tuple[str, str], dict[str, Any]]:
    _require(isinstance(value, list), "manifest.splits must be an array")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in value:
        entry = _mapping(raw, "manifest.splits item")
        key = (str(entry.get("crop", "")), str(entry.get("split", "")))
        _require(key not in indexed, f"Duplicate QC split manifest entry: {key}")
        indexed[key] = entry
    return indexed


def _finite_thresholds(value: object, *, crop: str) -> dict[str, float]:
    raw = _mapping(value, f"{crop}.thresholds")
    result: dict[str, float] = {}
    for key, item in raw.items():
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise QCContractFreezeError(f"{crop}: threshold {key!r} is not numeric") from exc
        _require(math.isfinite(number), f"{crop}: threshold {key!r} is not finite")
        result[str(key)] = number
    return result


def _positive_int(value: object, label: str) -> int:
    _require(not isinstance(value, bool), f"{label} must be a non-negative integer")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise QCContractFreezeError(f"{label} must be a non-negative integer") from exc
    _require(number >= 0 and number == value, f"{label} must be a non-negative integer")
    return number


def _parquet_row_count(path: Path) -> int:
    _require(path.is_file(), f"QC parquet is missing: {path}")
    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(path).metadata
        _require(metadata is not None, f"Parquet metadata is unavailable: {path}")
        return int(metadata.num_rows)
    except ImportError:
        import pandas as pd

        return int(len(pd.read_parquet(path)))
    except Exception as exc:
        if isinstance(exc, QCContractFreezeError):
            raise
        raise QCContractFreezeError(f"Cannot inspect QC parquet {path}: {exc}") from exc


def _file_record(path: Path, project_root: Path, *, row_count: int | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"Freeze input file is missing: {resolved}")
    try:
        display_path = str(resolved.relative_to(project_root))
    except ValueError:
        display_path = str(resolved)
    record: dict[str, Any] = {
        "path": display_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }
    if row_count is not None:
        record["row_count"] = int(row_count)
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QCContractFreezeError(message)


__all__ = [
    "FREEZE_MANIFEST_SCHEMA_VERSION",
    "OFFICIAL_QC_MODEL_BY_CROP",
    "OFFICIAL_QC_OBSERVATION_COLUMNS",
    "QCContractFreezeError",
    "REQUIRED_QC_SPLITS",
    "freeze_existing_qc_contract",
]
