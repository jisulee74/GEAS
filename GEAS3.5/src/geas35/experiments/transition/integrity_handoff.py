"""Step 11.6 hash chain, all-crop integrity gate, and Step 12 handoff."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from geas35.io_utils import write_json
from geas35.experiments.transition.qc_contract_freeze import OFFICIAL_QC_MODEL_BY_CROP
from geas35.rl.datasets import DEFAULT_SPLITS
from geas35.rl.mdp_v1 import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_TRANSITION_TARGET_COLUMNS,
    MdpV1Config,
)

INTEGRITY_VERSION = "geas35.transition.step11.6.v1"
RL_DATASET_VERSION = "geas35.rl_dataset.step11.5.v1"
INTEGRITY_MANIFEST_NAME = "step11_6_integrity_manifest.json"
STEP12_HANDOFF_NAME = "step12_handoff.json"
PREPROCESSING_CONFIG_PATHS = (
    "offline_dataset_preparation/configs/default.yaml",
    "offline_dataset_preparation/configs/crops/growth_stage_rules.csv",
    "offline_dataset_preparation/configs/qc/smartfarm_korea_domain_ranges.csv",
)


class Step116IntegrityError(ValueError):
    """Raised when any Step 11 provenance or dataset integrity check fails."""


def finalize_step11_integrity(
    *,
    project_root: str | Path,
    dataset_root: str | Path | None = None,
    rl_manifest_path: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    handoff_path: str | Path | None = None,
    run_id: str | None = None,
    created_at_utc: str | None = None,
    preprocessing_config_paths: Sequence[str | Path] = PREPROCESSING_CONFIG_PATHS,
) -> tuple[Path, Path]:
    """Validate the complete Step 11 chain and publish an immutable handoff."""

    root = Path(project_root).expanduser().resolve()
    datasets = _resolved(dataset_root, root / "offline_dataset_preparation/datasets")
    source_root = datasets / "02_split"
    qc_root = datasets / "03_quality_controlled"
    rl_root = datasets / "5_rl_dataset"
    rl_path = _resolved(rl_manifest_path, rl_root / "rl_dataset_manifest.json")
    destination = _resolved(output_manifest_path, rl_root / INTEGRITY_MANIFEST_NAME)
    handoff_destination = _resolved(handoff_path, rl_root / STEP12_HANDOFF_NAME)

    step11_1_path = qc_root / "step11_1_qc_contract_freeze_manifest.json"
    step11_2_path = qc_root / "step11_2_qc_schema_provenance_manifest.json"
    step11_3_path = qc_root / "step11_3_qc_test_completion_manifest.json"
    step11_1 = _read_object(step11_1_path, "Step 11.1 manifest")
    step11_2 = _read_object(step11_2_path, "Step 11.2 manifest")
    step11_3 = _read_object(step11_3_path, "Step 11.3 manifest")
    rl_manifest = _read_object(rl_path, "Step 11.5 RL manifest")

    _validate_parent_chain(
        root, step11_1_path, step11_1, step11_2_path, step11_2,
        step11_3_path, step11_3, rl_manifest,
    )
    split_entries = _index_rl_splits(rl_manifest)
    state_columns = _common_state_schema(split_entries)
    output_schema = _string_list(rl_manifest.get("output_schema"), "RL output_schema")
    target_columns = _string_list(
        rl_manifest.get("transition_target_columns"), "RL transition_target_columns"
    )
    _require(target_columns == list(MDP_V1_TRANSITION_TARGET_COLUMNS),
             "Step 11.5 target schema is not the official three-target schema")
    _require(all(column in output_schema for column in target_columns),
             "RL output schema is missing transition targets")
    _require(all(column in output_schema for column in MDP_V1_ACTION_COLUMNS),
             "RL output schema is missing official Action columns")

    config_records = _configuration_records(root, preprocessing_config_paths)
    schema_records = {
        "state": _schema_record(state_columns),
        "action": _schema_record(MDP_V1_ACTION_COLUMNS),
        "target": _schema_record(target_columns),
        "output": _schema_record(output_schema),
    }
    crop_records: dict[str, Any] = {}
    for crop, model in OFFICIAL_QC_MODEL_BY_CROP.items():
        frozen_crop = _mapping(_mapping(step11_1.get("crops"), "Step 11.1 crops").get(crop), f"Step 11.1 {crop}")
        application_record = _verified_record(root, frozen_crop.get("application_json"), f"{crop} application JSON")
        pickle_record = _verified_record(root, frozen_crop.get("model_pickle"), f"{crop} model pickle")
        application = _read_object(_path_from_record(root, application_record), f"{crop} application JSON")
        _require(application.get("model_name") == model, f"{crop}: selected QC model differs")
        thresholds = _mapping(application.get("threshold"), f"{crop} embedded threshold")
        _require(thresholds == frozen_crop.get("calibrated_thresholds"),
                 f"{crop}: embedded thresholds differ from Step 11.1 freeze")

        scaler_path = rl_root / crop / "observation_scaler.json"
        scaler = _read_object(scaler_path, f"{crop} scaler")
        scaler_record = _validate_scaler(
            scaler_path, scaler, crop=crop, state_columns=state_columns,
            train_entry=split_entries[(crop, "train")], qc_root=qc_root,
        )
        datasets_record: dict[str, Any] = {}
        for split in DEFAULT_SPLITS:
            entry = split_entries[(crop, split)]
            qc_path = qc_root / crop / f"{split}.parquet"
            rl_dataset_path = rl_root / crop / f"{split}.parquet"
            qc_expected = (
                _mapping(_mapping(_mapping(step11_2.get("crops"), "Step 11.2 crops").get(crop), crop).get(split), f"{crop}/{split}")
                if split != "test"
                else _mapping(_mapping(step11_3.get("tests"), "Step 11.3 tests").get(crop), f"{crop}/test")
            )
            qc_record = _validate_parquet(
                qc_path, expected_hash=str(qc_expected.get("sha256", "")),
                expected_rows=int(entry.get("input_rows", -1)), label=f"{crop}/{split} QC",
            )
            rl_record = _validate_parquet(
                rl_dataset_path, expected_hash=None,
                expected_rows=int(entry.get("output_rows", -1)), label=f"{crop}/{split} RL",
                expected_columns=output_schema,
            )
            _require(rl_record["row_count"] > 0, f"{crop}/{split}: RL dataset is empty")
            split_record: dict[str, Any] = {
                "generation_status": "generated_test_only" if split == "test" else "reused_existing",
                "quality_controlled": qc_record,
                "rl_dataset": rl_record,
            }
            if split == "test":
                source_expected = _mapping(qc_expected.get("source_02_split_test"), f"{crop} 02_split Test")
                split_record["source_02_split_test"] = _validate_parquet(
                    source_root / crop / "test.parquet",
                    expected_hash=str(source_expected.get("sha256", "")),
                    expected_rows=int(source_expected.get("row_count", -1)),
                    label=f"{crop} 02_split/Test",
                )
            datasets_record[split] = split_record

        crop_records[crop] = {
            "selected_qc_model": model,
            "application_json": application_record,
            "model_pickle": pickle_record,
            "threshold": {
                "source": f"{application_record['path']}#/threshold",
                "sha256": _canonical_sha256(thresholds),
                "values": thresholds,
            },
            "scaler": scaler_record,
            "datasets": datasets_record,
            "status": "passed",
        }

    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    identity = run_id or f"step11.6-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    payload = {
        "schema_version": INTEGRITY_VERSION,
        "step": "11.6",
        "status": "passed",
        "all_crop_integrity_gate": "passed",
        "run_identity": identity,
        "created_at_utc": timestamp,
        "code_version": _code_version(root),
        "parent_provenance": {
            "step11_1_freeze_manifest": _file_record(step11_1_path, root),
            "step11_2_schema_manifest": _file_record(step11_2_path, root),
            "step11_3_test_manifest": _file_record(step11_3_path, root),
            "step11_5_rl_manifest": _file_record(rl_path, root),
        },
        "preprocessing_configuration": {
            "mdp_v1_config": asdict(MdpV1Config()),
            "mdp_v1_config_sha256": _canonical_sha256(asdict(MdpV1Config())),
            "files": config_records,
        },
        "schemas": schema_records,
        "split_generation_policy": {
            "train": "reused_existing", "validation": "reused_existing",
            "test": "generated_test_only",
        },
        "crops": crop_records,
        "verified_crop_count": len(crop_records),
        "verified_split_count": len(split_entries),
    }
    write_json(destination, payload, convert=True)
    immutable_record = _file_record(destination, root)
    handoff = {
        "schema_version": "geas35.transition.step12_handoff.v1",
        "source_step": "11.6",
        "status": "ready",
        "run_identity": identity,
        "immutable_rl_manifest": immutable_record,
        "integrity_gate": "passed",
    }
    write_json(handoff_destination, handoff)
    _require(_sha256(destination) == immutable_record["sha256"],
             "Step 11.6 manifest changed while publishing handoff")
    return destination, handoff_destination


def _validate_parent_chain(
    root: Path, step11_1_path: Path, step11_1: dict[str, Any],
    step11_2_path: Path, step11_2: dict[str, Any], step11_3_path: Path,
    step11_3: dict[str, Any], rl_manifest: dict[str, Any],
) -> None:
    _require(step11_1.get("status") == "passed" and step11_1.get("step") == "11.1",
             "Step 11.1 manifest is not passed")
    _require(step11_2.get("status") == "passed" and step11_2.get("step") == "11.2",
             "Step 11.2 manifest is not passed")
    _require(step11_3.get("status") == "passed" and step11_3.get("step") == "11.3",
             "Step 11.3 manifest is not passed")
    _verify_parent(root, step11_2.get("step11_1_freeze_manifest"), step11_1_path, "Step 11.2 -> 11.1")
    _verify_parent(root, step11_3.get("parent_step11_2_manifest"), step11_2_path, "Step 11.3 -> 11.2")
    _require(rl_manifest.get("schema_version") == RL_DATASET_VERSION,
             "Step 11.5 RL manifest schema version is invalid")
    _require(rl_manifest.get("status") == "passed", "Step 11.5 RL manifest is not passed")
    _require(step11_1.get("selected_models") == OFFICIAL_QC_MODEL_BY_CROP,
             "Step 11.1 selected model mapping differs from the official mapping")
    _require(step11_3.get("selected_models") == OFFICIAL_QC_MODEL_BY_CROP,
             "Step 11.3 selected model mapping differs from the official mapping")


def _verify_parent(root: Path, record: object, expected_path: Path, label: str) -> None:
    value = _mapping(record, label)
    actual_path = _path_from_record(root, value)
    _require(actual_path == expected_path.resolve(), f"{label}: parent path differs")
    _require(_sha256(actual_path) == value.get("sha256"), f"{label}: parent hash differs")


def _index_rl_splits(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    values = manifest.get("splits")
    _require(isinstance(values, list), "Step 11.5 splits must be an array")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in values:
        item = _mapping(raw, "Step 11.5 split")
        key = (str(item.get("crop")), str(item.get("split")))
        _require(key not in indexed, f"Duplicate Step 11.5 split: {key}")
        indexed[key] = item
    expected = {(crop, split) for crop in OFFICIAL_QC_MODEL_BY_CROP for split in DEFAULT_SPLITS}
    _require(set(indexed) == expected, "Step 11.5 must contain exactly three crops x three splits")
    return indexed


def _common_state_schema(entries: Mapping[tuple[str, str], dict[str, Any]]) -> list[str]:
    schemas = {
        tuple(_string_list(entry.get("observation_columns"), "observation_columns"))
        for entry in entries.values()
    }
    _require(len(schemas) == 1, "Canonical State schema differs across crop/split")
    state = list(next(iter(schemas)))
    _require(bool(state), "Canonical State schema is empty")
    return state


def _validate_scaler(
    path: Path, scaler: dict[str, Any], *, crop: str, state_columns: Sequence[str],
    train_entry: dict[str, Any], qc_root: Path,
) -> dict[str, Any]:
    _require(scaler.get("crop") == crop and scaler.get("fit_split") == "train",
             f"{crop}: scaler was not fitted on Train")
    columns = _string_list(scaler.get("columns"), f"{crop} scaler columns")
    _require(set(columns).issubset(state_columns), f"{crop}: scaler contains non-State columns")
    mean = _mapping(scaler.get("mean"), f"{crop} scaler mean")
    scale = _mapping(scaler.get("scale"), f"{crop} scaler scale")
    _require(set(mean) == set(columns) == set(scale), f"{crop}: scaler keys differ")
    _require(all(float(scale[column]) > 0 for column in columns), f"{crop}: scaler has non-positive scale")
    _require(int(scaler.get("fit_valid_transition_rows", -1)) == int(train_entry.get("output_rows", -2)),
             f"{crop}: scaler fit row count differs from RL Train rows")
    raw_train = _mapping(scaler.get("raw_train"), f"{crop} scaler raw_train")
    train_path = qc_root / crop / "train.parquet"
    _require(Path(str(raw_train.get("path", ""))).expanduser().resolve() == train_path.resolve(),
             f"{crop}: scaler raw Train path differs")
    _require(raw_train.get("sha256") == _sha256(train_path), f"{crop}: scaler raw Train hash differs")
    return {**_file_record(path, path.parents[4]), "fit_split": "train", "columns": columns}


def _validate_parquet(
    path: Path, *, expected_hash: str | None, expected_rows: int, label: str,
    expected_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    actual_hash = _sha256(path)
    if expected_hash is not None:
        _require(actual_hash == expected_hash, f"{label} SHA-256 differs from manifest")
    frame = pd.read_parquet(path)
    _require(len(frame) == expected_rows, f"{label} row count differs: manifest={expected_rows}, actual={len(frame)}")
    if expected_columns is not None:
        _require(list(frame.columns) == list(expected_columns), f"{label} schema differs from manifest")
    return {
        "path": str(path.resolve()), "sha256": actual_hash,
        "size_bytes": path.stat().st_size, "row_count": len(frame),
        "column_count": len(frame.columns),
    }


def _configuration_records(root: Path, paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records = []
    for value in paths:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        _require(path.is_file(), f"Preprocessing config is missing: {path}")
        records.append(_file_record(path, root))
    return records


def _verified_record(root: Path, record: object, label: str) -> dict[str, Any]:
    value = _mapping(record, label)
    path = _path_from_record(root, value)
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(_sha256(path) == value.get("sha256"), f"{label} differs from frozen SHA-256")
    return _file_record(path, root)


def _path_from_record(root: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _schema_record(columns: Sequence[str]) -> dict[str, Any]:
    values = list(columns)
    return {"columns": values, "column_count": len(values), "sha256": _canonical_sha256(values)}


def _code_version(root: Path) -> dict[str, Any]:
    implementation = Path(__file__).resolve()
    result: dict[str, Any] = {"implementation_sha256": _sha256(implementation)}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        result.update({"git_commit": commit, "git_dirty": dirty})
    except (OSError, subprocess.SubprocessError):
        result.update({"git_commit": None, "git_dirty": None})
    return result


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": _display(path, root), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, label)


def _mapping(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return dict(value)


def _string_list(value: object, label: str) -> list[str]:
    _require(isinstance(value, list) and all(isinstance(item, str) for item in value),
             f"{label} must be a string array")
    return list(value)


def _resolved(value: str | Path | None, default: Path) -> Path:
    return default.resolve() if value is None else Path(value).expanduser().resolve()


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
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
        raise Step116IntegrityError(message)


__all__ = [
    "INTEGRITY_MANIFEST_NAME", "INTEGRITY_VERSION", "STEP12_HANDOFF_NAME",
    "Step116IntegrityError", "finalize_step11_integrity",
]
