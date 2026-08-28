from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geas35.experiments.transition.integrity_handoff import (
    INTEGRITY_VERSION,
    Step116IntegrityError,
    finalize_step11_integrity,
)
from geas35.experiments.transition.qc_contract_freeze import OFFICIAL_QC_MODEL_BY_CROP
from geas35.preprocessing import ACTION_COLUMNS, STATE_COLUMNS
from geas35.rl.datasets import prepare_rl_dataset_splits


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _qc_frame(crop: str, offset: float, rows: int = 6) -> pd.DataFrame:
    data = {column: np.zeros(rows) for column in (*STATE_COLUMNS, *ACTION_COLUMNS)}
    data.update({
        "reg_date": pd.date_range("2026-06-01", periods=rows, freq="5min"),
        "crop": [crop] * rows,
        "series_id": [f"{crop}_series"] * rows,
        "segment_id": [f"{crop}_segment"] * rows,
        "episode_id": [f"{crop}_episode"] * rows,
        "growth_stage_dat": pd.Series([10] * rows, dtype="Int64"),
        "growth_stage_order": pd.Series([1] * rows, dtype="Int64"),
        "in_temp_representative": np.arange(rows) + 20.0 + offset,
        "in_hum_representative": np.arange(rows) + 60.0 + offset,
        "in_co2_representative": np.arange(rows) * 10.0 + 500.0 + offset,
        "out_temp": np.full(rows, 15.0), "out_hum": np.full(rows, 55.0),
        "out_light": np.full(rows, 100.0), "out_light_sum": np.arange(rows) + 1.0,
        "out_windsp": np.full(rows, 1.0), "out_rain": np.zeros(rows),
    })
    return pd.DataFrame(data)


def _fixture(project: Path) -> tuple[Path, Path]:
    datasets = project / "offline_dataset_preparation/datasets"
    qc = datasets / "03_quality_controlled"
    source = datasets / "02_split"
    artifacts = project / "experiments/quality_control_model_selection/artifacts"
    frozen_crops = {}
    step2_crops = {}
    tests = {}
    for crop, model in OFFICIAL_QC_MODEL_BY_CROP.items():
        for split, offset in (("train", 0.0), ("validation", 10.0), ("test", 20.0)):
            frame = _qc_frame(crop, offset)
            qc_path = qc / crop / f"{split}.parquet"
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(qc_path, index=False)
            if split == "test":
                source_path = source / crop / "test.parquet"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(source_path, index=False)
        app_root = artifacts / crop / model
        app_root.mkdir(parents=True, exist_ok=True)
        model_path = app_root / "quality_model.pkl"
        model_path.write_bytes(f"{crop}-model".encode())
        thresholds = {"in_temp": 1.0, "in_hum": 2.0, "in_co2": 3.0}
        app_path = app_root / "quality_model_application.json"
        _write_json(app_path, {"model_name": model, "threshold": thresholds})
        frozen_crops[crop] = {
            "calibrated_thresholds": thresholds,
            "application_json": {"path": str(app_path.relative_to(project)), "sha256": _hash(app_path)},
            "model_pickle": {"path": str(model_path.relative_to(project)), "sha256": _hash(model_path)},
        }
        step2_crops[crop] = {}
        for split in ("train", "validation"):
            path = qc / crop / f"{split}.parquet"
            step2_crops[crop][split] = {"sha256": _hash(path)}
        test_path = qc / crop / "test.parquet"
        source_path = source / crop / "test.parquet"
        tests[crop] = {
            "sha256": _hash(test_path),
            "source_02_split_test": {"sha256": _hash(source_path), "row_count": 6},
        }

    step1_path = qc / "step11_1_qc_contract_freeze_manifest.json"
    _write_json(step1_path, {
        "step": "11.1", "status": "passed", "selected_models": OFFICIAL_QC_MODEL_BY_CROP,
        "crops": frozen_crops,
    })
    step2_path = qc / "step11_2_qc_schema_provenance_manifest.json"
    _write_json(step2_path, {
        "step": "11.2", "status": "passed", "step11_1_freeze_manifest": {
            "path": str(step1_path.relative_to(project)), "sha256": _hash(step1_path)},
        "crops": step2_crops,
    })
    step3_path = qc / "step11_3_qc_test_completion_manifest.json"
    _write_json(step3_path, {
        "step": "11.3", "status": "passed", "selected_models": OFFICIAL_QC_MODEL_BY_CROP,
        "parent_step11_2_manifest": {"path": str(step2_path.relative_to(project)), "sha256": _hash(step2_path)},
        "tests": tests,
    })
    prepare_rl_dataset_splits(dataset_root=datasets, crops=tuple(OFFICIAL_QC_MODEL_BY_CROP))
    config = project / "step11_config.yaml"
    config.write_text("step_minutes: 5\n", encoding="utf-8")
    return datasets, config


def test_step11_6_publishes_verified_hash_chain_and_handoff(tmp_path: Path) -> None:
    datasets, config = _fixture(tmp_path)
    manifest_path, handoff_path = finalize_step11_integrity(
        project_root=tmp_path,
        dataset_root=datasets,
        run_id="test-run",
        created_at_utc="2026-08-10T00:00:00Z",
        preprocessing_config_paths=(config,),
    )
    manifest = json.loads(manifest_path.read_text())
    handoff = json.loads(handoff_path.read_text())
    assert manifest["schema_version"] == INTEGRITY_VERSION
    assert manifest["all_crop_integrity_gate"] == "passed"
    assert manifest["verified_crop_count"] == 3
    assert manifest["verified_split_count"] == 9
    assert manifest["split_generation_policy"] == {
        "train": "reused_existing", "validation": "reused_existing", "test": "generated_test_only"
    }
    assert len(manifest["schemas"]["state"]["sha256"]) == 64
    assert len(manifest["schemas"]["action"]["sha256"]) == 64
    assert len(manifest["schemas"]["target"]["sha256"]) == 64
    assert len(manifest["crops"]["cucumber"]["threshold"]["sha256"]) == 64
    assert handoff["status"] == "ready"
    assert handoff["immutable_rl_manifest"]["sha256"] == _hash(manifest_path)
    assert "candidate" not in json.dumps(manifest).lower()


def test_step11_6_fails_closed_on_parent_hash_tamper(tmp_path: Path) -> None:
    datasets, config = _fixture(tmp_path)
    step1 = datasets / "03_quality_controlled/step11_1_qc_contract_freeze_manifest.json"
    step1.write_text(step1.read_text() + " ", encoding="utf-8")
    with pytest.raises(Step116IntegrityError, match="parent hash differs"):
        finalize_step11_integrity(
            project_root=tmp_path, dataset_root=datasets,
            preprocessing_config_paths=(config,),
        )
    assert not (datasets / "5_rl_dataset/step12_handoff.json").exists()


def test_step11_6_fails_closed_on_rl_row_count_mismatch(tmp_path: Path) -> None:
    datasets, config = _fixture(tmp_path)
    rl_manifest_path = datasets / "5_rl_dataset/rl_dataset_manifest.json"
    manifest = json.loads(rl_manifest_path.read_text())
    manifest["splits"][0]["output_rows"] += 1
    _write_json(rl_manifest_path, manifest)
    with pytest.raises(Step116IntegrityError, match="row count differs"):
        finalize_step11_integrity(
            project_root=tmp_path, dataset_root=datasets,
            preprocessing_config_paths=(config,),
        )
