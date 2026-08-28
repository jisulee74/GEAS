import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from geas35.experiments.transition.qc_contract_freeze import OFFICIAL_QC_MODEL_BY_CROP
from geas35.experiments.transition.qc_schema_validation import SCHEMA_VALIDATION_VERSION
from geas35.experiments.transition.qc_test_completion import (
    QCTestCompletionError, complete_missing_qc_test_only,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_frame(crop: str, split: str) -> pd.DataFrame:
    return pd.DataFrame({
        "reg_date": pd.date_range("2026-06-01", periods=3, freq="5min"),
        "crop": [crop] * 3, "split": [split] * 3,
        "series_id": [f"{crop}-series"] * 3,
        "segment_id": ["segment"] * 3, "episode_id": ["episode"] * 3,
        "in_temp": [24.0] * 3, "in_hum": [70.0] * 3,
        "in_co2": [500.0] * 3, "out_light": [100.0] * 3,
    })


def _enriched(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["in_temp_representative"] = out["in_temp"]
    out["in_hum_representative"] = out["in_hum"]
    out["in_co2_representative"] = out["in_co2"]
    out["growth_stage_dat"] = pd.Series([1] * len(out), dtype="Int64")
    out["growth_stage_order"] = pd.Series([1] * len(out), dtype="Int64")
    out["growth_stage_name"] = "stage"
    out["growth_stage_unmatched_flag"] = 0
    return out


def _tree(tmp_path: Path) -> tuple[Path, Path, dict[Path, str]]:
    project = tmp_path / "project"
    datasets = project / "offline_dataset_preparation/datasets"
    qc = datasets / "03_quality_controlled"
    records = {}
    protected = {}
    for crop in OFFICIAL_QC_MODEL_BY_CROP:
        records[crop] = {}
        for split in ("train", "validation"):
            path = qc / crop / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            _enriched(_base_frame(crop, split)).to_parquet(path, index=False)
            digest = _hash(path)
            protected[path] = digest
            records[crop][split] = {"sha256": digest}
        source = datasets / "02_split" / crop / "test.parquet"
        source.parent.mkdir(parents=True, exist_ok=True)
        _base_frame(crop, "test").to_parquet(source, index=False)
    freeze_crops = {}
    for crop, model in OFFICIAL_QC_MODEL_BY_CROP.items():
        application = project / f"artifacts/{crop}/quality_model_application.json"
        application.parent.mkdir(parents=True, exist_ok=True)
        application.write_text(json.dumps({"model_name": model, "test_used_for_selection": False,
            "automatic_best_model_selection": False}))
        freeze_crops[crop] = {"application_json": {
            "path": str(application.relative_to(project)), "sha256": _hash(application)}}
    freeze_path = qc / "step11_1_qc_contract_freeze_manifest.json"
    freeze_path.write_text(json.dumps({"selected_models": OFFICIAL_QC_MODEL_BY_CROP,
        "crops": freeze_crops}))
    manifest = {"schema_version": SCHEMA_VALIDATION_VERSION, "status": "passed", "crops": records,
        "step11_1_freeze_manifest": {"path": str(freeze_path.relative_to(project)),
            "sha256": _hash(freeze_path)}}
    (qc / "step11_2_qc_schema_provenance_manifest.json").write_text(json.dumps(manifest))
    return project, datasets, protected


def _fake_processor(dataset_root: Path, stage_name: str) -> None:
    stage = dataset_root / stage_name
    entries = []
    applications = {}
    for crop, model in OFFICIAL_QC_MODEL_BY_CROP.items():
        source = dataset_root / "02_split" / crop / "test.parquet"
        output = stage / crop / "test.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = _enriched(pd.read_parquet(source))
        frame.to_parquet(output, index=False)
        entries.append({"crop": crop, "split": "test", "output_rows": len(frame)})
        applications[crop] = {"quality_model": model}
    (stage / "quality_controlled_manifest.json").write_text(json.dumps({
        "processed_splits": ["test"], "test_processed": True,
        "growth_stage_splits": ["test"], "selected_models": OFFICIAL_QC_MODEL_BY_CROP,
        "crop_applications": applications, "splits": entries,
    }))


def test_step11_3_generates_test_only_and_preserves_train_validation(tmp_path: Path) -> None:
    project, datasets, protected = _tree(tmp_path)
    output = complete_missing_qc_test_only(
        project_root=project, dataset_root=datasets, processor=_fake_processor,
    )
    payload = json.loads(output.read_text())
    assert payload["step"] == "11.3" and payload["status"] == "passed"
    assert payload["test_used_for_qc_selection_or_calibration"] is False
    assert payload["leakage_audit"]["future_target_metric_prediction_columns_consumed"] == []
    assert all((datasets / "03_quality_controlled" / crop / "test.parquet").is_file()
               for crop in OFFICIAL_QC_MODEL_BY_CROP)
    assert {path: _hash(path) for path in protected} == protected


def test_step11_3_rejects_leakage_prone_test_source(tmp_path: Path) -> None:
    project, datasets, _ = _tree(tmp_path)
    path = datasets / "02_split/strawberry/test.parquet"
    frame = pd.read_parquet(path)
    frame["target_next_temp"] = 1.0
    frame.to_parquet(path, index=False)
    with pytest.raises(QCTestCompletionError, match="leakage-prone"):
        complete_missing_qc_test_only(
            project_root=project, dataset_root=datasets, processor=_fake_processor,
        )


def test_step11_3_refuses_to_overwrite_existing_qc_test(tmp_path: Path) -> None:
    project, datasets, _ = _tree(tmp_path)
    existing = datasets / "03_quality_controlled/melon/test.parquet"
    _base_frame("melon", "test").to_parquet(existing, index=False)
    with pytest.raises(QCTestCompletionError, match="already exists"):
        complete_missing_qc_test_only(
            project_root=project, dataset_root=datasets, processor=_fake_processor,
        )
