import json
from pathlib import Path

import pandas as pd
import pytest

from geas35.preprocessing import ACTION_COLUMNS, STATE_COLUMNS
from geas35.experiments.transition.qc_contract_freeze import (
    OFFICIAL_QC_MODEL_BY_CROP, OFFICIAL_QC_OBSERVATION_COLUMNS,
    freeze_existing_qc_contract,
)
from geas35.experiments.transition.qc_schema_validation import (
    QCSchemaValidationError, validate_existing_qc_schema_and_provenance,
)


def _frame(crop: str, split: str) -> pd.DataFrame:
    times = pd.date_range("2026-06-01 00:00", periods=3, freq="5min")
    data = {column: [1.0] * 3 for column in (*STATE_COLUMNS, *ACTION_COLUMNS)}
    data.update({
        "reg_date": times, "crop": [crop] * 3, "split": [split] * 3,
        "series_id": [f"{crop}-series"] * 3, "segment_id": ["segment-1"] * 3,
        "episode_date": ["2026-06-01"] * 3, "episode_id": ["episode-1"] * 3,
        "in_temp_representative": [24.0] * 3,
        "in_hum_representative": [70.0] * 3,
        "in_co2_representative": [500.0] * 3,
        "in_temp_representative_source": ["sensor1"] * 3,
        "in_hum_representative_source": ["sensor1"] * 3,
        "in_co2_representative_source": ["sensor1"] * 3,
        "growth_stage_dat": pd.Series([1, 1, 1], dtype="Int64"),
        "growth_stage_order": pd.Series([1, 1, 1], dtype="Int64"),
        "growth_stage_name": ["stage"] * 3,
        "growth_stage_unmatched_flag": [0, 0, 0],
        "rule_outlier_flag": [0, 0, 0], "ai_outlier_flag": [0, 0, 0],
        "invalid_flag": [0, 0, 0], "imputed_flag": [0, 0, 0],
        "quality_unconfirmed_flag": [0, 0, 0],
    })
    for column in STATE_COLUMNS:
        data[f"{column}_missing_flag"] = [0, 0, 0]
    for column in OFFICIAL_QC_OBSERVATION_COLUMNS:
        data[f"{column}_imputed_flag"] = [0, 0, 0]
    for column in ACTION_COLUMNS:
        data[f"{column}_restored_flag"] = [0, 0, 0]
    return pd.DataFrame(data)


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    datasets = project / "offline_dataset_preparation/datasets/03_quality_controlled"
    artifacts = project / "experiments/quality_control_model_selection/artifacts"
    crop_cycle = project / "offline_dataset_preparation/datasets/00_raw/crop_cycle_manifest.csv"
    crop_cycle.parent.mkdir(parents=True)
    crop_cycle.write_text("crop,start\n")
    thresholds = {column: float(i + 1) for i, column in enumerate(OFFICIAL_QC_OBSERVATION_COLUMNS)}
    splits = []
    applications = {}
    for crop, model in OFFICIAL_QC_MODEL_BY_CROP.items():
        model_dir = artifacts / crop / model
        model_dir.mkdir(parents=True)
        (model_dir / "quality_model.pkl").write_bytes(b"test-pickle")
        app = model_dir / "quality_model_application.json"
        app.write_text(json.dumps({"model_name": model, "model_pickle_path": "quality_model.pkl",
            "observation_columns": list(OFFICIAL_QC_OBSERVATION_COLUMNS), "threshold": thresholds}))
        applications[crop] = {"quality_model": model,
            "selected_quality_model_artifact": str(app.resolve()), "thresholds": thresholds,
            "control_log_path": None}
        for split in ("train", "validation"):
            frame = _frame(crop, split)
            path = datasets / crop / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            splits.append({"crop": crop, "split": split, "output_path": str(path.resolve()),
                "output_rows": len(frame), "output_columns": len(frame.columns),
                "missing_cells": 0, "restored_action_cells": 0,
                "rule_outlier_rows": 0, "rule_outlier_cells": 0,
                "ai_outlier_rows": 0, "invalid_rows": 0, "invalid_cells": 0,
                "imputed_rows": 0, "imputed_cells": 0,
                "provisional_imputed_cells": 0, "unconfirmed_rows": 0, "unconfirmed_cells": 0})
    manifest = {"selected_models": OFFICIAL_QC_MODEL_BY_CROP,
        "processed_splits": ["train", "validation"], "test_processed": False,
        "crop_applications": applications, "splits": splits,
        "ai_imputation_columns": list(OFFICIAL_QC_OBSERVATION_COLUMNS),
        "crop_cycle_manifest_path": str(crop_cycle.resolve())}
    (datasets / "quality_controlled_manifest.json").write_text(json.dumps(manifest))
    freeze_existing_qc_contract(project_root=project)
    return project, datasets


def test_step11_2_validates_schema_coverage_provenance_and_immutability(tmp_path: Path) -> None:
    project, datasets = _tree(tmp_path)
    output = validate_existing_qc_schema_and_provenance(project_root=project)
    payload = json.loads(output.read_text())
    record = payload["crops"]["melon"]["train"]
    assert payload["step"] == "11.2" and payload["status"] == "passed"
    assert record["immutable_input"] is True
    assert record["finite_coverage"]["in_co2_representative"]["finite_fraction"] == 1.0
    assert record["temporal_episode_validation"]["exact_step_minutes_within_episode"] == 5
    assert payload["provenance"]["action_restoration_source"]["melon"]["status"] == "legacy_unavailable"


def test_step11_2_fails_on_missing_required_source_column(tmp_path: Path) -> None:
    project, datasets = _tree(tmp_path)
    path = datasets / "cucumber/train.parquet"
    pd.read_parquet(path).drop(columns=["out_light"]).to_parquet(path, index=False)
    freeze_existing_qc_contract(project_root=project)
    with pytest.raises(QCSchemaValidationError, match="missing required source columns"):
        validate_existing_qc_schema_and_provenance(project_root=project)


def test_step11_2_fails_on_non_five_minute_step_inside_episode(tmp_path: Path) -> None:
    project, datasets = _tree(tmp_path)
    path = datasets / "strawberry/validation.parquet"
    frame = pd.read_parquet(path)
    frame.loc[1, "reg_date"] += pd.Timedelta(minutes=1)
    frame.to_parquet(path, index=False)
    freeze_existing_qc_contract(project_root=project)
    with pytest.raises(QCSchemaValidationError, match="non-5-minute"):
        validate_existing_qc_schema_and_provenance(project_root=project)
