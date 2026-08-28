import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from geas35.experiments.transition.qc_contract_freeze import (
    OFFICIAL_QC_MODEL_BY_CROP,
    OFFICIAL_QC_OBSERVATION_COLUMNS,
    QCContractFreezeError,
    freeze_existing_qc_contract,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    datasets = project / "offline_dataset_preparation/datasets/03_quality_controlled"
    artifacts = project / "experiments/quality_control_model_selection/artifacts"
    splits = []
    crop_applications = {}
    thresholds = {column: float(index + 1) for index, column in enumerate(OFFICIAL_QC_OBSERVATION_COLUMNS)}

    for crop, model in OFFICIAL_QC_MODEL_BY_CROP.items():
        model_dir = artifacts / crop / model
        model_dir.mkdir(parents=True)
        pickle_path = model_dir / "quality_model.pkl"
        pickle_path.write_bytes(b"non-empty-test-pickle")
        application_path = model_dir / "quality_model_application.json"
        application_path.write_text(
            json.dumps(
                {
                    "model_name": model,
                    "model_pickle_path": "quality_model.pkl",
                    "observation_columns": list(OFFICIAL_QC_OBSERVATION_COLUMNS),
                    "threshold": thresholds,
                }
            )
        )
        crop_applications[crop] = {
            "quality_model": model,
            "selected_quality_model_artifact": str(application_path.resolve()),
            "thresholds": thresholds,
        }
        for split, rows in (("train", 3), ("validation", 2)):
            parquet_path = datasets / crop / f"{split}.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"value": range(rows)}).to_parquet(parquet_path, index=False)
            splits.append(
                {
                    "crop": crop,
                    "split": split,
                    "output_path": str(parquet_path.resolve()),
                    "output_rows": rows,
                }
            )

    manifest_path = datasets / "quality_controlled_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selected_models": OFFICIAL_QC_MODEL_BY_CROP,
                "processed_splits": ["train", "validation"],
                "test_processed": False,
                "crop_applications": crop_applications,
                "splits": splits,
            }
        )
    )
    return project, datasets, artifacts


def test_step11_1_freezes_valid_inputs_without_modifying_them(tmp_path: Path) -> None:
    project, datasets, artifacts = _fixture_tree(tmp_path)
    protected = [
        path for path in project.rglob("*")
        if path.is_file() and path.name != "step11_1_qc_contract_freeze_manifest.json"
    ]
    before = {path: _sha256(path) for path in protected}

    output = freeze_existing_qc_contract(project_root=project)

    assert output == datasets / "step11_1_qc_contract_freeze_manifest.json"
    payload = json.loads(output.read_text())
    assert payload["step"] == "11.1"
    assert payload["status"] == "passed"
    assert payload["selected_models"] == OFFICIAL_QC_MODEL_BY_CROP
    assert payload["qc_observation_columns"] == list(OFFICIAL_QC_OBSERVATION_COLUMNS)
    assert payload["crops"]["strawberry"]["datasets"]["train"]["row_count"] == 3
    assert len(payload["crops"]["cucumber"]["model_pickle"]["sha256"]) == 64
    assert {path: _sha256(path) for path in protected} == before


def test_step11_1_fails_before_writing_when_threshold_contract_differs(tmp_path: Path) -> None:
    project, datasets, artifacts = _fixture_tree(tmp_path)
    application_path = artifacts / "melon/modern_tcn/quality_model_application.json"
    application = json.loads(application_path.read_text())
    application["threshold"].pop("in_co2_2")
    application_path.write_text(json.dumps(application))
    output = datasets / "step11_1_qc_contract_freeze_manifest.json"

    with pytest.raises(QCContractFreezeError, match="threshold keys/order"):
        freeze_existing_qc_contract(project_root=project)

    assert not output.exists()


def test_step11_1_fails_on_manifest_parquet_row_count_mismatch(tmp_path: Path) -> None:
    project, datasets, _ = _fixture_tree(tmp_path)
    manifest_path = datasets / "quality_controlled_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"][0]["output_rows"] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(QCContractFreezeError, match="row count mismatch"):
        freeze_existing_qc_contract(project_root=project)
