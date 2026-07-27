from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.quality.cli import run_from_config
from synthetic_experiment_fixtures import (
    median_smoke_registry,
    write_quality_smoke_fixture,
)


def test_quality_experiment_smoke_writes_application_artifacts(tmp_path: Path) -> None:
    fixture = write_quality_smoke_fixture(tmp_path)
    result = run_from_config(fixture.config_path, registry=median_smoke_registry())

    model_dir = fixture.output_dir / "median_smoke"
    application_path = model_dir / "quality_model_application.json"
    comparison_path = fixture.output_dir / "model_comparison.json"
    integrity_path = Path(result.artifact_integrity_path)

    assert application_path.exists()
    assert (model_dir / "quality_model.pkl").exists()
    assert comparison_path.exists()
    assert integrity_path.exists()
    assert result.integrity_result.passed

    application = json.loads(application_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert application["automatic_best_model_selection"] is False
    assert application["test_used_for_selection"] is False
    assert application["observation_columns"] == list(fixture.observation_columns)
    assert comparison["automatic_best_model_selection"] is False
    assert comparison["test_used_for_selection"] is False
