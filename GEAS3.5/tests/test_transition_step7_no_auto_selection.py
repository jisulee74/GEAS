from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.cli import run_from_config
from geas35.models import transition
from geas35.models.transition import load_transition_candidate_model
from synthetic_experiment_fixtures import write_transition_smoke_fixture


def test_step7_removes_auto_selection_and_requires_explicit_candidate(
    tmp_path: Path,
) -> None:
    fixture = write_transition_smoke_fixture(tmp_path)
    result = run_from_config(fixture.config_path)

    assert not hasattr(transition, "load_selected_transition_model")
    assert not hasattr(transition, "save_selected_transition_model")
    assert not hasattr(transition, "SELECTED_TRANSITION_MODEL_FILENAME")

    loaded = load_transition_candidate_model(
        fixture.output_dir,
        fixture.crop,
        model_name="linear_regression",
    )
    assert loaded.model_name == "linear_regression"

    crop_dir = fixture.output_dir / fixture.crop
    assert not (crop_dir / "selected_transition_model.json").exists()
    assert not (crop_dir / "selected_transition_model_test_metrics.json").exists()
    summary = json.loads(Path(result.experiment_summary_path).read_text(encoding="utf-8"))

    forbidden = {
        "selected_model",
        "selected_model_name",
        "winner",
        "recommended_model",
        "selection_report",
    }
    assert forbidden.isdisjoint(_keys(summary))
    assert summary["automatic_model_selection"] is False
    assert summary["validation_ranking"]["automatic_model_selection"] is False
    assert all(
        "selected" not in score
        for score in summary["validation_ranking"]["candidate_scores"]
    )


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        out = set(value)
        for item in value.values():
            out.update(_keys(item))
        return out
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out.update(_keys(item))
        return out
    return set()
