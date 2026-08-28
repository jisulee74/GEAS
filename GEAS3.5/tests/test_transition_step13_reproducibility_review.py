from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.reproducibility_review import (
    STEP13_VERSION,
    Step13ReviewError,
    _assert_no_selection,
    _same,
    run_step13_reproducibility_review,
)


def test_step13_reviews_all_actual_step12_candidates(tmp_path: Path) -> None:
    manifest_path = run_step13_reproducibility_review(
        project_root=PROJECT_ROOT,
        output_root=tmp_path / "step13",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == STEP13_VERSION
    assert manifest["status"] == "passed"
    assert manifest["reviewed_candidate_count"] == 24
    assert manifest["report_raw_artifact_match"] is True
    assert manifest["all_candidate_cold_load_smoke"] == "passed"
    assert manifest["automatic_model_selection"] is False
    assert manifest["step14_handoff_generated"] is False
    assert not any("selected" in path.name.lower() for path in (tmp_path / "step13").rglob("*"))
    for crop in ("strawberry", "melon", "cucumber"):
        assert manifest["crops"][crop]["status"] == "passed"
        assert len(manifest["crops"][crop]["cold_load_smoke"]) == 8
        assert all(
            value["status"] == "passed"
            for value in manifest["crops"][crop]["cold_load_smoke"].values()
        )


def test_step13_fails_on_report_or_selection_tampering(tmp_path: Path) -> None:
    with pytest.raises(Step13ReviewError, match="mismatch"):
        _same(1.0, 2.0, "tampered_metric")
    with pytest.raises(Step13ReviewError, match="auto-selection"):
        _assert_no_selection({"nested": {"selected_model": "x"}}, tmp_path / "bad.json")
