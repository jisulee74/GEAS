from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geas35.experiments.transition.deployment_handoff import (
    STEP14_COMPLETION_AUDIT_NAME,
    STEP14_VERSION,
    Step14HandoffError,
    create_step14_deployment_handoff,
)


def _write_external_decision(path: Path, *, candidate: str = "persistence") -> None:
    decisions = {}
    for crop in ("strawberry", "melon", "cucumber"):
        table = pd.read_csv(
            PROJECT_ROOT
            / f"experiments/transition_model_selection/artifacts/step12/{crop}/candidate_metrics.csv"
        )
        row = table.loc[table["candidate_name"] == candidate].iloc[0]
        decisions[crop] = {
            "candidate_name": candidate,
            "rationale": "Researcher-authored smoke decision based on validation evidence.",
            "validation_evidence": [{
                "table": "candidate_metrics",
                "metric": "validation_one_step_rmse",
                "value": float(row["validation_one_step_rmse"]),
            }],
            "resource_tradeoff": "Latency and model accuracy were reviewed by the researcher.",
            "deployment_constraints": ["Explicit artifact path loading is required."],
        }
    path.write_text(json.dumps({
        "schema_version": "geas35.transition.researcher_decision.v1",
        "decision_author": "pytest researcher",
        "decided_at_utc": "2026-08-12T00:00:00Z",
        "decision_source": "researcher",
        "framework_generated_recommendation": False,
        "decisions": decisions,
    }, indent=2), encoding="utf-8")


def test_step14_validates_external_decision_without_copying_candidates(tmp_path: Path) -> None:
    decision = tmp_path / "researcher_decision.json"
    _write_external_decision(decision)
    output = tmp_path / "step14"
    manifest_path, handoff_path = create_step14_deployment_handoff(
        project_root=PROJECT_ROOT,
        researcher_decision_path=decision,
        output_root=output,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == STEP14_VERSION
    assert manifest["status"] == "passed"
    assert manifest["researcher_decision_supplied_externally"] is True
    assert manifest["framework_generated_recommendation"] is False
    assert manifest["candidate_artifact_copy_count"] == 0
    assert manifest["explicit_candidate_count"] == 3
    assert set(handoff["candidates"]) == {"strawberry", "melon", "cucumber"}
    assert all(value["artifact_copied"] is False for value in handoff["candidates"].values())
    assert all(value["explicit_reward_handoff_verified"] is True
               for value in handoff["candidates"].values())
    assert not list(output.rglob("model.pkl"))
    assert not any("selected" in path.name.lower() for path in output.rglob("*"))



def test_step14_extra_trees_writes_completion_audit(tmp_path: Path) -> None:
    decision = tmp_path / "researcher_decision.json"
    _write_external_decision(decision, candidate="extra_trees")
    output = tmp_path / "step14"
    manifest_path, _ = create_step14_deployment_handoff(
        project_root=PROJECT_ROOT,
        researcher_decision_path=decision,
        output_root=output,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(
        (output / STEP14_COMPLETION_AUDIT_NAME).read_text(encoding="utf-8")
    )

    assert audit["status"] == "passed"
    assert set(audit["explicit_candidates"].values()) == {"extra_trees"}
    assert audit["checks"]["all_crops_explicit_extra_trees"] is True
    assert audit["checks"]["explicit_reward_handoff_verified"] is True
    assert audit["checks"]["candidate_artifact_copy_count"] == 0
    assert manifest["completion_audit"]["path"].endswith(STEP14_COMPLETION_AUDIT_NAME)

def test_step14_fails_closed_on_missing_or_tampered_researcher_evidence(tmp_path: Path) -> None:
    with pytest.raises(Step14HandoffError, match="Cannot read researcher decision"):
        create_step14_deployment_handoff(
            project_root=PROJECT_ROOT,
            researcher_decision_path=tmp_path / "missing.json",
            output_root=tmp_path / "missing-output",
        )
    decision = tmp_path / "researcher_decision.json"
    _write_external_decision(decision)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["decisions"]["strawberry"]["validation_evidence"][0]["value"] += 1.0
    decision.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Step14HandoffError, match="evidence value differs"):
        create_step14_deployment_handoff(
            project_root=PROJECT_ROOT,
            researcher_decision_path=decision,
            output_root=tmp_path / "tampered-output",
        )
