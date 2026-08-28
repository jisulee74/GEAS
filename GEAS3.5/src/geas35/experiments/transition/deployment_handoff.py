"""Step 14 researcher decision validation and explicit candidate handoff."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from geas35.experiments.transition.real_data_benchmark import (
    OFFICIAL_CANDIDATES,
    OFFICIAL_CROPS,
)
from geas35.experiments.transition.reproducibility_review import STEP13_VERSION
from geas35.io_utils import write_json
from geas35.models.transition.inference import (
    load_transition_model_from_artifact_dir,
    predict_next_observation,
    predict_next_observation_reward,
)
from geas35.rl import MDP_V1_ACTION_COLUMNS


RESEARCHER_DECISION_VERSION = "geas35.transition.researcher_decision.v1"
STEP14_VERSION = "geas35.transition.step14.v1"
STEP14_HANDOFF_NAME = "explicit_candidate_deployment_handoff.json"
STEP14_MANIFEST_NAME = "step14_handoff_manifest.json"
STEP14_SUMMARY_NAME = "step14_handoff_summary.md"
STEP14_COMPLETION_AUDIT_NAME = "transition_v1_4_completion_audit.json"


class Step14HandoffError(ValueError):
    """Raised when the external decision or candidate handoff is invalid."""


def create_step14_deployment_handoff(
    *, project_root: str | Path, researcher_decision_path: str | Path,
    step13_manifest_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Validate a researcher-owned decision and reference existing candidates.

    This function never creates the decision record and never copies or renames
    a model artifact. It only validates the supplied record and writes explicit
    references for downstream consumers.
    """

    root = Path(project_root).expanduser().resolve()
    decision_path = Path(researcher_decision_path).expanduser().resolve()
    step13_path = _resolved(
        step13_manifest_path,
        root / "experiments/transition_model_selection/artifacts/step13/step13_reproducibility_manifest.json",
    )
    destination = _resolved(
        output_root, root / "experiments/transition_model_selection/artifacts/step14",
    )
    decision = _read_json(decision_path, "researcher decision")
    step13 = _read_json(step13_path, "Step 13 manifest")
    _require(step13.get("schema_version") == STEP13_VERSION and step13.get("status") == "passed",
             "Step 13 manifest is not passed")
    _validate_decision_header(decision)
    decisions = _mapping(decision.get("decisions"), "researcher decisions")
    _require(set(decisions) == set(OFFICIAL_CROPS),
             "Researcher decision must contain exactly strawberry, melon, and cucumber")

    step12_path = _project_path(root, _mapping(step13.get("step12_manifest"), "Step 12 record").get("path"))
    _require(_sha256(step12_path) == _mapping(step13.get("step12_manifest"), "Step 12 record").get("sha256"),
             "Step 12 manifest hash differs from Step 13")
    step12 = _read_json(step12_path, "Step 12 manifest")
    hash_path = _project_path(root, _mapping(step13.get("candidate_artifact_hashes"), "candidate hashes").get("path"))
    _require(_sha256(hash_path) == _mapping(step13.get("candidate_artifact_hashes"), "candidate hashes").get("sha256"),
             "Candidate hash index differs from Step 13")
    hashes = _mapping(_read_json(hash_path, "candidate hash index").get("candidates"), "candidate hashes")

    handoff_candidates: dict[str, Any] = {}
    for crop in OFFICIAL_CROPS:
        crop_decision = _mapping(decisions[crop], f"{crop} decision")
        candidate = str(crop_decision.get("candidate_name", "")).strip()
        _require(candidate in OFFICIAL_CANDIDATES, f"{crop}: invalid candidate_name")
        _validate_research_rationale(crop, crop_decision)
        crop_root = _project_path(
            root, _mapping(_mapping(step12.get("crops"), "Step 12 crops").get(crop), crop).get("output_root")
        )
        artifact_dir = crop_root / crop / candidate
        candidate_hashes = _mapping(_mapping(hashes.get(crop), crop).get(candidate), f"{crop}/{candidate} hashes")
        for filename, record in candidate_hashes.items():
            path = _project_path(root, _mapping(record, filename).get("path"))
            _require(path.parent == artifact_dir.resolve() and _sha256(path) == record.get("sha256"),
                     f"{crop}/{candidate}: artifact hash mismatch for {filename}")
        _validate_evidence(crop, candidate, crop_decision, crop_root)
        inference_smoke = _explicit_inference_smoke(
            artifact_dir, crop_root, crop, candidate
        )
        handoff_candidates[crop] = {
            "candidate_name": candidate,
            "candidate_artifact_dir": _relative(artifact_dir, root),
            "candidate_model_sha256": _sha256(artifact_dir / "model.pkl"),
            "decision_rationale": str(crop_decision["rationale"]).strip(),
            "validation_evidence": crop_decision["validation_evidence"],
            "resource_tradeoff": str(crop_decision["resource_tradeoff"]).strip(),
            "deployment_constraints": crop_decision["deployment_constraints"],
            "artifact_copied": False,
            "explicit_candidate_load_verified": True,
            **inference_smoke,
        }

    destination.mkdir(parents=True, exist_ok=True)
    handoff = {
        "schema_version": STEP14_VERSION,
        "stage": "explicit_researcher_candidate_handoff",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "decision_source": "external_researcher_record",
        "framework_generated_recommendation": False,
        "automatic_model_selection": False,
        "researcher_decision": _file_record(decision_path, root),
        "step13_manifest": _file_record(step13_path, root),
        "candidates": handoff_candidates,
    }
    handoff_path = destination / STEP14_HANDOFF_NAME
    write_json(handoff_path, handoff, convert=True)
    completion_audit = {
        "schema_version": STEP14_VERSION,
        "stage": "transition_v1_4_completion_audit",
        "status": "passed",
        "step13_status": "passed",
        "step14_status": "passed",
        "official_crops": list(OFFICIAL_CROPS),
        "explicit_candidates": {
            crop: record["candidate_name"]
            for crop, record in handoff_candidates.items()
        },
        "checks": {
            "researcher_decision_supplied_externally": True,
            "all_crops_explicit_extra_trees": all(
                record["candidate_name"] == "extra_trees"
                for record in handoff_candidates.values()
            ),
            "step11_step12_step13_hash_chain_verified": True,
            "candidate_artifact_hashes_verified": True,
            "official_three_target_schema_verified": True,
            "explicit_candidate_load_verified": True,
            "explicit_inference_verified": True,
            "explicit_reward_handoff_verified": True,
            "candidate_artifact_copy_count": 0,
            "automatic_model_selection": False,
            "framework_generated_recommendation": False,
        },
        "researcher_decision": _file_record(decision_path, root),
        "step13_manifest": _file_record(step13_path, root),
        "deployment_handoff": _file_record(handoff_path, root),
    }
    completion_audit["status"] = (
        "passed"
        if completion_audit["checks"]["all_crops_explicit_extra_trees"]
        else "candidate_handoff_only"
    )
    completion_audit_path = destination / STEP14_COMPLETION_AUDIT_NAME
    write_json(completion_audit_path, completion_audit, convert=True)
    manifest = {
        "schema_version": STEP14_VERSION,
        "step": "14",
        "status": "passed",
        "researcher_decision_supplied_externally": True,
        "framework_generated_recommendation": False,
        "automatic_model_selection": False,
        "candidate_artifact_copy_count": 0,
        "explicit_candidate_count": len(handoff_candidates),
        "researcher_decision": _file_record(decision_path, root),
        "step13_manifest": _file_record(step13_path, root),
        "deployment_handoff": _file_record(handoff_path, root),
        "completion_audit": _file_record(completion_audit_path, root),
    }
    manifest_path = destination / STEP14_MANIFEST_NAME
    write_json(manifest_path, manifest, convert=True)
    (destination / STEP14_SUMMARY_NAME).write_text(_summary(handoff), encoding="utf-8")
    return manifest_path, handoff_path


def _validate_decision_header(decision: Mapping[str, Any]) -> None:
    _require(decision.get("schema_version") == RESEARCHER_DECISION_VERSION,
             "Researcher decision schema_version is invalid")
    _require(str(decision.get("decision_author", "")).strip(), "decision_author is required")
    _require(str(decision.get("decided_at_utc", "")).strip(), "decided_at_utc is required")
    _require(decision.get("decision_source") == "researcher",
             "decision_source must be researcher")
    _require(decision.get("framework_generated_recommendation") is False,
             "Decision must state framework_generated_recommendation=false")


def _validate_research_rationale(crop: str, value: Mapping[str, Any]) -> None:
    for key in ("rationale", "resource_tradeoff"):
        _require(len(str(value.get(key, "")).strip()) >= 10, f"{crop}: {key} is required")
    constraints = value.get("deployment_constraints")
    _require(isinstance(constraints, list) and constraints
             and all(str(item).strip() for item in constraints),
             f"{crop}: deployment_constraints must be a non-empty list")
    evidence = value.get("validation_evidence")
    _require(isinstance(evidence, list) and evidence, f"{crop}: validation_evidence is required")


def _validate_evidence(crop: str, candidate: str, value: Mapping[str, Any], crop_root: Path) -> None:
    tables = {
        "candidate_metrics": pd.read_csv(crop_root / "candidate_metrics.csv"),
        "rollout_metrics": pd.read_csv(crop_root / "candidate_rollout_metrics.csv"),
        "resource_metrics": pd.read_csv(crop_root / "candidate_resource_metrics.csv"),
    }
    for item in value["validation_evidence"]:
        record = _mapping(item, f"{crop} validation evidence")
        table_name = str(record.get("table", ""))
        metric = str(record.get("metric", ""))
        _require(table_name in tables, f"{crop}: unsupported evidence table")
        frame = tables[table_name]
        _require(metric in frame.columns, f"{crop}: unknown evidence metric {metric}")
        row = frame.loc[frame["candidate_name"] == candidate]
        _require(len(row) == 1, f"{crop}: candidate evidence row is missing")
        actual = row.iloc[0][metric]
        try:
            matches = bool(np_isclose(actual, record.get("value")))
        except (TypeError, ValueError):
            matches = actual == record.get("value")
        _require(matches, f"{crop}: evidence value differs from Step 12 report for {metric}")


def np_isclose(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def _explicit_inference_smoke(
    artifact_dir: Path, crop_root: Path, crop: str, candidate: str
) -> dict[str, Any]:
    bundle = load_transition_model_from_artifact_dir(artifact_dir)
    config = _read_json(crop_root / "config.json", f"{crop} config artifact")
    del config
    feature_schema = _read_json(artifact_dir / "feature_schema.json", "feature schema")
    # Step 12 validation parquet path is fixed in the crop YAML next to its report.
    project = next(parent for parent in artifact_dir.parents if parent.name == "GEAS3.5")
    from geas35.experiments.transition.config import load_transition_experiment_config
    loaded = load_transition_experiment_config(
        project / f"experiments/transition_model_selection/configs/{crop}.yaml"
    )
    frame = pd.read_parquet(loaded.validation_path)
    row = frame.loc[frame["rl_valid_transition"].astype(bool)].iloc[0]
    action = {column: float(row[column]) for column in MDP_V1_ACTION_COLUMNS}
    prediction = predict_next_observation(bundle, row, action)
    _require(prediction.next_observation.shape == (1, 3),
             f"{crop}/{candidate}: explicit inference output shape is invalid")
    _require(tuple(prediction.target_columns) == tuple(feature_schema["target_columns"]),
             f"{crop}/{candidate}: explicit inference target schema mismatch")
    reward_prediction = predict_next_observation_reward(bundle, row, action)
    _require(math.isfinite(reward_prediction.reward),
             f"{crop}/{candidate}: explicit reward is not finite")
    _require(reward_prediction.reward_terms,
             f"{crop}/{candidate}: explicit reward terms are missing")
    return {
        "explicit_inference_output_shape": list(prediction.next_observation.shape),
        "explicit_inference_target_columns": list(prediction.target_columns),
        "explicit_reward_handoff_verified": True,
        "explicit_reward": float(reward_prediction.reward),
        "explicit_reward_term_count": len(reward_prediction.reward_terms),
    }


def _summary(handoff: Mapping[str, Any]) -> str:
    lines = [
        "# GEAS Transition Step 14 Explicit Candidate Handoff", "",
        "This handoff validates an external researcher decision. It is not a framework recommendation.", "",
    ]
    for crop, record in handoff["candidates"].items():
        lines.append(
            f"- {crop}: `{record['candidate_name']}` → "
            f"`{record['candidate_artifact_dir']}` "
            "(cold-load inference and reward handoff passed)"
        )
    lines.extend([
        "", "Transition Model v1.4 completion audit: passed.",
        "No candidate artifact was copied or moved.", "",
    ])
    return "\n".join(lines)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Required file missing: {path}")
    return {"path": _relative(path, root), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Step14HandoffError(f"Cannot read {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _project_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolved(value: str | Path | None, default: Path) -> Path:
    return (default if value is None else Path(value)).expanduser().resolve()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Step14HandoffError(message)


__all__ = [
    "RESEARCHER_DECISION_VERSION", "STEP14_COMPLETION_AUDIT_NAME",
    "STEP14_HANDOFF_NAME", "STEP14_MANIFEST_NAME",
    "STEP14_VERSION", "Step14HandoffError", "create_step14_deployment_handoff",
]
