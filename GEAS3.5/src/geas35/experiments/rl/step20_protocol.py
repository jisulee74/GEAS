"""Step 20 protocol freeze and integrity records (no reduced-budget selection)."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

from geas35.experiments.rl.step20_hpo import (
    HpoDatabase, OFFICIAL_CROPS, STEP20_VERSION, normalized_objective_audit,
    validate_step20_config,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.resolve().relative_to(root).as_posix(), "sha256": _sha(path),
            "size_bytes": path.stat().st_size}


def _wilson_upper(successes: int, count: int, confidence: float = 0.95) -> float:
    if count <= 0:
        raise ValueError("Wilson interval requires observations")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / count
    denominator = 1.0 + z * z / count
    center = p + z * z / (2.0 * count)
    radius = z * ((p * (1.0 - p) / count + z * z / (4.0 * count * count)) ** 0.5)
    return min(1.0, (center + radius) / denominator)


def prepare_step20_protocol(*, project_root: str | Path,
                            output_root: str | Path | None = None) -> Path:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve() if output_root else (
        root / "experiments/rl_policy_training/artifacts/step20"
    )
    config_path = root / "experiments/rl_policy_training/configs/step20_hpo_v1.json"
    step15_path = root / "experiments/rl_policy_training/artifacts/step15/rl_protocol_manifest.json"
    step19_path = root / "experiments/rl_policy_training/artifacts/step19/step19_integrity_manifest.json"
    step17_coverage_path = root / "experiments/rl_policy_training/artifacts/step17/support_coverage_report.json"
    step19_diagnostics_path = root / "experiments/rl_policy_training/artifacts/step19/pilot_learning_diagnostics.json"
    config, step15, step19 = _read(config_path), _read(step15_path), _read(step19_path)
    if step19.get("status") != "passed":
        raise ValueError("Step 20 requires passed Step 19")
    seed_budget_path = root / step15["seed_budget_registry"]["path"]
    seed_budget = _read(seed_budget_path)
    validate_step20_config(config, seed_budget["hpo"])
    coverage = _read(step17_coverage_path)
    pilot = _read(step19_diagnostics_path)

    gates: dict[str, Any] = {}
    for crop in OFFICIAL_CROPS:
        overall = coverage["crops"][crop]["overall"]
        severe_limit = 1.0 - float(overall["joint_severe_coverage"])
        pilot_steps = int(pilot["crops"][crop]["training_steps"])
        fallback_events = round(float(pilot["crops"][crop]["ood_summary"]["fallback_rate"]) * pilot_steps)
        gates[crop] = {
            "hard_constraint_violations_max": 0,
            "severe_ood_rate_max": severe_limit,
            "severe_ood_derivation": "1 - Step17 Train joint severe coverage",
            "fallback_rate_max": _wilson_upper(fallback_events, pilot_steps),
            "fallback_derivation": "95% Wilson upper bound from Step19 Train pilot",
            "thresholds_hpo_tunable": False,
        }

    output.mkdir(parents=True, exist_ok=True)
    gate_path = output / "train_derived_safety_gates.json"
    audit_path = output / "normalized_objective_audit.json"
    status_path = output / "hpo_execution_status.json"
    db_path = output / "hpo_study.sqlite3"
    _write(gate_path, {"schema_version": STEP20_VERSION, "fit_split": "train_only", "crops": gates})
    _write(audit_path, {"schema_version": STEP20_VERSION, **normalized_objective_audit(),
                        "test_accessed": False})
    HpoDatabase(db_path, STEP20_VERSION)
    _write(status_path, {
        "schema_version": STEP20_VERSION,
        "status": "protocol_ready_official_trials_pending",
        "official_trials_required_per_crop": 30,
        "completed_trials": {crop: 0 for crop in OFFICIAL_CROPS},
        "best_config_selected": False,
        "smoke_results_eligible_for_selection": False,
        "test_accessed": False,
        "step21_started": False,
    })
    manifest = {
        "schema_version": STEP20_VERSION, "step": "20",
        "status": "protocol_ready_official_trials_pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "step20_only_step21_not_started",
        "config": _record(config_path, root), "step15": _record(step15_path, root),
        "step19": _record(step19_path, root),
        "implementations": {
            "hpo": _record(root / "src/geas35/experiments/rl/step20_hpo.py", root),
            "protocol": _record(root / "src/geas35/experiments/rl/step20_protocol.py", root),
            "runner": _record(root / "src/geas35/experiments/rl/step20_runner.py", root),
            "cli": _record(root / "experiments/rl_policy_training/scripts/run_step20_hpo.py", root),
        },
        "artifacts": {"safety_gates": _record(gate_path, root),
                      "objective_audit": _record(audit_path, root),
                      "execution_status": _record(status_path, root)},
        "database": {"path": db_path.resolve().relative_to(root).as_posix(),
                     "format": "sqlite3", "study_id": STEP20_VERSION,
                     "mutable_resumable_artifact": True},
        "checks": {"test_accessed": False, "step21_started": False,
                   "official_hpo_complete": False, "best_config_selected": False},
    }
    manifest_path = output / "step20_integrity_manifest.json"
    _write(manifest_path, manifest)
    return manifest_path


__all__ = ["prepare_step20_protocol"]
