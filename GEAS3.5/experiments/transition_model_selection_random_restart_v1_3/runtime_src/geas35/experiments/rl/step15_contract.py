"""Step 15 RL protocol freeze and Train-only support calibration."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from geas35.rl.mdp_v1 import (
    MDP_V1_ACTION_COLUMNS,
    MDP_V1_BINARY_ACTION_COLUMNS,
    MDP_V1_CONTINUOUS_ACTION_COLUMNS,
    MDP_V1_OBSERVATION_COLUMNS,
    MDP_V1_REWARD_TERM_KEYS,
    MDP_V1_STEP_MINUTES,
    MDP_V1_TRANSITION_TARGET_COLUMNS,
    MdpV1Config,
)
from geas35.rl.support_v1 import SupportBuildConfig, build_support_artifact

STEP15_VERSION = "geas35.rl.step15.v1"
OFFICIAL_CROPS = ("strawberry", "melon", "cucumber")


class Step15ContractError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Step15ContractError(f"Required file is missing: {path}")
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(path.resolve())
    return {"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Step15ContractError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Step15ContractError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def authorize_test_access(
    protocol_manifest_path: str | Path, authorization_path: str | Path
) -> Path:
    """Fail-closed Test path resolver for later final evaluation stages."""
    protocol_path = Path(protocol_manifest_path).resolve()
    protocol = _read(protocol_path)
    authorization = _read(Path(authorization_path).resolve())
    if protocol.get("schema_version") != STEP15_VERSION:
        raise Step15ContractError("Test access requires a frozen Step 15 protocol.")
    if authorization.get("schema_version") != "geas35.rl.test_access_authorization.v1":
        raise Step15ContractError("Invalid Test authorization schema.")
    if authorization.get("purpose") != "final_test" or authorization.get("authorized") is not True:
        raise Step15ContractError("Test access is not explicitly authorized for final_test.")
    if authorization.get("protocol_sha256") != _sha256(protocol_path):
        raise Step15ContractError("Test authorization does not match the frozen protocol hash.")
    test_path = Path(str(authorization.get("test_path", ""))).expanduser().resolve()
    split_record = protocol.get("split_access_policy")
    if not isinstance(split_record, Mapping) or not split_record.get("path"):
        raise Step15ContractError("Frozen protocol has no split-access policy reference.")
    project_root = next(
        (parent for parent in protocol_path.parents if parent.name == "GEAS3.5"),
        protocol_path.parent,
    )
    split_path = Path(str(split_record["path"])).expanduser()
    split_path = split_path.resolve() if split_path.is_absolute() else (
        project_root / split_path
    ).resolve()
    if _sha256(split_path) != split_record.get("sha256"):
        raise Step15ContractError("Frozen split-access policy hash differs.")
    split_policy = _read(split_path)
    allowed_test_paths = {
        Path(record["test"]["path"]).expanduser().resolve()
        for record in split_policy.get("datasets", {}).values()
    }
    if test_path not in allowed_test_paths:
        raise Step15ContractError("Authorized path is not a frozen Test dataset.")
    if not test_path.is_file():
        raise Step15ContractError("Authorized Test dataset does not exist.")
    return test_path


def run_step15_contract(
    *, project_root: str | Path, output_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    destination = Path(output_root).resolve() if output_root else (
        root / "experiments/rl_policy_training/artifacts/step15"
    )
    destination.mkdir(parents=True, exist_ok=True)
    frozen_config_path = Path(config_path).resolve() if config_path else (
        root / "experiments/rl_policy_training/configs/step15_protocol_v1.json"
    )
    frozen_config = _read(frozen_config_path)
    if (frozen_config.get("schema_version") != "geas35.rl.step15.config.v1"
            or frozen_config.get("protocol_version") != STEP15_VERSION):
        raise Step15ContractError("Step 15 config requires a new schema/protocol version.")
    existing_protocol_path = destination / "rl_protocol_manifest.json"
    if existing_protocol_path.exists():
        existing = _read(existing_protocol_path)
        previous = existing.get("protocol_config", {}).get("sha256")
        if previous is not None and previous != _sha256(frozen_config_path):
            raise Step15ContractError(
                "Step 15 config changed; use a new protocol version/output root."
            )
    step14_root = root / "experiments/transition_model_selection/artifacts/step14"
    step14_manifest_path = step14_root / "step14_handoff_manifest.json"
    step14_audit_path = step14_root / "transition_v1_4_completion_audit.json"
    step14_handoff_path = step14_root / "explicit_candidate_deployment_handoff.json"
    step14_manifest = _read(step14_manifest_path)
    step14_audit = _read(step14_audit_path)
    handoff = _read(step14_handoff_path)
    if step14_manifest.get("status") != "passed" or step14_audit.get("status") != "passed":
        raise Step15ContractError("Step 15 requires passed Step 14 artifacts.")
    candidates = handoff.get("candidates", {})
    if set(candidates) != set(OFFICIAL_CROPS) or any(
        candidates[crop].get("candidate_name") != "extra_trees" for crop in OFFICIAL_CROPS
    ):
        raise Step15ContractError("Step 15 requires explicit ExtraTrees handoff for all crops.")

    step11_path = root / "offline_dataset_preparation/datasets/5_rl_dataset/step11_6_integrity_manifest.json"
    step11 = _read(step11_path)
    if step11.get("status") != "passed" or step11.get("all_crop_integrity_gate") not in (True, "passed"):
        raise Step15ContractError("Step 11 immutable RL manifest is not passed.")

    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    split_policy = {
        "schema_version": STEP15_VERSION,
        "created_at_utc": created,
        "fit_permissions": {
            "train": ["support_fit", "ppo_fit", "hpo_trial_fit"],
            "validation": ["hpo_objective", "checkpoint_selection", "policy_comparison"],
            "test": ["single_final_evaluation_after_authorization"],
        },
        "support_fit_splits": ["train"],
        "support_threshold_refit_on_validation_or_test": False,
        "test_default_state": "locked_fail_closed",
        "test_authorization_schema": "geas35.rl.test_access_authorization.v1",
        "test_used_for_hpo": False,
        "test_used_for_checkpoint_selection": False,
        "test_used_for_threshold_calibration": False,
        "datasets": {
            crop: {
                split: step11["crops"][crop]["datasets"][split]["rl_dataset"]
                for split in ("train", "validation", "test")
            }
            for crop in OFFICIAL_CROPS
        },
    }
    split_policy_path = destination / "split_access_policy.json"
    _write(split_policy_path, split_policy)

    seed_budget = {
        "schema_version": STEP15_VERSION,
        "hpo": {**frozen_config["hpo"], "test_access": False},
        "final_training": {"seed_count": len(frozen_config["final_training"]["seeds"]),
                           "seeds": frozen_config["final_training"]["seeds"],
                           "test_access": False},
        "transition_seed": 42,
        "support_seed": 42,
        "checkpoint": {"atomic": True, "includes_rng_state": True,
                       "selection_split": "validation"},
    }
    seed_budget_path = destination / "seed_budget_registry.json"
    _write(seed_budget_path, seed_budget)

    environment_schema = {
        "schema_version": STEP15_VERSION,
        "mdp_version": "v1",
        "step_minutes": MDP_V1_STEP_MINUTES,
        "observation_columns": list(MDP_V1_OBSERVATION_COLUMNS),
        "action_columns": list(MDP_V1_ACTION_COLUMNS),
        "continuous_action_columns": list(MDP_V1_CONTINUOUS_ACTION_COLUMNS),
        "binary_action_columns": list(MDP_V1_BINARY_ACTION_COLUMNS),
        "transition_target_columns": list(MDP_V1_TRANSITION_TARGET_COLUMNS),
        "reward_term_keys": list(MDP_V1_REWARD_TERM_KEYS),
        "reward_config": vars(MdpV1Config()),
        "transition_output_count": 3,
        "exogenous_modes": {"offline": "recorded_weather", "operating": "forecast_weather_interface"},
        "implementation_scope": "contract_only_step16_environment_not_implemented",
    }
    environment_schema_path = destination / "environment_schema.json"
    _write(environment_schema_path, environment_schema)

    support_config = SupportBuildConfig(**frozen_config["support"])
    crop_records: dict[str, Any] = {}
    for crop in OFFICIAL_CROPS:
        candidate_dir = root / candidates[crop]["candidate_artifact_dir"]
        feature_schema_path = candidate_dir / "feature_schema.json"
        feature_schema = _read(feature_schema_path)
        if feature_schema.get("target_columns") != [
            "obs_indoor_temp_c", "obs_indoor_humidity_pct", "obs_indoor_co2_ppm"
        ]:
            raise Step15ContractError(f"{crop}: expected official three-target schema.")
        action_columns = list(feature_schema["action_columns"])
        state_columns = [c for c in feature_schema["input_columns"] if c not in action_columns]
        train_record = step11["crops"][crop]["datasets"]["train"]["rl_dataset"]
        train_path = Path(train_record["path"]).resolve()
        if _sha256(train_path) != train_record["sha256"]:
            raise Step15ContractError(f"{crop}: Train RL dataset hash mismatch.")
        columns = [*state_columns, *action_columns, "episode_id", "rl_valid_transition"]
        train = pd.read_parquet(train_path, columns=columns)
        train = train.loc[train["rl_valid_transition"].astype(bool)].reset_index(drop=True)
        crop_dir = destination / crop
        artifact_path, report_path = build_support_artifact(
            train, crop=crop, state_columns=state_columns,
            action_columns=action_columns, episode_column="episode_id",
            output_dir=crop_dir, config=support_config,
        )
        scaler_path = root / step11["crops"][crop]["scaler"]["path"]
        crop_records[crop] = {
            "candidate_name": "extra_trees",
            "candidate_model": _record(candidate_dir / "model.pkl", root),
            "feature_schema": _record(feature_schema_path, root),
            "observation_scaler": _record(scaler_path, root),
            "train_rl_dataset": train_record,
            "train_valid_rows_used": len(train),
            "validation_rows_used_for_support": 0,
            "test_rows_used_for_support": 0,
            "state_action_support": _record(artifact_path, root),
            "support_reference": _record(crop_dir / "support_reference.npz", root),
            "calibration_report": _record(report_path, root),
        }

    protocol = {
        "schema_version": STEP15_VERSION,
        "step": "15",
        "status": "passed",
        "created_at_utc": created,
        "scope": "protocol_and_train_only_support_freeze_no_step16_environment",
        "protocol_config": _record(frozen_config_path, root),
        "step14_manifest": _record(step14_manifest_path, root),
        "step14_completion_audit": _record(step14_audit_path, root),
        "step11_immutable_manifest": _record(step11_path, root),
        "split_access_policy": _record(split_policy_path, root),
        "seed_budget_registry": _record(seed_budget_path, root),
        "environment_schema": _record(environment_schema_path, root),
        "hpo_primary_metric": frozen_config["evaluation"]["primary_metric"],
        "episodic_return_role": frozen_config["evaluation"]["episodic_return_role"],
        "hard_safety_constraint_role": "infeasible_gate",
        "statistics": {
            "confidence_interval": frozen_config["evaluation"]["confidence_interval"],
            "comparison": frozen_config["evaluation"]["comparison"],
        },
        "support_contract": {
            "fit_split": "train_only",
            "episode_disjoint_calibration": True,
            "warning_coverage": support_config.warning_coverage,
            "severe_coverage": support_config.severe_coverage,
            "distance": "group_normalized_mixed_euclidean",
            "state_and_joint_scores": True,
            "local_action_support": True,
            "validation_or_test_refit": False,
        },
        "crops": crop_records,
    }
    protocol_path = destination / "rl_protocol_manifest.json"
    _write(protocol_path, protocol)
    integrity = {
        "schema_version": STEP15_VERSION,
        "step": "15",
        "status": "passed",
        "protocol_manifest": _record(protocol_path, root),
        "checks": {
            "step14_passed": True,
            "all_crops_extra_trees": True,
            "train_only_support_fit": True,
            "episode_disjoint_calibration": True,
            "validation_rows_used_for_support": 0,
            "test_rows_used_for_support": 0,
            "test_access_fail_closed": True,
            "step16_environment_implemented": False,
        },
    }
    integrity_path = destination / "step15_integrity_manifest.json"
    _write(integrity_path, integrity)
    return integrity_path
