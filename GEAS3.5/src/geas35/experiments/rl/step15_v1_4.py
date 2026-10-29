"""Freeze a v1.4 Transition-dependent RL protocol and Train-only support."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from geas35.models.transition.feature_variants import prepare_frame_for_feature_variant

from geas35.rl.mdp_v1 import (
    MDP_V1_ACTION_COLUMNS, MDP_V1_BINARY_ACTION_COLUMNS,
    MDP_V1_CONTINUOUS_ACTION_COLUMNS, MDP_V1_REWARD_TERM_KEYS,
    MDP_V1_STEP_MINUTES, MDP_V1_TRANSITION_TARGET_COLUMNS, MdpV1Config,
)
from geas35.rl.support_v1 import SupportBuildConfig, build_support_artifact

VERSION = "geas35.rl.step15.v1_4"
CROPS = ("strawberry", "melon", "cucumber")


class Step15V14Error(ValueError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Step15V14Error(f"Expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Step15V14Error(f"Required file is missing: {path}")
    try:
        name = path.resolve().relative_to(root).as_posix()
    except ValueError:
        name = str(path.resolve())
    return {"path": name, "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _resolve_record(root: Path, record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_file() or _sha(path) != record["sha256"]:
        raise Step15V14Error(f"Frozen dependency hash mismatch: {path}")
    return path


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def authorize_rl_test_access(
    protocol_manifest_path: str | Path, authorization_path: str | Path,
    *, crop: str, final_evaluation: bool,
) -> Path:
    """Return a frozen RL Test path only for a matching final evaluation token."""
    protocol_path = Path(protocol_manifest_path).resolve()
    protocol = _read(protocol_path)
    authorization = _read(Path(authorization_path).resolve())
    if protocol.get("schema_version") != VERSION or protocol.get("status") != "passed":
        raise Step15V14Error("RL Test requires a passed v1.4 protocol.")
    if not final_evaluation or authorization.get("purpose") != "final_policy_test":
        raise Step15V14Error("RL Test requires the final-evaluation flag.")
    if authorization.get("authorized") is not True or authorization.get("protocol_sha256") != _sha(protocol_path):
        raise Step15V14Error("RL Test authorization does not match the frozen protocol hash.")
    if crop not in CROPS or authorization.get("crop") != crop:
        raise Step15V14Error("RL Test authorization crop mismatch.")
    root = next((parent for parent in protocol_path.parents if parent.name == "GEAS3.5"), None)
    if root is None:
        raise Step15V14Error("Cannot resolve GEAS3.5 project root.")
    split = _read(_resolve_record(root, protocol["split_access_policy"]))
    test_path = _resolve_record(root, split["datasets"][crop]["test"])
    access_log = protocol_path.parent / "rl_test_access_log.json"
    if access_log.exists():
        raise Step15V14Error("RL Test access was already authorized once for this protocol.")
    _write(access_log, {"schema_version": VERSION, "crop": crop,
        "purpose": "final_policy_test", "authorized_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": _sha(protocol_path), "test_dataset_sha256": _sha(test_path),
        "access_count": 1})
    return test_path


def run_step15_v1_4(
    *, project_root: str | Path, output_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    destination = Path(output_root).resolve() if output_root else root / "experiments/rl_policy_training/artifacts/step15_v1_4"
    config_path = Path(config_path).resolve() if config_path else root / "experiments/rl_policy_training/configs/step15_protocol_v1_4.json"
    config = _read(config_path)
    if config.get("schema_version") != "geas35.rl.step15.config.v1_4" or config.get("protocol_version") != VERSION:
        raise Step15V14Error("Invalid Step 15 v1.4 config.")
    if config["hpo"].get("budget_status") != "pending_calibration" or "environment_steps_per_trial" in config["hpo"]:
        raise Step15V14Error("Step budget must come from Step 19.5, not Step 15.")
    existing = destination / "rl_protocol_manifest.json"
    if existing.exists():
        frozen = _read(existing)
        if frozen.get("protocol_config", {}).get("sha256") != _sha(config_path):
            raise Step15V14Error("Frozen config changed; create a new protocol version.")
        raise Step15V14Error("Step 15 v1.4 output is already frozen; refusing to overwrite it.")

    step14_path = root / "experiments/transition_model_selection/artifacts/step14_v1_4/step14_handoff_manifest.json"
    step14 = _read(step14_path)
    if step14.get("schema_version") != "geas35.transition.step14.v1_4" or step14.get("status") != "passed":
        raise Step15V14Error("Passed v1.4 Step 14 handoff is required.")
    handoff = _read(_resolve_record(root, step14["deployment_handoff"]))
    if handoff.get("status") != "passed" or set(handoff.get("candidates", {})) != set(CROPS):
        raise Step15V14Error("Step 14 handoff is incomplete.")
    step11_path = root / "offline_dataset_preparation/datasets/5_rl_dataset/step11_6_integrity_manifest.json"
    step11 = _read(step11_path)
    if step11.get("status") != "passed" or step11.get("all_crop_integrity_gate") not in (True, "passed"):
        raise Step15V14Error("Step 11 RL dataset integrity did not pass.")
    for crop in CROPS:
        choice = handoff["candidates"][crop]
        if choice.get("feature_variant") != "deduplicated_without_quality_flags" or choice.get("test_status") != "passed":
            raise Step15V14Error(f"{crop}: chosen variant/Test safety status is invalid.")

    destination.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    split_policy = {
        "schema_version": VERSION, "created_at_utc": created,
        "fit_permissions": {"train": ["ppo_fit", "support_fit"],
            "validation": ["hpo_objective", "checkpoint_selection", "policy_comparison"],
            "test": ["single_final_evaluation_after_authorization"]},
        "test_default_state": "locked_fail_closed", "support_fit_splits": ["train"],
        "test_used_for_hpo": False, "test_used_for_support": False,
        "test_access_log_required": True,
        "datasets": {crop: {split: step11["crops"][crop]["datasets"][split]["rl_dataset"]
            for split in ("train", "validation", "test")} for crop in CROPS},
    }
    split_path = destination / "split_access_policy.json"
    _write(split_path, split_policy)
    seed_budget = {"schema_version": VERSION, "hpo": config["hpo"],
        "final_training": config["final_training"], "support_seed": config["support"]["seed"],
        "step_budget_source": "step19_5_calibration_artifact_only", "test_access": False}
    seed_path = destination / "seed_budget_registry.json"
    _write(seed_path, seed_budget)
    environment = {"schema_version": VERSION, "step_minutes": MDP_V1_STEP_MINUTES,
        "action_columns": list(MDP_V1_ACTION_COLUMNS),
        "continuous_action_columns": list(MDP_V1_CONTINUOUS_ACTION_COLUMNS),
        "binary_action_columns": list(MDP_V1_BINARY_ACTION_COLUMNS),
        "transition_target_columns": list(MDP_V1_TRANSITION_TARGET_COLUMNS),
        "reward_term_keys": list(MDP_V1_REWARD_TERM_KEYS),
        "reward_config": asdict(MdpV1Config()), "crop_observation_columns": {},
        "exogenous_modes": {"offline": "recorded_weather", "operating": "forecast_weather_interface"},
        "implementation_scope": "protocol_and_support_only_step16_migration_pending"}
    support_config = SupportBuildConfig(**config["support"])
    crops: dict[str, Any] = {}
    for crop in CROPS:
        choice = handoff["candidates"][crop]
        artifact_dir = root / choice["candidate_artifact_dir"]
        model_path = artifact_dir / "model.pkl"
        if _sha(model_path) != choice["candidate_model_sha256"]:
            raise Step15V14Error(f"{crop}: model hash differs from Step 14.")
        feature_path = artifact_dir / "feature_schema.json"
        feature = _read(feature_path)
        actions = list(feature["action_columns"])
        states = list(feature["observation_columns"])
        if len(states) != 55 or len(actions) != 6 or list(feature["input_columns"]) != states + actions:
            raise Step15V14Error(f"{crop}: expected ordered 55-observation + 6-action schema.")
        if feature["schema_hash"] != choice["schema_hash"] or actions != list(MDP_V1_ACTION_COLUMNS):
            raise Step15V14Error(f"{crop}: handoff schema/action mismatch.")
        scaler_path = _resolve_record(root, choice["observation_scaler"])
        scaler = _read(scaler_path)
        if set(scaler["columns"]) - set(states):
            raise Step15V14Error(f"{crop}: scaler columns exceed the selected observation schema.")
        train_record = split_policy["datasets"][crop]["train"]
        train_path = _resolve_record(root, train_record)
        train = prepare_frame_for_feature_variant(
            pd.read_parquet(train_path), choice["feature_variant"]
        )
        columns = [*states, *actions, "episode_id", "rl_valid_transition"]
        train = train.loc[:, columns]
        train = train.loc[train["rl_valid_transition"].astype(bool)].reset_index(drop=True)
        artifact_path, report_path = build_support_artifact(train, crop=crop,
            state_columns=states, action_columns=actions, episode_column="episode_id",
            output_dir=destination / crop, config=support_config)
        support = _read(artifact_path)
        if not all(support["empirical_calibration_coverage"][key] >= coverage
            for key, coverage in (("state_warning", .95), ("state_severe", .99),
                                  ("joint_warning", .95), ("joint_severe", .99))):
            raise Step15V14Error(f"{crop}: Train conformal coverage below target.")
        environment["crop_observation_columns"][crop] = states
        crops[crop] = {"candidate_name": choice["candidate_name"], "feature_variant": choice["feature_variant"],
            "candidate_model": _record(model_path, root), "feature_schema": _record(feature_path, root),
            "observation_scaler": _record(scaler_path, root), "train_rl_dataset": train_record,
            "train_valid_rows_used": len(train), "validation_rows_used_for_support": 0,
            "test_rows_used_for_support": 0, "state_action_support": _record(artifact_path, root),
            "support_reference": _record(destination / crop / "support_reference.npz", root),
            "calibration_report": _record(report_path, root)}
    environment_path = destination / "environment_schema.json"
    _write(environment_path, environment)
    protocol = {"schema_version": VERSION, "step": 15, "status": "passed",
        "created_at_utc": created, "scope": "v1_4_protocol_and_train_only_support_no_rl_test",
        "protocol_config": _record(config_path, root), "step14_manifest": _record(step14_path, root),
        "step11_immutable_manifest": _record(step11_path, root),
        "split_access_policy": _record(split_path, root),
        "seed_budget_registry": _record(seed_path, root), "environment_schema": _record(environment_path, root),
        "evaluation": config["evaluation"],
        "implementation_files": [_record(root / relative, root) for relative in (
            "src/geas35/experiments/rl/step15_v1_4.py",
            "src/geas35/rl/support_v1.py",
            "src/geas35/rl/mdp_v1.py",
            "src/geas35/models/transition/feature_variants.py")],
        "support_contract": {"fit_split": "train_only",
            "episode_disjoint_calibration": True, "warning_coverage": .95, "severe_coverage": .99,
            "state_and_joint_scores": True, "local_action_support": True,
            "validation_or_test_refit": False}, "crops": crops}
    protocol_path = destination / "rl_protocol_manifest.json"
    _write(protocol_path, protocol)
    integrity = {"schema_version": VERSION, "step": 15, "status": "passed",
        "protocol_manifest": _record(protocol_path, root), "checks": {
            "step14_v1_4_passed": True, "selected_transition_hash_chain_verified": True,
            "train_only_support_fit": True, "episode_disjoint_calibration": True,
            "validation_rows_used_for_support": 0, "test_rows_used_for_support": 0,
            "rl_test_accessed": False, "test_access_fail_closed": True,
            "step16_environment_migrated": False}}
    integrity_path = destination / "step15_integrity_manifest.json"
    _write(integrity_path, integrity)
    return integrity_path


__all__ = ["VERSION", "Step15V14Error", "authorize_rl_test_access", "run_step15_v1_4"]
