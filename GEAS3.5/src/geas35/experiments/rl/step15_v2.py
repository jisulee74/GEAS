"""Step 15 v2 contract: fixed trial count, calibration-owned HPO budget."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
from typing import Any

STEP15_V2_VERSION = "geas35.rl.step15.v2"

def _read(path: Path) -> dict[str, Any]:
    value=json.loads(path.read_text());
    if not isinstance(value,dict): raise ValueError(f"Expected object: {path}")
    return value
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _record(path: Path, root: Path) -> dict[str, Any]:
    return {"path":path.resolve().relative_to(root).as_posix(),"sha256":_sha(path),"size_bytes":path.stat().st_size}
def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2)+"\n")

def prepare_step15_v2(project_root: str|Path) -> Path:
    root=Path(project_root).resolve(); out=root/"experiments/rl_policy_training/artifacts/step15_v2"
    config_path=root/"experiments/rl_policy_training/configs/step15_protocol_v2.json"
    v1_path=root/"experiments/rl_policy_training/artifacts/step15/rl_protocol_manifest.json"
    step19_path=root/"experiments/rl_policy_training/artifacts/step19/step19_integrity_manifest.json"
    config,v1,step19=_read(config_path),_read(v1_path),_read(step19_path)
    if config.get("protocol_version")!=STEP15_V2_VERSION: raise ValueError("Invalid Step 15 v2 config")
    hpo=config["hpo"]
    if "environment_steps_per_trial" in hpo or any("budget" in k and isinstance(v,(int,float,list)) for k,v in hpo.items()):
        raise ValueError("Step 15 v2 must not precommit a numeric HPO budget")
    if hpo["trials_per_crop"]!=30 or hpo["budget_source_step"]!="19.5": raise ValueError("Invalid v2 HPO contract")
    if v1.get("status")!="passed" or step19.get("status")!="passed": raise ValueError("Step 15 v2 requires passed v1 and Step 19")
    manifest={
      "schema_version":STEP15_V2_VERSION,"step":"15_v2","status":"passed",
      "created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
      "scope":"versioned_contract_only_reuses_frozen_v1_data_model_support",
      "config":_record(config_path,root),"inherited_v1":_record(v1_path,root),"step19":_record(step19_path,root),
      "hpo":{"trials_per_crop":30,"sampler_seed":hpo["sampler_seed"],"budget_source_step":"19.5",
             "budget_status":"pending_calibration","budget_artifact_schema":hpo["budget_artifact_schema"],
             "numeric_budget_precommitted":False},
      "split_access_policy":v1["split_access_policy"],"crops":v1["crops"],
      "checks":{"test_accessed":False,"v1_artifacts_modified":False,"step20_v2_started":False}
    }
    path=out/"rl_protocol_manifest.json"; _write(path,manifest); return path

__all__=["STEP15_V2_VERSION","prepare_step15_v2"]
