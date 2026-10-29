"""Fail-closed readiness audit for the unexecuted RL v2 campaign."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
from typing import Any
from geas35.experiments.rl.step20_v2 import OFFICIAL_CROPS,STEP20_V2_VERSION,TrialLeaseStore

def _read(path):return json.loads(Path(path).read_text())
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _record(path,root):
 path=Path(path);return {"path":path.resolve().relative_to(root).as_posix(),"sha256":_sha(path),"size_bytes":path.stat().st_size}
def build_v2_readiness_manifest(project_root:str|Path,test_summary:str)->Path:
 root=Path(project_root).resolve();base=root/"experiments/rl_policy_training/artifacts";step15=base/"step15_v2/rl_protocol_manifest.json";step19=base/"step19_5_v2/step19_5_readiness.json";step20=base/"step20_v2/step20_v2_readiness.json";snapshot=base/"step19_5_v2/v1_preservation_snapshot.json"
 d15,d19,d20,snap=map(_read,(step15,step19,step20,snapshot))
 if d15["status"]!="passed" or d19["status"]!="ready_to_run" or d20["status"]!="ready_to_run_waiting_for_calibration":raise ValueError("v2 stages are not in pre-execution readiness state")
 if (base/"step19_5_v2/calibrated_budget.json").exists():raise ValueError("Real calibrated budget exists; pre-execution audit refuses it")
 if any((base/"step19_5_v2"/crop/"learning_curves.json").exists() for crop in OFFICIAL_CROPS):raise ValueError("Real calibration curves already exist")
 for crop in OFFICIAL_CROPS:
  store=TrialLeaseStore(base/"step20_v2"/crop/"hpo_study.sqlite3",f"{STEP20_V2_VERSION}.{crop}")
  if store.rows():raise ValueError("Step 20 v2 trial DB is not empty")
 for record in snap["v1_files"]:
  path=root/record["path"]
  if _sha(path)!=record["sha256"]:raise ValueError(f"v1 preservation hash changed: {path}")
 implementations=[root/"src/geas35/experiments/rl/step15_v2.py",root/"src/geas35/experiments/rl/step19_5_calibration.py",root/"src/geas35/experiments/rl/step19_5_runner.py",root/"src/geas35/experiments/rl/step20_v2.py",root/"src/geas35/experiments/rl/step20_v2_runner.py",root/"src/geas35/rl/parallel_v2.py"]
 manifest={"schema_version":"geas35.rl.v2.readiness.v1","status":"ready_to_run","created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"test_summary":test_summary,"stages":{"step15_v2":"implemented_passed","step19_5":"implemented_not_executed","step20_v2":"implemented_waiting_for_calibration"},"contracts":{"step15":_record(step15,root),"step19_5":_record(step19,root),"step20":_record(step20,root),"v1_snapshot":_record(snapshot,root)},"implementations":[_record(path,root) for path in implementations],"checks":{"v1_preserved":True,"actual_calibration_started":False,"actual_hpo_started":False,"test_accessed":False,"background_process_started":False}}
 path=base/"v2_readiness_manifest.json";path.write_text(json.dumps(manifest,indent=2)+"\n");return path
__all__=["build_v2_readiness_manifest"]
