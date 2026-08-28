"""Step 20 v2 calibrated-budget gate and crash-safe trial leasing."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json,sqlite3,time
from pathlib import Path
from typing import Any,Mapping

STEP20_V2_VERSION="geas35.rl.step20.v2"; OFFICIAL_CROPS=("strawberry","melon","cucumber")
def _read(path:Path)->dict[str,Any]:
 value=json.loads(path.read_text());
 if not isinstance(value,dict):raise ValueError(f"Expected object: {path}")
 return value
def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _write(path:Path,value:Any)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)

def load_calibrated_budget(project_root:str|Path)->tuple[dict[str,Any],str]:
 root=Path(project_root).resolve();config=_read(root/"experiments/rl_policy_training/configs/step20_hpo_v2.json")
 source=config["budget_source"]
 if source.get("numeric_cli_override_allowed") is not False:raise ValueError("Numeric budget override must be disabled")
 path=root/source["path"];artifact=_read(path)
 if artifact.get("schema_version")!=source["schema"] or artifact.get("status")!="passed":raise ValueError("Step 19.5 calibrated budget is not passed")
 if artifact.get("test_accessed") is not False or artifact.get("numeric_budget_precommitted") is not False:raise ValueError("Invalid calibrated budget provenance")
 integrity=_read(path.parent/"step19_5_integrity_manifest.json")
 if integrity["budget"]["sha256"]!=_sha(path):raise ValueError("Calibrated budget hash mismatch")
 for crop in OFFICIAL_CROPS:
  rungs=artifact["crops"][crop]["rung_budgets"]
  if len(rungs)<2 or any(not isinstance(v,int) or v<=0 for v in rungs) or rungs!=sorted(set(rungs)):raise ValueError(f"{crop}: invalid calibrated rungs")
 return artifact,_sha(path)

class TrialLeaseStore:
 def __init__(self,path:str|Path,study_id:str):
  self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.study_id=study_id
  with self.connect() as db:db.execute("""CREATE TABLE IF NOT EXISTS trials(study_id TEXT,trial INTEGER,rung INTEGER,status TEXT,owner TEXT,lease_until REAL,heartbeat REAL,attempt INTEGER DEFAULT 0,payload TEXT,result TEXT,error TEXT,PRIMARY KEY(study_id,trial,rung))""")
 def connect(self):
  db=sqlite3.connect(self.path,timeout=60,isolation_level=None);db.execute("PRAGMA busy_timeout=60000");db.execute("PRAGMA journal_mode=WAL");return db
 def initialize(self,count:int,rung:int,payloads:Mapping[int,Mapping[str,Any]]|None=None):
  with self.connect() as db:
   db.execute("BEGIN IMMEDIATE")
   for trial in range(count):db.execute("INSERT OR IGNORE INTO trials(study_id,trial,rung,status,payload) VALUES(?,?,?,?,?)",(self.study_id,trial,rung,"pending",json.dumps(dict((payloads or {}).get(trial,{})))))
   db.commit()
 def initialize_selected(self,trials:Mapping[int,Mapping[str,Any]],rung:int):
  with self.connect() as db:
   db.execute("BEGIN IMMEDIATE")
   for trial,payload in trials.items():db.execute("INSERT OR IGNORE INTO trials(study_id,trial,rung,status,payload) VALUES(?,?,?,?,?)",(self.study_id,int(trial),rung,"pending",json.dumps(dict(payload))))
   db.commit()
 def claim(self,owner:str,rung:int,lease_seconds:float=300)->dict[str,Any]|None:
  now=time.time()
  with self.connect() as db:
   db.execute("BEGIN IMMEDIATE")
   row=db.execute("SELECT trial,payload,attempt FROM trials WHERE study_id=? AND rung=? AND (status='pending' OR (status='running' AND lease_until<?)) ORDER BY trial LIMIT 1",(self.study_id,rung,now)).fetchone()
   if row is None:db.commit();return None
   db.execute("UPDATE trials SET status='running',owner=?,lease_until=?,heartbeat=?,attempt=? WHERE study_id=? AND trial=? AND rung=?",(owner,now+lease_seconds,now,row[2]+1,self.study_id,row[0],rung));db.commit()
   return {"trial":row[0],"payload":json.loads(row[1] or "{}"),"attempt":row[2]+1}
 def heartbeat(self,owner:str,trial:int,rung:int,lease_seconds:float=300):
  now=time.time()
  with self.connect() as db:
   changed=db.execute("UPDATE trials SET heartbeat=?,lease_until=? WHERE study_id=? AND trial=? AND rung=? AND status='running' AND owner=?",(now,now+lease_seconds,self.study_id,trial,rung,owner)).rowcount
   if changed!=1:raise ValueError("Lease ownership lost")
 def complete(self,owner:str,trial:int,rung:int,result:Mapping[str,Any]):
  with self.connect() as db:
   changed=db.execute("UPDATE trials SET status='complete',result=?,lease_until=NULL WHERE study_id=? AND trial=? AND rung=? AND owner=? AND status='running'",(json.dumps(dict(result)),self.study_id,trial,rung,owner)).rowcount
   if changed!=1:raise ValueError("Cannot complete an unowned lease")
 def fail(self,owner:str,trial:int,rung:int,error:str):
  with self.connect() as db:db.execute("UPDATE trials SET status='failed',error=?,lease_until=NULL WHERE study_id=? AND trial=? AND rung=? AND owner=?",(error,self.study_id,trial,rung,owner))
 def rows(self):
  with self.connect() as db:values=db.execute("SELECT trial,rung,status,owner,attempt,payload,result,error FROM trials WHERE study_id=? ORDER BY rung,trial",(self.study_id,)).fetchall()
  return [{"trial":r[0],"rung":r[1],"status":r[2],"owner":r[3],"attempt":r[4],"payload":json.loads(r[5] or "{}"),"result":None if r[6] is None else json.loads(r[6]),"error":r[7]} for r in values]

def prepare_step20_v2(project_root:str|Path,require_calibrated_budget:bool=False)->Path:
 root=Path(project_root).resolve();out=root/"experiments/rl_policy_training/artifacts/step20_v2";config_path=root/"experiments/rl_policy_training/configs/step20_hpo_v2.json";config=_read(config_path)
 if config.get("protocol_version")!=STEP20_V2_VERSION or "rungs_environment_steps" in json.dumps(config):raise ValueError("Step 20 v2 config must not contain numeric rung budgets")
 budget_hash=None;status="ready_to_run_waiting_for_calibration"
 if require_calibrated_budget:
  _,budget_hash=load_calibrated_budget(root);status="calibrated_ready"
 for crop in OFFICIAL_CROPS:(out/crop).mkdir(parents=True,exist_ok=True);TrialLeaseStore(out/crop/"hpo_study.sqlite3",f"{STEP20_V2_VERSION}.{crop}")
 manifest={"schema_version":STEP20_V2_VERSION,"status":status,"created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"config_sha256":_sha(config_path),"budget_sha256":budget_hash,"crop_databases":"independent","test_accessed":False,"official_hpo_started":False,"step21_started":False}
 path=out/"step20_v2_readiness.json";_write(path,manifest);return path

__all__=["STEP20_V2_VERSION","TrialLeaseStore","load_calibrated_budget","prepare_step20_v2"]
