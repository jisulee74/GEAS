"""Parallel, calibrated-budget Step 20 v2 campaign runner."""
from __future__ import annotations
from datetime import datetime,timezone
import json,math,multiprocessing as mp,os,socket,threading,time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from dataclasses import asdict
from geas35.experiments.rl.step20_hpo import DeterministicSearchSpace,trial_rank_key
from geas35.experiments.rl.step20_runner import _artifact_hashes,_evaluate,_ppo_config,_train_to_budget
from geas35.rl.hybrid_ppo import HybridActorCritic,HybridPPOTrainer,save_ppo_checkpoint,seed_everything
from geas35.experiments.rl.checkpoint_v2 import load_ppo_checkpoint_v2 as load_ppo_checkpoint
import geas35.experiments.rl.step20_runner as _step20_runner
from geas35.rl.parallel_v2 import ProcessVectorEnv,VectorEnvSpec,collect_vectorized_rollout
from geas35.experiments.rl.step20_v2 import OFFICIAL_CROPS,STEP20_V2_VERSION,TrialLeaseStore,load_calibrated_budget,prepare_step20_v2
from geas35.rl.model_driven_env import load_model_driven_env_from_step15

def _read(path):return json.loads(Path(path).read_text())
def _write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,default=str)+"\n");tmp.replace(path)

def _train_to_budget_v2(**kwargs):
 original=_step20_runner.load_ppo_checkpoint
 _step20_runner.load_ppo_checkpoint=load_ppo_checkpoint
 try:return _step20_runner._train_to_budget(**kwargs)
 finally:_step20_runner.load_ppo_checkpoint=original

def _train_vectorized(root:Path,crop:str,config:dict[str,Any],seed:int,budget:int,checkpoint:Path,device:str,transition_n_jobs:int,vector_envs:int):
 if vector_envs<2:return _train_to_budget_v2(root=root,crop=crop,config_payload=config,seed=seed,budget_steps=budget,checkpoint=checkpoint,device=device,resume=True,transition_n_jobs=transition_n_jobs)
 rollout=int(config["rollout_length"])
 if rollout%vector_envs or budget%vector_envs:raise ValueError("Calibrated budget and rollout length must be divisible by vector_envs")
 hashes=_artifact_hashes(root,crop);expected=_ppo_config(config)
 if checkpoint.exists():
  trainer,saved=load_ppo_checkpoint(checkpoint,expected_artifact_hashes=hashes,device=device,restore_rng=True)
  if asdict(trainer.config)!=asdict(expected):raise ValueError("Resume config differs")
  saved_vector_envs=int(saved.get("extra_state",{}).get("vector_envs",1))
  if saved_vector_envs!=vector_envs:raise ValueError("Resume vector_envs differs from checkpoint")
  completed=int(saved.get("extra_state",{}).get("training_steps",0))
 else:
  seed_everything(seed);trainer=HybridPPOTrainer(HybridActorCritic(expected),expected,device=device);completed=0
 history_path=checkpoint.with_suffix(".training_history.json")
 history=_read(history_path).get("history",[]) if history_path.exists() else []
 specs=[VectorEnvSpec("geas_model_driven",{"project_root":str(root),"crop":crop,"split":"train","transition_n_jobs":transition_n_jobs}) for _ in range(vector_envs)]
 with ProcessVectorEnv(specs) as environments:
  while completed<budget:
   count=min(rollout,budget-completed)
   if count%vector_envs:raise ValueError("Remaining calibrated budget is not vector divisible")
   buffer=collect_vectorized_rollout(environments,trainer,steps=count,base_seed=seed+trainer.update_count,crop=crop,config_id=f"trial_seed_{seed}")
   metrics=trainer.update(buffer);completed+=count;history.append({"training_steps":completed,**metrics})
   _write(history_path,{"schema_version":STEP20_V2_VERSION,"crop":crop,"vector_envs":vector_envs,"history":history,"test_accessed":False})
   save_ppo_checkpoint(checkpoint,trainer,artifact_hashes=hashes,extra_state={"crop":crop,"training_steps":completed,"hpo_step":"20_v2","vector_envs":vector_envs,"test_accessed":False})
 return trainer,history

def _worker(root_text:str,crop:str,rung:int,device:str,transition_n_jobs:int,vector_envs:int,worker_index:int):
 root=Path(root_text);out=root/"experiments/rl_policy_training/artifacts/step20_v2";store=TrialLeaseStore(out/crop/"hpo_study.sqlite3",f"{STEP20_V2_VERSION}.{crop}");owner=f"{socket.gethostname()}:{os.getpid()}:{worker_index}"
 gates=_read(root/"experiments/rl_policy_training/artifacts/step20/train_derived_safety_gates.json")["crops"][crop]
 while True:
  claim=store.claim(owner,rung,lease_seconds=180)
  if claim is None:return
  trial,payload=claim["trial"],claim["payload"]
  heartbeat_stop=threading.Event()
  def keep_alive():
   while not heartbeat_stop.wait(60):
    store.heartbeat(owner,trial,rung,lease_seconds=180)
  heartbeat_thread=threading.Thread(target=keep_alive,daemon=True);heartbeat_thread.start()
  try:
   seed=int(payload["seed"]);config=payload["config"];budget=int(payload["budget_steps"])
   checkpoint=out/crop/f"trial_{trial:03d}"/(f"replication_seed_{seed}.pt" if payload.get("replication") else "checkpoint.pt")
   trainer,history=_train_vectorized(root,crop,config,seed,budget,checkpoint,device,transition_n_jobs,vector_envs)
   metrics,feasible,reasons,pairing=_evaluate(root=root,crop=crop,trainer=trainer,validation_episode_indices=(),evaluation_seed=42,gates=gates,transition_n_jobs=transition_n_jobs)
   store.complete(owner,trial,rung,{"config":config,"seed":seed,"budget_steps":budget,"metrics":metrics,"feasible":feasible,"failure_reasons":reasons,"checkpoint":checkpoint.relative_to(root).as_posix(),"updates":len(history),"paired_validation":pairing,"test_accessed":False,"replication":bool(payload.get("replication"))})
  except BaseException as exc:
   store.fail(owner,trial,rung,f"{type(exc).__name__}: {exc}");raise
  finally:
   heartbeat_stop.set();heartbeat_thread.join(timeout=2)

def _run_workers(root:Path,crop:str,rung:int,workers:int,device:str,transition_n_jobs:int,vector_envs:int):
 if workers<1:raise ValueError("trial_workers must be positive")
 context=mp.get_context("spawn");processes=[context.Process(target=_worker,args=(str(root),crop,rung,device,transition_n_jobs,vector_envs,index)) for index in range(workers)]
 for process in processes:process.start()
 for process in processes:process.join()
 failed=[p.exitcode for p in processes if p.exitcode!=0]
 if failed:raise RuntimeError(f"Step 20 v2 workers failed: {failed}")

def _completed(store:TrialLeaseStore,rung:int):
 return [row for row in store.rows() if row["rung"]==rung and row["status"]=="complete"]

def run_crop_hpo_v2(project_root:str|Path,crop:str,device:str="cpu",trial_workers:int=1,transition_n_jobs:int=1,vector_envs:int=1)->Path:
 root=Path(project_root).resolve();budget_artifact,budget_hash=load_calibrated_budget(root)
 readiness_path=root/"experiments/rl_policy_training/artifacts/step20_v2/step20_v2_readiness.json"
 if not readiness_path.exists():raise ValueError("Run Step 20 v2 --prepare once before crop workers")
 readiness=_read(readiness_path)
 if readiness.get("status")!="calibrated_ready" or readiness.get("budget_sha256")!=budget_hash:raise ValueError("Step 20 v2 readiness does not match calibrated budget")
 config=_read(root/"experiments/rl_policy_training/configs/step20_hpo_v2.json");rungs=budget_artifact["crops"][crop]["rung_budgets"];out=root/"experiments/rl_policy_training/artifacts/step20_v2";store=TrialLeaseStore(out/crop/"hpo_study.sqlite3",f"{STEP20_V2_VERSION}.{crop}")
 probe=load_model_driven_env_from_step15(project_root=root,crop=crop,split="train");sampler=DeterministicSearchSpace(config["search_space"],int(config["sampler_seed"]));payloads={trial:{"config":sampler.sample_trial(trial,probe.observation_shape[0]),"seed":int(config["sampler_seed"])+trial,"budget_steps":rungs[0]} for trial in range(int(config["trials_per_crop"]))}
 store.initialize(len(payloads),0,payloads);_run_workers(root,crop,0,trial_workers,device,transition_n_jobs,vector_envs)
 for rung_index,budget in enumerate(rungs[1:],start=1):
  previous=_completed(store,rung_index-1);feasible=[]
  for row in previous:
   result=row["result"]
   if result["feasible"]:feasible.append({"trial_number":row["trial"],"feasible":True,"metrics":result["metrics"],"result":result})
  count=max(1,int(math.ceil(len(previous)*float(config["promotion_fraction"]))));selected=sorted(feasible,key=trial_rank_key,reverse=True)[:count]
  store.initialize_selected({item["trial_number"]:{"config":item["result"]["config"],"seed":item["result"]["seed"],"budget_steps":budget} for item in selected},rung_index)
  _run_workers(root,crop,rung_index,trial_workers,device,transition_n_jobs,vector_envs)
 final_rows=_completed(store,len(rungs)-1);feasible=[]
 for row in final_rows:
  result=row["result"]
  if result["feasible"]:feasible.append({"trial_number":row["trial"],"feasible":True,"metrics":result["metrics"],"result":result})
 top=sorted(feasible,key=trial_rank_key,reverse=True)[:int(config["top_configurations_for_multi_seed"])]
 for seed_index,seed in enumerate(config["top_configuration_seeds"]):
  replication_rung=100+seed_index;store.initialize_selected({item["trial_number"]:{"config":{**item["result"]["config"],"ppo":{**item["result"]["config"]["ppo"],"seed":seed}},"seed":seed,"budget_steps":rungs[-1],"replication":True} for item in top},replication_rung);_run_workers(root,crop,replication_rung,trial_workers,device,transition_n_jobs,vector_envs)
 status=out/crop/"crop_hpo_status.json";_write(status,{"schema_version":STEP20_V2_VERSION,"status":"complete","crop":crop,"calibrated_rungs":rungs,"budget_sha256":budget_hash,"trial_workers":trial_workers,"transition_n_jobs":transition_n_jobs,"vector_envs":vector_envs,"test_accessed":False})
 return status

def finalize_step20_v2(project_root:str|Path)->Path:
 root=Path(project_root).resolve();out=root/"experiments/rl_policy_training/artifacts/step20_v2";budget,budget_hash=load_calibrated_budget(root);config=_read(root/"experiments/rl_policy_training/configs/step20_hpo_v2.json");best={}
 for crop in OFFICIAL_CROPS:
  store=TrialLeaseStore(out/crop/"hpo_study.sqlite3",f"{STEP20_V2_VERSION}.{crop}");rows=store.rows();required=int(config["trials_per_crop"])
  if len([r for r in rows if r["rung"]==0 and r["status"]=="complete"])!=required:raise RuntimeError(f"{crop}: official trials incomplete")
  candidates=[]
  for trial in sorted({r["trial"] for r in rows if r["rung"]>=100}):
   reps=[r["result"] for r in rows if r["trial"]==trial and r["rung"]>=100 and r["status"]=="complete"]
   if len(reps)!=len(config["top_configuration_seeds"]):continue
   metrics={key:float(np.mean([r["metrics"][key] for r in reps])) for key in reps[0]["metrics"]};candidates.append({"trial_number":trial,"feasible":all(r["feasible"] for r in reps),"metrics":metrics,"config":reps[0]["config"]})
  feasible=[c for c in candidates if c["feasible"]]
  if not feasible:raise RuntimeError(f"{crop}: multi-seed candidates incomplete or infeasible")
  selected=max(feasible,key=trial_rank_key);best[crop]=selected
 report=out/"hpo_report.json";_write(report,{"schema_version":STEP20_V2_VERSION,"status":"passed","budget_sha256":budget_hash,"crops":best,"test_accessed":False})
 _write(out/"step20_v2_readiness.json",{"schema_version":STEP20_V2_VERSION,"status":"passed","completed_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"budget_sha256":budget_hash,"test_accessed":False,"step21_started":False})
 return report

__all__=["finalize_step20_v2","run_crop_hpo_v2"]
