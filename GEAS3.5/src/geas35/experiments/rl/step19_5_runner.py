"""Resumable real-data runner for Step 19.5; no Test access."""
from __future__ import annotations
from dataclasses import asdict
import csv,fcntl,hashlib,json,multiprocessing as mp,os,socket,threading,time
from pathlib import Path
from typing import Any
from geas35.experiments.rl.step19_5_calibration import analyze_crop_learning_curves,create_budget_artifact,representative_configurations,OFFICIAL_CROPS,STEP19_5_VERSION
from geas35.experiments.rl.step19_5_parallel import CalibrationLeaseStore
from geas35.experiments.rl.checkpoint_v2 import load_ppo_checkpoint_v2
import geas35.experiments.rl.step20_runner as _step20_runner
from geas35.experiments.rl.step20_runner import _evaluate,_train_to_budget
from geas35.rl.hybrid_ppo import HybridPPOConfig
from geas35.rl.model_driven_env import load_model_driven_env_from_step15

def _read(path):return json.loads(Path(path).read_text())
def _write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,default=str)+"\n");tmp.replace(path)
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare_step19_5(project_root:str|Path)->Path:
 root=Path(project_root).resolve();out=root/"experiments/rl_policy_training/artifacts/step19_5_v2";out.mkdir(parents=True,exist_ok=True)
 config=_read(root/"experiments/rl_policy_training/configs/step19_5_calibration_v2.json")
 search=_read(root/"experiments/rl_policy_training/configs/step20_hpo_v1.json")["search_space"]
 default=HybridPPOConfig();defaults={"learning_rate":default.learning_rate,"rollout_length":2048,"minibatch_size":default.minibatch_size,"update_epochs":default.update_epochs,"gamma":default.gamma,"gae_lambda":default.gae_lambda,"clip_range":default.clip_range,"entropy_coefficient":default.entropy_coefficient,"value_coefficient":default.value_coefficient,"target_kl":default.target_kl,"network_width":default.hidden_sizes[0],"network_depth":len(default.hidden_sizes),"beta_concentration_floor":default.beta_concentration_floor}
 reps=representative_configurations(search,defaults,int(config["representative_configurations"]),42)
 path=out/"representative_configurations.json";_write(path,{"schema_version":STEP19_5_VERSION,"selection":"default_plus_deterministic_maximin","configurations":reps,"outcome_blind":True})
 v1=root/"experiments/rl_policy_training/artifacts/step20";files=[]
 for item in sorted(p for p in v1.rglob("*") if p.is_file() and not p.name.endswith(("-wal","-shm"))):files.append({"path":item.relative_to(root).as_posix(),"sha256":_sha(item),"size_bytes":item.stat().st_size})
 _write(out/"v1_preservation_snapshot.json",{"schema_version":"geas35.rl.v1_preservation.v2","status":"superseded_stopped_preserved","v1_files":files,"v1_used_for_v2_selection":False})
 readiness=out/"step19_5_readiness.json";_write(readiness,{"schema_version":STEP19_5_VERSION,"status":"ready_to_run","actual_calibration_started":False,"test_accessed":False,"representatives_sha256":_sha(path)})
 return readiness

def _payload(parameters:dict[str,Any],observation_dim:int,seed:int)->dict[str,Any]:
 p=dict(parameters);width=int(p.pop("network_width"));depth=int(p.pop("network_depth"));rollout=int(p.pop("rollout_length"))
 ppo=HybridPPOConfig(observation_dim=observation_dim,hidden_sizes=tuple([width]*depth),seed=seed,learning_rate=float(p.pop("learning_rate")),minibatch_size=int(p.pop("minibatch_size")),update_epochs=int(p.pop("update_epochs")),gamma=float(p.pop("gamma")),gae_lambda=float(p.pop("gae_lambda")),clip_range=float(p.pop("clip_range")),entropy_coefficient=float(p.pop("entropy_coefficient")),value_coefficient=float(p.pop("value_coefficient")),target_kl=float(p.pop("target_kl")),beta_concentration_floor=float(p.pop("beta_concentration_floor")))
 if p:raise ValueError(f"Unused representative parameters: {sorted(p)}")
 return {"ppo":json.loads(json.dumps(asdict(ppo))),"rollout_length":rollout}



def _train_to_budget_v2(**kwargs):
 original=_step20_runner.load_ppo_checkpoint
 _step20_runner.load_ppo_checkpoint=load_ppo_checkpoint_v2
 try:return _step20_runner._train_to_budget(**kwargs)
 finally:_step20_runner.load_ppo_checkpoint=original

def _parallel_state_path(out:Path,crop:str)->Path:return out/crop/"parallel_state.json"

def _job_payloads(reps,seeds,observation_dim,budget):
 payloads={};job_id=0
 for representative in reps:
  for seed in seeds:
   payloads[job_id]={"config_id":representative["config_id"],"seed":int(seed),"budget_steps":int(budget),"config":_payload(representative["parameters"],observation_dim,int(seed))};job_id+=1
 return payloads

def _execute_job(root:Path,crop:str,budget:int,claim:dict[str,Any],owner:str,store:CalibrationLeaseStore,device:str,transition_n_jobs:int)->None:
 payload=claim["payload"];job_id=int(claim["job_id"]);seed=int(payload["seed"]);config_id=str(payload["config_id"])
 checkpoint=root/"experiments/rl_policy_training/artifacts/step19_5_v2"/crop/config_id/f"seed_{seed}.pt"
 gates=_read(root/"experiments/rl_policy_training/artifacts/step20/train_derived_safety_gates.json")["crops"][crop]
 stop=threading.Event()
 def keep_alive():
  while not stop.wait(60):store.heartbeat(owner,budget,job_id,lease_seconds=180)
 thread=threading.Thread(target=keep_alive,daemon=True);thread.start()
 try:
  trainer,history=_train_to_budget_v2(root=root,crop=crop,config_payload=payload["config"],seed=seed,budget_steps=budget,checkpoint=checkpoint,device=device,resume=True,transition_n_jobs=transition_n_jobs)
  metrics,feasible,_,_=_evaluate(root=root,crop=crop,trainer=trainer,validation_episode_indices=(),evaluation_seed=42,gates=gates,transition_n_jobs=transition_n_jobs)
  latest=history[-1];row={"crop":crop,"config_id":config_id,"seed":seed,"steps":budget,"split":"validation","test_accessed":False,"safety_gate_passed":feasible,"mean_reward_per_scheduled_valid_step":metrics["mean_reward_per_scheduled_valid_step"],"policy_loss":latest["policy_loss"],"value_loss":latest["value_loss"],"approx_kl":latest["approx_kl"],"entropy":latest["entropy"],"explained_variance":latest["explained_variance"],"gradient_norm":latest["gradient_norm"],"in_support_rate":latest["in_support_rate"],"warning_ood_rate":latest["warning_ood_rate"],"severe_ood_rate":latest["severe_ood_rate"],"projection_rate":latest["projection_rate"],"fallback_rate":latest["fallback_rate"],"termination_rate":latest["severe_state_termination_rate"]}
  store.complete(owner,budget,job_id,{"row":row,"checkpoint":checkpoint.relative_to(root).as_posix(),"test_accessed":False})
 except BaseException as exc:
  store.fail(owner,budget,job_id,f"{type(exc).__name__}: {exc}");raise
 finally:
  stop.set();thread.join(timeout=2)

def _fixed_budget_worker(root_text:str,crop:str,budget:int,device:str,transition_n_jobs:int,worker_index:int)->None:
 root=Path(root_text);out=root/"experiments/rl_policy_training/artifacts/step19_5_v2";store=CalibrationLeaseStore(out/crop/"calibration_jobs.sqlite3",crop);owner=f"{socket.gethostname()}:{os.getpid()}:{worker_index}"
 while True:
  claim=store.claim(owner,budget,lease_seconds=180)
  if claim is None:return
  _execute_job(root,crop,budget,claim,owner,store,device,transition_n_jobs)

def _following_worker(root_text:str,crop:str,device:str,transition_n_jobs:int,idle_timeout_seconds:float,worker_index:int)->None:
 root=Path(root_text);out=root/"experiments/rl_policy_training/artifacts/step19_5_v2";state_path=_parallel_state_path(out,crop);store=CalibrationLeaseStore(out/crop/"calibration_jobs.sqlite3",crop);owner=f"{socket.gethostname()}:{os.getpid()}:extra-{worker_index}";idle_since=time.monotonic()
 while True:
  state=_read(state_path)
  if state.get("status") in {"passed","failed","stopped"}:return
  budget=int(state["active_budget"]);claim=store.claim(owner,budget,lease_seconds=180)
  if claim is not None:
   _execute_job(root,crop,budget,claim,owner,store,device,transition_n_jobs);idle_since=time.monotonic();continue
  if time.monotonic()-idle_since>=idle_timeout_seconds:return
  time.sleep(2)

def _spawn_workers(target,args_prefix,workers:int)->None:
 if workers<1:raise ValueError("workers must be positive")
 context=mp.get_context("spawn");processes=[context.Process(target=target,args=(*args_prefix,index)) for index in range(workers)]
 for process in processes:process.start()
 for process in processes:process.join()
 failed=[process.exitcode for process in processes if process.exitcode!=0]
 if failed:raise RuntimeError(f"Step 19.5 workers failed: {failed}")

def add_crop_calibration_workers(project_root:str|Path,crop:str,workers:int,device:str="cpu",transition_n_jobs:int=1,idle_timeout_seconds:float=3600)->Path:
 root=Path(project_root).resolve();out=root/"experiments/rl_policy_training/artifacts/step19_5_v2";state_path=_parallel_state_path(out,crop)
 if not state_path.exists() or _read(state_path).get("status")!="running":raise RuntimeError(f"{crop}: no running coordinator")
 _spawn_workers(_following_worker,(str(root),crop,device,transition_n_jobs,idle_timeout_seconds),workers)
 return state_path

def run_crop_calibration(project_root:str|Path,crop:str,device:str="cpu",transition_n_jobs:int=1,workers:int=1)->Path:
 root=Path(project_root).resolve();lock_path=root/"experiments/rl_policy_training/artifacts/step19_5_v2"/crop/"coordinator.lock";lock_path.parent.mkdir(parents=True,exist_ok=True)
 with lock_path.open("a+") as lock:
  try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as exc:raise RuntimeError(f"{crop}: coordinator is already running; use --add-workers") from exc
  try:return _run_crop_calibration_coordinator(root,crop,device,transition_n_jobs,workers)
  finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def _run_crop_calibration_coordinator(project_root:str|Path,crop:str,device:str="cpu",transition_n_jobs:int=1,workers:int=1)->Path:
 root=Path(project_root).resolve();out=root/"experiments/rl_policy_training/artifacts/step19_5_v2";config=_read(root/"experiments/rl_policy_training/configs/step19_5_calibration_v2.json")
 readiness=out/"step19_5_readiness.json";_write(readiness,{"schema_version":STEP19_5_VERSION,"status":"running_parallel","actual_calibration_started":True,"active_crop":crop,"coordinator_pid":os.getpid(),"workers":workers,"test_accessed":False})
 reps=_read(out/"representative_configurations.json")["configurations"];seeds=config["seeds"]
 probe=load_model_driven_env_from_step15(project_root=root,crop=crop,split="train");base=max(int(r["parameters"]["rollout_length"]) for r in reps)
 curves_path=out/crop/"learning_curves.json";curves=_read(curves_path)["rows"] if curves_path.exists() else []
 completed_steps=sorted({int(r["steps"]) for r in curves});budget=base if not completed_steps else completed_steps[-1]*2
 store=CalibrationLeaseStore(out/crop/"calibration_jobs.sqlite3",crop);state_path=_parallel_state_path(out,crop)
 while True:
  payloads=_job_payloads(reps,seeds,probe.observation_shape[0],budget);store.initialize(budget,payloads);store.retry_failed(budget)
  _write(state_path,{"schema_version":STEP19_5_VERSION,"status":"running","crop":crop,"active_budget":budget,"jobs":len(payloads),"coordinator_pid":os.getpid(),"primary_workers":workers,"dynamic_workers_supported":True,"test_accessed":False})
  _spawn_workers(_fixed_budget_worker,(str(root),crop,budget,device,transition_n_jobs),workers)
  while True:
   rows=store.rows(budget);failed=[row for row in rows if row["status"]=="failed"]
   if failed:
    _write(state_path,{**_read(state_path),"status":"failed","errors":[row["error"] for row in failed]});raise RuntimeError(f"{crop} budget {budget}: worker failure")
   if len(rows)==len(payloads) and all(row["status"]=="complete" for row in rows):break
   now=time.time();claimable=any(row["status"]=="pending" or (row["status"]=="running" and row["lease_until"] is not None and row["lease_until"]<now) for row in rows)
   if claimable:_spawn_workers(_fixed_budget_worker,(str(root),crop,budget,device,transition_n_jobs),1)
   else:time.sleep(5)
  existing={(int(row["steps"]),str(row["config_id"]),int(row["seed"])) for row in curves}
  for job in rows:
   row=job["result"]["row"];key=(int(row["steps"]),str(row["config_id"]),int(row["seed"]))
   if key not in existing:curves.append(row);existing.add(key)
  curves.sort(key=lambda row:(int(row["steps"]),str(row["config_id"]),int(row["seed"])))
  _write(curves_path,{"schema_version":STEP19_5_VERSION,"crop":crop,"rows":curves,"test_accessed":False})
  result=analyze_crop_learning_curves(curves,config);_write(out/crop/"calibration_analysis.json",result)
  if result["status"]=="passed":
   _write(state_path,{**_read(state_path),"status":"passed","rung_budgets":result["rung_budgets"]});return out/crop/"calibration_analysis.json"
  budget*=2

def finalize_calibration(project_root:str|Path)->Path:
 root=Path(project_root).resolve();out=root/"experiments/rl_policy_training/artifacts/step19_5_v2"
 curves={crop:_read(out/crop/"learning_curves.json")["rows"] for crop in OFFICIAL_CROPS}
 for crop,rows in curves.items():
  csv_path=out/crop/"learning_curves.csv"
  with csv_path.open("w",newline="") as stream:
   writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  analysis=_read(out/crop/"calibration_analysis.json")
  _write(out/crop/"ranking_stability_matrix.json",{"schema_version":STEP19_5_VERSION,"crop":crop,"comparisons":analysis["ranking_comparisons"]})
 budget=create_budget_artifact(root,curves)
 _write(out/"step19_5_readiness.json",{"schema_version":STEP19_5_VERSION,"status":"passed","actual_calibration_started":True,"calibrated_budget_sha256":_sha(budget),"test_accessed":False})
 return budget

__all__=["add_crop_calibration_workers","finalize_calibration","prepare_step19_5","run_crop_calibration"]
