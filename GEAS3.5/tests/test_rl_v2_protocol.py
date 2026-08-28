from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import sys
import numpy as np
import pytest
PROJECT_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT_ROOT/"src"))
from geas35.experiments.rl.step19_5_calibration import analyze_crop_learning_curves,create_budget_artifact
from geas35.experiments.rl.step20_v2 import TrialLeaseStore,load_calibrated_budget
from geas35.rl.parallel_v2 import ProcessVectorEnv,VectorEnvSpec,collect_vectorized_rollout,deterministic_episode_seed
from geas35.rl.hybrid_ppo import HybridActorCritic,HybridPPOConfig,HybridPPOTrainer

DIAGNOSTICS=("policy_loss","value_loss","approx_kl","entropy","explained_variance","gradient_norm","in_support_rate","warning_ood_rate","severe_ood_rate","projection_rate","fallback_rate","termination_rate")
def config():return json.loads((PROJECT_ROOT/"experiments/rl_policy_training/configs/step19_5_calibration_v2.json").read_text())
def rows(*,unstable=False,safety_delay=None):
 out=[];steps=(100,200,400,800,1600,3200,6400,12800);names=[f"c{i}" for i in range(5)]
 for si,step in enumerate(steps):
  for ci,name in enumerate(names):
   rank=ci if not unstable or si%2==0 else 4-ci
   # Large changes through 800, then flat enough for the 3-interval plateau at 6400.
   reward=10-rank+(4-si if si<4 else 0)*2
   for seed in (42,43,44):
    item={"config_id":name,"seed":seed,"steps":step,"split":"validation","test_accessed":False,
          "mean_reward_per_scheduled_valid_step":reward+(seed-43)*0.001,
          "safety_gate_passed":False if safety_delay and step<safety_delay else True}
    item.update({key:0.1 for key in DIAGNOSTICS});out.append(item)
 return out

def test_step15_v2_has_no_numeric_hpo_budget_and_v1_snapshot_is_intact():
 d=json.loads((PROJECT_ROOT/"experiments/rl_policy_training/configs/step15_protocol_v2.json").read_text())
 assert d["hpo"]["trials_per_crop"]==30
 assert "environment_steps_per_trial" not in d["hpo"]
 assert d["hpo"]["budget_source_step"]=="19.5"
 snapshot=json.loads((PROJECT_ROOT/"experiments/rl_policy_training/artifacts/step19_5_v2/v1_preservation_snapshot.json").read_text())
 for record in snapshot["v1_files"]:
  path=PROJECT_ROOT/record["path"]
  assert hashlib.sha256(path.read_bytes()).hexdigest()==record["sha256"]

def test_learning_curve_stability_selects_observed_rungs_and_instability_blocks():
 passed=analyze_crop_learning_curves(rows(),config())
 assert passed["status"]=="passed"
 assert len(passed["rung_budgets"]) in (2,3)
 assert all(step in passed["steps"] for step in passed["rung_budgets"])
 assert passed["rung_budgets"]==sorted(set(passed["rung_budgets"]))
 blocked=analyze_crop_learning_curves(rows(unstable=True),config())
 assert blocked["status"]=="calibration_inconclusive" and blocked["rung_budgets"]==[]

def test_budget_artifact_hash_is_required_by_step20_v2(tmp_path):
 cfg19=tmp_path/"experiments/rl_policy_training/configs/step19_5_calibration_v2.json";cfg19.parent.mkdir(parents=True);cfg19.write_text((PROJECT_ROOT/"experiments/rl_policy_training/configs/step19_5_calibration_v2.json").read_text())
 cfg20=tmp_path/"experiments/rl_policy_training/configs/step20_hpo_v2.json";cfg20.write_text((PROJECT_ROOT/"experiments/rl_policy_training/configs/step20_hpo_v2.json").read_text())
 create_budget_artifact(tmp_path,{crop:rows() for crop in ("strawberry","melon","cucumber")})
 artifact,digest=load_calibrated_budget(tmp_path)
 assert artifact["status"]=="passed" and len(digest)==64
 path=tmp_path/"experiments/rl_policy_training/artifacts/step19_5_v2/calibrated_budget.json";path.write_text(path.read_text()+" ")
 with pytest.raises(ValueError,match="hash mismatch"):load_calibrated_budget(tmp_path)

def test_atomic_lease_reclaims_crashed_worker_and_preserves_single_owner(tmp_path):
 db=TrialLeaseStore(tmp_path/"crop.sqlite3","study");db.initialize(2,0,{0:{"x":0},1:{"x":1}})
 first=db.claim("dead",0,lease_seconds=.01);assert first["trial"]==0
 time.sleep(.02);reclaimed=db.claim("replacement",0);assert reclaimed["trial"]==0 and reclaimed["attempt"]==2
 with pytest.raises(ValueError,match="ownership lost"):db.heartbeat("dead",0,0)
 db.complete("replacement",0,0,{"ok":True});second=db.claim("other",0);assert second["trial"]==1

def test_process_vector_env_matches_scalar_semantics_and_seed_mapping():
 seeds=[deterministic_episode_seed(42,"melon","c0",i,0) for i in range(2)]
 assert seeds==[deterministic_episode_seed(42,"melon","c0",i,0) for i in range(2)] and seeds[0]!=seeds[1]
 specs=[VectorEnvSpec("synthetic_counter",{"horizon":2}) for _ in range(2)]
 with ProcessVectorEnv(specs) as env:
  obs,info=env.reset(seeds);assert obs.shape==(2,1)
  obs,reward,terminated,truncated,info=env.step([[1.],[2.]])
  assert reward.tolist()==[1.,2.] and not terminated.any()
  obs,reward,terminated,truncated,info=env.step([[1.],[2.]])
  assert terminated.tolist()==[True,True]

def test_vectorized_rollout_feeds_exact_batch_to_ppo():
 specs=[VectorEnvSpec("synthetic_counter",{"horizon":2}) for _ in range(2)]
 trainer=HybridPPOTrainer(HybridActorCritic(HybridPPOConfig(observation_dim=1,hidden_sizes=(8,),update_epochs=1,minibatch_size=4)),HybridPPOConfig(observation_dim=1,hidden_sizes=(8,),update_epochs=1,minibatch_size=4))
 with ProcessVectorEnv(specs) as env:
  buffer=collect_vectorized_rollout(env,trainer,steps=4,base_seed=42,crop="melon",config_id="fixture")
 assert len(buffer)==4
 metrics=trainer.update(buffer)
 assert np.isfinite([value for value in metrics.values() if not isinstance(value,bool)]).all()
