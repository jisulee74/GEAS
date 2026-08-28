from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT_ROOT/"src"))
from geas35.experiments.rl.step20_v2 import STEP20_V2_VERSION,TrialLeaseStore,prepare_step20_v2

def _lease_worker(path,owner):
 store=TrialLeaseStore(path,"stress")
 while True:
  claim=store.claim(owner,0,lease_seconds=5)
  if claim is None:return
  store.complete(owner,claim["trial"],0,{"owner":owner})

def test_trial_workers_claim_every_trial_exactly_once(tmp_path):
 path=tmp_path/"crop.sqlite3";store=TrialLeaseStore(path,"stress");store.initialize(50,0)
 context=mp.get_context("spawn");workers=[context.Process(target=_lease_worker,args=(path,f"w{i}")) for i in range(4)]
 for worker in workers:worker.start()
 for worker in workers:worker.join()
 assert [worker.exitcode for worker in workers]==[0,0,0,0]
 rows=store.rows();assert len(rows)==50 and all(row["status"]=="complete" for row in rows)
 assert len({row["trial"] for row in rows})==50

def test_crop_databases_are_physically_independent_and_no_training_starts(tmp_path):
 # Recreate only the minimum config tree needed for readiness preparation.
 source=PROJECT_ROOT/"experiments/rl_policy_training/configs/step20_hpo_v2.json"
 target=tmp_path/"experiments/rl_policy_training/configs/step20_hpo_v2.json";target.parent.mkdir(parents=True);target.write_text(source.read_text())
 readiness=prepare_step20_v2(tmp_path,require_calibrated_budget=False)
 assert readiness.name=="step20_v2_readiness.json"
 paths=[readiness.parent/crop/"hpo_study.sqlite3" for crop in ("strawberry","melon","cucumber")]
 assert all(path.exists() for path in paths) and len({path.resolve() for path in paths})==3
 for crop,path in zip(("strawberry","melon","cucumber"),paths):
  assert TrialLeaseStore(path,f"{STEP20_V2_VERSION}.{crop}").rows()==[]
