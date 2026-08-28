#!/usr/bin/env python3
import argparse,os
from pathlib import Path
import sys
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[variable]="1"
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"src"))
from geas35.experiments.rl.step20_v2 import prepare_step20_v2
from geas35.experiments.rl.step20_v2_runner import finalize_step20_v2,run_crop_hpo_v2
def main():
 p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--prepare",action="store_true");mode.add_argument("--run-crop",action="store_true");mode.add_argument("--finalize",action="store_true")
 p.add_argument("--crop",choices=("strawberry","melon","cucumber"));p.add_argument("--device",default="cpu");p.add_argument("--trial-workers",type=int,default=1);p.add_argument("--vector-envs",type=int,default=1);p.add_argument("--transition-n-jobs",type=int,default=1);a=p.parse_args()
 if a.prepare:print(prepare_step20_v2(ROOT,require_calibrated_budget=True))
 elif a.finalize:print(finalize_step20_v2(ROOT))
 else:
  if not a.crop:p.error("--run-crop requires --crop")
  print(run_crop_hpo_v2(ROOT,a.crop,a.device,a.trial_workers,a.transition_n_jobs,a.vector_envs))
if __name__=="__main__":main()
