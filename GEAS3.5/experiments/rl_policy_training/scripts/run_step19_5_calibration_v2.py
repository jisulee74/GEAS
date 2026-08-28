#!/usr/bin/env python3
import argparse,os
from pathlib import Path
import sys
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[variable]="1"
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"src"))
from geas35.experiments.rl.step19_5_runner import add_crop_calibration_workers,finalize_calibration,prepare_step19_5,run_crop_calibration
def main():
 p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--prepare",action="store_true");mode.add_argument("--run-crop",action="store_true");mode.add_argument("--add-workers",action="store_true");mode.add_argument("--finalize",action="store_true")
 p.add_argument("--crop",choices=("strawberry","melon","cucumber"));p.add_argument("--device",default="cpu");p.add_argument("--transition-n-jobs",type=int,default=1);p.add_argument("--workers",type=int,default=1);p.add_argument("--idle-timeout-seconds",type=float,default=3600);a=p.parse_args()
 if a.prepare:print(prepare_step19_5(ROOT))
 elif a.finalize:print(finalize_calibration(ROOT))
 elif a.add_workers:
  if not a.crop:p.error("--add-workers requires --crop")
  print(add_crop_calibration_workers(ROOT,a.crop,a.workers,a.device,a.transition_n_jobs,a.idle_timeout_seconds))
 else:
  if not a.crop:p.error("--run-crop requires --crop")
  print(run_crop_calibration(ROOT,a.crop,a.device,a.transition_n_jobs,a.workers))
if __name__=="__main__":main()
