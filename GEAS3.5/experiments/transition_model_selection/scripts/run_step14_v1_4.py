#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(key,"1")
PROJECT_ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(PROJECT_ROOT/"src"))
from geas35.experiments.transition.deployment_handoff_v1_4 import run_step14_v1_4
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--decision-author",default="researcher"); p.add_argument("--step13-manifest"); p.add_argument("--output-root"); a=p.parse_args()
    print(run_step14_v1_4(project_root=PROJECT_ROOT,decision_author=a.decision_author,step13_manifest_path=a.step13_manifest,output_root=a.output_root))
