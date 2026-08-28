#!/usr/bin/env python3
from __future__ import annotations
import argparse,os,sys
from pathlib import Path
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable]="1"
PROJECT_ROOT=Path(__file__).resolve().parents[3]
SRC_ROOT=PROJECT_ROOT/'src'
if str(SRC_ROOT) not in sys.path: sys.path.insert(0,str(SRC_ROOT))
from geas35.experiments.rl.step16_environment import run_step16_environment_handoff

def main(argv=None):
    parser=argparse.ArgumentParser(description='Run GEAS RL Step 16 environment handoff.')
    parser.add_argument('--project-root',default=str(PROJECT_ROOT)); parser.add_argument('--output-root')
    args=parser.parse_args(argv)
    result=run_step16_environment_handoff(project_root=args.project_root,output_root=args.output_root)
    print(f'Step 16 integrity manifest: {result}')
    return 0
if __name__=='__main__': raise SystemExit(main())
