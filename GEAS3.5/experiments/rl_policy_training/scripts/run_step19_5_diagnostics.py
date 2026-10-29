#!/usr/bin/env python3
"""Train/Validation-only diagnostics; deliberately no training or Test options."""
import argparse
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from geas35.experiments.rl.step19_5_diagnostics import CROPS,STAGES,run_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=(*STAGES,'all'),required=True)
    parser.add_argument('--crop',choices=(*CROPS,'all'),default='all')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    args=parser.parse_args();crops=('cucumber','melon','strawberry') if args.crop=='all' else (args.crop,)
    if args.stage=='all':
        run_stage(ROOT,'inventory',device=args.device)
        # One crop and one worker prove the evaluator path before broad scheduling.
        run_stage(ROOT,'evaluator-parity',(crops[0],),args.device,1)
        if len(crops)>1:run_stage(ROOT,'evaluator-parity',crops[1:],args.device,args.workers)
        for stage in ('rollout-diagnosis','learning-analysis','report'):
            print(run_stage(ROOT,stage,crops,args.device,args.workers),flush=True)
    else:print(run_stage(ROOT,args.stage,crops,args.device,args.workers),flush=True)


if __name__=='__main__':main()
