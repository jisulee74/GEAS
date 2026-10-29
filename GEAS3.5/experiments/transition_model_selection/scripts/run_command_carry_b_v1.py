from pathlib import Path
import sys,os,argparse,traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.models.transition.command_carry_b_v1 import *
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--crop',choices=['melon','cucumber','strawberry'],required=True);a=ap.parse_args();p=protocol();chosen=p['prior_selected'][a.crop]
 for name in [chosen]+[n for n in p['candidates'][a.crop] if n!=chosen]:
  for seed in p['candidates'][a.crop][name]['seeds']:
   try:fresh_fit(a.crop,name,seed if name=='mlp' else None)
   except Exception:traceback.print_exc()
