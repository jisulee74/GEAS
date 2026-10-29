from pathlib import Path
import sys,os,argparse,traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.models.transition.evaluate_command_carry_b_v1 import *
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--crop',choices=['melon','cucumber','strawberry'],required=True);a=ap.parse_args();p=protocol();chosen=p['prior_selected'][a.crop]
 for name in [chosen,'persistence']+[n for n in p['candidates'][a.crop] if n not in (chosen,'persistence')]:
  for seed in p['candidates'][a.crop][name]['seeds']:
   label=name+(f'_seed_{seed}' if name=='mlp' else '')
   if not (D/a.crop/label/'fit/completed.json').exists():continue
   try:evaluate(a.crop,label)
   except Exception as e:
    traceback.print_exc();out=D/a.crop/label/'rollout';out.mkdir(exist_ok=True);write(out/'failure.json',{'type':type(e).__name__,'message':str(e)})
