from pathlib import Path
import sys,os,argparse,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.models.transition.command_carry_b_v1 import *
from geas35.models.transition.evaluate_command_carry_b_v1 import evaluate

def select(crop):
 pro=protocol();rows=[];failure=[]
 for name,spec in pro['candidates'][crop].items():
  choices=[]
  for seed in spec['seeds']:
   label=name+(f'_seed_{seed}' if name=='mlp' else '');p=D/crop/label/'rollout/completed.json'
   if not p.exists():
    assert (D/crop/label/'fit/failure.json').exists() or (D/crop/label/'rollout/failure.json').exists(),('incomplete',crop,label)
    failure.append(label);continue
   x=json.loads(p.read_text());
   if x['eligible']:choices.append(x)
  if choices:
   choices.sort(key=lambda x:(x['score'],x['seed'] or 0));rows.append(choices[len(choices)//2] if name=='mlp' else choices[0])
 baseline=json.loads((D/crop/'persistence/rollout/completed.json').read_text());rows.sort(key=lambda x:(x['score'],x['label']));leader=rows[0] if rows else None
 selection={'crop':crop,'status':'BLOCKED','leader':leader['label'] if leader else None,'selected':None,'eligible_ranking':[{'label':r['label'],'score':r['score']} for r in rows],'persistence_score':baseline['score'],'failures':failure,'protocol_sha256':sha(D/'protocol.json'),'approval_reason':None}
 if leader is None:selection['approval_reason']='no eligible candidate'
 elif leader['score']>=baseline['score']:selection['approval_reason']='frozen Persistence weighted-score improvement guard failed'
 else:
  label=leader['label'];name=leader['name'];seed=leader['seed'] if name=='mlp' else None
  repeat=fresh_fit(crop,name,seed,repeat=True);dest=D/crop/label
  a=pd.read_parquet(dest/'fit/validation_predictions.parquet').to_numpy(float);b=pd.read_parquet(dest/'repeat/validation_predictions.parquet').to_numpy(float)
  np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-8)
  rr=evaluate(crop,label,repeat=True)
  with np.load(dest/'rollout/predictions.npz') as p1,np.load(dest/'repeat_rollout/predictions.npz') as p2:
   np.testing.assert_array_equal(p1['start_row_id'],p2['start_row_id']);np.testing.assert_array_equal(p1['max_horizon'],p2['max_horizon']);np.testing.assert_allclose(p1['predictions'],p2['predictions'],rtol=1e-6,atol=1e-8,equal_nan=True)
  assert rr['eligible'] and np.isclose(rr['score'],leader['score'],rtol=1e-6,atol=1e-8)
  selection.update(status='PASS',selected=label,approval_reason='minimum eligible fixed score; improves Persistence; fresh same-seed full-window repeat and reload verified',reproducibility={'one_step_max_abs_diff':float(np.max(np.abs(a-b))),'rollout_score_diff':rr['score']-leader['score'],'model_hash_equal':repeat['model_sha256']==leader['model_sha256'],'repeat_model_sha256':repeat['model_sha256'],'rtol':1e-6,'atol':1e-8},model_sha256=leader['model_sha256'])
 write(D/crop/'selection.json',selection);print(crop,selection['status'],selection['selected'],selection['approval_reason'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--crop',choices=['melon','cucumber','strawberry'],required=True);a=ap.parse_args();select(a.crop)
