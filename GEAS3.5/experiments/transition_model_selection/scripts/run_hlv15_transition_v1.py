"""Fresh Transition Phase1/2 only, byte-identical frozen protocol + new input binding."""
from pathlib import Path
import os,sys,json,hashlib,shutil,platform,importlib.util,importlib.metadata,argparse,traceback,runpy
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'))
D=R/'experiments/transition_model_selection/artifacts/command_carry_hlv15_b60_r6_v1'
P=R/'experiments/rl_policy_training/artifacts/past_only_qc_hlv15_b60_r6_v1'
OLD=R/'experiments/transition_model_selection/artifacts/command_carry_stale_b60_r6_v1'
HASH='8a96e21c8929339f11eefd5c08b96e92bd12649507802abe2d3694d1cdb2e9f0'
ORDER=('persistence','linear_regression','linear_svr','knn','extra_trees','lightgbm','xgboost','mlp')
EXPECTED={'melon':(28713,5985,[28499,28180,27544]),'cucumber':(14192,4663,[13978,13678,13111]),'strawberry':(43909,17399,[43585,43103,42156])}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def freeze():
 import numpy as np,pandas as pd
 from geas35.rl.vent_command_memory_b_v3 import OBS,ACTIONS
 assert sha(OLD/'protocol.json')==HASH
 assert json.loads((P/'gate.json').read_text())['status']=='PASS'
 D.mkdir(exist_ok=False)
 pro=json.loads((OLD/'protocol.json').read_text())
 for x in pro['configuration_hashes']:assert sha(R/x['path'])==x['sha256']
 for rel,h in json.loads((P/'output_hashes.json').read_text()).items():assert sha(P/rel)==h,rel
 schema=json.loads((P/'schema.json').read_text());assert schema['observation_columns']==OBS and len(OBS)==55 and len(ACTIONS)==6
 populations=[];inputs={str(p.relative_to(R)):sha(p) for p in sorted(P.rglob('*')) if p.is_file()}
 for crop,(nt,nv,expected_windows) in EXPECTED.items():
  assert set(pro['candidates'][crop])==set(ORDER) and pro['candidates'][crop]['mlp']['seeds']==[42,43,44]
  for split,n in [('train',nt),('validation',nv)]:
   b=pd.read_parquet(P/crop/f'{split}_physical.parquet');reg=pd.read_parquet(P/crop/f'{split}_registry.parquet');w=pd.read_parquet(P/crop/f'{split}_windows.parquet');scaled=pd.read_parquet(P/crop/f'{split}_scaled.parquet')
   assert len(b)==len(reg)==len(scaled)==n and b.row_id.equals(reg.row_id) and b.row_id.equals(scaled.row_id)
   assert b.split.eq(split).all() and b.row_id.is_unique
   wc=[int(w.horizon.eq(h).sum()) for h in (3,6,12)]
   if split=='train':assert wc==expected_windows
   assert min(wc)>0
   populations.append(dict(crop=crop,split=split,rows=n,windows=dict(zip((3,6,12),wc)),physical_sha256=sha(P/crop/f'{split}_physical.parquet'),scaled_sha256=sha(P/crop/f'{split}_scaled.parquet'),registry_sha256=sha(P/crop/f'{split}_registry.parquet'),windows_sha256=sha(P/crop/f'{split}_windows.parquet')))
  bind=json.loads((P/crop/'binding.json').read_text());assert bind['namespace']==P.name and bind['crop']==crop and bind['scaler_sha256']==sha(P/crop/'observation_scaler.json')
 shutil.copyfile(OLD/'protocol.json',D/'protocol.json');write(D/'protocol_lock.json',dict(sha256=HASH,exact_copy_of=str((OLD/'protocol.json').relative_to(R))))
 codefiles=[Path(__file__),R/'src/geas35/models/transition/command_carry_b_v1.py',R/'src/geas35/models/transition/evaluate_command_carry_b_v1.py',R/'src/geas35/models/transition/batch_sensor_rollout_b_v1.py',R/'src/geas35/models/transition/hyperparameters.py',R/'src/geas35/models/transition/action_response.py',R/'experiments/transition_model_selection/scripts/select_command_carry_b_v1.py',R/'experiments/transition_model_selection/scripts/validate_batch_sensor_b_v1.py']
 versions={name:importlib.metadata.version(name) for name in pro['dependencies']}
 assert versions==pro['dependencies'],('dependency drift',versions,pro['dependencies'])
 binding=dict(namespace=D.name,phase0_namespace=str(P.relative_to(R)),protocol_sha256=HASH,date_kst='2026-10-08',python=platform.python_version(),dependencies=versions,input_hashes=inputs,populations=populations,
  execution_order=ORDER,execution_override='user: no prior winner preference; protocol execution/prior_selected and old input fields retained as historical provenance only; this binding defines current input paths',
  policy='candidate params/seeds/score/gates unchanged; full Validation window registry; no Test/HPO; fresh fit validation_dataset=None; MLP internal Train early-stopping only',
  one_step_gate='verify frozen all-horizon finite/zero physical gate also at one-step; no per-target negative-skill rejection',
  code_hashes={str(p.relative_to(R)):sha(p) for p in codefiles},restrictions=['no Step14/15/16','no support','no domain-v2 or C/E','no reward changes','no PPO/HPO','no Test'],current_BLOCKED=['strawberry Step16','overall Step17','online deployment'])
 write(D/'execution_binding.json',binding);write(D/'execution_lock.json',dict(sha256=sha(D/'execution_binding.json')))
 print('FROZEN',D.name,'protocol',HASH,'populations',populations,flush=True)

def configure():
 import geas35.models.transition.command_carry_b_v1 as core
 import geas35.models.transition.evaluate_command_carry_b_v1 as ev
 core.D=D;core.P=P;ev.D=D;ev.P=P
 original=ev.evaluate
 def evaluate(crop,label,repeat=False):
  result=original(crop,label,repeat=repeat)
  fit=json.loads((D/crop/label/('repeat' if repeat else 'fit')/'completed.json').read_text())
  gate=fit['one_step_nan_inf']==0 and fit['one_step_physical_violation_rate']==0
  result['one_step_finite_physical_gate']=gate
  result['eligible']=bool(result['eligible'] and gate)
  core.write(D/crop/label/('repeat_rollout' if repeat else 'rollout')/'completed.json',result)
  return result
 ev.evaluate=evaluate
 return core,ev

def verify_lock():
 assert sha(D/'protocol.json')==sha(OLD/'protocol.json')==HASH
 assert sha(D/'execution_binding.json')==json.loads((D/'execution_lock.json').read_text())['sha256']
 b=json.loads((D/'execution_binding.json').read_text())
 for rel,h in b['input_hashes'].items():assert sha(R/rel)==h,rel
 for rel,h in b['code_hashes'].items():assert sha(R/rel)==h,rel
 return b

def run(crop):
 binding=verify_lock();core,ev=configure();assert json.loads((D/'batch_runtime_parity.json').read_text())['passed']
 pro=core.protocol()
 for name in ORDER:
  for seed in pro['candidates'][crop][name]['seeds']:
   label=name+(f'_seed_{seed}' if name=='mlp' else '')
   try:
    core.fresh_fit(crop,name,seed if name=='mlp' else None)
    ev.evaluate(crop,label)
   except Exception as exc:
    traceback.print_exc();dest=D/crop/label
    if not (dest/'fit/failure.json').exists():
     out=dest/'rollout';out.mkdir(exist_ok=True);core.write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),protocol_sha256=HASH))
 spec=importlib.util.spec_from_file_location('hlv15_selector',R/'experiments/transition_model_selection/scripts/select_command_carry_b_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.D=D;m.P=P
 try:m.select(crop)
 except Exception as exc:
  traceback.print_exc();core.write(D/crop/'selection.json',dict(crop=crop,status='BLOCKED',selected=None,approval_reason='repeat/reload or selection gate exception',error=repr(exc),protocol_sha256=HASH))
 verify_lock()
 write(D/crop/'execution_complete.json',dict(crop=crop,protocol_sha256=HASH,input_hashes_unchanged=True,old_weights_used=False,Test_accessed=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['freeze','parity','run'],required=True);ap.add_argument('--crop',choices=tuple(EXPECTED));a=ap.parse_args()
 if a.stage=='freeze':freeze()
 elif a.stage=='parity':
  verify_lock();configure();runpy.run_path(str(R/'experiments/transition_model_selection/scripts/validate_batch_sensor_b_v1.py'),run_name='__main__')
 else:run(a.crop)
