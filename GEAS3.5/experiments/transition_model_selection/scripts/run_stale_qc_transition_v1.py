"""Explicit new namespaces; reuse frozen algorithms without editing old modules/artifacts."""
from pathlib import Path
import os,sys,json,hashlib,importlib.util,argparse,traceback,runpy
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
D=ROOT/'experiments/transition_model_selection/artifacts/command_carry_stale_b60_r6_v1';P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1';OLD=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def freeze():
 assert json.loads((P/'gate_results.json').read_text())['status']=='PASS'
 D.mkdir(exist_ok=False);p=json.loads((OLD/'protocol.json').read_text());original=sha(OLD/'protocol.json')
 p.update(version='transition.command_carry_stale_b60_r6.v1',phase0_namespace=str(P.relative_to(ROOT)),parent_protocol_sha256=original,offline_gate='research only',online_deployment_gate='BLOCKED unchanged',stale_recovery_contract='B60 causal-forward and6 changing qualifying observations recovery; operational project choice not physical truth',strawberry_interpretation='domain exclusion, not newly learned high CO2 dynamics; high CO2/high vent training coverage and legacy assumptions remain',input_hashes=[{'path':str(f.relative_to(ROOT)),'sha256':sha(f)} for f in sorted(P.rglob('*')) if f.is_file()])
 assert p['candidates']==json.loads((OLD/'protocol.json').read_text())['candidates'];write(D/'protocol.json',p);write(D/'protocol_lock.json',{'sha256':sha(D/'protocol.json')})
 print('FROZEN',sha(D/'protocol.json'),flush=True)
def configure():
 import geas35.models.transition.command_carry_b_v1 as core
 core.P=P;core.D=D
 import geas35.models.transition.evaluate_command_carry_b_v1 as evaluation
 evaluation.P=P;evaluation.D=D
 return core,evaluation
def run(crop):
 core,evaluation=configure();pro=core.protocol();assert json.loads((P/'gate_results.json').read_text())['status']=='PASS';assert json.loads((D/'batch_runtime_parity.json').read_text())['passed']
 chosen=pro['prior_selected'][crop]
 for name in [chosen]+[n for n in pro['candidates'][crop] if n!=chosen]:
  for seed in pro['candidates'][crop][name]['seeds']:
   label=name+(f'_seed_{seed}' if name=='mlp' else '')
   try:core.fresh_fit(crop,name,seed if name=='mlp' else None)
   except Exception:traceback.print_exc();continue
   try:evaluation.evaluate(crop,label)
   except Exception as e:
    traceback.print_exc();out=D/crop/label/'rollout';out.mkdir(exist_ok=True);write(out/'failure.json',{'type':type(e).__name__,'message':str(e),'protocol_sha256':sha(D/'protocol.json')})
 spec=importlib.util.spec_from_file_location('selector',ROOT/'experiments/transition_model_selection/scripts/select_command_carry_b_v1.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.D=D;mod.P=P;mod.select(crop)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['freeze','parity','run'],required=True);ap.add_argument('--crop',choices=['melon','cucumber','strawberry']);a=ap.parse_args()
 if a.stage=='freeze':freeze()
 elif a.stage=='parity':
  configure();runpy.run_path(str(ROOT/'experiments/transition_model_selection/scripts/validate_batch_sensor_b_v1.py'),run_name='__main__')
 else:run(a.crop)
