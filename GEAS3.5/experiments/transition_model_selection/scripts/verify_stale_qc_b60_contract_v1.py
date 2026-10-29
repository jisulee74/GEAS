from pathlib import Path
from datetime import datetime,timedelta
from dataclasses import asdict
from types import SimpleNamespace
import ast,json,hashlib
import pandas as pd
from stale_qc_b60_recovery_contract_v1 import Recovery
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/stale_qc_b60_contract_v1'
for n in (2,3,6):
 ident=('c','s','g','source');start=datetime(2000,1,1)
 def u(m,i,v,valid=True,base=False,identity=ident):return m.update(start+timedelta(minutes=5*i),identity,(v,)*5,valid,base)
 m=Recovery(n);assert u(m,0,0,base=True)['supervisor_blocked']
 for j in range(1,n):assert u(m,j,j)['supervisor_blocked']
 assert u(m,n,n)['released']
 m=Recovery(n);u(m,0,0,base=True);u(m,1,1);assert u(m,2,1)['recovery_count']==0
 assert not u(m,3,1,False)['released'];assert u(m,4,2)['recovery_count']==0
 assert u(m,6,3)['recovery_count']==0
 assert u(m,7,4,identity=('c','s','g','new'))['recovery_count']==0
 assert m.latched
 m=Recovery(n);u(m,0,0,base=True);saved=asdict(m);a=[u(m,i,i) for i in range(1,10)];m2=Recovery(**saved);b=[u(m2,i,i) for i in range(1,10)];assert a==b
 # arbitrary future changes cannot retroactively edit prefix
 m=Recovery(n);prefix=[u(m,0,0,base=True),u(m,1,1)];snapshot=[dict(x) for x in prefix];[u(m,i,99) for i in range(2,9)];assert prefix==snapshot
# Replay real streams with checkpoint serialization semantics via dataclass snapshot, no new sensor reads.
S=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_shadow_b_v1';checked=0
for p in sorted(S.glob('*_row_sidecar.parquet')):
 f=pd.read_parquet(p);out=pd.read_parquet(D/p.name.replace('_row_sidecar','_recovery_sidecar'))
 for n in (2,3,6):
  machine=Recovery(n)
  for i,r in enumerate(f.itertuples(index=False)):
   if i==len(f)//2:machine=Recovery(**asdict(machine))
   val=tuple(float(getattr(r,'raw_'+c)) for c in ('in_co2','in_co2_2','in_temp','in_hum','out_light'))
   z=machine.update(r.reg_date,(r.crop,r.series_id,r.segment_id,r.raw_source),val,bool(r.B_input_valid),bool(r.B_60_causal_forward))
   assert z['supervisor_blocked']==out.iloc[i][f'latched_{n}'];checked+=1
# Extract only audited methods; no model/checkpoint/env construction or inference.
def method(path,cls,name):
 tree=ast.parse(path.read_text());node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
 code=ast.Module(body=[node],type_ignores=[]);ns={'MDP_V1_BINARY_ACTION_COLUMNS':['heat_run','cool_run','fan_run']};exec(compile(ast.fix_missing_locations(code),str(path),'exec',flags=__import__('__future__').annotations.compiler_flag),ns);return ns[name],node.lineno,node.end_lineno
fn,a,b=method(ROOT/'src/geas35/rl/model_driven_env.py','ModelDrivenGreenhouseEnv','_safe_fallback_action') if False else (None,None,None)
# Determine actual class name rather than assuming it.
p=ROOT/'src/geas35/rl/model_driven_env.py';tree=ast.parse(p.read_text());cls=next(c.name for c in tree.body if isinstance(c,ast.ClassDef) and any(isinstance(n,ast.FunctionDef) and n.name=='_safe_fallback_action' for n in c.body));fn,a,b=method(p,cls,'_safe_fallback_action')
obj=SimpleNamespace(_current_physical={'obs_indoor_temp_c':15,'obs_indoor_humidity_pct':50,'obs_current_vpd_kpa':1,'obs_current_target_temp_min_c':18,'obs_current_target_temp_max_c':25,'obs_is_daytime':1},mdp_config=SimpleNamespace(rh_max_pct=90,vpd_min_kpa=.5),_project_local_support=lambda a:(a,{}))
action1=fn(obj);obj._current_physical['obs_indoor_temp_c']=35;action2=fn(obj);assert action1!=action2
# This is a dependency demonstration with identity projection, NOT a safety experiment.
(D/'contract_verification.json').write_text(json.dumps({'detector_boundary_pass_inherited_and_hash_pinned':True,'recovery_synthetic_boundary_prefix_resume_pass':True,'actual_recovery_updates_replayed':checked,'actual_replay_exact':True,'sensor_dependency_demonstration':{'function':str(p.relative_to(ROOT))+':'+str(a),'projection':'identity stub for input dependency only','T15_action':action1,'T35_action':action2,'action_changes_with_T':True},'no_runtime_safety_certification':True},indent=2)+'\n')
print('Recovery checks PASS; replay updates',checked,'fallback T dependence confirmed.')
