"""Frozen seven-case isolated diagnostic; authoritative source remains unchanged."""
from pathlib import Path
import json,sys,platform,importlib.metadata,hashlib,inspect,textwrap,copy,traceback
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import geas35.rl.profile_a_offline_v1 as module
from geas35.rl.profile_a_offline_v1 import ProfileAOfflineEnv,OfflineCollector,ACT
from geas35.rl.profile_a_main_trainer_v1 import config,act,value_fn,equal,clean,write,EXPERIMENT_HASH
from geas35.rl.hybrid_ppo import HybridActorCritic
O=Path(__file__).resolve().parent;P=O.parent;R=P.parents[3];A=R/'experiments/rl_policy_training/artifacts'
EXPECTED='b78010c734ab6952569633d762bae8e903e2c8af354f482cce9fc54ddf41a64d'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def verify():
 c=load(P/'final_action_shadow_contract.json');h=c.pop('contract_payload_sha256');assert h==EXPECTED==digest(c)
 assert len(c['authoritative_inputs'])==55
 for x in c['authoritative_inputs']:assert sha(R/x['path'])==x['sha256'],x['path']
 e=c['environment'];assert sys.executable==e['sys_executable'] and platform.python_version()==e['python']
 assert {k:importlib.metadata.version(k) for k in e['dependencies']}==e['dependencies']
 return c
READS=[]
def guards():
 st=ExitStack();reader=pd.read_parquet
 def read(p,*args,**kwargs):
  assert Path(p).name in ['train_physical.parquet','train_sensor_prehistory.parquet'],str(p)
  READS.append(str(p));return reader(p,*args,**kwargs)
 st.enter_context(patch('pandas.read_parquet',side_effect=read))
 for name in ['torch.optim.Adam.step','torch.optim.SGD.step','torch.autograd.backward','torch.autograd.grad','torch.save']:
  st.enter_context(patch(name,side_effect=AssertionError('FORBIDDEN '+name)))
 st.enter_context(torch.no_grad());return st

def params(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):
  a=v.detach().cpu().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def model_for(seed,u):
 cfg,_=config(load(A/'cucumber_profile_a_ppo_experiment_prereg_v1/ppo_experiment_contract.json'),seed)
 x=torch.load(A/f'cucumber_profile_a_ppo_main_v1/seed{seed}/update{u}.pt',map_location='cpu',weights_only=False)
 assert x['experiment_hash']==EXPERIMENT_HASH and x['update']==u
 m=HybridActorCritic(cfg).to(dtype=torch.float32,device='cpu');m.load_state_dict(x['policy']);m.eval();return m

# Only two textual insertions to frozen step: pure final-query hook, counter selector.
source=textwrap.dedent(inspect.getsource(ProfileAOfflineEnv._step))
needle="final = np.array([weather[k] for k in ACT])"
assert source.count(needle)==1
instrumented=source.replace(needle,needle+"\n    self._observe_final(scaled, final)")
needle2="('FAILURE' if self.counter >= 3 else None)";assert instrumented.count(needle2)==1
shadow_source=instrumented.replace(needle2,"('FAILURE' if self.final_counter >= 3 else None)")
ns=dict(vars(module));exec(compile(instrumented,str(O/'legacy_instrumented_step.py'),'exec'),ns);legacy_step=ns['_step']
ns=dict(vars(module));exec(compile(shadow_source,str(O/'shadow_step.py'),'exec'),ns);shadow_step=ns['_step']
class DiagnosticEnv(ProfileAOfflineEnv):
 def _observe_final(self,scaled,final):
  score=self.support.scores(scaled,pd.DataFrame([final],columns=ACT))
  sd,jd=float(score['state'][0]),float(score['joint'][0]);assert np.isfinite([sd,jd]).all()
  self.final_counter=self.final_counter+1 if jd>self.support.config['thresholds']['joint_severe'] else 0
  self.final_query={'state_distance':sd,'joint_distance':jd,'counter_after':self.final_counter}

def safe(x):
 if isinstance(x,pd.Series):return safe(x.to_dict())
 if isinstance(x,pd.Timestamp):return str(x)
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 return clean(x)
def history(env):
 u=env.runtime.adapter.updater
 return {k:safe(v) for k,v in vars(u).items() if k in ['_history','_sensor_history','_last_action','_last_last_action']}

def run(case,branch,model,legacy=None):
 env=DiagnosticEnv(R,'development');env._step=(legacy_step if branch=='L' else shadow_step).__get__(env,DiagnosticEnv)
 ob,_=env.reset(case['run_id'],case['seed']);env.final_counter=0;env.final_query=None;collector=OfflineCollector(env)
 original_advance=env.runtime.advance;capture={}
 def advance(action,exogenous):
  capture['input']=env.runtime.adapter.model_input(action).to_dict();capture['exogenous']=safe(exogenous)
  result=original_advance(action,exogenous);capture['history_after']=history(env);return result
 env.runtime.advance=advance
 original_exo=env.runtime.adapter.exogenous
 def exo(frame):
  x=original_exo(frame)
  assert not any(k in x.index for k in env.runtime.adapter.obs if k.startswith('obs_indoor_'))
  return x
 env.runtime.adapter.exogenous=exo
 historical=[json.loads(l) for l in (A/f"cucumber_profile_a_failure_isolation_v1/execution_v1/F_seed{case['seed']}.jsonl").open() if json.loads(l)['run_id']==case['run_id']]
 trace=[];path=O/f"{branch}_{case['case_id']}.jsonl";before=params(model)
 with path.open('x') as f:
  while not env.done:
   i=env.cursor;row=env.run.iloc[i];current={k:float(env.runtime.adapter.current[k]) for k in env.obs};hb=history(env)
   proposal,lp,v=act(model,ob,True);pb=env.counter;fb=env.final_counter;capture.clear();env.final_query=None
   ob,reward,_,_,info=collector.step(proposal,lp,v,lambda o:value_fn(model,o))
   if info['category'] in ['SYSTEM_INVALID','UNSCORABLE']:raise AssertionError((case['case_id'],branch,i,info))
   pred=info.get('prediction');nxt={k:float(env.runtime.adapter.current[k]) for k in env.obs} if pred is not None else None
   t=safe({'case_id':case['case_id'],'branch':branch,'step':i,'decision_timestamp':str(row.reg_date),'outcome_timestamp':str(row.target_timestamp) if pred is not None else None,'current_state':current,'next_state':nxt,'proposal_log_prob':lp,'value':v,'reward':reward,'info':info,'proposal_counter_before':pb,'proposal_counter_after':env.counter,'final_counter_before':fb,'final_counter_after':env.final_counter if env.final_query is not None else None,'final_counter_last_valid':env.final_counter,'final_query':env.final_query,'transition_input':capture.get('input'),'exogenous':capture.get('exogenous'),'history_before':hb,'history_after':capture.get('history_after')})
   if branch=='L':
    assert i<len(historical);ex=historical[i]
    for k,actual in [('current_state',current),('next_state',nxt),('info',clean(info))]:assert equal(actual,ex[k]),(case['case_id'],i,k)
    assert env.counter==ex['consecutive_joint_severe'] and str(row.reg_date)==ex['timestamp']
    if 'policy_log_prob' in ex:assert lp==ex['policy_log_prob']
   elif i<=case['legacy_stop_step_zero_based']:
    ex=copy.deepcopy(legacy[i]);now=copy.deepcopy(t)
    for z in [ex,now]:z.pop('branch')
    if i==case['legacy_stop_step_zero_based']:
     assert ex['info']['category']=='FAILURE' and now['info']['category'] is None
     ex['info']['category']=None
    assert equal(ex,now),(case['case_id'],i,'PRE_DIVERGENCE_EXACT')
   if pred is not None:
    assert len(capture['input'])==62 and len(nxt)==56
    assert np.isfinite(list(nxt.values())).all()
    for k,a in zip(['obs_prev_'+k for k in ACT],info['final_action']):assert np.isclose(nxt[k],a,rtol=1e-6,atol=1e-8)
   f.write(json.dumps(t,ensure_ascii=False,allow_nan=False)+'\n');f.flush();trace.append(t)
   assert len(trace)<=case['scheduled_transitions']
   if len(trace)%50==0:print(branch,case['seed'],case['run_id'][:10],len(trace),'steps',flush=True)
 s=env.staging
 accounting={'run_id':case['run_id'],'parent_episode_id':str(env.run.iloc[0].episode_id),'scheduled_steps':len(env.run),'category':s.category,'admitted':len(s.rows),'numeric_return':s.numeric_return,'attempted':len(trace),'binding':env.checkpoint_metadata(),'events':s.events}
 if branch=='L':
  reference=load(A/f"cucumber_profile_a_policy_review_v1/replay_seed{case['seed']}_update{case['update']}.json")
  assert equal(accounting,next(x for x in reference['runs'] if x['run_id']==case['run_id'])),('ACCOUNTING',case['case_id'])
  assert len(trace)==len(historical) and trace[-1]['decision_timestamp']==case['legacy_stop_decision'] and trace[-1]['outcome_timestamp']==case['legacy_stop_outcome_timestamp']
 assert before==params(model)
 write(O/f"{branch}_{case['case_id']}_accounting.json",accounting)
 print(branch,case['seed'],case['run_id'][:10],s.category,len(s.rows),'DONE',flush=True)
 return trace,{'case_id':case['case_id'],'branch':branch,'before':before,'after':params(model),'identical':True}

def main():
 c=verify();write(O/'input_gate.json',{'PASS':True,'contract_payload_sha256':EXPECTED,'environment':c['environment'],'verified_input_hashes':55})
 for n,s in [('legacy_instrumented_step.py',instrumented),('shadow_step.py',shadow_source)]:
  (O/n).write_text(s)
 write(O/'implementation_delta.json',{'parent_step_sha256':hashlib.sha256(source.encode()).hexdigest(),'legacy_delta':'insert pure final support observer; no decision change','shadow_delta':'same observer; category expression self.counter replaced with self.final_counter only','parent_source_unchanged':True})
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 legacy={};cert=[]
 with guards():
  models={(x['seed'],x['update']):model_for(x['seed'],x['update']) for x in c['selected_policies']}
  for case in c['primary_population']:
   tr,ha=run(case,'L',models[case['seed'],case['update']]);legacy[case['case_id']]=tr;cert.append(ha)
  verify();write(O/'legacy_parity_validation.json',{'PASS':True,'cases':list(legacy),'count':7,'all_historical_states_info_next_and_accounting_exact':True})
  for case in c['primary_population']:
   tr,ha=run(case,'S',models[case['seed'],case['update']],legacy[case['case_id']]);cert.append(ha)
  verify()
 write(O/'parameter_hash_integrity.json',{'PASS':True,'parameters':cert,'all_55_input_hashes_unchanged':True,'actual_before_after_tensor_digest':True})
 write(O/'execution_gate.json',{'PASS':True,'legacy_parity':True,'shadow_predivergence_parity':True,'parquet_reads':sorted(set(READS)),'optimizer_executed':False,'Test_accessed':False,'authoritative_environment_modified':False})
 print('ALL SEVEN L/S GATES PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(O/'technical_failure.json',{'verdict':'FINAL-ACTION SHADOW — BLOCKED','error':str(e),'traceback':traceback.format_exc()});raise
