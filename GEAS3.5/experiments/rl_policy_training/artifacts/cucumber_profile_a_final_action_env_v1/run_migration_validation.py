"""Gated legacy replay, seven-case regression replay, then full Development.
No training, no default dispatch mutation; immutable selected checkpoint inputs.
"""
from pathlib import Path
import json,sys,platform,importlib.metadata,hashlib,copy,traceback,xml.etree.ElementTree as ET
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from geas35.rl.profile_a_offline_v1 import ProfileAOfflineEnv, OfflineCollector, ACT
from geas35.rl.profile_a_final_action_env_v1 import FinalActionProfileAEnv,VERSION
from geas35.rl.profile_a_main_trainer_v1 import config,act,value_fn,equal,clean,write,EXPERIMENT_HASH
from geas35.rl.hybrid_ppo import HybridActorCritic
O=Path(__file__).resolve().parent;R=O.parents[3];A=R/'experiments/rl_policy_training/artifacts';E=A/'cucumber_profile_a_final_action_shadow_v1/execution_v1'
H='73516c9044969a1825fe6782fe1696bd5d0bdc23efd05207c26f3bd7062aa343'
READS=[];PARAMS=[];BINDINGS=[]
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def verify():
 c=load(A/'cucumber_profile_a_final_action_migration_review_v1/migration_contract.json');h=c.pop('contract_payload_sha256');assert h==H==digest(c)
 for x in c['inputs']:assert sha(R/x['path'])==x['sha256'],x['path']
 e=c['regression_gates']['canonical_environment'];assert sys.executable==e['sys_executable'] and platform.python_version()==e['python'];assert {k:importlib.metadata.version(k) for k in e['dependencies']}==e['dependencies']
 return c

def safe(x):
 if isinstance(x,pd.Series):return safe(x.to_dict())
 if isinstance(x,pd.Timestamp):return str(x)
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [safe(v) for v in x]
 return clean(x)
def hist(e):return digest(safe({k:v for k,v in vars(e.runtime.adapter.updater).items() if k in ['_history','_sensor_history','_last_action','_last_last_action']}))
def phash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):
  a=v.detach().cpu().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def guards():
 st=ExitStack();reader=pd.read_parquet
 def read(p,*a,**k):
  assert Path(p).name in ['train_physical.parquet','train_sensor_prehistory.parquet'],str(p);READS.append(str(p));return reader(p,*a,**k)
 st.enter_context(patch('pandas.read_parquet',side_effect=read))
 for n in ['torch.optim.Adam.step','torch.optim.SGD.step','torch.autograd.backward','torch.autograd.grad','torch.save','torch.optim.lr_scheduler.LRScheduler.step']:
  st.enter_context(patch(n,side_effect=AssertionError('FORBIDDEN '+n)))
 st.enter_context(torch.no_grad());return st

def make_model(s,u):
 cfg,_=config(load(A/'cucumber_profile_a_ppo_experiment_prereg_v1/ppo_experiment_contract.json'),s)
 x=torch.load(A/f'cucumber_profile_a_ppo_main_v1/seed{s}/update{u}.pt',map_location='cpu',weights_only=False)
 assert x['experiment_hash']==EXPERIMENT_HASH and x['update']==u
 m=HybridActorCritic(cfg).to(dtype=torch.float32,device='cpu');m.load_state_dict(x['policy']);m.eval()
 return m,x['metadata']
NEW_FIELDS={'environment_semantic_version','proposal_joint_distance','proposal_joint_status','proposal_severe_streak','proposal_streak_before','final_streak_before','final_streak_last_valid','counter_authority','final_joint_distance','final_joint_status','final_severe_streak','final_support_unavailable_reason'}
def baseinfo(info):
 i=copy.deepcopy(info)
 for k in NEW_FIELDS:i.pop(k,None)
 i['order']=[x for x in i['order'] if x!='final_joint_ood']
 return i

def run(env,m,s,u,rid,phase,expected=None,old_account=None,previous=None):
 ob,_=env.reset(rid,s);col=OfflineCollector(env);fcount=0;capture={};rows=[];events=[];initial=phash(m)
 orig=env.runtime.advance
 def advance(action,exo):
  capture['input']=env.runtime.adapter.model_input(action).to_dict()
  assert len(capture['input'])==62
  assert not any(k.startswith('obs_indoor_') for k in exo.index)
  result=orig(action,exo);return result
 env.runtime.advance=advance
 try:
  with (O/f'{phase}_seed{s}_{rid}.jsonl').open('x') as file:
   while not env.done:
    j=env.cursor;row=env.run.iloc[j];state={k:float(env.runtime.adapter.current[k]) for k in env.obs};history_before=hist(env);scaled=pd.DataFrame([ob],columns=env.obs)
    a,lp,v=act(m,ob,True);cb=env.counter;fb=fcount;capture.clear()
    ob,reward,_,_,info=col.step(a,lp,v,lambda o:value_fn(m,o));events.append(info)
    assert info['category'] not in ['SYSTEM_INVALID','UNSCORABLE'],(phase,s,rid,j,info)
    final=info.get('final_action');fd=None
    if final is not None:
     q=env.support.scores(scaled,pd.DataFrame([final],columns=ACT));assert float(q['state'][0])==info['state_distance'];fd=float(q['joint'][0]);assert np.isfinite(fd)
     fcount=fcount+1 if fd>env.support.config['thresholds']['joint_severe'] else 0
     if isinstance(env,FinalActionProfileAEnv):assert fcount==env.final_counter and fd==info['final_joint_distance']
    nxt={k:float(env.runtime.adapter.current[k]) for k in env.obs} if info.get('prediction') is not None else None
    t=safe({'seed':s,'update':u,'run_id':rid,'step':j,'decision_timestamp':str(row.reg_date),'outcome_timestamp':str(row.target_timestamp) if nxt is not None else None,'current_state':state,'next_state':nxt,'proposal_log_prob':lp,'value':v,'reward':reward,'info':info,'proposal_counter_before':cb,'proposal_counter_after':env.counter,'final_counter_before':fb,'final_counter_after':fcount if fd is not None else None,'final_counter_last_valid':fcount,'final_joint_distance':fd,'transition_input':capture.get('input'),'history_before_hash':history_before,'history_after_hash':hist(env) if nxt is not None else None})
    if expected is not None:
     ex=expected[j];assert equal(state,ex['current_state']) and equal(clean(info),ex['info']),(phase,s,rid,j,'HISTORICAL_EXACT')
     assert env.counter==ex['consecutive_joint_severe'] and str(row.reg_date)==ex['timestamp']
    if previous is not None:
     assert equal(t,previous[j]),(phase,s,rid,j,'DETERMINISM')
    if nxt is not None:
     assert np.isfinite(list(nxt.values())).all()
     for name,val in zip(['obs_prev_'+k for k in ACT],final):assert np.isclose(nxt[name],val,rtol=1e-6,atol=1e-8)
    file.write(json.dumps(t,ensure_ascii=False,allow_nan=False)+'\n');rows.append(t)
    assert len(rows)<=len(env.run)
  st=env.staging
  acc={'run_id':rid,'parent_episode_id':str(env.run.iloc[0].episode_id),'scheduled_steps':len(env.run),'category':st.category,'admitted':len(st.rows),'numeric_return':st.numeric_return,'attempted':len(rows),'binding':env.checkpoint_metadata(),'events':st.events}
  if old_account is not None:assert equal(acc,old_account),(s,rid,'ACCOUNTING_EXACT')
  if expected is not None:assert len(rows)==len(expected)
  assert initial==phash(m)
  PARAMS.append({'phase':phase,'seed':s,'run_id':rid,'before':initial,'after':phash(m),'identical':True})
  write(O/f'{phase}_seed{s}_{rid}_accounting.json',acc)
  print(phase,s,rid[:10],st.category,'steps',len(st.rows),flush=True)
  return rows,acc
 finally:env.runtime.advance=orig

def parity_until_divergence(old,new):
 first=None
 for i in range(min(len(old),len(new))):
  a=copy.deepcopy(old[i]);b=copy.deepcopy(new[i]);a['info']=baseinfo(a['info']);b['info']=baseinfo(b['info'])
  if a['info'].get('reason')=='executed_joint_support_failure':a['info'].pop('reason')
  if b['info'].get('reason')=='executed_joint_support_failure':b['info'].pop('reason')
  ca,cb=a['info']['category'],b['info']['category'];a['info']['category']=None;b['info']['category']=None
  assert equal(a,b),(i,'COMMON_PREFIX_EXACT')
  if ca!=cb:
   first=old[i]['decision_timestamp'];break
 return first

def main():
 c=verify();write(O/'runtime_input_gate.json',{'PASS':True,'environment':c['regression_gates']['canonical_environment'],'migration_payload':H,'new_implementation_hash':sha(R/'src/geas35/rl/profile_a_final_action_env_v1.py')})
 xml=ET.parse(O/'synthetic_regressions.xml');tests=xml.findall('.//testcase');assert len(tests)==15 and not xml.findall('.//failure') and not xml.findall('.//error')
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 results={r['name']:{'PASS':True,'evidence':'synthetic_regressions.xml'} for r in c['post_migration_regressions'] if r['name'] not in ['legacy_exact','semantic_divergence_parity','deterministic_replay']}
 legacy_traces={};reg_new={};accs=[]
 with guards():
  models={};metadata={}
  for s,u in [(42,8),(43,8),(44,4)]:models[s],metadata[s]=make_model(s,u)
  legacy=ProfileAOfflineEnv(R,'development');assert len(legacy.cohort.runs)==25 and sum(map(len,legacy.cohort.runs.values()))==2811
  for s,u in [(42,8),(43,8),(44,4)]:
   historical=[json.loads(l) for l in (A/f'cucumber_profile_a_policy_review_v1/trace_seed{s}_update{u}.jsonl').open()];by={}
   for t in historical:by.setdefault(t['run_id'],[]).append(t)
   reference=load(A/f'cucumber_profile_a_policy_review_v1/replay_seed{s}_update{u}.json')
   for rid in sorted(legacy.cohort.runs):
    tr,acc=run(legacy,models[s],s,u,rid,'legacy',by[rid],next(x for x in reference['runs'] if x['run_id']==rid));legacy_traces[s,rid]=tr
  results['legacy_exact']={'PASS':True,'seed_runs':75,'historical_info_states_accounting_exact':True}
  write(O/'regression_progress.json',results)
  new=FinalActionProfileAEnv(R,semantic_version=VERSION)
  for s,u in [(42,8),(43,8),(44,4)]:
   path=A/f'cucumber_profile_a_ppo_main_v1/seed{s}/update{u}.pt'
   try:new.validate_resume(metadata[s]);raise AssertionError('legacy accepted')
   except ValueError as e:assert 'SEMANTIC_MISMATCH' in str(e)
   b=new.frozen_policy_evaluation_binding(path,seed=s,update=u,original_training_binding=metadata[s]);BINDINGS.append(b)
  shadow=load(A/'cucumber_profile_a_final_action_shadow_v1/final_action_shadow_contract.json')
  for case in shadow['primary_population']:
   s,u,rid=case['seed'],case['update'],case['run_id'];tr,ac=run(new,models[s],s,u,rid,'regression1');parity_until_divergence(legacy_traces[s,rid],tr)
   # Prior seven-case shadow exact physical/reward/counter agreement.
   prior=[json.loads(l) for l in (E/f"S_{case['case_id']}.jsonl").open()];assert len(prior)==len(tr)
   for x,y in zip(prior,tr):
    for k in ['current_state','next_state','proposal_log_prob','reward','transition_input','proposal_counter_after','final_counter_after']:assert equal(x[k],y[k]),(rid,k)
   fresh=FinalActionProfileAEnv(R,semantic_version=VERSION)
   repeat,acc2=run(fresh,models[s],s,u,rid,'regression2',previous=tr);assert equal(ac,acc2)
   reg_new[s,rid]=tr
  results['semantic_divergence_parity']={'PASS':True,'seven_cases':7,'includes_shared_legacy_stop':True,'prior_shadow_physical_parity':True}
  results['deterministic_replay']={'PASS':True,'independent_instances':True,'seven_selected_seed_cases_twice_exact':True,'accounting_exact':True}
  assert len(results)==18 and all(v['PASS'] for v in results.values());verify()
  write(O/'regression_results.json',{'PASS':True,'passed':18,'total':18,'results':results,'reevaluation_started_yet':False})
  print('18/18 REGRESSIONS PASS; FULL DEVELOPMENT NOW AUTHORIZED',flush=True)
  for s,u in [(42,8),(43,8),(44,4)]:
   for rid in sorted(new.cohort.runs):
    tr,ac=run(new,models[s],s,u,rid,'development');div=parity_until_divergence(legacy_traces[s,rid],tr)
    if (s,rid) in reg_new:assert equal(tr,reg_new[s,rid]),'SHADOW_CASE_FULL_EVAL_DISCREPANCY'
    accs.append({'seed':s,'update':u,'run_id':rid,'accounting':ac,'first_semantic_divergence_timestamp':div})
  verify()
 write(O/'cross_environment_evaluation_bindings.json',BINDINGS)
 write(O/'parameter_integrity.json',{'PASS':True,'checks':PARAMS,'frozen_inputs_unchanged':True})
 write(O/'development_reevaluation.json',{'PASS':True,'seed_runs':len(accs),'results':accs,'selected_policy_parameters_unchanged':True})
 write(O/'execution_guard_report.json',{'PASS':True,'parquet_reads':sorted(set(READS)),'no_optimizer':True,'no_grad':True,'no_checkpoint_write':True,'no_Test':True,'no_Validation_policy_evaluation':True,'no_P0_live':True,'legacy_source_unchanged':True})
 print('FULL75 COMPLETE; ALL GATES PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(O/'technical_failure.json',{'verdict':'FINAL-ACTION ENV v1 — BLOCKED','error':str(e),'traceback':traceback.format_exc()});raise
