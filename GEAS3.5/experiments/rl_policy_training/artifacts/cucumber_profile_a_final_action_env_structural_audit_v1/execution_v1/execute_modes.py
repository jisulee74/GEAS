from pathlib import Path
import sys,json,hashlib,platform,importlib.metadata,copy,traceback
R=Path('/NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/GEAS/GEAS3.5');O=Path(__file__).resolve().parent;P=O.parent;A=R/'experiments/rl_policy_training/artifacts';E=A/'cucumber_profile_a_final_action_env_v1';OLD=A/'cucumber_profile_a_failure_isolation_v1/execution_v1'
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(E));sys.path.insert(0,str(OLD))
from run_migration_validation import guards,make_model,phash,hist,safe,READS
from run_modes import meta,next_record,state_distance,pipeline
from geas35.rl.profile_a_final_action_env_v1 import FinalActionProfileAEnv,VERSION
from geas35.rl.profile_a_offline_v1 import ACT,OfflineCollector
from geas35.rl.profile_a_main_trainer_v1 import act,value_fn,clean,equal
import numpy as np,pandas as pd,torch
H='070a6e7622b1ddb85630ec201263eaf10655e327dd35b4c728417d4db811401c'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(O/n).write_text(json.dumps(safe(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def lines(p):return [json.loads(l) for l in Path(p).open()]
def emit(f,x):f.write(json.dumps(safe(x),ensure_ascii=False,allow_nan=False)+'\n')
def verify():
 c=load(P/'structural_audit_contract.json');h=c.pop('contract_payload_sha256');assert H==h==hashlib.sha256(json.dumps(c,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 for x in c['authoritative_inputs']:assert sha(R/x['path'])==x['sha256'],x['path']
 for n,h in c['companion_sha256'].items():assert sha(P/n)==h
 e=c['environment'];assert sys.executable==e['sys_executable'] and platform.python_version()==e['python'];assert {k:importlib.metadata.version(k) for k in e['dependencies']}==e['dependencies']
 return c

def main():
 c=verify();write('input_gate.json',{'PASS':True,'payload':H,'frozen_inputs':419,'environment':c['environment']});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 cases=load(P/'failure_provenance_matrix.json')['cases'];write('failure_provenance_matrix.json',cases)
 reuse=[];params=[];bindings=[]
 with guards():
  env=FinalActionProfileAEnv(R,semantic_version=VERSION);th=env.support.config['thresholds'];assert th==c['frozen_thresholds']
  # R0 exact recorded scan includes all fixed COMPLETE reference, no policy or model.
  with (O/'recorded_nodes.jsonl').open('x') as f:
   for rid,frame in sorted(env.cohort.runs.items()):
    if len(frame)>1:np.testing.assert_allclose(frame.iloc[:-1][['next_'+k for k in env.obs]].to_numpy(float),frame.iloc[1:][env.obs].to_numpy(float),rtol=1e-6,atol=1e-8)
    for j,row in frame.iterrows():
     emit(f,{'run_id':rid,'parent_episode_id':str(row.episode_id),'crop':str(row.crop),'series':str(row.series_id),'segment':str(row.segment_id),'timestamp':str(row.reg_date),'state':{k:float(row[k]) for k in env.obs},'state_distance':state_distance(env,row),'endpoint_only':False})
    row=next_record(frame.iloc[-1],env.obs);emit(f,{'run_id':rid,'parent_episode_id':str(row.episode_id),'crop':str(row.crop),'series':str(row.series_id),'segment':str(row.segment_id),'timestamp':str(row.reg_date),'state':{k:float(row[k]) for k in env.obs},'state_distance':state_distance(env,row),'endpoint_only':True})
  for seed,u in [(42,8),(43,8),(44,4)]:
   model,metadata=make_model(seed,u);before=phash(model);bindings.append(env.frozen_policy_evaluation_binding(A/f'cucumber_profile_a_ppo_main_v1/seed{seed}/update{u}.pt',seed=seed,update=u,original_training_binding=metadata))
   oldf=lines(OLD/f'F_seed{seed}.jsonl')
   for case in [x for x in cases if x['seed']==seed and x['group']!='C']:
    rid=case['run_id'];fv=lines(E/f'development_seed{seed}_{rid}.jsonl');legacy=lines(E/f'legacy_seed{seed}_{rid}.jsonl');prior=[t for t in oldf if t['run_id']==rid];ft=fv[-1]['decision_timestamp'];frame=env.cohort.runs[rid];assert ft in set(frame.reg_date.astype(str))
    parity=None
    if case['group']=='A':
     assert len(fv)==len(prior)==len(legacy)
     for x,y,z in zip(fv,prior,legacy):
      for k in ['current_state','next_state','decision_timestamp']:assert equal(x[k],y[k])
      for k in ['current_state','next_state','history_before_hash','history_after_hash','transition_input']:assert equal(x[k],z[k])
      for k in ['proposed','projected_action','fallback_action','intermediate_action','final_action','state_distance','joint_distance']:assert equal(x['info'].get(k),y['info'].get(k))
     parity=True
    reuse.append({'seed':seed,'run_id':rid,'group':case['group'],'prior_F_exact_prefix':parity,'migrated_legacy_history_digest_parity':parity,'old_diagnostic_full_buffer_provenance_proven':False,'reuse':False,'reason':'Old R1/TF1/TF4 logs do not persist full reset/rolling buffer provenance; recompute only state-case diagnostic paths through Fv1 failure. Fv1 remains stored and unchanged.'})
    # Fresh R1/TF1, no recurrence; no new Fv1.
    with (O/f'R1_seed{seed}_{rid}.jsonl').open('x') as f1,(O/f'TF1_seed{seed}_{rid}.jsonl').open('x') as f2:
     pc=fc=0
     for j,row in frame.iterrows():
      if str(row.reg_date)>ft:break
      current=row.copy();ob=env.runtime.scaler.transform_frame(current.to_frame().T)[env.obs].iloc[0].to_numpy(float);action,_,_=act(model,ob,True);i=pipeline(env,current,action)
      final=i['final_action'];jd=None if final is None else float(env.support.scores(env.runtime.scaler.transform_frame(current.to_frame().T)[env.obs],pd.DataFrame([final],columns=ACT))['joint'][0]);pc=pc+1 if i['joint_distance']>th['joint_severe'] else 0;fc=fc+1 if jd is not None and jd>th['joint_severe'] else 0
      t=meta(env,row,rid,seed,'R1',j,current);t.update(info=i,final_joint_distance=jd,diagnostic_proposal_counter=pc,diagnostic_final_counter=fc);emit(f1,t)
      env.runtime.reset(row,env.cohort.history);s=env.runtime.adapter.current.copy();np.testing.assert_allclose(s[env.obs].to_numpy(float),row[env.obs].to_numpy(float),rtol=1e-6,atol=1e-8)
      ob=env.runtime.scaler.transform_frame(s.to_frame().T)[env.obs].iloc[0].to_numpy(float);aa,_,_=act(model,ob,True);assert equal(action.tolist(),aa.tolist());ii=pipeline(env,s,aa);assert equal(i['final_action'],ii['final_action'])
      t=meta(env,row,rid,seed,'TF1',j,s);t.update(info=ii,history_before_hash=hist(env),next_state=None,next_state_distance=None,outcome_timestamp=None,status=ii['reason'])
      if ii['final_action'] is not None:
       pred,state=env.runtime.advance(dict(zip(ACT,ii['final_action'])),env.runtime.adapter.exogenous(next_record(row,env.obs)));t.update(next_state={k:float(state[k]) for k in env.obs},next_state_distance=state_distance(env,state),outcome_timestamp=str(row.target_timestamp),history_after_hash=hist(env),status='ONE_STEP_DISCARDED')
      emit(f2,t)
    # TF4 own migrated policy, commands and history. No model prediction.
    tf=FinalActionProfileAEnv(R,semantic_version=VERSION);calls=[0]
    def inject(action,exogenous):
     row=tf.run.iloc[tf.cursor];pred=pd.Series({k:float(row['next_'+k]) for k in c['target_order']});calls[0]+=1;state=tf.runtime.adapter.advance(prediction=pred.to_dict(),action=action,exogenous=exogenous);np.testing.assert_allclose(state[c['target_order']].to_numpy(float),pred.to_numpy(float),rtol=1e-6,atol=1e-8);return pred,state
    tf.runtime.advance=inject
    from unittest.mock import patch
    with (O/f'TF4_seed{seed}_{rid}.jsonl').open('x') as f,patch.object(tf.runtime.model,'predict',side_effect=AssertionError('TF4_MODEL_FORBIDDEN')):
     ob,_=tf.reset(rid,seed);collector=OfflineCollector(tf)
     while not tf.done:
      j=tf.cursor;row=tf.run.iloc[j]
      if str(row.reg_date)>ft:break
      s=tf.runtime.adapter.current.copy();hb=hist(tf);a,lp,v=act(model,ob,True);n=calls[0];ob,reward,_,_,ii=collector.step(a,lp,v,lambda o:value_fn(model,o));has=ii.get('prediction') is not None;assert calls[0]-n==int(has);assert ii['category'] not in ['SYSTEM_INVALID','UNSCORABLE'],ii
      t=meta(tf,row,rid,seed,'TF4',j,s);t.update(info=ii,history_before_hash=hb,history_after_hash=hist(tf) if has else None,next_state={k:float(tf.runtime.adapter.current[k]) for k in tf.obs} if has else None,next_state_distance=state_distance(tf,tf.runtime.adapter.current) if has else None,outcome_timestamp=str(row.target_timestamp) if has else None,synthetic=True);emit(f,t)
    assert before==phash(model);print('DIAGNOSTICS',seed,rid[:10],case['group'],'through',ft,flush=True)
   params.append({'seed':seed,'before':before,'after':phash(model),'identical':before==phash(model)})
  verify()
 write('reuse_parity_validation.json',reuse);write('policy_parameter_integrity.json',params);write('policy_bindings.json',bindings);write('mode_gate.json',{'PASS':True,'new_Fv1':False,'new_diagnostic_rollout':True,'no_grad':True,'read_paths':sorted(set(READS)),'optimizer':False,'checkpoint_write':False});print('MODES PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write('technical_failure.json',{'gate':'diagnostic_modes','error':str(e),'traceback':traceback.format_exc()});raise
