"""Validation recorded-command simulation only. No training/Test entry point."""
from pathlib import Path
import os,sys,json,gzip,argparse,hashlib
from dataclasses import asdict
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from geas35.rl.step16_b60_r6_env import B60R6ResearchEnv,VERSION
from geas35.rl.model_driven_env import ModelDrivenEnvConfig
from geas35.rl.mdp_v1 import MdpV1Config,compute_mdp_v1_reward,MDP_V1_REWARD_TERM_KEYS,_reward_penalty_label
from geas35.models.transition.handoff_b60_r6_v1 import sha,SELECTED
D=ROOT/'experiments/rl_policy_training/artifacts/step16_b60_r6_v1';S=ROOT/'experiments/rl_policy_training/artifacts/step15_b60_r6_v1';H=ROOT/'experiments/transition_model_selection/artifacts/step14_b60_r6_v1';P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1'
def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n');tmp.replace(p)
def freeze():
 assert read(S/'final_gate.json')['status']=='PASS';D.mkdir(exist_ok=False)
 inputs=[]
 for folder in (S,H,P):
  for p in sorted(folder.rglob('*')):
   if p.is_file():inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 paths=[Path(__file__),ROOT/'src/geas35/rl/step16_b60_r6_env.py',ROOT/'src/geas35/rl/model_driven_env.py',ROOT/'src/geas35/rl/mdp_v1.py',ROOT/'src/geas35/rl/sensor_history_env_v4.py',ROOT/'src/geas35/rl/causal_contract_env_v2.py']
 pro={'version':VERSION,'selected':SELECTED,'inputs':inputs,'code':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in paths],'env_config':asdict(ModelDrivenEnvConfig()),'reward_config':asdict(MdpV1Config()),'split':'validation_only','repetitions':2,'parity_tolerance':{'rtol':1e-6,'atol':1e-8},'registry':'all eligible transition rows exactly once, separate contiguous mdp rollout IDs; terminal target included','practical_rollout_guard':{'minimum_executed_scheduled_ratio':.5,'minimum_episode_fraction_reaching_15min_or_full_short_run':.5,'at_least_one_60min_run_if_scheduled_exists':True,'interpretation':'prospectively fixed Step16 research operability guard, not OOD threshold or reward relaxation'},'physical_bounds':[[-60,80],[0,100],[0,10000]],'abrupt_change_diagnostic_only':[5,20,500],'OOD_attribution':'recorded-severe overlap vs rollout-only; rollout-only after changed actions cannot isolate pure model error','raw_stale_on_simulation':False,'online_deployment':'BLOCKED','no_training_Test_HPO':True}
 write(D/'protocol.json',pro);write(D/'protocol_lock.json',{'sha256':sha(D/'protocol.json')});print('FROZEN',flush=True)
def check():
 pro=read(D/'protocol.json');assert sha(D/'protocol.json')==read(D/'protocol_lock.json')['sha256']
 for r in pro['inputs']+pro['code']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
 return pro

def run(crop):
 pro=check();out=D/crop;out.mkdir(exist_ok=True);env=B60R6ResearchEnv(root=ROOT,crop=crop);obs=list(env.observation_columns);act=list(env.action_columns);terms=[_reward_penalty_label(x) for x in MDP_V1_REWARD_TERM_KEYS]
 recorded=pd.read_parquet(S/crop/'validation_support_scores.parquet').set_index('row_id');thresholds=env.support.config['thresholds'];epresults=[];allrewards=[];contributions=[];actions=[];physical=[];resetchecks=[]
 for index,episode in enumerate(env.episodes):
  file=out/'episodes'/f'{index:04d}.json';tracefile=file.with_suffix('.jsonl.gz');file.parent.mkdir(exist_ok=True)
  if file.exists():
   saved=read(file);assert saved['protocol_sha256']==sha(D/'protocol.json') and saved['trace_sha256']==sha(tracefile);epresults.append(saved);continue
  repeats=[];trace=[]
  for repeat in range(2):
   _,reset=env.reset(seed=42,options={'episode_index':index});initial=env._current_physical[obs].to_numpy(float).copy();done=False;rows=[];altered=0
   # Changing current proposed action cannot change reset observation/history.
   x=env._current_scaled.copy();old=x[obs].to_numpy(float).copy();x[act]=1-x[act].to_numpy(float);np.testing.assert_array_equal(x[obs].to_numpy(float),old)
   while not done:
    pos=env._position;source=env.frame.iloc[pos];current=env._current_physical.copy();scaled=env._current_scaled[obs].to_numpy(float).copy();proposed=source[act].to_numpy(float);prev=dict(env._last_action);pp=dict(env._last_last_action);record=recorded.loc[source.row_id]
    beforesteps=env._executed_steps;_,reward,terminated,truncated,info=env.step(proposed);done=terminated or truncated;executed=env._executed_steps>beforesteps;row={'episode_index':index,'episode_id':episode['episode_id'],'rollout_id':episode['rollout_id'],'step':beforesteps,'row_id':str(source.row_id),'timestamp':str(source.reg_date),'current_physical':current[obs].to_numpy(float).tolist(),'current_scaled':scaled.tolist(),'proposed':proposed.tolist(),'reward':reward,'recorded_state_distance':float(record.state_distance),'recorded_joint_distance':float(record.joint_distance),'recorded_state_severe':bool(record.state_severe),'info':info,'executed':executed,'previous_action':prev,'previous_previous_action':pp,'changed_actions_before':altered}
    assert np.isfinite(reward)
    if executed:
     final=np.array([info['executed_action'][a] for a in act]);altered+=int(not np.allclose(final,proposed,rtol=0,atol=1e-12))
     mi=env.model.last_input;np.testing.assert_allclose(mi[obs].to_numpy(float)[0],scaled,rtol=1e-6,atol=1e-8);np.testing.assert_array_equal(mi[act].to_numpy(float)[0],final)
     np.testing.assert_array_equal(env._current_physical[['obs_prev_'+a for a in act]].to_numpy(float),final)
     base,rt=compute_mdp_v1_reward(env._current_physical,info['executed_action'],prev_action=prev,prev_prev_action=pp,config=env.mdp_config)
     np.testing.assert_allclose(base,info['base_reward'],rtol=1e-6,atol=1e-8)
     for term in terms:np.testing.assert_allclose(rt[term],info['reward_terms'][term],rtol=1e-6,atol=1e-8)
     expected=max(env.env_config.worst_step_reward,base-info['warning_ood_penalty'])
     if info['joint_ood_level']=='severe':expected=env.env_config.worst_step_reward
     expected+=info['pessimistic_terminal_cost'];np.testing.assert_allclose(reward,expected,rtol=1e-6,atol=1e-8)
     prediction=np.array([info['transition_prediction_physical'][c] for c in obs[:3]]);actual=env.scaler.inverse_transform_frame(env.frame.iloc[[env._position]])[obs[:3]].to_numpy(float)[0];nextstate=env._current_physical[obs].to_numpy(float)
     row.update(final=final.tolist(),prediction=prediction.tolist(),next_physical=nextstate.tolist(),recorded_next=actual.tolist(),step_delta=(prediction-current[obs[:3]].to_numpy(float)).tolist(),recorded_path_difference=(prediction-actual).tolist(),initial_drift=(prediction-initial[:3]).tolist(),action_changed=not np.allclose(final,proposed,rtol=0,atol=1e-12))
     assert np.isfinite(prediction).all() and np.isfinite(nextstate).all()
    else:np.testing.assert_allclose(reward,-env._remaining_steps(),rtol=0,atol=0)
    rows.append(row)
   repeats.append(rows)
  assert len(repeats[0])==len(repeats[1]);np.testing.assert_allclose([r['reward'] for r in repeats[0]],[r['reward'] for r in repeats[1]],rtol=1e-6,atol=1e-8)
  for a,b in zip(*repeats):
   assert a['info']['termination_reason']==b['info']['termination_reason'] and a['executed']==b['executed']
   for key in ('current_physical','current_scaled','proposed','prediction','next_physical','final'):
    if key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-8)
   for key in ('state_ood_score','joint_ood_score'):np.testing.assert_allclose(a['info'][key],b['info'][key],rtol=1e-6,atol=1e-8)
  trace=repeats[0]
  with gzip.open(tracefile,'wt',encoding='utf-8') as stream:
   for row in trace:stream.write(json.dumps(row,default=str)+'\n')
  last=trace[-1];n=sum(r['executed'] for r in trace);reason=last['info']['termination_reason'];classification=None
  if reason=='severe_state_ood':classification='recorded_state_already_severe' if last['recorded_state_severe'] else ('rollout_only_after_modified_actions' if last['changed_actions_before'] else 'rollout_only_unchanged_recorded_actions')
  result={'episode_index':index,**episode,'original_episode_id':episode['episode_id'],'executed_steps':n,'termination_reason':reason,'OOD_classification':classification,'recorded_state_severe_at_termination':last['recorded_state_severe'],'reward_sum':sum(r['reward'] for r in trace),'terminal_cost':sum(r['info'].get('pessimistic_terminal_cost',0) for r in trace),'fallback_steps':sum(r['info'].get('safety_fallback_applied',False) for r in trace),'changed_action_steps':sum(r.get('action_changed',False) for r in trace),'repeat_parity':True,'protocol_sha256':sha(D/'protocol.json'),'trace_sha256':sha(tracefile)}
  write(file,result);epresults.append(result);write(out/'status.json',{'status':'running','complete_episodes':index+1,'total_episodes':len(env.episodes)});print(crop,index+1,'/',len(env.episodes),n,'/',episode['scheduled_valid_steps'],reason,flush=True)
 # Aggregate exclusively from durable traces, including reused complete jobs.
 for e in epresults:
  with gzip.open(out/'episodes'/f"{e['episode_index']:04d}.jsonl.gz",'rt') as stream:rows=[json.loads(line) for line in stream]
  for row in rows:
   info=row['info'];common={'crop':crop,'episode_index':e['episode_index'],'step':row['step'],'row_id':row['row_id']}
   allrewards.append({**common,'executed':row['executed'],'reward':row['reward'],'terminal_cost':info.get('pessimistic_terminal_cost',0),'warning_penalty':info.get('warning_ood_penalty',0),'base_reward':info.get('base_reward',0),**{t:info.get('reward_terms',{}).get(t,0) for t in terms}})
   if row['executed']:
    changes=info.get('support_projection',{}).get('changes',{})
    for j,a in enumerate(act):actions.append({**common,'action':a,'proposed':row['proposed'][j],'final':row['final'][j],'changed':not np.isclose(row['proposed'][j],row['final'][j],rtol=0,atol=1e-12),'local_support_limited':a in changes,'local_projected':changes.get(a,{}).get('to',row['proposed'][j]),'fallback':info.get('safety_fallback_applied',False)})
    for j,target in enumerate(obs[:3]):physical.append({**common,'target':target,'prediction':row['prediction'][j],'step_delta':row['step_delta'][j],'initial_drift':row['initial_drift'][j],'recorded_path_difference':row['recorded_path_difference'][j],'physical_violation':not pro['physical_bounds'][j][0]<=row['prediction'][j]<=pro['physical_bounds'][j][1],'abrupt_flag':abs(row['step_delta'][j])>pro['abrupt_change_diagnostic_only'][j],'horizon_minutes':(row['step']+1)*5,'changed_actions_before':row['changed_actions_before']})
   if info['state_ood_level']=='severe' or info['joint_ood_level']=='severe':
    state=np.array(row['current_scaled']);weighted=state*env.support.state_weights;_,neighbors=env.support._state_nn.kneighbors(weighted[None,:]);k=neighbors[0,-1];components=((state-env.support.reference_state_raw[k])*env.support.state_weights)**2;order=np.argsort(components)[::-1][:10]
    for j in order:contributions.append({**common,'timestamp':row['timestamp'],'termination_reason':info['termination_reason'],'feature':obs[j],'scaled_value':state[j],'physical_value':row['current_physical'][j],'neighbor_scaled_value':env.support.reference_state_raw[k,j],'squared_distance_component':components[j],'component_fraction':components[j]/components.sum() if components.sum() else 0,'state_distance':info['state_ood_score'],'state_threshold_ratio':info['state_ood_score']/thresholds['state_severe'],'joint_distance':info['joint_ood_score'],'joint_threshold_ratio':info['joint_ood_score']/thresholds['joint_severe'],'recorded_state_distance':row['recorded_state_distance'],'recorded_joint_distance':row['recorded_joint_distance'],'recorded_already_severe':row['recorded_state_severe'],'changed_actions_before':row['changed_actions_before']})
 ep=pd.DataFrame(epresults);ep.to_csv(out/'episode_summary.csv',index=False);reward=pd.DataFrame(allrewards);reward.to_parquet(out/'reward_steps.parquet',index=False);action=pd.DataFrame(actions);action.to_parquet(out/'action_steps.parquet',index=False);phys=pd.DataFrame(physical);phys.to_parquet(out/'physical_steps.parquet',index=False);pd.DataFrame(contributions).to_csv(out/'ood_kth_neighbor_contributions.csv',index=False)
 rewardsummary=[];validreward=reward.loc[reward.executed]
 for term in terms+['warning_penalty','terminal_cost','reward']:
  x=validreward[term] if term in terms else reward[term];den=validreward[terms].sum().sum();rewardsummary.append({'term':term,'n':len(x),'mean':x.mean(),'p50':x.quantile(.5),'p90':x.quantile(.9),'p99':x.quantile(.99),'min':x.min(),'max':x.max(),'sum':x.sum(),'six_term_absolute_share':x.sum()/den if term in terms and den else None})
 pd.DataFrame(rewardsummary).to_csv(out/'reward_summary.csv',index=False)
 action.groupby('action').agg(attempts=('changed','size'),projection_or_fallback_change_rate=('changed','mean'),local_support_limited_rate=('local_support_limited','mean'),fallback_rate=('fallback','mean')).to_csv(out/'action_summary.csv')
 phys.groupby('target').agg(rows=('prediction','size'),minimum=('prediction','min'),maximum=('prediction','max'),physical_violations=('physical_violation','sum'),abrupt_changes=('abrupt_flag','sum'),max_abs_step_change=('step_delta',lambda x:x.abs().max())).to_csv(out/'physical_summary.csv')
 phys.loc[phys.horizon_minutes.isin([15,30,60])].groupby(['target','horizon_minutes']).agg(rows=('prediction','size'),mean_signed_drift=('initial_drift','mean'),mean_absolute_drift=('initial_drift',lambda x:x.abs().mean()),recorded_path_rmse=('recorded_path_difference',lambda x:np.sqrt(np.mean(x*x)))).to_csv(out/'horizon_drift.csv')
 coverage=ep.executed_steps.sum()/ep.scheduled_valid_steps.sum();reached=np.mean(ep.executed_steps>=np.minimum(ep.scheduled_valid_steps,3));long_ok=not (ep.scheduled_valid_steps>=12).any() or (ep.executed_steps>=12).any();physical_ok=not phys.physical_violation.any();practical=coverage>=.5 and reached>=.5 and long_ok
 result={'crop':crop,'environment_contract':'PASS','official_binding':'PASS','reproducibility':'PASS','reward_recomputed_finite':'PASS','physical_sanity':'PASS' if physical_ok else 'BLOCKED','episode_runs':len(ep),'original_episodes':int(ep.original_episode_id.nunique()),'scheduled_steps':int(ep.scheduled_valid_steps.sum()),'executed_steps':int(ep.executed_steps.sum()),'executed_fraction':float(coverage),'fraction_reaching_15min_or_short_run_end':float(reached),'termination_counts':ep.termination_reason.value_counts().to_dict(),'state_OOD_classification':ep.OOD_classification.value_counts().to_dict(),'fallback_steps':int(ep.fallback_steps.sum()),'changed_action_steps':int(ep.changed_action_steps.sum()),'terminal_cost_sum':float(ep.terminal_cost.sum()),'reward_sum':float(ep.reward_sum.sum()),'physical_violations':int(phys.physical_violation.sum()),'nan_inf':0,'practical_rollout_gate':'PASS' if practical else 'BLOCKED','Step17_entry':'PASS' if practical and physical_ok else 'BLOCKED','online_deployment':'BLOCKED','threshold_reward_support_changed':False}
 write(out/'result.json',result);write(out/'status.json',{'status':'complete'});print('COMPLETE',crop,result,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);ap.add_argument('--crop',choices=list(SELECTED));a=ap.parse_args()
 if a.stage=='freeze':freeze()
 else:run(a.crop)
