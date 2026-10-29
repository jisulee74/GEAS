"""Read-only Step16 trace/reward audit. Never instantiate an environment/model."""
from pathlib import Path
import sys,json,gzip,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.mdp_v1 import MdpV1Config,compute_mdp_v1_reward,MDP_V1_CONTINUOUS_ACTION_COLUMNS,MDP_V1_BINARY_ACTION_COLUMNS
D=ROOT/'experiments/rl_policy_training/artifacts/step16_hlv15_b60_r6_v1'
O=ROOT/'experiments/rl_policy_training/artifacts/step17_entry_audit_hlv15_v1'
S=ROOT/'experiments/rl_policy_training/artifacts/step15_hlv15_b60_r6_v1'
CROPS=['melon','cucumber','strawberry'];TERMS=['temp','rh','vpd','cond','hv','act']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
def table(f):
 f=f.copy()
 for c in f:
  if pd.api.types.is_float_dtype(f[c]):f[c]=f[c].map(lambda x:f'{x:.6g}')
 return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None)])
def stats(x):
 x=pd.Series(x,dtype=float)
 return {'n':len(x),'min':x.min(),'p1':x.quantile(.01),'p10':x.quantile(.1),'p50':x.quantile(.5),'p90':x.quantile(.9),'p99':x.quantile(.99),'max':x.max(),'mean':x.mean()}
def save(name,rows):
 f=pd.DataFrame(rows);f.to_csv(O/(name+'.csv'),index=False);return f
O.mkdir(exist_ok=False)
protocol={'scope':'read-only existing Step16 traces and pure reward recomputation; no rollout/model loading/fit/Test',
 'gate_before_audit':{'OOD_ENV':'retain fixed Step16 guard; unexplained divergence must not structurally prevent trajectories; replay is not evidence of PPO policy coverage','implementation':'recomputed and independently checked reward matches trace; no nonfinite inputs','reward':'BLOCKED if key objectives effectively inactive and saturated single penalty plus terminal dominates, as requested; no new tuned numerical admission cutoff','entry_labels':['STEP17 ENTRY PASS — offline research only','STEP17 ENTRY BLOCKED — REWARD','STEP17 ENTRY BLOCKED — OOD/ENV','STEP17 ENTRY BLOCKED — MULTIPLE']},
 'diagnostics':{'zero':'exact zero','saturation':'raw normalized temp >=1-1e-12; numerical comparison only','near_unique':'round to 8 decimals; descriptive only','action_events':['projection proposed-final change','executed action vs previous executed command change'],'lag':'same run executed observations after event, lags1/3/6; no boundary crossing; right censor reported','contribution':'six penalties pre-clipping; abs sum / abs final return is accounting ratio not causal additive attribution'},
 'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file()],
 'code':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [Path(__file__),ROOT/'src/geas35/rl/mdp_v1.py',ROOT/'src/geas35/rl/model_driven_env.py']], 'online_deployment':'BLOCKED'}
write(O/'protocol.json',protocol);write(O/'protocol_lock.json',{'sha256':sha(O/'protocol.json')})
pro=read(D/'protocol.json');assert sha(D/'protocol.json')==read(D/'protocol_lock.json')['sha256']
for r in pro['inputs']+pro['code']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
for name,digest in read(D/'artifact_integrity.json').items():assert sha(D/name)==digest
cfg=MdpV1Config(**pro['reward_config']);allsteps=[];states=[];ood=[];survive=[];terminal=[];events=[];actiontables=[];diagnostics=[]
for crop in CROPS:
 sc=read(S/crop/'state_action_support.json');obs=sc['state_columns'];acts=sc['action_columns'];ep=pd.read_csv(D/crop/'episode_summary.csv');result=read(D/crop/'result.json');assert result['practical_rollout_gate']=='PASS'
 sr={'crop':crop,'runs':len(ep),'executed_runs':int((ep.executed_steps>0).sum()),'executed_rate':result['executed_fraction'],'normal_rate':float(ep.termination_reason.eq('episode_boundary').mean()),'ood_rate':float(ep.termination_reason.isin(['severe_state_ood','repeated_severe_action_ood']).mean()),**{'length_steps_'+k:v for k,v in stats(ep.executed_steps).items()}}
 for h in [3,6,12]:sr[f'reach_{h*5}min_runs']=int((ep.executed_steps>=h).sum());sr[f'reach_{h*5}min_rate_all_runs']=float((ep.executed_steps>=h).mean())
 survive.append(sr)
 totalterm=ep.terminal_cost.abs().sum()
 for reason,g in ep.groupby('termination_reason'):
  terminal.append({'crop':crop,'reason':reason,'runs':len(g),'terminal_sum':g.terminal_cost.sum(),'mean_magnitude':g.terminal_cost.abs().mean(),'median_magnitude':g.terminal_cost.abs().median(),'max_magnitude':g.terminal_cost.abs().max(),'share':g.terminal_cost.abs().sum()/totalterm if totalterm else 0})
 a=pd.read_csv(D/crop/'action_summary.csv');a.insert(0,'crop',crop);actiontables.append(a)
 for e in ep.to_dict('records'):
  p=D/crop/'episodes'/f"{e['episode_index']:04d}.jsonl.gz";assert sha(p)==read(p.with_suffix('').with_suffix('.json'))['trace_sha256']
  with gzip.open(p,'rt') as f:rows=[json.loads(l) for l in f]
  last=rows[-1];kind='state' if e['termination_reason']=='severe_state_ood' else 'joint'
  if e['termination_reason'] in ['severe_state_ood','repeated_severe_action_ood']:
   key='recorded_'+kind+'_distance';threshold=sc['thresholds'][kind+'_severe'];overlap=last[key]>threshold;prior=last['changed_actions_before'];pathnormal=all(r[key]<=threshold for r in rows)
   label='A recorded-domain limitation' if overlap else ('B safety/action-changed path' if prior else ('C unexplained free-rollout divergence' if pathnormal else 'mixed: earlier recorded severe'))
   ood.append({'crop':crop,'run':e['episode_index'],'kind':kind,'reason':e['termination_reason'],'class':label,'recorded_overlap':overlap,'prior_action_changes':prior,'recorded_path_normal':pathnormal,'timestamp':last['timestamp']})
  executed=[]
  for r in rows:
   info=r['info'];entry={'crop':crop,'run':e['episode_index'],'step':r['step'],'timestamp':r['timestamp'],'executed':r['executed'],'terminal_event':info['termination_reason'] is not None,'reason':info['termination_reason'],'reward':r['reward'],'base':info.get('base_reward',0),'warning':info.get('warning_ood_penalty',0),'terminal':info.get('pessimistic_terminal_cost',0),**{t:info.get('reward_terms',{}).get(t+'_penalty',0) for t in TERMS}}
   entry['adjustment']=entry['reward']-entry['terminal']-(entry['base']-entry['warning']);allsteps.append(entry)
   if not r['executed']:continue
   x=dict(zip(obs,r['next_physical']));action=dict(zip(acts,r['final']));prev=r['previous_action'];pp=r['previous_previous_action'];rt=info['reward_terms']
   base,calc=compute_mdp_v1_reward(pd.Series(x),action,prev_action=prev,prev_prev_action=pp,config=cfg)
   np.testing.assert_allclose(base,info['base_reward'],rtol=1e-6,atol=1e-8)
   for k in rt:np.testing.assert_allclose(calc[k],rt[k],rtol=1e-6,atol=1e-8)
   # Independent scalar formula, without runtime/model/updater invocation.
   temp=x['obs_indoor_temp_c'];rh=x['obs_indoor_humidity_pct'];vpd=x['obs_current_vpd_kpa'];margin=x['obs_current_condensation_margin_c'];lo=x['obs_current_target_temp_min_c'];hi=x['obs_current_target_temp_max_c'];err=max(lo-temp,0)+max(temp-hi,0)
   ach=np.clip((3+x['obs_outdoor_wind_speed'])*action['vent_pct'],0,15);qh=15000*action['heat_run'];qv=100*ach*max(temp-x['obs_outdoor_temp_c'],0);hv=max(qv-.25*qh,0)/qh if action['heat_run']>.5 and qh>5 else 0
   reversal=sum((abs(action[k]-prev[k])+abs(prev[k]-pp[k]))/2 if (action[k]-prev[k])*(prev[k]-pp[k])<0 else 0 for k in MDP_V1_CONTINUOUS_ACTION_COLUMNS)
   reversal+=sum(2 if (action[k]-prev[k])*(prev[k]-pp[k])<0 else 0 for k in MDP_V1_BINARY_ACTION_COLUMNS)
   reversal/=len(MDP_V1_CONTINUOUS_ACTION_COLUMNS)+2*len(MDP_V1_BINARY_ACTION_COLUMNS)
   raw=[min(err/3,1),np.clip((rh-90)/10,0,1),np.clip((.3-vpd)/.3,0,1),np.clip((.8-margin)/.8,0,1),min(hv/.5,1),np.clip(reversal,0,1)]
   np.testing.assert_allclose(raw,[rt['raw_'+t+'_penalty'] for t in TERMS],rtol=1e-6,atol=1e-8)
   st={**entry,**x,'temperature_error':err,'target_min':lo,'target_max':hi,'hv_heat_on':action['heat_run']>.5,'hv_q_heat':qh,'hv_q_ventloss':qv,'hv_threshold':qh*.25,'hv_excess':hv,'trigger_rh':rh>90,'trigger_vpd':vpd<.3,'trigger_cond':margin<.8,'trigger_hv':hv>0,'trigger_temp':err>0,'trigger_act':reversal>0,'temp_saturated':err>=3,'temp_partial':0<err<3,'temp_zero':err==0}
   assert np.isfinite([temp,rh,vpd,margin,err,qh,qv,hv,*raw,*[x[k] for k in obs]]).all();states.append(st);executed.append((r,entry))
  for i,(r,baseentry) in enumerate(executed):
   for j,action in enumerate(acts):
    for event,changed in [('projection',abs(r['proposed'][j]-r['final'][j])>1e-12),('command_delta',abs(r['final'][j]-r['previous_action'][action])>1e-12)]:
     if not changed:continue
     for h in [1,3,6]:
      available=i+h<len(executed);v={'crop':crop,'run':e['episode_index'],'step':r['step'],'timestamp':r['timestamp'],'action':action,'event':event,'lag':h,'available':available,'proposed':r['proposed'][j],'final':r['final'][j]}
      for term in TERMS:
       v['delta_'+term]=(executed[i+h][1][term]-baseentry[term]) if available else np.nan
       v['any_change_'+term]=any(abs(executed[k][1][term]-baseentry[term])>1e-12 for k in range(i+1,i+h+1)) if available else np.nan
      events.append(v)
 print('AUDITED',crop,flush=True)
step=pd.DataFrame(allsteps);state=pd.DataFrame(states);step.to_parquet(O/'reward_rows.parquet',index=False);state.to_parquet(O/'reward_trigger_states.parquet',index=False);pd.DataFrame(events).to_parquet(O/'action_reward_lag_events.parquet',index=False)
sv=save('survivability',survive);od=save('ood_run_classification',ood);tc=save('terminal_by_reason',terminal);actions=pd.concat(actiontables);actions.to_csv(O/'action_intervention.csv',index=False)
assert len(od.loc[(od.crop=='strawberry')&(od.kind=='state')&(~od.recorded_overlap)&(od.prior_action_changes==0)])==0
odsum=[]
for c in CROPS:
 for kind in ['state','joint']:
  for cls in ['A recorded-domain limitation','B safety/action-changed path','C unexplained free-rollout divergence','mixed: earlier recorded severe']:
   n=int(((od.crop==c)&(od.kind==kind)&(od['class']==cls)).sum());nr=int(sv.loc[sv.crop==c,'runs'].iloc[0]);den=int(((od.crop==c)&(od.kind==kind)).sum());odsum.append({'crop':c,'kind':kind,'class':cls,'runs':n,'all_runs_rate':n/nr,'same_kind_ood_rate':n/den if den else 0})
osum=save('ood_summary',odsum)
components=[];triggers=[];saturation=[];range_rows=[];dynamics=[];reconcile=[];hvdetail=[]
for c in CROPS:
 f=step.loc[step.crop==c];st=state.loc[state.crop==c];dense=f.loc[f.executed];six=dense[TERMS].abs().sum().sum();finalabs=abs(f.reward.sum())
 for term in TERMS+['warning','terminal','base','adjustment']:
  x=dense[term] if term in TERMS+['base'] else f[term];components.append({'crop':c,'term':term,'rows':len(x),'nonzero_rows':int((x!=0).sum()),'nonzero_rate':float((x!=0).mean()),'absolute_contribution':float(x.abs().sum()),'signed_sum':float(x.sum()),'six_absolute_share':float(x.abs().sum()/six) if term in TERMS else None,'ratio_to_abs_final_return':float(x.abs().sum()/finalabs),'note':'pre-clip accounting ratio; not independently attributable final contribution'})
 for term in TERMS:
  triggers.append({'crop':c,'term':term,'executed_rows':len(st),'trigger_rows':int(st['trigger_'+term].sum()),'nonzero_rows':int((st[term]!=0).sum()),'raw_invalid_rows':0,'branch_evaluated_rows':len(st),'implementation':'PASS'})
 saturation.append({'crop':c,'rows':len(st),'zero_rate':st.temp_zero.mean(),'partial_rate':st.temp_partial.mean(),'saturated_rate':st.temp_saturated.mean(),**{'error_c_'+k:v for k,v in stats(st.temperature_error).items()}})
 for col in ['obs_indoor_humidity_pct','obs_current_vpd_kpa','obs_current_condensation_margin_c','obs_indoor_temp_c','obs_outdoor_temp_c','temperature_error','target_min','target_max','hv_q_ventloss','hv_threshold']:
  range_rows.append({'crop':c,'variable':col,**stats(st[col])})
 hvdetail.append({'crop':c,'heat_on_rows':int(st.hv_heat_on.sum()),'heat_off_rows':int((~st.hv_heat_on).sum()),'heat_on_ventloss_exceeds_alpha_qheat':int((st.hv_heat_on&(st.hv_q_ventloss>st.hv_threshold)).sum()),'heat_off_ventloss_positive':int(((~st.hv_heat_on)&(st.hv_q_ventloss>0)).sum()),'conflict_nonzero':int((st.hv>0).sum())})
 for name,g in [('all_calls',f),('nonterminal_executed',f.loc[f.executed&~f.terminal_event]),('terminal_calls',f.loc[f.terminal_event]),('executed_base',dense)]:
  x=g['base'] if name=='executed_base' else g.reward;dynamics.append({'crop':c,'scope':name,**stats(x),'variance_population':x.var(ddof=0),'unique_exact':x.nunique(),'unique_rounded_8_decimals':x.round(8).nunique(),'zero_rate':(x==0).mean(),'nonzero_rate':(x!=0).mean(),'dynamic_range':x.max()-x.min()})
 non=f.loc[f.executed&~f.terminal_event,'reward'];end=f.loc[f.terminal_event,'reward'];reconcile.append({'crop':c,'six_sum':dense[TERMS].sum().sum(),'base_sum':f.base.sum(),'warning_sum':f.warning.sum(),'clip_fallback_adjustment':f.adjustment.sum(),'terminal_sum':f.terminal.sum(),'final_sum':f.reward.sum(),'terminal_abs_final_ratio':abs(f.terminal.sum())/finalabs,'terminal_to_nonterminal_mean_abs_scale':end.abs().mean()/non.abs().mean(),'terminal_to_nonterminal_median_abs_scale':end.abs().median()/non.abs().median()})
 np.testing.assert_allclose(f.base.sum()-f.warning.sum()+f.adjustment.sum()+f.terminal.sum(),f.reward.sum(),rtol=1e-6,atol=1e-8)
component=save('reward_component_activity',components);trigger=save('reward_trigger_checks',triggers);sat=save('temperature_saturation',saturation);ranges=save('reward_input_ranges',range_rows);dyn=save('reward_dynamics',dynamics);rec=save('reward_accounting',reconcile);hv=save('heat_vent_trigger_details',hvdetail)
assert not trigger.loc[trigger.term.isin(['rh','vpd','cond','hv']),'trigger_rows'].any()
for c,expected in [('melon',.9809),('cucumber',.9984),('strawberry',.9945)]:assert abs(component.loc[(component.crop==c)&(component.term=='temp'),'six_absolute_share'].iloc[0]-expected)<.00005
lags=[];ev=pd.DataFrame(events)
for (crop,action,event,h),g in ev.groupby(['crop','action','event','lag']):
 v=g.loc[g.available]
 for term in TERMS:
  lags.append({'crop':crop,'action':action,'event':event,'lag':h,'term':term,'event_rows':len(g),'observed_lag_rows':len(v),'right_censored':len(g)-len(v),'mean_signed_change':v['delta_'+term].mean(),'mean_abs_change':v['delta_'+term].abs().mean(),'nonzero_endpoint_change_rows':int((v['delta_'+term].abs()>1e-12).sum()),'any_change_within_h_rows':int(v['any_change_'+term].sum())})
lag=save('action_reward_lag_summary',lags)
# Qualitative gate follows the user-fixed criteria, not a fitted numeric threshold.
ood_ok=all(read(D/c/'result.json')['practical_rollout_gate']=='PASS' for c in CROPS) and not (od['class']=='C unexplained free-rollout divergence').any()
reward_bad=all(component.loc[(component.crop==c)&component.term.isin(['rh','vpd','cond','hv']),'nonzero_rows'].sum()==0 for c in CROPS)
verdict='STEP17 ENTRY BLOCKED — REWARD' if reward_bad and ood_ok else ('STEP17 ENTRY BLOCKED — MULTIPLE' if reward_bad else ('STEP17 ENTRY PASS — offline research only' if ood_ok else 'STEP17 ENTRY BLOCKED — OOD/ENV'))
write(O/'analysis_summary.json',{'verdict':verdict,'OOD_ENV':'no independent entry blocker established in fixed recorded replay' if ood_ok else 'requires further audit','reward_implementation':'PASS for all visited finite states; independent scalar and pure function parity','reward_informativeness':'BLOCKED: humidity/VPD/condensation/HV inactive; temp saturation; act is reversal cost not energy cost','strawberry_state_rollout_only_no_prior_action_changes':0,'Step16':'PASS unchanged','Step17_executed':False,'online_deployment':'BLOCKED','next_only':'reward contract redesign preregistration'})
for r in protocol['inputs']+protocol['code']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
for r in pro['inputs']+pro['code']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
write(O/'checks.json',{'trace_hashes':'PASS','original_inputs_unchanged':'PASS','all_pure_reward_recomputation':'PASS','independent_scalar_formula_parity':'PASS','reward_trigger_nonzero_consistency':'PASS','known_temp_shares':'PASS','strawberry_zero_unchanged_state_divergence':'PASS','new_rollout_or_training':False})
print(verdict,flush=True)
