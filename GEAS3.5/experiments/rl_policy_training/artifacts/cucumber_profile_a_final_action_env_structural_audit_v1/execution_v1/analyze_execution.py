from execute_modes import *
from collections import Counter,defaultdict
from itertools import product
from geas35.rl.bound_support_four_channel_v1 import load_bound_support
from geas35.rl.mdp_v1 import project_mdp_v1_action_constraints
from geas35.rl.profile_a_offline_v1 import CFG

def tri(xs):return False if any(x is False for x in xs) else (None if any(x is None for x in xs) else True)
def stats(xs):
 a=np.asarray([x for x in xs if x is not None],float);return {'n':len(a),'missing':len(xs)-len(a),**{k:(float(v) if len(a) else None) for k,v in zip(['min','p25','median','p75','p90','max'],np.quantile(a,[0,.25,.5,.75,.9,1],method='linear') if len(a) else [0]*6)}}
def main():
 assert load(O/'mode_gate.json')['PASS'];c=verify();th=c['frozen_thresholds'];cases=load(O/'failure_provenance_matrix.json');cm={(x['seed'],x['run_id']):x for x in cases}
 b=A/'step15_hlv15_b60_r6_4ch_v1/cucumber/binding.json';rt,sup=load_bound_support(b,manifest_sha256=sha(b),crop='cucumber',root=R);cols=rt.obs;targets=c['target_order'];refhash=load(b)['files']['reference']['sha256']
 def scaled(s):return rt.scaler.transform_frame(pd.DataFrame([s]))[cols]
 def sd(s):return float(sup._state_nn.kneighbors(scaled(s).to_numpy(float)*sup.state_weights)[0][0,-1])
 def jd(s,act):return float(sup.scores(scaled(s),pd.DataFrame([act],columns=ACT))['joint'][0])
 def weather(s,a):
  v,f=project_mdp_v1_action_constraints(dict(zip(ACT,a)),s,CFG);return [v[k] for k in ACT],f
 def decomp(s,d,joint=None):
  z=scaled(s).to_numpy(float)[0];w=z*sup.state_weights;ref=sup._state_reference;nn=sup._state_nn
  if joint is not None:w=np.r_[w,np.asarray(joint)*sup.action_weights]/np.sqrt(2);ref=sup._joint_reference;nn=sup._joint_nn
  ds,ix=nn.kneighbors(w.reshape(1,-1));idx=int(ix[0,-1]);diff=w-ref[idx];sq=diff**2;assert np.isclose(ds[0,-1],d,rtol=1e-6,atol=1e-8);assert np.isclose(sq.sum(),d*d,rtol=1e-6,atol=1e-8)
  raw_neighbor=rt.scaler.inverse_transform_frame(pd.DataFrame([sup.reference_state_raw[idx]],columns=cols)).iloc[0]
  rows=[]
  for j,k in enumerate(cols):rows.append({'feature':k,'physical_query':s[k],'physical_neighbor':float(raw_neighbor[k]),'raw_difference':s[k]-float(raw_neighbor[k]),'scaled_difference':float(z[j]-sup.reference_state_raw[idx,j]),'weight':float(sup.state_weights[j]),'weighted_difference':float(diff[j]),'squared_contribution':float(sq[j]),'fraction':float(sq[j]/sq.sum()) if sq.sum() else None})
  if joint is not None:
   for j,k in enumerate(ACT):rows.append({'feature':k,'query':joint[j],'neighbor':float(sup.reference_action_raw[idx,j]),'weighted_difference':float(diff[56+j]),'squared_contribution':float(sq[56+j])})
  return {'distance':d,'k':50,'reference_hash':refhash,'kth_index':idx,'nearest_index':int(ix[0,0]),'nearest_distance':float(ds[0,0]),'returned_neighbor_indices':ix[0].tolist(),'returned_neighbor_distances':ds[0].tolist(),'exact_ties_at_k_among_returned':int(sum(ds[0]==ds[0,-1])),'all_reference_exact_tie_indices':np.flatnonzero(np.linalg.norm(ref-w,axis=1)==ds[0,-1]).tolist(),'closure_sum':float(sq.sum()),'closure_PASS':True,'state_block':float(sq[:56].sum()),'action_block':float(sq[56:].sum()),'features':rows,'top10':[x['feature'] for x in sorted(rows,key=lambda x:-x['squared_contribution'])[:10]],'original_row_identity':'reference_hash + exact array row index; original Train row mapping not assumed'}
 rec=defaultdict(dict)
 for t in lines(O/'recorded_nodes.jsonl'):rec[t['run_id']][t['timestamp']]=t
 contributions=[];stated=[];joint_events=[];feasible=[];history=[];drifts=[];reference=[];allmetrics=[];votes=[];aligned_paths=[]
 matrix=load(E/'legacy_vs_migrated_run_matrix.json')
 for row in matrix:
  seed,rid=row['seed'],row['run_id'];case=cm.get((seed,rid));group=case['group'] if case else 'REFERENCE_COMPLETE';fv=lines(E/f'development_seed{seed}_{rid}.jsonl');nodes={};rr=rec[rid]
  for t in fv:
   time=t['decision_timestamp'];assert time in rr;assert rr[time]['parent_episode_id']==row['parent_episode_id'];assert t['run_id']==rid and t['seed']==seed
   nodes[time]={'state':t['current_state'],'distance':t['info']['state_distance']}
   assert np.isfinite(list(t['current_state'].values())).all()
   if t['next_state'] is not None:
    assert pd.Timestamp(t['outcome_timestamp'])-pd.Timestamp(time)==pd.Timedelta(minutes=5);assert t['outcome_timestamp'] in rr
    nodes[t['outcome_timestamp']]={'state':t['next_state'],'distance':sd(t['next_state'])}
  for t in fv:assert np.isclose(sd(t['current_state']),t['info']['state_distance'],rtol=1e-6,atol=1e-8)
  firstwarn=next((tm for tm,n in sorted(nodes.items()) if n['distance']>th['state_warning']),None);firstsev=next((tm for tm,n in sorted(nodes.items()) if n['distance']>th['state_severe']),None)
  commands=[k for k in cols if k.startswith('obs_prev_')];derived=[k for k in cols if k.startswith('obs_derived_') or k in ['obs_current_vpd_kpa','obs_current_dewpoint_c','obs_current_condensation_margin_c']]
  changed={};dchanged={};diffcount=0;devs={k:[] for k in targets};full_deltas=[]
  for tm,n in sorted(nodes.items()):
   rs=rr[tm]['state'];cs=n['state'];cmd=[k for k in commands if not np.isclose(cs[k],rs[k],rtol=1e-6,atol=1e-8)];dd=[k for k in derived if not np.isclose(cs[k],rs[k],rtol=1e-6,atol=1e-8)]
   if cmd and not changed:changed={'timestamp':tm,'features':cmd}
   if dd and not dchanged:dchanged={'timestamp':tm,'features':dd}
   diffcount+=bool(cmd)
   for k in targets:devs[k].append(cs[k]-rs[k])
   if case:full_deltas.append({'timestamp':tm,'recorded_state_distance':rr[tm]['state_distance'],'Fv1_state_distance':n['distance'],'differences':{k:cs[k]-rs[k] for k in cols}})
  if case:history.append({'seed':seed,'run_id':rid,'group':group,'first_command_divergence':changed or None,'first_derived_divergence':dchanged or None,'first_state_warning':firstwarn,'first_state_severe':firstsev,'full_feature_differences':full_deltas,'causal_proof':False})
  finals=[t for t in fv if t['final_joint_distance'] is not None];executed=[t for t in fv if t['next_state'] is not None];projection=0;fb=0
  last_counter=0
  for t in finals:
   assert t['final_counter_before']==last_counter
   last_counter=last_counter+1 if t['final_joint_distance']>th['joint_severe'] else 0
   assert t['final_counter_after']==last_counter
   assert np.isclose(jd(t['current_state'],t['info']['final_action']),t['final_joint_distance'],rtol=1e-6,atol=1e-8)
   pre=t['info']['intermediate_action'];w,flags=weather(t['current_state'],pre);assert equal(w,t['info']['final_action'])
   projection+=not equal(t['info']['proposed'],t['info']['projected_action']);fb+=t['info']['fallback_action'] is not None
  m={'seed':seed,'run_id':rid,'group':group,'initial_state_distance':fv[0]['info']['state_distance'],'maximum_state_distance':max(n['distance'] for n in nodes.values()),'proposal_severe_n':sum(t['info']['joint_distance']>th['joint_severe'] for t in fv),'proposal_denominator':len(fv),'final_severe_n':sum(t['final_joint_distance']>th['joint_severe'] for t in finals),'final_denominator':len(finals),'projection_n':projection,'fallback_n':fb,'executed_denominator':len(executed),'command_different_nodes':diffcount,'aligned_nodes':len(nodes),'first_command_divergence':changed or None,'four_channel_deviation':{k:{'signed':stats(v),'absolute':stats([abs(x) for x in v])} for k,v in devs.items()}}
  for n,d in [('proposal_severe','proposal'),('final_severe','final'),('projection','executed'),('fallback','executed')]:m[n+'_rate']=m[n+'_n']/m[d+'_denominator'] if m[d+'_denominator'] else None
  m['command_divergence_rate']=diffcount/len(nodes);allmetrics.append(m)
  if not case:reference.append(m);continue
  ft=fv[-1]['decision_timestamp'];start=fv[0]['decision_timestamp']
  if group in ['A','B']:
   modes={m:lines(O/f'{m}_seed{seed}_{rid}.jsonl') for m in ['R1','TF1','TF4']};tf1={t['target_timestamp']:t for t in modes['TF1']};tf4={}
   for t in modes['TF4']:
    tf4[t['decision_timestamp']]={'state':t['current_state'],'distance':t['info']['state_distance']}
    if t['next_state'] is not None:tf4[t['outcome_timestamp']]={'state':t['next_state'],'distance':t['next_state_distance']}
   pred=tf1.get(ft);tf1sev=None if not pred or pred['next_state'] is None else pred['next_state_distance']>th['state_severe'];rsev=rr[ft]['state_distance']>th['state_severe'];times=[tm for tm in sorted(rr) if start<=tm<=ft]
   rnormal=all(rr[tm]['state_distance']<=th['state_severe'] for tm in times)
   pnormal=tri([None if tf1.get(tm,{}).get('next_state') is None else tf1[tm]['next_state_distance']<=th['state_severe'] for tm in times[1:]]) if len(times)>1 else None
   tnormal=None if ft not in tf4 else all(n['distance']<=th['state_severe'] for tm,n in tf4.items() if tm<=ft)
   thassev=True if any(n['distance']>th['state_severe'] for tm,n in tf4.items() if tm<=ft) else (False if ft in tf4 else None)
   labs={'RECORDED_DOMAIN':bool(rsev),'POLICY_ONE_STEP_ASSOCIATED':tri([not rsev,tf1sev]),'FREE_RECURRENCE_ASSOCIATED':tri([rnormal,pnormal,True]),'FOUR_CHANNEL_RECURRENCE_ASSOCIATED':tnormal,'COMMAND_HISTORY_OR_DERIVED_ASSOCIATED':thassev}
   labs['MIXED_OR_UNRESOLVED']=sum(v is True for v in labs.values())!=1 or any(v is None for v in labs.values())
   policy=labs['POLICY_ONE_STEP_ASSOCIATED'] is True;structure=any(labs[k] is True for k in ['RECORDED_DOMAIN','FREE_RECURRENCE_ASSOCIATED','FOUR_CHANNEL_RECURRENCE_ASSOCIATED','COMMAND_HISTORY_OR_DERIVED_ASSOCIATED']);unknown=any(v is None for v in labs.values());vote='UNRESOLVED' if unknown else ('MIXED' if policy and structure else ('POLICY' if policy else ('STRUCTURE' if structure else 'UNRESOLVED')))
   stated.append({'seed':seed,'run_id':rid,'group':group,'failure_timestamp':ft,'R0_distance':rr[ft]['state_distance'],'Fv1_distance':nodes[ft]['distance'],'TF1_predecessor_distance':None if pred is None else pred['next_state_distance'],'TF4_distance':tf4.get(ft,{}).get('distance'),'TF4_first_severe':next((tm for tm,n in sorted(tf4.items()) if n['distance']>th['state_severe']),None),'associations':labs,'exclusive_vote':vote})
   votes.append({'seed':seed,'run_id':rid,'group':group,'vote':vote})
   points={ft:'state_failure'}
   if firstwarn is not None:points[firstwarn]='first_state_warning' if firstwarn!=ft else 'first_warning_and_failure'
   for tm,role in points.items():
    for mode,node in [('Fv1',nodes.get(tm)),('R0',{'state':rr[tm]['state'],'distance':rr[tm]['state_distance']}),('TF4',tf4.get(tm)),('TF1',{'state':tf1[tm]['next_state'],'distance':tf1[tm]['next_state_distance']} if tf1.get(tm,{}).get('next_state') is not None else None)]:
     if node:contributions.append({'seed':seed,'run_id':rid,'timestamp':tm,'role':role,'mode':mode,**decomp(node['state'],node['distance'])})
   horizons=[(f'+{h}min',str(pd.Timestamp(start)+pd.Timedelta(minutes=h))) for h in [5,15,30,60]]
   horizons += [('before_first_warning',str(pd.Timestamp(firstwarn)-pd.Timedelta(minutes=5)) if firstwarn else None),('before_first_severe',str(pd.Timestamp(firstsev)-pd.Timedelta(minutes=5)) if firstsev else None)]
   for label,tm in horizons:
    n=nodes.get(tm);rs=rr.get(tm);avail=n is not None and rs is not None
    drifts.append({'seed':seed,'run_id':rid,'anchor':label,'timestamp':tm,'available':avail,'reason':None if avail else 'NO_ALIGNED_EXECUTED_NODE_NO_FILL','recorded_distance':rs['state_distance'] if avail else None,'free_distance':n['distance'] if avail else None,'distance_delta':n['distance']-rs['state_distance'] if avail else None,'channels':{k:{'signed':n['state'][k]-rs['state'][k],'absolute':abs(n['state'][k]-rs['state'][k])} for k in targets} if avail else None})
   with (O/f'aligned_seed{seed}_{rid}.jsonl').open('x') as f:
    r1={t['decision_timestamp']:t for t in modes['R1']}
    for tm in times:emit(f,{'seed':seed,'run_id':rid,'parent_episode_id':row['parent_episode_id'],'timestamp':tm,'crop':rr[tm]['crop'],'series':rr[tm]['series'],'segment':rr[tm]['segment'],'R0':rr[tm],'R1':r1.get(tm),'TF1_predecessor':tf1.get(tm),'TF4':tf4.get(tm),'Fv1':nodes.get(tm),'missing_reason':'UNKNOWN_IF_NULL_NO_POSTTERMINATION_FILL'})
  else:
   terminal=finals[-3:];assert [t['final_counter_after'] for t in terminal]==[1,2,3];assert fv[-1]['info']['reason']=='executed_joint_support_failure'
   cvotes=[]
   for t in finals:
    if t['final_joint_distance']<=th['joint_severe']:continue
    primary=t in terminal;s=t['current_state'];i=t['info'];loc=sup.local_action_support(scaled(s))[0];assert equal(loc,i['support']);pre=i['intermediate_action'];pdistance=jd(s,pre);post=t['final_joint_distance'];dec=decomp(s,post,i['final_action']);ww,wflags=weather(s,pre);assert equal(ww,i['final_action'])
    def admiss(a):
     return a is not None and len(a)==6 and np.isfinite(a).all() and all((loc['continuous'][k][0]<=a[j]<=loc['continuous'][k][1]) if k in loc['continuous'] else a[j] in loc['binary'][k] for j,k in enumerate(ACT))
    event={'seed':seed,'run_id':rid,'decision_timestamp':t['decision_timestamp'],'outcome_timestamp':t['outcome_timestamp'],'primary_terminal_event':primary,'state_distance':i['state_distance'],'proposal':i['proposed'],'proposal_distance':i['joint_distance'],'projection':i['projected_action'],'projection_distance':jd(s,i['projected_action']) if i['projected_action'] is not None else None,'fallback':i['fallback_action'],'fallback_distance':jd(s,i['fallback_action']) if i['fallback_action'] is not None else None,'preweather':pre,'preweather_distance':pdistance,'final':i['final_action'],'final_distance':post,'final_counter_before':t['final_counter_before'],'final_counter_after':t['final_counter_after'],'local_envelope':loc,'weather_context':{k:v for k,v in s.items() if 'outdoor' in k or 'rain' in k},'weather_flags':wflags,'joint_decomposition':dec}
    if primary:
     domains=[]
     for k in ACT:
      if k in loc['continuous']:
       lo,hi=loc['continuous'][k];assert np.isfinite([lo,hi]).all() and lo<=hi;domains.append(sorted(set([lo,(lo+hi)/2,hi])))
      else:domains.append(loc['binary'][k])
     assert all(domains),'EMPTY_CANDIDATE_DOMAIN';grid=set(product(*domains));assert len(grid)<=216
     for ac in [i['proposed'],i['projected_action'],i['fallback_action'],pre,i['final_action']]:
      if admiss(ac):grid.add(tuple(ac))
     assert len(grid)<=221;results=[]
     for ac in sorted(grid):
      final,flags=weather(s,list(ac));d=jd(s,final);assert np.isfinite(d);results.append({'preweather':list(ac),'weather_final':final,'final_local_admissible':admiss(final),'distance':d,'nonsevere':bool(d<=th['joint_severe']),'weather_flags':flags})
     witness=next((x for x in results if x['nonsevere']),None);fc='SUPPORTED_ALTERNATIVE_EXISTS' if witness else 'NO_SUPPORTED_ACTION_FOUND_IN_FROZEN_CANDIDATE_DOMAIN'
     feasible.append({'seed':seed,'run_id':rid,'timestamp':t['decision_timestamp'],'candidate_n':len(results),'classification':fc,'first_lexicographic_witness':witness,'candidates':results,'no_Transition_reward_or_execution':True})
     labs={'LOCAL_GLOBAL_GEOMETRY_MISMATCH':bool(admiss(pre) and pdistance>th['joint_severe']),'WEATHER_CAP_INDUCED':bool(pdistance<=th['joint_severe'] and post>th['joint_severe'] and not equal(pre,i['final_action'])),'STATE_CONTEXT_DOMINATED':bool(i['state_distance']>th['state_warning'] and witness is None and dec['state_block']>dec['action_block']),'POLICY_INTERVENTION_PATH':witness is not None};labs['MIXED_OR_UNRESOLVED']=sum(labs.values())!=1;event.update(labels=labs,feasibility=fc);cvotes.append((witness is not None,witness is None))
    joint_events.append(event)
   pol=any(x[0] for x in cvotes);st=any(x[1] for x in cvotes);votes.append({'seed':seed,'run_id':rid,'group':'C','vote':'MIXED' if pol and st else ('POLICY' if pol else 'STRUCTURE')})
  print('ANALYZED',seed,rid[:10],group,flush=True)
 # Prefix evidence uses stored traces only, and checks state/action/history directly through legacy censoring.
 prefix=[]
 for row in matrix:
  seed,rid=row['seed'],row['run_id'];l=lines(E/f'legacy_seed{seed}_{rid}.jsonl');f=lines(E/f'development_seed{seed}_{rid}.jsonl')
  for x,y in zip(l,f):
   for k in ['current_state','next_state','decision_timestamp','outcome_timestamp','transition_input','history_before_hash','history_after_hash']:assert equal(x[k],y[k]),(seed,rid,k)
   for k in ['proposed','projected_action','fallback_action','intermediate_action','final_action','prediction','state_distance','joint_distance']:assert equal(x['info'].get(k),y['info'].get(k))
  prefix.append({'seed':seed,'run_id':rid,'shared_steps':min(len(l),len(f)),'PASS':True})
 seven=load(E/'historical_seven_consistency.json');assert all(x['prior_shadow_exact_consistency'] and x['termination_timestamp_difference_minutes']>0 for x in seven);regressions=sum(x['legacy_outcome']=='COMPLETE' and x['migrated_outcome']!='COMPLETE' for x in matrix);assert regressions==0
 v=Counter(x['vote'] for x in votes);per={s:Counter(x['vote'] for x in votes if x['seed']==s) for s in [42,43,44]};dom=next((k for k in ['POLICY','STRUCTURE'] if v[k]>=10 and all(per[s][k]>per[s]['STRUCTURE' if k=='POLICY' else 'POLICY'] for s in per)),None)
 validity='ENVIRONMENT SEMANTICS VALID / STRUCTURAL POLICY-DOMAIN ISSUE REMAINS'
 retrain='RETRAINING MAY ADDRESS POLICY-DEPENDENT FAILURES' if dom=='POLICY' else ('RETRAINING NOT YET JUSTIFIED — DOMAIN/ROLLOUT STRUCTURE DOMINANT' if dom=='STRUCTURE' else 'RETRAINING EVIDENCE MIXED')
 summary={'exclusive_counts':dict(v),'per_seed_counts':{str(s):dict(vv) for s,vv in per.items()},'dominance':dom or 'MIXED/UNRESOLVED','retraining':retrain,'HPO':'HPO_NOT_YET_JUSTIFIED','environment_validity':validity,'migration':'Previously proposal-censored downstream failure modes exposed; no new shared-path divergence or COMPLETE regression detected. Not a guarantee of domain quality.','legacy_COMPLETE_to_failure':regressions,'seven_shadow_consistent':True,'candidate_outcomes':dict(Counter(x['classification'] for x in feasible)),'recovery_invariant_violation':False,'recovery_interpretation':'Frozen recovery attempts local/fallback action, never guarantees global support; all actual order/weather/query/counter checks verified; no same explicit invariant violation in both C runs.'}
 assert len(stated)==17 and len(reference)==56 and len(feasible)==6 and len(votes)==19
 groupstats={}
 for g in ['A','B','C','REFERENCE_COMPLETE']:
  ms=[x for x in allmetrics if x['group']==g];groupstats[g]={k:stats([x[k] for x in ms]) for k in ['initial_state_distance','maximum_state_distance','proposal_severe_rate','final_severe_rate','projection_rate','fallback_rate','command_divergence_rate']};groupstats[g]['pooled_rates']={n: {'numerator':sum(x[n+'_n'] for x in ms),'denominator':sum(x[d+'_denominator'] for x in ms)} for n,d in [('proposal_severe','proposal'),('final_severe','final'),('projection','executed'),('fallback','executed')]}
 per_seed_reference={str(seed):{k:stats([x[k] for x in reference if x['seed']==seed]) for k in ['initial_state_distance','maximum_state_distance','proposal_severe_rate','final_severe_rate','projection_rate','fallback_rate','command_divergence_rate']} for seed in [42,43,44]}
 for n,x in [('state_failure_diagnostics.json',stated),('executed_joint_failure_diagnostics.json',joint_events),('candidate_support_feasibility.json',feasible),('support_contribution_summary.json',contributions),('complete_reference_summary.json',{'cases':reference,'per_seed_reference':per_seed_reference,'group_comparison':groupstats,'all_case_metrics':allmetrics}),('command_history_divergence.json',history),('recorded_free_drift.json',drifts),('exclusive_votes.json',votes),('migration_prefix_parity.json',prefix),('decision_summary.json',summary)]:write(n,x)
 verify();write('analysis_gate.json',{'PASS':True,'state_cases':17,'joint_terminal_events':6,'contribution_queries':len(contributions),'closure_PASS':True,'shared_prefix_PARITY':True,'new_Fv1':False});print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write('technical_failure.json',{'gate':'analysis','error':str(e),'traceback':traceback.format_exc()});raise
