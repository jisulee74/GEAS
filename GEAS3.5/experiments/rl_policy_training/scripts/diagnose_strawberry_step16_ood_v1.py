"""Read-only A/B diagnosis with teacher forcing; no fits or official mutations."""
from pathlib import Path
import os,sys,json,gzip,argparse
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from geas35.models.transition.handoff_b60_r6_v1 import sha,Identity
from geas35.rl.bound_support_b60_r6_v1 import load_bound_research_artifacts
from geas35.rl.sensor_history_env_v4 import SensorHistoryStateUpdater
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1'
S=ROOT/'experiments/rl_policy_training/artifacts/step15_b60_r6_v1'
OLD=ROOT/'experiments/rl_policy_training/artifacts/step16_b60_r6_v1'
D=ROOT/'experiments/rl_policy_training/artifacts/strawberry_step16_ood_diagnosis_v1'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def group(c):
 if c in ('obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm'):return 'endogenous_indoor'
 if c.startswith('obs_outdoor_') or c=='obs_rain_flag' or c in ('obs_derived_dli_mol_m2','obs_derived_light_sum','obs_derived_clear_sky_sum','obs_derived_light_sum_ratio','obs_derived_light_eta_minutes'):return 'exogenous_weather'
 if c.startswith(('obs_growth_','obs_hour_','obs_solar_period_','obs_daylight_condition_','obs_target_','obs_current_target_')) or c=='obs_is_daytime':return 'growth_context'
 if c.startswith('obs_prev_') or c in ('obs_derived_vent_loss_proxy','obs_derived_ramp_limit_pct'):return 'command_memory_action_derived'
 return 'rolling_derived_state'
def trace(index):
 path=OLD/'strawberry/episodes'/f'{index:04d}.jsonl.gz';r=read(path.with_suffix('').with_suffix('.json'));assert r['trace_sha256']==sha(path)
 with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]
def freeze():
 D.mkdir(exist_ok=False);schema=read(P/'schema.json');obs=schema['observation_columns'];f=pd.read_csv(OLD/'ood_termination_attribution.csv');f=f.loc[f.crop.eq('strawberry')];state=f.loc[f.termination_reason.eq('severe_state_ood')]
 cohorts={'A':state.loc[state.classification.eq('recorded_severe_overlap')].episode_index.astype(int).tolist(),'B':state.loc[state.classification.eq('rollout_only_no_prior_action_changes')].episode_index.astype(int).tolist(),'safety_changed':state.loc[state.classification.eq('rollout_only_after_safety_action_changes')].episode_index.astype(int).tolist()};assert [len(cohorts[k]) for k in cohorts]==[39,7,6]
 files=[]
 for folder in (P/'strawberry',S/'strawberry',OLD/'strawberry',ROOT/'experiments/transition_model_selection/artifacts/step14_b60_r6_v1/strawberry'):
  files += [p for p in sorted(folder.rglob('*')) if p.is_file()]
 files += [P/'schema.json',P/'runtime_contract.json',S/'research_environment_binding.json',OLD/'ood_termination_attribution.csv',OLD/'final_gate.json',OLD/'reward_sanity_flags.csv',Path(__file__),ROOT/'src/geas35/rl/sensor_history_env_v4.py',ROOT/'src/geas35/rl/causal_contract_env_v2.py']
 pro={'namespace':D.name,'cohorts':cohorts,'groups':{c:group(c) for c in obs},'rtol':1e-6,'atol':1e-8,'coverage_bins':'Train physical marginal quantiles [0,.1,.25,.5,.75,.9,1]; outside-range bins separate; joint cell counts diagnostic, not support gate','group_distances':'same frozen coordinate weights; full-space kth neighbor contributions AND subgroup nearest/kth distances diagnostic only; no threshold fit','teacher_forcing':'reset through each termination timestamp for A and B; inject recorded next T/RH/CO2 only; no policy interpretation','free_path':'unchanged immutable Step16 trace; B action equality checked against recorded actions every executed step','inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files],'decisions':'strawberry Step16 BLOCKED; overall Step17 BLOCKED unchanged; no Test or fit or official changes'}
 write(D/'protocol.json',pro);write(D/'protocol_lock.json',{'sha256':sha(D/'protocol.json')});print('FROZEN',cohorts,flush=True)
def run():
 pro=read(D/'protocol.json');assert sha(D/'protocol.json')==read(D/'protocol_lock.json')['sha256']
 for r in pro['inputs']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
 entry=read(S/'research_environment_binding.json')['crops']['strawberry'];rt,support=load_bound_research_artifacts(ROOT/entry['path'],binding_sha256=entry['sha256'],crop='strawberry',root=ROOT);obs=rt.obs;actions=rt.actions;threshold=support.config['thresholds']['state_severe'];weights=support.state_weights
 train=pd.read_parquet(P/'strawberry/train_physical.parquet');val=pd.read_parquet(P/'strawberry/validation_physical.parquet');ts=pd.read_parquet(P/'strawberry/train_scaled.parquet');vs=pd.read_parquet(P/'strawberry/validation_scaled.parquet');qc=pd.read_parquet(P/'strawberry/validation_prepared_qc.parquet');vl=val.set_index('row_id',verify_integrity=True);vsl=vs.set_index('row_id',verify_integrity=True)
 registry=pd.read_parquet(S/'strawberry/fit_row_registry.parquet');refpos=registry.loc[registry.role.eq('reference'),'position'].to_numpy(int);refphysical=train.iloc[refpos];np.testing.assert_array_equal(ts.iloc[refpos][obs].to_numpy(float),support.reference_state_raw)
 quantiles=[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1];distrows=[]
 for label,frame in [('Train',train),('Validation',val),('support_reference',refphysical)]:
  for c in obs:
   x=frame[c].to_numpy(float);distrows.append({'population':label,'feature':c,'group':group(c),'rows':len(x),**{str(q):float(np.quantile(x,q)) for q in quantiles}})
 pd.DataFrame(distrows).to_csv(D/'all_feature_distributions.csv',index=False)
 feature=[];groups=[];neighbors=[];coverage=[];cases=[];paths=[];differences=[];first=[];tf_summary=[];one_steps=[];safety=[]
 joint_defs={'indoor':obs[:3],'DAT_wind_DLI':['obs_growth_stage_dat','obs_outdoor_wind_speed','obs_derived_dli_mol_m2'],'temp_outdoor_vent':['obs_indoor_temp_c','obs_outdoor_temp_c','obs_prev_vent_pct'],'temp_diff_vent_light':['obs_derived_temp_diff_in_out_c','obs_prev_vent_pct','obs_outdoor_light'],'CO2_vent':['obs_indoor_co2_ppm','obs_prev_vent_pct'],'DAT_indoor':['obs_growth_stage_dat',*obs[:3]]}
 edges={c:np.unique(np.quantile(train[c].to_numpy(float),[0,.1,.25,.5,.75,.9,1])) for c in obs}
 def bins(x,c):return np.where(np.asarray(x)>edges[c][-1],len(edges[c])+1,np.searchsorted(edges[c],x,side='right'))
 for index in pro['cohorts']['A']:
  rows=trace(index);last=rows[-1];record=vl.loc[last['row_id']];state=vsl.loc[last['row_id'],obs].to_numpy(float);dist,indices=support._state_nn.kneighbors((state*weights)[None,:]);near=indices[0];comp=((state-support.reference_state_raw[near[-1]])*weights)**2
  np.testing.assert_allclose(np.sqrt(comp.sum()),last['recorded_state_distance'],rtol=1e-6,atol=1e-8);assert dist[0,-1]>threshold
  common={'episode_index':index,'row_id':last['row_id'],'timestamp':record.reg_date,'episode_id':record.episode_id,'DAT':float(record.obs_growth_stage_dat),'state_distance':dist[0,-1],'severe_ratio':dist[0,-1]/threshold,'nearest_distance':dist[0,0]}
  for rank,k in enumerate(near,1):neighbors.append({**common,'rank':rank,'train_position':int(refpos[k]),'train_row_id':str(train.iloc[refpos[k]].row_id),'train_timestamp':train.iloc[refpos[k]].reg_date,'distance':dist[0,rank-1],**{c:float(refphysical.iloc[k][c]) for c in obs}})
  for j,c in enumerate(obs):
   x=float(record[c]);a=train[c].to_numpy(float);b=val[c].to_numpy(float);nn=refphysical.iloc[near][c].to_numpy(float);binvalue=bins(x,c)
   feature.append({**common,'feature':c,'group':group(c),'physical':x,'scaled':state[j],'squared_distance_component':comp[j],'component_share':comp[j]/comp.sum(),'nearest_physical':float(refphysical.iloc[near[0]][c]),'kth_physical':float(refphysical.iloc[near[-1]][c]),'nearest50_min':nn.min(),'nearest50_p50':np.median(nn),'nearest50_max':nn.max(),'Train_min':a.min(),'Train_p01':np.quantile(a,.01),'Train_p50':np.median(a),'Train_p99':np.quantile(a,.99),'Train_max':a.max(),'Validation_min':b.min(),'Validation_p50':np.median(b),'Validation_max':b.max(),'Train_cell_count':int((bins(a,c)==binvalue).sum()),'Validation_cell_count':int((bins(b,c)==binvalue).sum()),'outside_Train_range':bool(x<a.min() or x>a.max())})
  shares={}
  for g in sorted(set(pro['groups'].values())):
   ix=np.array([j for j,c in enumerate(obs) if group(c)==g]);squared=((support.reference_state_raw[:,ix]-state[ix])*weights[ix])**2;dd=np.sqrt(squared.sum(axis=1));shares[g]=float(comp[ix].sum()/comp.sum());groups.append({**common,'group':g,'squared_distance_component':float(comp[ix].sum()),'share':shares[g],'subspace_nearest_reference_distance':float(dd.min()),'subspace_k50_reference_distance':float(np.partition(dd,49)[49]),'official_threshold_applied_to_subspace':False})
  jc={}
  for label,cols in joint_defs.items():
   masks=[]
   for frame in (train,val,refphysical):
    mask=np.ones(len(frame),bool)
    for c in cols:mask &= bins(frame[c].to_numpy(float),c)==bins(float(record[c]),c)
    masks.append(int(mask.sum()))
   ix=[obs.index(c) for c in cols];dd=np.sqrt(np.sum(((ts[cols].to_numpy(float)-state[ix])*weights[ix])**2,axis=1));jc[label]=masks[0]
   coverage.append({**common,'combination':label,'features':'|'.join(cols),'Train_count':masks[0],'Validation_count':masks[1],'reference_count':masks[2],'nearest_eligible_Train_distance':dd.min(),'region_definition':'same Train-quantile cell including dedicated out-of-range bins'})
  outside=[c for c in obs if float(record[c])<train[c].min() or float(record[c])>train[c].max()];indooroutside=[c for c in obs[:3] if c in outside]
  contextshare=shares['growth_context']+shares['exogenous_weather'];typ='A_endogenous_range_outside' if indooroutside else ('B_context_dominated_with_indoor_cell_coverage' if contextshare>.5 and jc['indoor']>0 else 'D_mixed_or_joint_coverage_not_resolved')
  cases.append({**common,'coverage_type':typ,'outside_Train_features':'|'.join(outside),'endogenous_outside_Train':'|'.join(indooroutside),'indoor_joint_Train_count':jc['indoor'],'context_weather_share':contextshare,'largest_group':max(shares,key=shares.get)})
 pd.DataFrame(feature).to_csv(D/'A_all_55_feature_contributions_coverage.csv',index=False);pd.DataFrame(groups).to_csv(D/'A_group_decomposition.csv',index=False);pd.DataFrame(neighbors).to_parquet(D/'A_nearest_50_train_neighbors.parquet',index=False);pd.DataFrame(coverage).to_csv(D/'A_joint_coverage.csv',index=False);pd.DataFrame(cases).to_csv(D/'A_case_summary.csv',index=False)
 for cohort in ('A','B'):
  for index in pro['cohorts'][cohort]:
   rows=trace(index);start=vl.loc[rows[0]['row_id']];qpos=pd.Index(qc.reg_date).get_loc(start.reg_date);initial=qc.iloc[qpos].copy();initial[obs]=start[obs];u=SensorHistoryStateUpdater(Identity(),qc.iloc[max(0,qpos-12):qpos]);u.reset(initial);recstates=[];tfstates=[];freestates=[];times=[];matched=True
   for n,row in enumerate(rows):
    recorded=vl.loc[row['row_id']];rec=recorded[obs].to_numpy(float);tf=u.current_physical[obs].to_numpy(float);free=np.array(row['current_physical']);recstates.append(rec);tfstates.append(tf);freestates.append(free);times.append(recorded.reg_date)
    if cohort=='B' and row['executed']:
     np.testing.assert_allclose(row['final'],recorded[actions].to_numpy(float),rtol=0,atol=1e-12);assert row['changed_actions_before']==0
    if row['executed']:
     nxt=qc.iloc[qpos+n+1].copy();assert pd.Timestamp(nxt.reg_date)==pd.Timestamp(recorded.target_timestamp);nxt[obs]=recorded[['next_'+c for c in obs]].to_numpy(float)
     # Teacher forcing uses recorded action regardless of safety changes in A.
     u.advance(current_scaled=u.current_physical,recorded_next_scaled=nxt,prediction_physical=dict(zip(obs[:3],nxt[obs[:3]].to_numpy(float))),action=recorded[actions].to_dict())
   arrs={'recorded':np.array(recstates),'teacher_forced':np.array(tfstates),'free':np.array(freestates)};scores={}
   for path,arr in arrs.items():
    scaled=rt.scaler.transform_frame(pd.DataFrame(arr,columns=obs))[obs].to_numpy(float);scores[path]=support._state_nn.kneighbors(scaled*weights,return_distance=True)[0][:,-1]
    if cohort=='B':
     for step in range(len(arr)):paths.append({'cohort':cohort,'episode_index':index,'step':step,'timestamp':times[step],'path':path,'state_distance':scores[path][step],'severe_ratio':scores[path][step]/threshold,'severe':bool(scores[path][step]>threshold),**dict(zip(obs,arr[step]))})
   np.testing.assert_allclose(scores['free'][-1],rows[-1]['info']['state_ood_score'],rtol=1e-6,atol=1e-8)
   for name in ('teacher_forced','free'):
    mismatch=~np.isclose(arrs[name],arrs['recorded'],rtol=1e-6,atol=1e-8)
    for j,c in enumerate(obs):
     hit=np.flatnonzero(mismatch[:,j]);differences.append({'cohort':cohort,'episode_index':index,'path':name,'feature':c,'group':group(c),'first_divergence_step':int(hit[0]) if len(hit) else None,'first_timestamp':str(times[hit[0]]) if len(hit) else None,'max_abs_difference':float(np.max(np.abs(arrs[name][:,j]-arrs['recorded'][:,j]))),'termination_difference':float(arrs[name][-1,j]-arrs['recorded'][-1,j])})
    hits=np.argwhere(mismatch);earliest=int(hits[:,0].min()) if len(hits) else None;first.append({'cohort':cohort,'episode_index':index,'path':name,'first_step':earliest,'first_features':'|'.join(obs[j] for j in np.flatnonzero(mismatch[earliest])) if earliest is not None else '', 'teacher_forced_contract_mismatch':bool(mismatch.any()) if name=='teacher_forced' else None})
   tf_summary.append({'cohort':cohort,'episode_index':index,'states':len(rows),'teacher_forced_max_abs_difference':float(np.abs(arrs['teacher_forced']-arrs['recorded']).max()),'teacher_forced_all55_parity':bool(np.isclose(arrs['teacher_forced'],arrs['recorded'],rtol=1e-6,atol=1e-8).all()),'recorded_first_severe_step':next((j for j,x in enumerate(scores['recorded']) if x>threshold),None),'teacher_first_severe_step':next((j for j,x in enumerate(scores['teacher_forced']) if x>threshold),None),'free_first_severe_step':next((j for j,x in enumerate(scores['free']) if x>threshold),None),'recorded_terminal_distance':scores['recorded'][-1],'teacher_terminal_distance':scores['teacher_forced'][-1],'free_terminal_distance':scores['free'][-1]})
   if cohort=='B':
    executed=[r for r in rows if r['executed']];x=vl.loc[[r['row_id'] for r in executed]];prediction=rt.predict(x[obs],x[actions].to_numpy(float)).to_numpy(float);truth=x[['target_next_'+c.removeprefix('obs_') for c in obs[:3]]].to_numpy(float)
    for n,r in enumerate(executed):
     for j,c in enumerate(obs[:3]):one_steps.append({'episode_index':index,'step':n,'target':c,'teacher_forced_one_step_prediction':prediction[n,j],'recorded_target':truth[n,j],'one_step_error':prediction[n,j]-truth[n,j],'free_prediction':r['prediction'][j],'free_path_difference':r['prediction'][j]-truth[n,j],'free_minus_teacher_one_step_prediction':r['prediction'][j]-prediction[n,j]})
   print('TEACHER',cohort,index,'states',len(rows),'all55',tf_summary[-1]['teacher_forced_all55_parity'],flush=True)
 pd.DataFrame(paths).to_parquet(D/'B_three_paths_all55.parquet',index=False);pd.DataFrame(differences).to_csv(D/'first_divergence_by_feature.csv',index=False);pd.DataFrame(first).to_csv(D/'first_divergence_by_run.csv',index=False);pd.DataFrame(tf_summary).to_csv(D/'teacher_forcing_summary.csv',index=False);pd.DataFrame(one_steps).to_csv(D/'B_one_step_vs_free_errors.csv',index=False)
 for index in pro['cohorts']['safety_changed']:
  rows=trace(index);r=next(x for x in rows if x.get('action_changed',False));info=r['info'];safety.append({'episode_index':index,'first_changed_step':r['step'],'timestamp':r['timestamp'],'proposed':json.dumps(r['proposed']),'final':json.dumps(r['final']),'support_projection':json.dumps(info.get('support_projection')),'weather_constraint':json.dumps(info.get('weather_constraint')),'fallback':info.get('safety_fallback_applied',False),'state_distance_before':info['state_ood_score'],'joint_distance_before':info['joint_ood_score'],'steps_until_first_state_severe':rows[-1]['step']-r['step'],'interpretation':'safety-modified path; not isolated Transition fault'})
 pd.DataFrame(safety).to_csv(D/'safety_changed_6_preserved.csv',index=False)
 joint=pd.read_csv(OLD/'ood_termination_attribution.csv');joint=joint.loc[joint.crop.eq('strawberry')&joint.termination_reason.eq('repeated_severe_action_ood')];assert len(joint)==16 and joint.classification.eq('recorded_severe_overlap').sum()==15;joint.to_csv(D/'joint_severe_16_preserved.csv',index=False)
 for r in pro['inputs']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
 write(D/'execution_complete.json',{'diagnosis_complete':True,'A_cases':39,'B_cases':7,'teacher_states':sum(x['states'] for x in tf_summary),'all_teacher_parity':all(x['teacher_forced_all55_parity'] for x in tf_summary),'original_hashes_unchanged':True,'strawberry_Step16':'BLOCKED','overall_Step17':'BLOCKED','no_fit_Test_modifications':True});print('DIAGNOSIS COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);a=ap.parse_args();freeze() if a.stage=='freeze' else run()
