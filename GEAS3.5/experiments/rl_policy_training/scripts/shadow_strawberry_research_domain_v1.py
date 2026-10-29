"""Freeze Train-only low-dimensional domain first, then separate shadow evaluation."""
from pathlib import Path
import os,sys,json,argparse
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from geas35.models.transition.handoff_b60_r6_v1 import sha
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1';S=ROOT/'experiments/rl_policy_training/artifacts/step15_b60_r6_v1/strawberry';D=ROOT/'experiments/rl_policy_training/artifacts/strawberry_research_domain_shadow_v1'
LEVEL=['IN_DOMAIN','LOW_SUPPORT','OUT_OF_DOMAIN']
META=['row_id','crop','series_id','segment_id','episode_id','reg_date','source_confidence','mdp_v1_rollout_id']
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def binvalue(x,edge):
 x=np.asarray(x,dtype=float);return np.where(x>edge[-1],len(edge)+1,np.searchsorted(edge,x,side='right')).astype(int)
def keymatrix(frame,cols,edges):return np.column_stack([binvalue(frame[c].to_numpy(float),edges[c]) for c in cols])
def occupancy(frame,cols,edges):
 keys=keymatrix(frame,cols,edges);f=pd.DataFrame(keys,columns=cols);f['episode']=frame.episode_id.astype(str).to_numpy();return {'|'.join(map(str,idx if isinstance(idx,tuple) else (idx,))):{'rows':int(len(g)),'episodes':int(g.episode.nunique())} for idx,g in f.groupby(cols,sort=True)}
def classify(frame,rule,kind='state'):
 n=len(frame);level=np.zeros(n,dtype=int);why=[[] for _ in range(n)];details=[]
 specs=rule['state_cells'] if kind=='state' else rule['action_cells']
 for name,spec in specs.items():
  keys=keymatrix(frame,spec['columns'],rule['edges']);counts=[];episodes=[]
  for k in keys:
   evidence=spec['occupancy'].get('|'.join(map(str,k)),{'rows':0,'episodes':0});counts.append(evidence['rows']);episodes.append(evidence['episodes'])
  counts=np.array(counts);episodes=np.array(episodes);lev=np.where(counts==0,2,np.where((counts<rule['minimum_rows'])|(episodes<rule['minimum_episodes']),1,0));level=np.maximum(level,lev)
  for i in np.flatnonzero(lev):why[i].append(name+(':unobserved_cell' if lev[i]==2 else ':sparse_or_single_episode'))
  details.append(pd.DataFrame({'position':np.arange(n),'condition':name,'Train_rows':counts,'Train_episodes':episodes,'level':lev}))
 if kind=='state':
  dt=frame.obs_growth_stage_dat.to_numpy(float);db=np.floor(dt/30).astype(int);count=np.array([rule['DAT_30d'].get(str(v),{'rows':0})['rows'] for v in db]);ep=np.array([rule['DAT_30d'].get(str(v),{'episodes':0})['episodes'] for v in db]);lev=np.where(count==0,2,np.where((count<50)|(ep<2),1,0));level=np.maximum(level,lev)
  for i in np.flatnonzero(lev):why[i].append('DAT_30d:unobserved' if lev[i]==2 else 'DAT_30d:sparse')
  finite=np.isfinite(frame[rule['observation_columns']].to_numpy(float)).all(axis=1);level[~finite]=2
  for i in np.flatnonzero(~finite):why[i].append('invalid_observation')
  for c in rule['marginal_axes']:
   x=frame[c].to_numpy(float);lo,hi=rule['ranges'][c];outside=(x<lo)|(x>hi)
   level[outside]=2
   for i in np.flatnonzero(outside):why[i].append(c+':outside_Train_range')
  unknown=~frame.source_confidence.isin(rule['known_source_confidence']).to_numpy();level[unknown]=np.maximum(level[unknown],1)
  for i in np.flatnonzero(unknown):why[i].append('source_provenance_unverified')
 return level,why,pd.concat(details,ignore_index=True)
def freeze():
 D.mkdir(exist_ok=False);schema=read(P/'schema.json');obs=schema['observation_columns'];actions=schema['actions'];config=read(S/'state_action_support.json');assert len(obs)==55 and config['state_columns']==obs and config['neighbors']==50
 # Recipe locked before Train retention and before any new Validation read.
 recipe={'version':D.name,'state_candidate_count':1,'quantile_edges':[0,.1,.25,.5,.75,.9,1],'DAT_bin_width':30,'minimum_rows':50,'minimum_distinct_episodes':2,'rationale':'k50 motivates minimum observations, not a calibrated probability; >=2 episodes prevents a single episode being the only evidence; project conservative candidate, no tuning','no_series_exclusion':True,'current_action_separate':True,'source_rule':'unknown source LOW_SUPPORT; known legacy source not physical ground truth','out_of_domain':'nonfinite or monitored axis outside Train range or any selected low-dimensional combination entirely absent','low_support':'observed cell but <50 rows or <2 episodes, or unverified source; OUT precedence','in_domain':'all selected conditions supported; still subject to UNCHANGED local Step15 OOD and action safety','no_55dim_cells':True,'maximum_joint_dimension':3,'prior_Validation_exposure':'previous audits known and user-requested axes retained; this is not a prospectively blinded study; no Validation rows/results/cutoff search in rule construction'}
 write(D/'recipe_before_Train_retention.json',recipe)
 cols=META+obs+actions;train=pd.read_parquet(P/'strawberry/train_physical.parquet',columns=cols)
 state_cols={
 'growth_weather:DAT_wind':['obs_growth_stage_dat','obs_outdoor_wind_speed'],
 'growth_weather:DAT_DLI':['obs_growth_stage_dat','obs_derived_dli_mol_m2'],
 'growth_weather:wind_DLI':['obs_outdoor_wind_speed','obs_derived_dli_mol_m2'],
 'growth_weather:light_DLI':['obs_outdoor_light','obs_derived_dli_mol_m2'],
 'growth_indoor:DAT_T':['obs_growth_stage_dat',obs[0]],
 'growth_indoor:DAT_RH':['obs_growth_stage_dat',obs[1]],
 'growth_indoor:DAT_CO2':['obs_growth_stage_dat',obs[2]],
 'indoor_command:T_RH':obs[:2],
 'indoor_command:RH_CO2':obs[1:3],
 'indoor_command:T_prevvent':[obs[0],'obs_prev_vent_pct'],
 'indoor_command:RH_prevvent':[obs[1],'obs_prev_vent_pct'],
 'indoor_command:CO2_prevvent':[obs[2],'obs_prev_vent_pct'],
 'vent_physics:T_Tout_prevvent':[obs[0],'obs_outdoor_temp_c','obs_prev_vent_pct'],
 'vent_physics:ventloss_prevvent':['obs_derived_vent_loss_proxy','obs_prev_vent_pct']}
 action_cols={'command_pair:'+a:['obs_prev_'+a,a] for a in actions}
 axes=sorted(set(c for cs in state_cols.values() for c in cs));edgecols=sorted(set(axes+actions+['obs_prev_'+a for a in actions]));edges={c:np.unique(np.quantile(train[c],recipe['quantile_edges'])).tolist() for c in edgecols}
 # Persist mappings before classifying either split; occupancy is descriptive, no model fitting.
 cells={name:{'columns':cs,'occupancy':occupancy(train,cs,edges)} for name,cs in state_cols.items()};acells={name:{'columns':cs,'occupancy':occupancy(train,cs,edges)} for name,cs in action_cols.items()}
 dat={str(int(b)):{'rows':len(g),'episodes':g.episode_id.nunique(),'min_DAT':g.obs_growth_stage_dat.min(),'max_DAT':g.obs_growth_stage_dat.max()} for b,g in train.groupby(np.floor(train.obs_growth_stage_dat/30).astype(int))}
 bindings=[P/'strawberry/train_physical.parquet',P/'strawberry/train_provenance.parquet',P/'strawberry/scaler_fit_registry.parquet',P/'strawberry/observation_scaler.json',P/'schema.json',P/'runtime_contract.json',S/'state_action_support.json',S/'support_reference.npz',S/'fit_row_registry.parquet',Path(__file__)]
 rule={'version':D.name,'observation_columns':obs,'actions':actions,'minimum_rows':50,'minimum_episodes':2,'state_cells':cells,'action_cells':acells,'edges':edges,'DAT_30d':dat,'marginal_axes':axes,'ranges':{c:[float(train[c].min()),float(train[c].max())] for c in axes},'known_source_confidence':train.source_confidence.unique().tolist(),'train_series_metadata_only':train.series_id.unique().tolist(),'recipe_sha256':sha(D/'recipe_before_Train_retention.json'),'input_hashes':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in bindings],'role':'shadow research validity metadata; not physical safety; no feature/scaler/support modification'}
 write(D/'frozen_domain_rule.json',rule);write(D/'rule_lock.json',{'sha256':sha(D/'frozen_domain_rule.json'),'recipe_sha256':sha(D/'recipe_before_Train_retention.json'),'Validation_read_before_rule_freeze':False})
 lev,why,detail=classify(train,rule);al,aw,ad=classify(train,rule,'action');reg=train[META].copy();reg['domain']=np.array(LEVEL)[lev];reg['action_domain']=np.array(LEVEL)[al];reg['reason']=['|'.join(x) for x in why];reg.to_parquet(D/'Train_domain_sidecar.parquet',index=False);detail.to_parquet(D/'Train_condition_evidence.parquet',index=False)
 distributions=[]
 for c in axes:
  distributions.append({'feature':c,'min':train[c].min(),'p01':train[c].quantile(.01),'p50':train[c].median(),'p99':train[c].quantile(.99),'max':train[c].max()})
 pd.DataFrame(distributions).to_csv(D/'Train_axes_summary.csv',index=False)
 datrows=[]
 for b in range(int(np.floor(train.obs_growth_stage_dat.min()/30))-1,int(np.floor(train.obs_growth_stage_dat.max()/30))+2):
  evidence=dat.get(str(b),{'rows':0,'episodes':0});label='UNOBSERVED' if evidence['rows']==0 else 'SUPPORTED' if evidence['rows']>=50 and evidence['episodes']>=2 else 'SPARSE_EDGE';datrows.append({'DAT_bin':b,'start':b*30,'end_exclusive':(b+1)*30,**evidence,'coverage':label})
 pd.DataFrame(datrows).to_csv(D/'Train_DAT_coverage.csv',index=False)
 write(D/'Train_retention.json',{'rows':len(train),'state_counts':reg.domain.value_counts().to_dict(),'action_counts':reg.action_domain.value_counts().to_dict(),'in_sample_only':True,'uses_Validation':False});print('RULE FROZEN',sha(D/'frozen_domain_rule.json'),read(D/'Train_retention.json'),flush=True)
def check():
 rule=read(D/'frozen_domain_rule.json');assert sha(D/'frozen_domain_rule.json')==read(D/'rule_lock.json')['sha256'];assert sha(D/'recipe_before_Train_retention.json')==rule['recipe_sha256']
 for x in rule['input_hashes']:assert sha(ROOT/x['path'])==x['sha256']
 return rule
def shadow():
 rule=check();obs=rule['observation_columns'];actions=rule['actions'];vpath=P/'strawberry/validation_physical.parquet'
 extras=['next_'+c for c in obs]+['target_timestamp'];val=pd.read_parquet(vpath,columns=META+obs+actions+extras);vhash=sha(vpath)
 lev,why,detail=classify(val,rule);actionlevel,actionwhy,ad=classify(val,rule,'action');target=val[META].copy();target[obs]=val[['next_'+c for c in obs]].to_numpy(float);target['reg_date']=val.target_timestamp;tl,tw,_=classify(target,rule)
 transition=np.maximum.reduce([lev,tl,actionlevel]);out=val[META].copy();out['row_domain']=np.array(LEVEL)[lev];out['target_domain']=np.array(LEVEL)[tl];out['action_domain']=np.array(LEVEL)[actionlevel];out['transition_domain']=np.array(LEVEL)[transition];out['state_reasons']=['|'.join(x) for x in why];out['action_reasons']=['|'.join(x) for x in actionwhy];out['target_reasons']=['|'.join(x) for x in tw];out['unseen_series_metadata']=~val.series_id.isin(rule['train_series_metadata_only']);out.to_parquet(D/'Validation_shadow_sidecar.parquet',index=False);detail.to_parquet(D/'Validation_condition_evidence.parquet',index=False)
 # Explicit invariants: series label and current action do not influence observation domain.
 altered=val.copy();altered['series_id']='unseen_series_probe';altered[actions]=1-altered[actions].to_numpy(float);lev2,why2,_=classify(altered,rule);np.testing.assert_array_equal(lev,lev2);assert why==why2
 again,againwhy,_=classify(val,rule);np.testing.assert_array_equal(lev,again);assert why==againwhy
 registry=[]
 for run,g in out.groupby('mdp_v1_rollout_id',sort=False):
  values=g.transition_domain.map({v:i for i,v in enumerate(LEVEL)}).to_numpy();registry.append({'rollout_id':str(run),'episode_id':g.episode_id.iloc[0],'series':g.series_id.iloc[0],'rows':len(g),'start_row_id':g.row_id.iloc[0],'start_state_domain':g.row_domain.iloc[0],'whole_run_domain':LEVEL[int(values.max())],'in_domain_transition_fraction':float(np.mean(values==0)),'longest_in_domain_sequence':max((len(x) for x in np.split(np.arange(len(values)),np.flatnonzero(np.diff(values==0))+1) if len(x) and values[x[0]]==0),default=0)})
 runs=pd.DataFrame(registry);runs.to_csv(D/'Validation_run_shadow.csv',index=False)
 step=ROOT/'experiments/rl_policy_training/artifacts/step16_b60_r6_v1';prev=ROOT/'experiments/rl_policy_training/artifacts/strawberry_step16_ood_diagnosis_v1';audit=ROOT/'experiments/rl_policy_training/artifacts/strawberry_support_coverage_audit_v1'
 eps=pd.read_csv(step/'strawberry/episode_summary.csv',usecols=['episode_index','rollout_id','termination_reason'],dtype={'rollout_id':str});joined=runs.merge(eps,left_on='rollout_id',right_on='rollout_id',validate='one_to_one');assert len(joined)==len(runs);joined.to_csv(D/'Validation_Step16_outcome_shadow.csv',index=False)
 lookup=out.set_index('row_id');cases=pd.read_csv(prev/'A_case_summary.csv',usecols=['episode_index','row_id']);cases=cases.join(lookup[['row_domain','transition_domain','series_id','state_reasons']],on='row_id');cases.to_csv(D/'recorded_severe39_domain.csv',index=False)
 sampling=pd.read_csv(audit/'case_classification.csv');sampling=sampling.loc[sampling.full_train_radius_count.ge(50)&sampling.reference_radius_count.lt(50),['episode_index','row_id']].join(lookup[['row_domain','transition_domain','series_id','state_reasons']],on='row_id');assert len(sampling)==16;sampling.to_csv(D/'reference_shortage16_domain.csv',index=False)
 clean=pd.read_csv(prev/'B_whole_path_counterfactual_classification.csv');clean=clean.loc[clean.whole_path_classification.eq('teacher_normal_free_only_severe'),['episode_index']];clean=clean.merge(joined[['episode_index','start_row_id','start_state_domain','series']],on='episode_index',validate='one_to_one');assert len(clean)==5;clean.to_csv(D/'clean5_start_domain.csv',index=False)
 # Diagnose residual series dependence only after rule lock; no rule feedback.
 out.groupby(['series_id','row_domain']).size().rename('rows').reset_index().to_csv(D/'Validation_series_domain.csv',index=False)
 reasonrows=[]
 for series,g in out.groupby('series_id'):
  for reason,count in g.state_reasons.str.split('|').explode().value_counts().items():
   if reason:reasonrows.append({'series':series,'reason':reason,'rows':int(count),'fraction':count/len(g)})
 pd.DataFrame(reasonrows).to_csv(D/'Validation_series_physical_reasons.csv',index=False)
 support=pd.read_parquet(S/'validation_support_scores.parquet',columns=['row_id','state_severe','state_distance']);cross=out[['row_id','row_domain']].merge(support,on='row_id',validate='one_to_one');cross.groupby('row_domain').agg(rows=('row_id','size'),frozen_state_severe_rows=('state_severe','sum'),frozen_state_severe_rate=('state_severe','mean')).to_csv(D/'domain_vs_existing_support.csv')
 stats={'row_counts':out.row_domain.value_counts().to_dict(),'transition_counts':out.transition_domain.value_counts().to_dict(),'run_counts':runs.whole_run_domain.value_counts().to_dict(),'run_start_counts':runs.start_state_domain.value_counts().to_dict(),'rows':len(out),'runs':len(runs),'recorded_severe39':cases.row_domain.value_counts().to_dict(),'reference_shortage16':sampling.row_domain.value_counts().to_dict(),'clean5':clean.start_state_domain.value_counts().to_dict(),'rule_sha256':sha(D/'frozen_domain_rule.json'),'series_and_current_action_invariance':True,'deterministic_replay':True,'official_changes':False,'strawberry_Step16':'BLOCKED','overall_Step17':'BLOCKED'}
 check();assert sha(vpath)==vhash
 bindings=[vpath,step/'strawberry/episode_summary.csv',prev/'A_case_summary.csv',prev/'B_whole_path_counterfactual_classification.csv',audit/'case_classification.csv',S/'validation_support_scores.parquet'];write(D/'shadow_only_input_hashes.json',[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in bindings]);write(D/'shadow_summary.json',stats);print('SHADOW COMPLETE',stats,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','shadow']);a=ap.parse_args();freeze() if a.stage=='freeze' else shadow()
