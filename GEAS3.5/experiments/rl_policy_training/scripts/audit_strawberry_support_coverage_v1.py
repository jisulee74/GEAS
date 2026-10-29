"""Fixed-population support coverage audit, no fit or recalibration."""
from pathlib import Path
import os,sys,json,gzip
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from geas35.models.transition.handoff_b60_r6_v1 import sha
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1/strawberry'
S=ROOT/'experiments/rl_policy_training/artifacts/step15_b60_r6_v1/strawberry'
PRE=ROOT/'experiments/rl_policy_training/artifacts/strawberry_step16_ood_diagnosis_v1'
STEP=ROOT/'experiments/rl_policy_training/artifacts/step16_b60_r6_v1/strawberry'
D=ROOT/'experiments/rl_policy_training/artifacts/strawberry_support_coverage_audit_v1'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def table(f):
 f=f.copy()
 for c in f:
  if pd.api.types.is_float_dtype(f[c]):f[c]=f[c].map(lambda x:f'{x:.6g}')
 return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None)])
def strata(f):
 f=f.copy();dt=pd.to_datetime(f.reg_date);f['month']=dt.dt.month;f['year_month']=dt.dt.strftime('%Y-%m');f['season']=f.month.map({12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'});f['DAT_30d']=np.floor(f.obs_growth_stage_dat/30).astype(int);f['daynight']=np.where(f.obs_is_daytime>.5,'day','night');f['period_proxy']=f.series_id.astype(str)+'|'+f.segment_id.astype(str);return f
D.mkdir(exist_ok=False)
config=read(S/'state_action_support.json');obs=config['state_columns'];w=np.array(config['state_weights']);threshold=config['thresholds']['state_severe'];k=config['neighbors'];assert k==50 and len(obs)==55
joint={
'DAT_wind':['obs_growth_stage_dat','obs_outdoor_wind_speed'],
'DAT_DLI':['obs_growth_stage_dat','obs_derived_dli_mol_m2'],
'DAT_indoor':['obs_growth_stage_dat',*obs[:3]],
'wind_DLI':['obs_outdoor_wind_speed','obs_derived_dli_mol_m2'],
'T_Tout_prevvent':['obs_indoor_temp_c','obs_outdoor_temp_c','obs_prev_vent_pct'],
'indoor':obs[:3],
'indoor_prevvent':obs[:3]+['obs_prev_vent_pct'],
'indoor_prevvent_DAT':obs[:3]+['obs_prev_vent_pct','obs_growth_stage_dat'],
'prevvent_ventloss_tempdiff':['obs_prev_vent_pct','obs_derived_vent_loss_proxy','obs_derived_temp_diff_in_out_c'],
'current_previous_vent':['vent_pct','obs_prev_vent_pct'],
'current_previous_commands':config['action_columns']+['obs_prev_'+c for c in config['action_columns']],
'wind_light_day_target':['obs_outdoor_wind_speed','obs_outdoor_light','obs_is_daytime','obs_current_target_temp_c']}
rolling=['obs_derived_dli_mol_m2','obs_derived_light_sum','obs_derived_low_vpd_minutes_1h','obs_derived_rh90_minutes_1h','obs_derived_heating_degree_minutes','obs_derived_cooling_degree_minutes','obs_derived_ach_min_day','obs_derived_ach_min_night','obs_current_dewpoint_c','obs_current_condensation_margin_c','obs_derived_condensation_margin_pred_10m','obs_derived_condensation_risk_10m','obs_derived_temp_diff_in_out_c']
for c in rolling:joint['rolling_context:'+c]=[c,'obs_growth_stage_dat','obs_is_daytime','obs_current_target_temp_c']
files=[Path(__file__)]+[p for folder in (P,S,PRE) for p in sorted(folder.rglob('*')) if p.is_file()]
pro={'version':D.name,'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files],'geometry':'exact frozen 55 scaled coordinates/weights/k50; no feature removal or support fitting','diagnostic_radius':threshold,'radius_interpretation':'reuse frozen severe distance ONLY as common geometric radius for neighbor counts in different pools; never assign full-Train PASS; original threshold is calibrated only to frozen reference','sufficient_local_neighbors_definition':'>=50 within common diagnostic radius, descriptive not operational admission','bins':'eligible Train quantiles [0,.1,.25,.5,.75,.9,1]; exact maximum vs above maximum separated','joint_combinations':joint,'DAT_strata':'floor(DAT/30), no crop-specific tuning','season':'DJF/MAM/JJA/SON; raw timestamps KST','greenhouse_period':'series_id+segment_id metadata proxy, not externally verified crop-cycle definition','classification_4way':'data_gap=full radius count<50 OR absent source/series OR zero prespecified core joint cell; sampling_gap=full>=50 and reference<50; data only/full sparse, sampling only/ref scarce, both, or unclear','final_A_G':'overlapping evidence tags; A full radius<50; B full>=50 reference<50; C absent source/series or DAT bin or season-month combination; D command-related zero/sparse cell; E weather/context zero/sparse cell; F rolling-context zero/sparse cell; G multiple categories or none','sparse_joint_cell':'1..49 rows; absent=0; finer joint bins may be empty without physical impossibility','official_decisions':'strawberry Step16 BLOCKED / overall Step17 BLOCKED unchanged'}
write(D/'protocol.json',pro);write(D/'protocol_lock.json',{'sha256':sha(D/'protocol.json')});print('FROZEN',flush=True)
train=strata(pd.read_parquet(P/'train_physical.parquet'));val=strata(pd.read_parquet(P/'validation_physical.parquet'));ts=pd.read_parquet(P/'train_scaled.parquet');vs=pd.read_parquet(P/'validation_scaled.parquet').set_index('row_id');vl=val.set_index('row_id');registry=pd.read_parquet(S/'fit_row_registry.parquet');ri=registry.loc[registry.role.eq('reference'),'position'].to_numpy(int);ci=registry.loc[registry.role.eq('calibration'),'position'].to_numpy(int)
assert not set(train.iloc[ri].episode_id)&set(train.iloc[ci].episode_id)
# Reproduce original episode allocation (without changing any support).
episodes=np.array(sorted(train.episode_id.astype(str).unique()));rng=np.random.default_rng(config['seed']);rng.shuffle(episodes);ncal=min(len(episodes)-1,max(1,round(len(episodes)*.2)));held=set(episodes[:ncal]);cm=train.episode_id.astype(str).isin(held).to_numpy();rp=np.flatnonzero(~cm);cp=np.flatnonzero(cm)
assert set(ci)<=set(cp) and set(ri)<=set(rp)
ref=np.load(S/'support_reference.npz')['reference_state'];np.testing.assert_array_equal(ts.iloc[ri][obs].to_numpy(float),ref)
pools={'full_train':np.arange(len(train)),'reference':ri,'calibration_sample':ci,'reference_episode_pool':rp,'heldout_calibration_episode_pool':cp};arr=ts[obs].to_numpy(float)
columns=list(dict.fromkeys(obs+config['action_columns']));edges={c:np.unique(np.quantile(train[c],[0,.1,.25,.5,.75,.9,1])) for c in columns}
def bins(x,c):return np.where(np.asarray(x)>edges[c][-1],len(edges[c])+1,np.searchsorted(edges[c],x,side='right'))
write(D/'joint_bin_edges.json',{c:e.tolist() for c,e in edges.items()})
# Population composition, sample retention and episode-calibration partition.
stratum_cols=['source_confidence','series_id','segment_id','episode_id','DAT_30d','season','month','year_month','daynight','period_proxy']
strat=[]
for label,frame in [('full_train',train),('reference',train.iloc[ri]),('calibration_sample',train.iloc[ci]),('reference_episode_pool',train.iloc[rp]),('heldout_calibration_episode_pool',train.iloc[cp]),('validation',val)]:
 for c in stratum_cols:
  for v,count in frame[c].value_counts(dropna=False).items():strat.append({'population':label,'dimension':c,'stratum':str(v),'count':int(count),'population_rows':len(frame),'fraction':count/len(frame)})
pd.DataFrame(strat).to_csv(D/'population_strata.csv',index=False)
queries=pd.read_csv(PRE/'A_case_summary.csv');distance=[];neighbors=[];cells=[];stratum_query=[];marginal=[];case_rows=[]
core=['DAT_wind','DAT_DLI','DAT_indoor','wind_DLI','T_Tout_prevvent']
command=['indoor_prevvent','indoor_prevvent_DAT','prevvent_ventloss_tempdiff','current_previous_vent','current_previous_commands']
weather=['DAT_wind','DAT_DLI','wind_DLI','T_Tout_prevvent','wind_light_day_target']
for q in queries.to_dict('records'):
 index=q['episode_index'];record=vl.loc[q['row_id']];x=vs.loc[q['row_id'],obs].to_numpy(float);dd=np.sqrt(np.sum(((arr-x)*w)**2,axis=1));common={'episode_index':index,'row_id':q['row_id'],'timestamp':record.reg_date,'DAT':record.obs_growth_stage_dat,'source':record.source_confidence,'series':record.series_id,'segment':record.segment_id,'season':record.season,'month':record.month};count_by={};dist_by={}
 for pool,ix in pools.items():
  order=np.argsort(dd[ix])[:50];positions=ix[order];values=dd[positions];count_by[pool]=int((dd[ix]<=threshold).sum());dist_by[pool]=float(values[-1]);distance.append({**common,'pool':pool,'population_rows':len(ix),'nearest':values[0],'k50':values[-1],'kth_available_rank':len(values),'frozen_reference_severe_threshold':threshold,'within_common_radius':count_by[pool],'official_reclassification':False})
  for rank,position in enumerate(positions,1):
   row=train.iloc[position];neighbors.append({**common,'pool':pool,'rank':rank,'distance':dd[position],'train_row_id':row.row_id,'train_timestamp':row.reg_date,**{'train_'+c:row[c] for c in stratum_cols}})
 np.testing.assert_allclose(dist_by['reference'],q['state_distance'],rtol=1e-6,atol=1e-8)
 missing={}
 for c in stratum_cols:
  values={label:int(frame[c].eq(record[c]).sum()) for label,frame in [('full_train',train),('reference',train.iloc[ri]),('calibration',train.iloc[ci]),('validation',val)]};missing[c]=values['full_train']==0;stratum_query.append({**common,'dimension':c,'value':record[c],**values})
 for c in obs:
  a=train[c].to_numpy(float);v=float(record[c]);marginal.append({**common,'feature':c,'value':v,'train_min':a.min(),'train_p01':np.quantile(a,.01),'train_p50':np.median(a),'train_p99':np.quantile(a,.99),'train_max':a.max(),'validation_min':val[c].min(),'validation_p99':val[c].quantile(.99),'validation_max':val[c].max(),'outside_full_train_range':v<a.min() or v>a.max()})
 cell_counts={}
 for label,cols in joint.items():
  masks={}
  for pop,f in [('full_train',train),('reference',train.iloc[ri]),('calibration',train.iloc[ci]),('validation',val)]:
   mask=np.ones(len(f),bool)
   for c in cols:mask &= bins(f[c].to_numpy(float),c)==bins(float(record[c]),c)
   masks[pop]=int(mask.sum())
  cell_counts[label]=masks['full_train'];cells.append({**common,'combination':label,'features':'|'.join(cols),'cell':json.dumps({c:int(bins(float(record[c]),c)) for c in cols}),'physical_query_values':json.dumps({c:float(record[c]) for c in cols}),**masks,'full_train_status':'absent' if masks['full_train']==0 else 'sparse' if masks['full_train']<50 else '>=50_rows'})
 data_gap=count_by['full_train']<50 or missing['source_confidence'] or missing['series_id'] or any(cell_counts[c]==0 for c in core)
 sampling_gap=count_by['full_train']>=50 and count_by['reference']<50
 typ='both' if data_gap and sampling_gap else 'full_train_sparse_or_missing_domain' if data_gap else 'reference_sampling_scarcity' if sampling_gap else 'unclear'
 tags=[]
 if count_by['full_train']<50:tags.append('A')
 if sampling_gap:tags.append('B')
 if any(missing[c] for c in ('source_confidence','series_id','DAT_30d')) or not ((train.season==record.season)&(train.month==record.month)&(train.series_id==record.series_id)).any():tags.append('C')
 if any(cell_counts[c]<50 for c in command):tags.append('D')
 if any(cell_counts[c]<50 for c in weather):tags.append('E')
 if any(cell_counts['rolling_context:'+c]<50 for c in rolling):tags.append('F')
 if len(tags)>1 or not tags:tags.append('G')
 case_rows.append({**common,'fourway_class':typ,'A_G_tags':'|'.join(tags),'full_train_radius_count':count_by['full_train'],'reference_radius_count':count_by['reference'],'sampled_calibration_radius_count':count_by['calibration_sample'],'reference_episode_pool_radius_count':count_by['reference_episode_pool'],'heldout_calibration_episode_pool_radius_count':count_by['heldout_calibration_episode_pool'],'neighbors_lost_by_episode_holdout':count_by['heldout_calibration_episode_pool'],'neighbors_lost_by_reference_row_sampling':count_by['reference_episode_pool']-count_by['reference'],'expected_reference_neighbors_after_uniform_row_sampling':count_by['reference_episode_pool']*len(ri)/len(rp),'full_k50':dist_by['full_train'],'reference_k50':dist_by['reference'],'missing_series':missing['series_id'],'missing_DAT_bin':missing['DAT_30d'],'indoor_cell_count':cell_counts['indoor'],'indoor_prevvent_cell_count':cell_counts['indoor_prevvent'],'indoor_prevvent_DAT_cell_count':cell_counts['indoor_prevvent_DAT']})
 print('CASE',index,typ,count_by['full_train'],count_by['reference'],flush=True)
for name,rows in [('distance_pools',distance),('joint_cells',cells),('query_strata',stratum_query),('marginal_ranges',marginal),('case_classification',case_rows)]:pd.DataFrame(rows).to_csv(D/(name+'.csv'),index=False)
pd.DataFrame(neighbors).to_parquet(D/'nearest_50_neighbors_all_pools.parquet',index=False)
f=pd.DataFrame(case_rows);tagrows=[]
for label in 'ABCDEFG':
 subset=f.loc[f.A_G_tags.str.split('|').map(lambda tags:label in tags)];tagrows.append({'category':label,'count':len(subset),'example_episode_indices':','.join(map(str,subset.episode_index.head(5)))})
pd.DataFrame(tagrows).to_csv(D/'overlapping_category_counts.csv',index=False)
# Sparse full->reference omissions by the queried case strata; separate from entire pool representation.
bias=[]
for c in ['source','series','segment','DAT','season','month']:
 for value,g in f.groupby(c):bias.append({'dimension':c,'value':value,'cases':len(g),'full_at_least50_reference_below50':int(((g.full_train_radius_count>=50)&(g.reference_radius_count<50)).sum()),'median_full_neighbors':g.full_train_radius_count.median(),'median_reference_neighbors':g.reference_radius_count.median()})
pd.DataFrame(bias).to_csv(D/'query_sampling_gap_strata.csv',index=False)
# Clean5 appendix at starts, not mixed with39 query conclusions.
clean=pd.read_csv(PRE/'B_whole_path_counterfactual_classification.csv');clean=clean.loc[clean.whole_path_classification.eq('teacher_normal_free_only_severe')];assert len(clean)==5;appendix=[]
for index in clean.episode_index.astype(int):
 p=STEP/'episodes'/f'{index:04d}.jsonl.gz';meta=read(p.with_suffix('').with_suffix('.json'));assert sha(p)==meta['trace_sha256']
 with gzip.open(p,'rt') as stream:first=json.loads(next(stream))
 x=vs.loc[first['row_id'],obs].to_numpy(float);dd=np.sqrt(np.sum(((arr-x)*w)**2,axis=1));row={'episode_index':index,'timestamp':first['timestamp']}
 for label,ix in [('full_train',pools['full_train']),('reference',ri),('calibration',ci)]:
  values=np.sort(dd[ix]);row[label+'_nearest']=values[0];row[label+'_k50']=values[49]
 row['frozen_severe_threshold']=threshold;row['reference_severe_margin']=threshold-row['reference_k50'];row['relative_margin']=row['reference_severe_margin']/threshold;appendix.append(row)
pd.DataFrame(appendix).to_csv(D/'clean_rollout5_start_coverage_appendix.csv',index=False)
for r in pro['inputs']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
write(D/'audit_complete.json',{'status':'AUDIT_COMPLETE','queries':39,'full_train_sufficient_reference_sparse':int(((f.full_train_radius_count>=50)&(f.reference_radius_count<50)).sum()),'full_train_below50':int((f.full_train_radius_count<50).sum()),'fourway_counts':f.fourway_class.value_counts().to_dict(),'original_hashes_unchanged':True,'strawberry_Step16':'BLOCKED','overall_Step17':'BLOCKED','official_changes':False,'no_Test_fit_PPO':True})
print('AUDIT COMPLETE',read(D/'audit_complete.json'),flush=True)
