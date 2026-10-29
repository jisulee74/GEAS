"""Read-only source audit. Explicit Train/Validation predicates; no fit or dataset build."""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.datasets import _transition_exclusion_masks,_exclusive_reasons,MDP_V1_REQUIRED_SOURCE_ACTION_COLUMNS
from geas35.rl.mdp_v1 import mdp_v1_observation_columns,MDP_V1_TRANSITION_TARGET_COLUMNS
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry'
OLD=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_diagnosis_b_v1'
D=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_source_audit_b_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,x): (D/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
inputs={}
def read(p,**kw):
 inputs[str(p.relative_to(ROOT))]=sha(p)
 return pd.read_parquet(p,**kw)
def csv(name,f):f.to_csv(D/name,index=False)
def runs(f,columns,near=0):
 f=f.sort_values(['series_id','segment_id','reg_date']).reset_index(drop=True)
 same=f.reg_date.diff().eq(pd.Timedelta(minutes=5))
 for k in ('series_id','segment_id'):same &= f[k].eq(f[k].shift())
 for c in columns:same &= (f[c]-f[c].shift()).abs().le(near)
 rows=[]
 for _,g in f.groupby((~same).cumsum()):
  if len(g)<2:continue
  row={'start':g.reg_date.iloc[0],'end':g.reg_date.iloc[-1],'rows':len(g),'elapsed_minutes':(g.reg_date.iloc[-1]-g.reg_date.iloc[0]).total_seconds()/60,'nominal_covered_minutes':len(g)*5,'series':g.series_id.iloc[0],'raw_rows':int(g.raw_idx.notna().sum()),'unique_raw_ids':g.raw_idx.nunique(),'unique_raw_reg_date':g.raw_reg_date.nunique(),'raw_measurement_timestamp_available':False,'co2_min':g.raw_in_co2.min(),'co2_max':g.raw_in_co2.max()}
  for c in ('raw_in_co2','raw_in_co2_2','raw_out_light','raw_in_temp','raw_in_hum','raw_cont_skyl_vol','raw_cont_skyr_vol'):
   row[c+'_unique']=g[c].nunique();row[c+'_changes']=int(g[c].diff().ne(0).iloc[1:].sum())
  rows.append(row)
 return pd.DataFrame(rows)
if __name__=='__main__':
 D.mkdir(exist_ok=False)
 write('protocol.json',{'scope':'strawberry Train/Validation read-only source audit','threshold_co2_ppm':1500,'high_vent':.75,'exact_plateau':'identical raw value(s), contiguous 5min same series/segment; daily episode boundary does not break physical continuity','near_plateau':'adjacent change <=1ppm, descriptive only','no_changes':'no fit, Test, QC change, episode exclusion, dataset/scaler/support rebuild','timestamp_caution':'DB reg_date and new idx do not prove a new physical measurement','timezone':'KST external system contract'})
 allframes=[];rawschemas={}
 for split in ('train','validation'):
  q=read(P/f'{split}_prepared_qc.parquet');a=read(P/f'{split}_provenance.parquet');assert q.reg_date.equals(a.reg_date)
  rawparts=[]
  for series,g in q.groupby('series_id'):
   path=ROOT/f'offline_dataset_preparation/datasets/00_raw/strawberry/{series}.parquet'
   raw=read(path,filters=[('reg_date','in',g.reg_date.tolist())]);rawschemas[series]=list(raw.columns)
   assert raw.reg_date.isin(g.reg_date).all();assert not raw.duplicated(['series_id','reg_date']).any()
   rawparts.append(raw)
  raw=pd.concat(rawparts,ignore_index=True)
  cols=['idx','reg_date','series_id','source_segment_id','source_file_path','iot_data_idx','in_co2','in_co2_2','out_light','out_windsp','in_temp','in_temp2','in_hum','in_hum2','cont_skyl_vol','cont_skyr_vol']
  for c in cols:
   if c not in ('reg_date','series_id','source_segment_id','source_file_path'):raw[c]=pd.to_numeric(raw[c],errors='coerce')
  raw=raw[cols].rename(columns={c:'raw_'+c for c in cols if c!='series_id'})
  f=q.merge(raw,left_on=['series_id','reg_date'],right_on=['series_id','raw_reg_date'],how='left',validate='one_to_one')
  f['split']=split;f['B_eligible']=a.eligible.to_numpy();f['B_reason']=a.reason.to_numpy();f['current_vent']=a.current_vent_pct.to_numpy();f['previous_vent']=a.previous_vent_pct.to_numpy();f['previous_previous_vent']=a.previous_previous_vent_pct.to_numpy();f['source_confidence']=a.source_confidence.to_numpy();f['command_change']=f.current_vent-f.previous_vent
  f['original_measurement_timestamp']=pd.NaT;f['separate_collection_timestamp']=pd.NaT
  allframes.append(f)
  for label,columns,near in [('sensor1',['raw_in_co2'],0),('sensor2',['raw_in_co2_2'],0),('joint',['raw_in_co2','raw_in_co2_2'],0),('sensor1_near1ppm',['raw_in_co2'],1),('co2_light_temperature_humidity',['raw_in_co2','raw_in_co2_2','raw_out_light','raw_in_temp','raw_in_hum'],0)]:
   r=runs(f,columns,near);r['split']=split;r['definition']=label;csv(f'{split}_{label}_plateaus.csv',r)
  if split=='train':
   membership=read(OLD/'A_B_train_membership.parquet');assert q.reg_date.equals(membership.reg_date)
   masks=_transition_exclusion_masks(q,mdp_v1_observation_columns(q),MDP_V1_REQUIRED_SOURCE_ACTION_COLUMNS);reasons=_exclusive_reasons(masks,q.index)
   assert np.array_equal(reasons.eq(''),membership.in_A_train)
   f['in_A']=membership.in_A_train.to_numpy();f['A_reason']=reasons.to_numpy()
   f['population']=np.select([~f.in_A,f.in_A&~f.B_eligible],['already_excluded_A','additional_excluded_B'],default='B_eligible')
   badcols=mdp_v1_observation_columns(q)
   f['A_nonfinite_features']=['|'.join(c for c in badcols if not np.isfinite(float(row[c]))) for _,row in q.iterrows()]
   for k,v in masks.items():f['A_flag_'+k]=v.to_numpy()
   high=f.loc[f.obs_indoor_co2_ppm.ge(1500)].copy()
   csv('train_high_co2_all_rows.csv.gz',high)
   csv('train_high_exclusion_combinations.csv',high.groupby(['population','A_reason','B_reason'],dropna=False).size().reset_index(name='rows'))
   csv('train_high_examples.csv',high.groupby(['population','A_reason','B_reason'],dropna=False).head(2))
   # Preserve B event provenance for every high row, including exact unknown/boundary fields.
   csv('train_high_command_provenance.csv',a.loc[q.obs_indoor_co2_ppm.ge(1500)])
   write('A_rule_reproduction.json',{'all_train_membership_exact_match':True,'rows':len(q),'high_partition':high.population.value_counts().to_dict(),'exclusive_A_reasons':high.A_reason.value_counts().to_dict(),'overlapping_A_masks':{k:int(v[q.obs_indoor_co2_ppm.ge(1500)].sum()) for k,v in masks.items()}})
 frames=pd.concat(allframes,ignore_index=True)
 worst=frames.loc[frames.split.eq('validation')&frames.reg_date.ge('2025-11-10')&frames.reg_date.lt('2025-11-12')].copy()
 csv('worst_episode_5min_source_trace.csv.gz',worst)
 write('raw_schema.json',rawschemas)
 # Check representative choice directly against raw pre-QC values and per-sensor flags.
 def usable(f,c):
  z=f[c+'_raw_value'].notna()
  for suffix in ('_missing_flag','_rule_outlier_flag','_ai_outlier_flag'):z &= ~f[c+suffix].fillna(0).astype(bool)
  return z
 u1=usable(worst,'in_co2');u2=usable(worst,'in_co2_2');expected=np.select([u1,~u1&u2],['sensor1','sensor2'],default='other')
 assert (worst.in_co2_representative_source.loc[u1].eq('sensor1')).all()
 np.testing.assert_allclose(worst.loc[u1,'obs_indoor_co2_ppm'],worst.loc[u1,'in_co2_raw_value'],rtol=1e-6,atol=1e-8)
 write('representative_audit.json',{'rows':len(worst),'representative_source':worst.in_co2_representative_source.value_counts().to_dict(),'both_usable':int((u1&u2).sum()),'both_usable_difference_ge500':int((u1&u2&(worst.in_co2_raw_value-worst.in_co2_2_raw_value).abs().ge(500)).sum()),'missing_disagreement_columns':[c for c in worst if 'disagree' in c],'sensor1_priority_reproduced':True,'CO2_domain_ppm':[200,5000]})
 # Conditional coverage: only fixed descriptive bins, never filter training or adjust models.
 frames['hour']=frames.reg_date.dt.hour;frames['date']=frames.reg_date.dt.date.astype(str)
 for name,col,edges in [('DAT','obs_growth_stage_dat',[-np.inf,30,60,90,120,180,np.inf]),('light','obs_outdoor_light',[-np.inf,0,10,200,500,800,np.inf]),('T','obs_indoor_temp_c',[-np.inf,15,25,35,45,np.inf]),('RH','obs_indoor_humidity_pct',[-np.inf,40,60,80,np.inf])]:frames[name+'_bin']=pd.cut(frames[col],edges,right=False).astype(str)
 frames['vent_bin']=pd.cut(frames.current_vent,[-np.inf,.01,.25,.5,.75,1.00001,np.inf],right=False).astype(str)
 frames['co2_bin']=pd.cut(frames.obs_indoor_co2_ppm,[-np.inf,600,1000,1500,np.inf],right=False).astype(str)
 frames['command_pattern']=np.select([frames.command_change.isna(),frames.command_change.abs().lt(1e-12),frames.command_change.gt(0)],['unknown','unchanged','opening'],default='closing')
 dims=['co2_bin','vent_bin','DAT_bin','light_bin','T_bin','RH_bin','command_pattern']
 csv('coverage_joint_cells.csv',frames.groupby(['split','B_eligible','series_id']+dims,dropna=False).size().reset_index(name='rows'))
 marginal=[]
 for col in ['series_id','date','hour']+dims:
  for population,mask in [('all',pd.Series(True,index=frames.index)),('co2_ge1500',frames.obs_indoor_co2_ppm.ge(1500)),('co2_ge1500_vent_ge75',frames.obs_indoor_co2_ppm.ge(1500)&frames.current_vent.ge(.75))]:
   z=frames.loc[mask].groupby(['split','B_eligible',col],dropna=False).size().reset_index(name='rows').rename(columns={col:'value'});z['dimension']=col;z['population']=population;marginal.append(z)
 csv('coverage_marginals.csv',pd.concat(marginal,ignore_index=True))
 highval=frames.loc[frames.split.eq('validation')&frames.B_eligible&frames.obs_indoor_co2_ppm.ge(1500)]
 train=frames.loc[frames.split.eq('train')]
 conditions={'high_CO2':lambda f:f.obs_indoor_co2_ppm.ge(1500),'high_CO2_high_vent':lambda f:f.obs_indoor_co2_ppm.ge(1500)&f.current_vent.ge(.75),'high_vent_only':lambda f:f.current_vent.ge(.75)}
 # Ranges from observed validation failure domain are descriptive post-hoc audit, not selection criteria.
 for col in ['obs_growth_stage_dat','obs_outdoor_light','obs_indoor_temp_c','obs_indoor_humidity_pct']:
  lo,hi=highval[col].min(),highval[col].max();conditions['high_CO2_in_validation_'+col+'_range']=lambda f,col=col,lo=lo,hi=hi:f.obs_indoor_co2_ppm.ge(1500)&f[col].between(lo,hi)
 rows=[]
 for name,fn in conditions.items():
  row={'condition':name,'Train_QC':int(fn(train).sum()),'Train_A':int((fn(train)&train.in_A.fillna(False)).sum()),'Train_B':int((fn(train)&train.B_eligible).sum()),'Validation_B':int(fn(frames.loc[frames.split.eq('validation')&frames.B_eligible]).sum())};rows.append(row)
 csv('coverage_condition_ladder.csv',pd.DataFrame(rows));csv('validation_high_rows.csv',highval)
 csv('high_population_ranges.csv',frames.loc[frames.obs_indoor_co2_ppm.ge(1500)].groupby(['split','B_eligible'])[['obs_indoor_co2_ppm','obs_growth_stage_dat','obs_outdoor_light','obs_indoor_temp_c','obs_indoor_humidity_pct','current_vent']].agg(['min','median','max']).reset_index())
 write('input_manifest.json',{'inputs':inputs,'raw_read_predicate':'exact reg_date membership of frozen Train/Validation prepared QC and matching series only','Test_accessed':False})
 for p,h in inputs.items():assert sha(ROOT/p)==h
 write('verification.json',{'original_hashes_unchanged':True,'A_membership_exact':True,'representative_priority_exact':True,'Test_accessed':False,'fit_performed':False,'dataset_regenerated':False,'raw_timestamp_is_not_measurement_proof':True})
 print('Audit tables complete:',D)
