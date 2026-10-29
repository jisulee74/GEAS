"""Evidence summaries; no change to source/QC/eligibility."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.mdp_v1 import mdp_v1_observation_columns,MDP_V1_ACTION_COLUMNS
D=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_source_audit_b_v1';P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry'
def write(n,x):(D/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
q=pd.read_parquet(P/'train_prepared_qc.parquet');p=pd.read_parquet(P/'train_provenance.parquet');h=pd.read_csv(D/'train_high_co2_all_rows.csv.gz',parse_dates=['reg_date']);v=pd.read_csv(D/'validation_high_rows.csv',parse_dates=['reg_date']);w=pd.read_csv(D/'worst_episode_5min_source_trace.csv.gz',parse_dates=['reg_date','raw_reg_date']);db=pd.read_csv(D/'db_worst_episode_raw_rows.csv.gz',parse_dates=['reg_date'])
joined=w.merge(db,on='reg_date',suffixes=('','_db'),validate='one_to_one')
for c in ['idx','in_co2','in_co2_2','in_temp','in_temp2','in_hum','in_hum2','out_light','out_windsp','cont_skyl_vol','cont_skyr_vol']:
 right=c+'_db' if c in w else c
 np.testing.assert_allclose(pd.to_numeric(joined['raw_'+c]),pd.to_numeric(joined[right]),equal_nan=True,rtol=0,atol=0)
# Read only original split Train to inspect numeric values before QC; no pipeline is run.
src=ROOT/'offline_dataset_preparation/datasets/02_split/strawberry/train.parquet';pre=pd.read_parquet(src,columns=['reg_date','series_id','segment_id','episode_id','out_windsp','out_light'])
keys=['reg_date','series_id','segment_id','episode_id'];z=h.merge(pre.rename(columns={'out_windsp':'preQC_wind','out_light':'preQC_light'}),on=keys,validate='one_to_one')
for c in ['preQC_wind','preQC_light']:z[c]=pd.to_numeric(z[c],errors='coerce')
z[['reg_date','population','A_reason','B_reason','raw_idx','raw_out_windsp','preQC_wind','out_windsp','out_windsp_rule_outlier_flag','out_windsp_imputed_flag','out_windsp_unconfirmed_flag','raw_out_light','preQC_light','out_light','out_light_rule_outlier_flag','out_light_imputed_flag','out_light_unconfirmed_flag']].to_csv(D/'train_high_raw_to_QC.csv',index=False)
counts={}
for pop,g in z.groupby('population'):
 counts[pop]={'rows':len(g),'raw_rows':int(g.raw_idx.notna().sum()),'inserted_rows':int(g.is_resampled_row.sum()),'wind_preQC_gt30':int(g.preQC_wind.gt(30).sum()),'wind_preQC_missing':int(g.preQC_wind.isna().sum()),'wind_QC_nan':int(g.out_windsp.isna().sum()),'light_preQC_gt1400':int(g.preQC_light.gt(1400).sum()),'light_QC_nan':int(g.out_light.isna().sum()),'both_QC_nan':int((g.out_windsp.isna()&g.out_light.isna()).sum()),'QC_nan_neither':int((g.out_windsp.notna()&g.out_light.notna()).sum())}
write('train_exclusion_source_summary.json',counts)
obs=q[mdp_v1_observation_columns(q)].copy()
for action in MDP_V1_ACTION_COLUMNS:obs['obs_prev_'+action]=p['previous_'+action]
obs['obs_derived_vent_loss_proxy']=p.vent_loss_B
obs=obs[mdp_v1_observation_columns(q)]
rows=[]
extra=z.loc[z.population.eq('additional_excluded_B')]
for _,r in extra.iterrows():
 idx=q.index[q.reg_date.eq(r.reg_date)&q.series_id.eq(r.series_id)][0]
 bad=lambda row:'|'.join(c for c in obs if pd.isna(row[c]) or not np.isfinite(float(row[c])))
 rows.append({'reg_date':r.reg_date,'reason':r.B_reason,'current_bad_features':bad(obs.iloc[idx]),'next_bad_features':bad(obs.iloc[idx+1]),'current_raw_snapshot':bool(p.loc[idx,'cycle_present']),'previous_raw_timestamp':p.loc[idx,'legacy_previous_timestamp'],'previous_previous_raw_timestamp':p.loc[idx,'legacy_previous_previous_timestamp'],'next_timestamp':q.loc[idx+1,'reg_date']})
pd.DataFrame(rows).to_csv(D/'B_additional_16_exact_causes.csv',index=False)
plateau=w.loc[w.reg_date.between('2025-11-10 19:20','2025-11-11 09:35')]
flags=[c for c in plateau if any(c.startswith(x) for x in ('in_co2_','in_co2_2_')) and c.endswith(('missing_flag','ffill_flag','rule_outlier_flag','ai_outlier_flag','imputed_flag'))]
write('worst_plateau_summary.json',{'joint_CO2_start':'2025-11-10 18:45','joint_CO2_end':'2025-11-11 09:35','joint_CO2_rows':179,'joint_CO2_elapsed_minutes':890,'multi_sensor_start':'2025-11-10 19:20','multi_sensor_end':'2025-11-11 09:35','multi_sensor_rows':len(plateau),'multi_sensor_elapsed_minutes':855,'new_raw_rows':int(plateau.raw_idx.nunique()),'raw_first_id':int(plateau.raw_idx.iloc[0]),'raw_last_id':int(plateau.raw_idx.iloc[-1]),'B_eligible_rows':int(plateau.B_eligible.sum()),'values':{c:plateau[c].unique().tolist() for c in ['raw_in_co2','raw_in_co2_2','raw_in_temp','raw_in_temp2','raw_in_hum','raw_in_hum2','raw_out_light','raw_out_windsp','raw_cont_skyl_vol','raw_cont_skyr_vol']},'QC_CO2_flags_sum':plateau[flags].sum().to_dict(),'source_measurement_update_timestamp':'not stored','classification':'C / undetermined; frozen acquisition snapshot strongly suspected, not confirmed'})
# Baseline run-length distribution, counts are descriptive and run boundaries censored at split/gaps.
stats=[]
for split in ('train','validation'):
 for label in ('sensor1','joint','co2_light_temperature_humidity'):
  r=pd.read_csv(D/f'{split}_{label}_plateaus.csv')
  for high,mask in [('all',pd.Series(True,index=r.index)),('high',r.co2_min.ge(1500)),('below1500',r.co2_max.lt(1500))]:
   g=r.loc[mask];stats.append({'split':split,'definition':label,'population':high,'runs_ge2':len(g),'rows_in_runs':int(g.rows.sum()),'median_minutes':g.elapsed_minutes.median(),'p95_minutes':g.elapsed_minutes.quantile(.95),'max_minutes':g.elapsed_minutes.max(),'runs_ge60min':int(g.elapsed_minutes.ge(60).sum())})
pd.DataFrame(stats).to_csv(D/'plateau_duration_reference.csv',index=False)
# Dominant excluded plateau blocks: count exact Train membership inside each interval.
r=pd.read_csv(D/'train_co2_light_temperature_humidity_plateaus.csv',parse_dates=['start','end']);rows=[]
for _,a in r.loc[r.co2_min.ge(1500)&r.rows.ge(12)].iterrows():
 g=z.loc[z.reg_date.between(a.start,a.end)]
 rows.append({'start':a.start,'end':a.end,'raw_rows':a.rows,'co2':a.co2_min,**g.population.value_counts().to_dict(),'raw_wind_values':g.raw_out_windsp.dropna().unique().tolist(),'raw_light_values':g.raw_out_light.dropna().unique().tolist()})
write('train_high_long_plateau_exclusions.json',rows)
# Numeric overlap: the most consequential joint hole existed already in A.
train_high=z;val=v.loc[v.current_vent.ge(.75)];combo=[]
for name,condition in [('high_vent',lambda f:f.current_vent.ge(.75)),('high_T_ge35',lambda f:f.obs_indoor_temp_c.ge(35)),('high_vent_T_ge35',lambda f:f.current_vent.ge(.75)&f.obs_indoor_temp_c.ge(35)),('high_vent_light500_800',lambda f:f.current_vent.ge(.75)&f.obs_outdoor_light.between(500,800)),('high_DAT_lt30_vent_ge75',lambda f:f.obs_growth_stage_dat.lt(30)&f.current_vent.ge(.75))]:
 combo.append({'condition_with_CO2_ge1500':name,'Train_QC':int(condition(z).sum()),'Train_A':int((condition(z)&z.in_A).sum()),'Train_B':int((condition(z)&z.B_eligible).sum()),'Validation_B':int(condition(v).sum())})
pd.DataFrame(combo).to_csv(D/'coverage_key_combinations.csv',index=False)
write('supplemental_verification.json',{'live_DB_vs_saved_raw_exact':True,'matched_rows':len(joined),'extra_input':str(src.relative_to(ROOT)),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'excluded_rows_reintroduced':False,'source_values_changed':False})
print(json.dumps(counts,ensure_ascii=False,indent=2));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(combo).to_string(index=False))
