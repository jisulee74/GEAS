"""Train/Validation temporal contrasts only; correlation does not identify causality."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.models.transition.causal_state import load_frozen_observation_scaler
D=ROOT/'experiments/rl_policy_training/artifacts/step19_5_vent_timestamp_v1'
protocol=json.loads((ROOT/'experiments/rl_policy_training/artifacts/step15_v1_4/rl_protocol_manifest.json').read_text())
rows=[];examples=[];files=[];cutoff=pd.Timestamp('2026-01-22 09:05')
def record(p):files.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
def corr(a,b):
 v=pd.concat([a,b],axis=1).dropna()
 return float(v.iloc[:,0].corr(v.iloc[:,1])) if len(v)>1 and all(v.std()>0) else None
for crop in ('melon','cucumber','strawberry'):
 sp=ROOT/protocol['crops'][crop]['observation_scaler']['path'];scaler,h=load_frozen_observation_scaler(sp);assert h==protocol['crops'][crop]['observation_scaler']['sha256'];record(sp)
 for split in ('train','validation'):
  p=ROOT/f'offline_dataset_preparation/datasets/5_rl_dataset/{crop}/{split}.parquet';record(p);f=scaler.inverse_transform_frame(pd.read_parquet(p));t=pd.to_datetime(f.reg_date)
  ok=t.diff().eq(pd.Timedelta(minutes=5)) & t.shift(-1).sub(t).eq(pd.Timedelta(minutes=5)) & t.dt.date.eq(t.shift().dt.date) & t.dt.date.eq(t.shift(-1).dt.date)
  for c in ('crop','series_id','segment_id','episode_id'):
   if c in f:ok &= f[c].eq(f[c].shift()) & f[c].eq(f[c].shift(-1))
  ok &= f.rl_valid_transition.astype(bool) & f.rl_valid_transition.shift().fillna(False).astype(bool)
  v=pd.to_numeric(f.vent_pct);prev=v.shift();factor=(f.obs_indoor_temp_c-f.obs_outdoor_temp_c).clip(lower=0)*(1-(f.obs_outdoor_light.clip(lower=0)/1000).clip(0,1))
  a=v*factor;b=prev*factor;observed=f.obs_derived_vent_loss_proxy
  source=pd.Series(np.where(t<=cutoff,'state_table_before_cutoff','api_ratio_after_cutoff'),index=f.index)
  for src in source.unique():
   mask=ok & source.eq(src);n=int(mask.sum())
   if not n:continue
   deltaT=f.obs_indoor_temp_c.shift(-1)-f.obs_indoor_temp_c;deltaRH=f.obs_indoor_humidity_pct.shift(-1)-f.obs_indoor_humidity_pct
   rows.append({'crop':crop,'split':split,'source':src,'triplets':n,'vent_changed':int((~np.isclose(v,prev,rtol=1e-6,atol=1e-8)&mask).sum()),'vent_change_fraction':float((~np.isclose(v[mask],prev[mask],rtol=1e-6,atol=1e-8)).mean()),'current_definition_max_abs_error':float((a[mask]-observed[mask]).abs().max()),'previous_definition_mae':float((b[mask]-observed[mask]).abs().mean()),'previous_definition_max_abs_error':float((b[mask]-observed[mask]).abs().max()),'corr_current_vent_next_deltaT':corr(v[mask],deltaT[mask]),'corr_previous_vent_next_deltaT':corr(prev[mask],deltaT[mask]),'corr_current_vent_next_deltaRH':corr(v[mask],deltaRH[mask]),'corr_previous_vent_next_deltaRH':corr(prev[mask],deltaRH[mask])})
   selected=f.index[mask & (~np.isclose(a,b,rtol=1e-6,atol=1e-8))][:8]
   for i in selected:examples.append({'crop':crop,'split':split,'source':src,'timestamp':str(t.loc[i]),'vent_previous':float(prev.loc[i]),'vent_current':float(v.loc[i]),'T_t':float(f.loc[i,'obs_indoor_temp_c']),'T_next':float(f.obs_indoor_temp_c.shift(-1).loc[i]),'RH_t':float(f.loc[i,'obs_indoor_humidity_pct']),'RH_next':float(f.obs_indoor_humidity_pct.shift(-1).loc[i]),'proxy_recorded':float(observed.loc[i]),'proxy_A':float(a.loc[i]),'proxy_B':float(b.loc[i])})
pd.DataFrame(rows).to_csv(D/'temporal_contrasts.csv',index=False);pd.DataFrame(examples).to_csv(D/'changed_action_examples.csv',index=False)
record(Path(__file__));record(ROOT/'offline_dataset_preparation/scripts/00_extract_raw.py');record(ROOT/'offline_dataset_preparation/scripts/01_resample.py')
(D/'manifest.json').write_text(json.dumps({'files':files,'splits':['train','validation'],'test_accessed':False,'new_fit':False,'meaning':'observational contrasts, not causal identification'},indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
