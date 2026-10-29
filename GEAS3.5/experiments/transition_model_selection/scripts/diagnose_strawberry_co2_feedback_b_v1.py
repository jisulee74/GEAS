"""Fixed-model diagnostic oracle feedback; not a deployable evaluation or fit."""
from pathlib import Path
import os,sys,json,pickle
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd,numpy as np
from geas35.models.transition.command_carry_b_v1 import OBS,ACTIONS,sha
from geas35.models.transition.batch_sensor_rollout_b_v1 import BatchSensorUpdater
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry';A=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1/strawberry';D=A.parents[1]/'strawberry_co2_diagnosis_b_v1'
# Correct sibling of command_carry_b_v1.
D=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_diagnosis_b_v1'
b=pd.read_parquet(P/'validation_physical.parquet');base=pd.read_parquet(P/'validation_prepared_qc.parquet');z=np.load(A/'lightgbm/rollout/predictions.npz');start=pd.Series(np.arange(len(b)),index=b.row_id).loc[z['start_row_id']].to_numpy();qpos=pd.Index(base.reg_date).get_indexer(b.iloc[start].reg_date)
with (A/'lightgbm/fit/model.pkl').open('rb') as f:model=pickle.load(f)
means=np.array([model.observation_scaler.means.get(c,0.) for c in OBS]);scales=np.array([model.observation_scaler.scales.get(c,1.) for c in OBS]);rows=[]
for mode in ('co2_oracle_feedback','temperature_humidity_oracle_feedback'):
 u=BatchSensorUpdater(base,qpos,b.iloc[start][OBS].to_numpy(float));preds=np.full((len(start),12),np.nan)
 for step in range(12):
  ix=np.flatnonzero(z['max_horizon']>step);tr=b.iloc[start[ix]+step];truth=tr[['next_'+c for c in OBS]].to_numpy(float);act=tr[ACTIONS].to_numpy(float);x=np.column_stack([(u.state[ix]-means)/scales,act]);pr=model.predict(pd.DataFrame(x,columns=OBS+ACTIONS)).next_observation.to_numpy(float);preds[ix,step]=pr[:,2]
  # Preserve pre-injection forecast in metrics; inject observed targets only into next inputs.
  if mode=='co2_oracle_feedback':pr[:,2]=truth[:,2]
  else:pr[:,:2]=truth[:,:2]
  u.advance(ix,pr,truth,act,tr.target_timestamp.to_numpy(dtype='datetime64[ns]').astype('int64'))
 np.savez_compressed(D/f'{mode}.npz',start_row_id=z['start_row_id'],max_horizon=z['max_horizon'],predictions=preds)
 for h in (3,6,12):
  ix=z['max_horizon']>=h;e=preds[ix,:h]-z['truth'][ix,:h,2];rows.append({'mode':mode,'horizon':h*5,'windows':int(ix.sum()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'endpoint_rmse':float(np.sqrt(np.mean(e[:,-1]**2)))})
pd.DataFrame(rows).to_csv(D/'feedback_oracle_metrics.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
