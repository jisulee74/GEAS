from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from geas35.rl.sensor_history_env_v4 import SensorHistoryStateUpdater, SENSOR_FEATURES
from geas35.rl.causal_contract_env_v2 import CausalContractStateUpdater
from geas35.rl.vent_command_memory_b_v3 import OBS,ACTIONS
from geas35.rl.mdp_v1 import compute_mdp_v1_reward

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v3/strawberry'
class Identity:
 def transform_frame(self,f):return f.copy()
 def inverse_transform_frame(self,f):return f.copy()
@pytest.fixture(scope='module')
def data():
 return pd.read_parquet(D/'train_prepared_qc.parquet'),pd.read_parquet(D/'train_physical.parquet'),pd.read_parquet(D/'train_provenance.parquet')
@pytest.mark.parametrize('day',['2024-11-11','2024-11-16','2024-11-30','2024-12-01','2024-12-08','2024-12-09','2025-01-23'])
def test_seven_real_failed_resets(data,day):
 base,b,a=data;t=pd.Timestamp(day+' 00:10:00');i=base.index[base.reg_date.eq(t)][0];row=b.loc[b.reg_date.eq(t)].iloc[0];initial=base.iloc[i].copy();initial[OBS]=row[OBS]
 u=SensorHistoryStateUpdater(Identity(),base.iloc[i-12:i]);u.reset(initial)
 np.testing.assert_allclose(u.current_physical[OBS].to_numpy(float),initial[OBS].to_numpy(float),rtol=1e-6,atol=1e-8)
 nxt=base.iloc[i+1].copy();nxt[OBS]=row[['next_'+c for c in OBS]].to_numpy();action={c:row[c] for c in ACTIONS};pred={c:nxt[c] for c in ['obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm']}
 got=u.advance(current_scaled=initial,recorded_next_scaled=nxt,prediction_physical=pred,action=action)
 np.testing.assert_allclose(got[OBS].to_numpy(float),nxt[OBS].to_numpy(float),rtol=1e-6,atol=1e-8)
 ar=a.iloc[i];reward=compute_mdp_v1_reward(got,action,prev_action={k:ar['previous_'+k] for k in ACTIONS},prev_prev_action={k:ar['previous_previous_'+k] for k in ACTIONS})[0]
 assert np.isclose(reward,row.reward,rtol=1e-6,atol=1e-8)
 assert all(r.episode_id==initial.episode_id for r in u._history)
 assert len({r.episode_id for r in u._sensor_history})==2

@pytest.mark.parametrize('boundary',['episode','midnight','midrun','gap','crop','series_id','segment_id','source_table','physical_discontinuity','duplicate','future'])
def test_sensor_boundaries_and_command_isolation(data,boundary):
 base,_,_=data;initial=base.iloc[100].copy();t=pd.Timestamp('2025-01-02 00:00' if boundary=='midnight' else '2025-01-02 12:00');initial.reg_date=t
 initial['obs_indoor_temp_c']=20.;initial['obs_indoor_humidity_pct']=99.;initial['source_table']='source_a';initial['physical_discontinuity']=False
 rows=[]
 for k in range(12,0,-1):
  p=initial.copy();p.reg_date=t-pd.Timedelta(minutes=5*k);p.episode_id='older' if boundary!='midrun' else initial.episode_id;rows.append(p)
 past=pd.DataFrame(rows)
 blocked=boundary in ('gap','crop','series_id','segment_id','source_table','physical_discontinuity','duplicate')
 if boundary=='gap':past=past.iloc[:-1]
 elif boundary in ('crop','series_id','segment_id','source_table'):past.loc[past.index[-1],boundary]='changed'
 elif boundary=='physical_discontinuity':initial[boundary]=True
 elif boundary=='duplicate':past=pd.concat([past,past.iloc[[-1]]])
 elif boundary=='future':
  future=initial.copy();future.reg_date+=pd.Timedelta(minutes=5);future.obs_indoor_humidity_pct=0.;past=pd.concat([past,future.to_frame().T])
 old=CausalContractStateUpdater(Identity(),past);old.reset(initial);u=SensorHistoryStateUpdater(Identity(),past);u.reset(initial)
 assert u.current_physical.obs_derived_low_vpd_minutes_1h==(5. if blocked else 60.)
 assert u.current_physical.obs_derived_rh90_minutes_1h==(5. if blocked else 60.)
 rest=[c for c in OBS if c not in SENSOR_FEATURES]
 np.testing.assert_allclose(u.current_physical[rest].to_numpy(float),old.current_physical[rest].to_numpy(float),equal_nan=True)
 assert [r.reg_date for r in u._history]==[r.reg_date for r in old._history]
 assert all(r.reg_date<=t for r in u._sensor_history)

def test_long_rollout_uses_predictions_not_future_recorded_sensors(data):
 base,_,_=data;initial=base.iloc[100].copy();initial.reg_date=pd.Timestamp('2025-01-01 23:00');initial.obs_indoor_temp_c=20.;initial.obs_indoor_humidity_pct=99.
 u=SensorHistoryStateUpdater(Identity());u.reset(initial);current=u.current_physical
 for step in range(301):
  nxt=current.copy();nxt.reg_date+=pd.Timedelta(minutes=5);nxt.obs_indoor_humidity_pct=99.
  got=u.advance(current_scaled=current,recorded_next_scaled=nxt,
    prediction_physical={'obs_indoor_temp_c':20.,'obs_indoor_humidity_pct':20.,'obs_indoor_co2_ppm':500.},action={c:.2 if c=='vent_pct' else 0. for c in ACTIONS})
  assert len(u._sensor_history)<=12
  if step>=12:
   assert got.obs_derived_low_vpd_minutes_1h==0.
   assert got.obs_derived_rh90_minutes_1h==0.
   assert got.obs_derived_ach_min_day==.1
  assert got.obs_prev_vent_pct==.2
  current=got
 assert len(u._history)==288
 # An advance across a gap must also drop the sensor buffer.
 nxt=current.copy();nxt.reg_date+=pd.Timedelta(minutes=10)
 got=u.advance(current_scaled=current,recorded_next_scaled=nxt,
   prediction_physical={'obs_indoor_temp_c':20.,'obs_indoor_humidity_pct':99.,'obs_indoor_co2_ppm':500.},action={c:0. for c in ACTIONS})
 assert len(u._sensor_history)==1
 assert got.obs_derived_low_vpd_minutes_1h==5.

def test_real_train_scaler_runtime_roundtrip(data):
 import json
 from geas35.rl.mdp_v1 import MdpV1ObservationScaler
 base,b,_=data;t=pd.Timestamp('2025-01-23 00:10');i=base.index[base.reg_date.eq(t)][0];row=b.loc[b.reg_date.eq(t)].iloc[0]
 scaler=MdpV1ObservationScaler(**json.loads((ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry/observation_scaler.json').read_text()))
 initial=base.iloc[i].copy();initial[OBS]=row[OBS];nxt=base.iloc[i+1].copy();nxt[OBS]=row[['next_'+c for c in OBS]].to_numpy()
 current_scaled=scaler.transform_frame(initial.to_frame().T).iloc[0];next_scaled=scaler.transform_frame(nxt.to_frame().T).iloc[0]
 u=SensorHistoryStateUpdater(scaler,base.iloc[i-12:i]);u.reset(current_scaled)
 np.testing.assert_allclose(u.current_physical[OBS].to_numpy(float),initial[OBS].to_numpy(float),rtol=1e-6,atol=1e-8)
 got=u.advance(current_scaled=current_scaled,recorded_next_scaled=next_scaled,prediction_physical={k:nxt[k] for k in ('obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm')},action={k:row[k] for k in ACTIONS})
 np.testing.assert_allclose(got[OBS].to_numpy(float),next_scaled[OBS].to_numpy(float),rtol=1e-6,atol=1e-8)
