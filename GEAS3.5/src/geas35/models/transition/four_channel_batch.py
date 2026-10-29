"""Four-channel batched state math (no CO2, no fabricated second-sensor history); validated against scalar v4 before model scoring.

Only continuous frozen windows are accepted. Each batch row is an independent
window, with recorded past sensors and predicted future endogenous states.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[4]
_SCHEMA=json.loads((_ROOT/'experiments/rl_policy_training/artifacts/transition_4ch_migration_prereg_v1/migration_preregistration.json').read_text())['schema']
OBS=_SCHEMA['observation_columns']; ACTIONS=_SCHEMA['action_columns']
from geas35.rl.recorded_light_env_v1 import SOLAR_FEATURES
from geas35.models.transition.causal_state import CausalStateUpdater
I={c:j for j,c in enumerate(OBS)}
COPY=[I[c] for c in OBS if CausalStateUpdater._recorded_column_allowed(c) or c in SOLAR_FEATURES]

def thermo(temp,rh):
 es=.6108*np.exp(17.27*temp/(temp+237.3));vpd=np.maximum(es-es*np.clip(rh,0,100)/100,0)
 g=np.log(np.clip(rh,1e-6,100)/100)+17.62*temp/(243.12+temp);dew=243.12*g/(17.62-g)
 return vpd,dew,temp-dew

def run_lengths(frame,episode=False):
 t=frame.reg_date.to_numpy(dtype='datetime64[ns]').astype('int64');ok=np.r_[False,np.diff(t)==300_000_000_000]
 for c in ('crop','series_id','segment_id')+ (('episode_id',) if episode else ()):
  x=frame[c].astype(str).to_numpy();ok &= np.r_[False,x[1:]==x[:-1]]
 if episode:ok &= np.r_[False,(t[1:]//86400_000_000_000)==(t[:-1]//86400_000_000_000)]
 else:
  for c in ('sensor_source','source_table','source_confidence'):
   if c in frame:
    x=frame[c].astype(str).to_numpy();ok &= np.r_[False,x[1:]==x[:-1]] & frame[c].notna().to_numpy()
  for c in ('sensor_discontinuity','physical_discontinuity'):
   if c in frame:ok &= ~frame[c].fillna(True).astype(bool).to_numpy()
 idx=np.arange(len(frame));return idx-np.maximum.accumulate(np.where(~ok,idx,0))+1

class FourChannelBatchUpdater:
 def __init__(self,base,positions,initial):
  assert np.asarray(initial).shape[1]==56
  self.state=np.asarray(initial,float).copy();self.times=base.reg_date.to_numpy(dtype='datetime64[ns]').astype('int64')[positions]
  temperature=base['obs_indoor_temp_c'].to_numpy(float);humidity=base['obs_indoor_humidity_pct'].to_numpy(float);v,d,m=thermo(temperature,humidity);self.low=np.zeros((len(positions),12));self.rh=np.zeros_like(self.low);self.margin=np.full((len(positions),7),np.nan)
  sensor_count=run_lengths(base);cond_count=run_lengths(base,True)
  for k in range(12):
   valid=sensor_count[positions]>k;ix=np.maximum(positions-k,0);self.low[:,11-k]=valid&(v[ix]<.5);self.rh[:,11-k]=valid&(humidity[ix]>=90)
  for k in range(7):
   valid=cond_count[positions]>k;self.margin[:,6-k]=np.where(valid,m[np.maximum(positions-k,0)],np.nan)
  self._sensor_features(np.arange(len(positions)))
 def _sensor_features(self,ix):
  lo=self.low[ix].sum(axis=1)*5;rh=self.rh[ix].sum(axis=1)*5;severe=(lo>=20)|(rh>=20);moderate=((lo>=10)|(rh>=10))&~severe
  self.state[ix,I['obs_derived_low_vpd_minutes_1h']]=lo;self.state[ix,I['obs_derived_rh90_minutes_1h']]=rh
  self.state[ix,I['obs_derived_ach_min_day']]=np.where(severe,.1*1.5,np.where(moderate,.1*1.2,.1));self.state[ix,I['obs_derived_ach_min_night']]=np.where(severe,.05*1.3,np.where(moderate,.05*1.1,.05))
 def advance(self,ix,pred,recorded,action,times):
  before=self.state[ix].copy();out=before.copy();out[:,COPY]=recorded[:,COPY];out[:,:4]=pred
  for j,c in enumerate(ACTIONS):out[:,I['obs_prev_'+c]]=action[:,j]
  temp,rh=pred[:,0],pred[:,2];v,dew,margin=thermo(temp,rh)
  for c,x in [('obs_current_vpd_kpa',v),('obs_current_dewpoint_c',dew),('obs_current_condensation_margin_c',margin)]:out[:,I[c]]=x
  continuous=times-self.times[ix]==300_000_000_000;day=times//86400_000_000_000==self.times[ix]//86400_000_000_000
  self.low[ix]=np.column_stack([self.low[ix,1:]*continuous[:,None],v<.5]);self.rh[ix]=np.column_stack([self.rh[ix,1:]*continuous[:,None],rh>=90])
  hist=self.margin[ix].copy();hist[~(continuous&day)]=np.nan;hist=np.column_stack([hist[:,1:],margin]);self.margin[ix]=hist
  # Least-squares line at offsets -30..0 minutes, forecast +10 minutes.
  finite=np.isfinite(hist);n=finite.sum(axis=1);x=np.arange(-30,1,5)[None,:];sx=(finite*x).sum(axis=1);sy=np.where(finite,hist,0).sum(axis=1);sxx=(finite*x*x).sum(axis=1);sxy=np.where(finite,hist*x,0).sum(axis=1)
  denom=n*sxx-sx*sx;slope=np.divide(n*sxy-sx*sy,denom,out=np.zeros(len(ix)),where=denom!=0);intercept=np.divide(sy-slope*sx,n,out=margin.copy(),where=n!=0);forecast=np.where(n>=2,intercept+slope*10,margin)
  out[:,I['obs_derived_condensation_margin_pred_10m']]=forecast;out[:,I['obs_derived_condensation_risk_10m']]=(forecast<.8).astype(float)
  diff=temp-out[:,I['obs_outdoor_temp_c']];out[:,I['obs_derived_temp_diff_in_out_c']]=diff
  out[:,I['obs_derived_vent_loss_proxy']]=np.clip(action[:,0],0,1)*np.maximum(diff,0)*(1-np.clip(out[:,I['obs_outdoor_light']]/1000,0,1))
  dt=(times-self.times[ix])/60e9;target=out[:,I['obs_current_target_temp_c']]
  out[:,I['obs_derived_heating_degree_minutes']]=np.where(day,before[:,I['obs_derived_heating_degree_minutes']],0)+np.maximum(target-temp,0)*dt
  out[:,I['obs_derived_cooling_degree_minutes']]=np.where(day,before[:,I['obs_derived_cooling_degree_minutes']],0)+np.maximum(temp-target,0)*dt
  out[:,I['obs_derived_ramp_limit_pct']]=15.
  self.state[ix]=out;self.times[ix]=times;self._sensor_features(ix)
  return self.state[ix].copy()
