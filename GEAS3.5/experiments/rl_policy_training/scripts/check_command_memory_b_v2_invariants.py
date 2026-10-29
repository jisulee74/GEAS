from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.mdp_v1 import prepare_mdp_v1_frame
from geas35.rl.vent_command_memory_b_v2 import build,OBS
D=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v2'
results=[]
for split in ('train','validation'):
 p=D/'phase0_verified/cucumber'/f'{split}_physical.parquet'
 b=pd.read_parquet(p)
 if b.empty:results.append({'split':split,'status':'not_evaluable_no_eligible_rows'});continue
 base=prepare_mdp_v1_frame(pd.read_parquet(ROOT/f'offline_dataset_preparation/datasets/03_quality_controlled/cucumber/{split}.parquet'))
 cycles=pd.read_parquet(D/'raw'/f'cucumber_{split}_optimal_monitoring_copy.parquet');events=pd.read_parquet(D/'raw'/f'cucumber_{split}_send_ds_control_api_history.parquet');audit=pd.read_parquet(D/'phase0_verified/cucumber'/f'{split}_events.parquet')
 candidates=audit.loc[audit.classification.eq('automatic_payload_linked')&audit.channel.eq(1)&audit.sensor_timestamp.isin(b.reg_date),'sensor_timestamp'].drop_duplicates()
 for t in candidates.iloc[np.unique(np.linspace(0,len(candidates)-1,min(3,len(candidates))).astype(int))] if len(candidates) else []:
  low=t-pd.Timedelta(minutes=60);high=t+pd.Timedelta(minutes=15)
  sub=base.loc[base.reg_date.ge(low)&base.reg_date.lt(high)].copy();cc=cycles.loc[cycles.reg_date.ge(low)&cycles.reg_date.lt(high)].copy();ee=events.loc[events.reg_date.ge(low)&events.reg_date.lt(high)].copy()
  x,_,_=build(sub,cc,ee,split);event_id=audit.loc[audit.sensor_timestamp.eq(t)&audit.channel.eq(1)&audit.classification.eq('automatic_payload_linked'),'idx'].iloc[0]
  new=100 if float(cc.loc[cc.reg_date.eq(t),'pred_ltw'].iloc[0])<50 else 0
  cc.loc[cc.reg_date.eq(t),'pred_ltw']=new;ee.loc[ee.idx.eq(event_id),'ratio']=str(new)
  y,_,_=build(sub,cc,ee,split)
  xx=x.set_index('reg_date').loc[t];yy=y.set_index('reg_date').loc[t]
  np.testing.assert_allclose(xx[OBS].to_numpy(float),yy[OBS].to_numpy(float),rtol=1e-6,atol=1e-8)
  assert xx.vent_pct!=yy.vent_pct
  factor=max(float(yy.next_obs_indoor_temp_c)-float(yy.next_obs_outdoor_temp_c),0)*(1-np.clip(float(yy.next_obs_outdoor_light)/1000,0,1))
  assert np.isclose(float(yy.next_obs_derived_vent_loss_proxy),float(yy.vent_pct)*factor,rtol=1e-6,atol=1e-8)
  assert yy.target_next_indoor_temp_c==xx.target_next_indoor_temp_c
  results.append({'split':split,'timestamp':str(t),'changed_event_id':int(event_id),'observation_current_unchanged':True,'transition_action_changed':True,'next_previous_command_vent_formula_matches':True,'target_unchanged':True})
(D/'actual_invariants.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
