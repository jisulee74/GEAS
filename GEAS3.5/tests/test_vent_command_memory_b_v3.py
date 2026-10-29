import numpy as np
import pandas as pd
import pytest
from test_vent_command_memory_b_v2 import source
from geas35.rl.vent_command_memory_b_v3 import build,restore,admission,value,OBS,ACTIONS,LEGACY_COLS,fingerprint

def legacy(n=12):
 b,c,e=source(n);s=b[['reg_date']].copy();s['idx']=range(n)
 for j,k in enumerate(LEGACY_COLS):s[k]=20 if j<2 else 0
 return b,c,e,s

def test_repeated_identical_collapse_provenance():
 b,c,e=source();extra=e.copy();extra.idx+=10000;extra.reg_date+=pd.Timedelta(seconds=2);e=pd.concat([e,extra])
 x,a,ea=build(b,c,e,'train');y,aa,_=build(b,c,e,'train',collapse=False)
 assert len(x)==7 and y.empty;assert a.collapsed_events.eq(11).all();assert len(ea)==len(e);assert ea.idx.is_unique

def test_conflict_offcycle_no_collapse():
 b,c,e=source();extra=e.iloc[[44]].copy();extra.idx+=10000;extra.ratio='90';extra.reg_date+=pd.Timedelta(seconds=2)
 x,a,_=build(b,c,pd.concat([e,extra]),'train');assert a.loc[4,'conflicting_command'];assert not a.loc[4,'eligible']
 extra.ratio='20';extra.reg_date+=pd.Timedelta(minutes=2);_,a,_=build(b,c,pd.concat([e,extra]),'train');assert a.loc[4,'intervening_override'];assert not a.loc[4,'eligible']

def test_operation_alias_and_conflicts():
 for ch in [7,8,11,12]:
  assert value(ch,100,'OPEN')==value(ch,100,'ON')==1
  assert value(ch,0,'CLOSE')==value(ch,0,'OFF')==0
  assert np.isnan(value(ch,100,'OFF'))
 assert value(1,0,'OPEN')==0

def test_legacy_previous_not_current_and_no_fill():
 b,c,e,s=legacy();x,a,_=build(b,c,e,'train',source='legacy',snapshots=s);assert len(x)==9
 s.loc[4,['cont_skyl_vol','cont_skyr_vol']]=80;y,aa,_=build(b,c,e,'train',source='legacy',snapshots=s)
 xx=x.set_index('reg_date').loc[b.reg_date[4]];yy=y.set_index('reg_date').loc[b.reg_date[4]]
 np.testing.assert_allclose(xx[OBS].to_numpy(float),yy[OBS].to_numpy(float));assert yy.vent_pct==.8;assert yy.next_obs_prev_vent_pct==.8
 assert a.loc[:1,'command_memory_unknown'].all()
 s.loc[4,'cont_skyl_vol']=np.nan;_,a,_=build(b,c,e,'train',source='legacy',snapshots=s);assert not a.loc[4:6,'eligible'].any()

@pytest.mark.parametrize('key',['crop','series_id','segment_id','episode_id','gap'])
def test_legacy_boundaries(key):
 b,c,e,s=legacy()
 if key=='gap':b.loc[5:,'reg_date']+=pd.Timedelta(minutes=5);s.loc[5:,'reg_date']+=pd.Timedelta(minutes=5)
 else:b.loc[5:,key]='new'
 _,a,_=build(b,c,e,'train',source='legacy',snapshots=s);assert a.loc[5:6,'command_memory_unknown'].all();assert not a.loc[4:6,'eligible'].any()

def test_raw_missing_snapshot_never_interpolated():
 b,c,e,s=legacy();s=s.drop(index=4);_,a,_=build(b,c,e,'train',source='legacy',snapshots=s);assert not a.loc[4:6,'eligible'].any()

def test_test_locked_and_determinism():
 b,c,e,s=legacy();x,a,_=build(b,c,e,'train',source='legacy',snapshots=s);y,aa,_=build(b,c,e,'train',source='legacy',snapshots=s);assert fingerprint(x)==fingerprint(y);assert fingerprint(a)==fingerprint(aa)
 with pytest.raises(ValueError):build(b,c,e,'test',source='legacy',snapshots=s)

def test_reward_reference_and_train_only_scaler():
 from geas35.rl.mdp_v1 import compute_mdp_v1_reward
 from geas35.rl.vent_command_memory_b_v3 import fit_b_scaler
 b,c,e,s=legacy();s.loc[2,['cont_skyl_vol','cont_skyr_vol']]=90
 x,a,_=build(b,c,e,'train',source='legacy',snapshots=s)
 row=x.set_index('reg_date').loc[b.reg_date[4]];nxt=b.iloc[5].copy()
 for col in OBS:nxt[col]=row['next_'+col]
 expected=compute_mdp_v1_reward(nxt,{k:row[k] for k in ACTIONS},prev_action={k:a.loc[4,'previous_'+k] for k in ACTIONS},prev_prev_action={k:a.loc[4,'previous_previous_'+k] for k in ACTIONS})[0]
 assert row.reward==pytest.approx(expected)
 scaler=fit_b_scaler(x);assert scaler.means['obs_indoor_temp_c']==pytest.approx(x.obs_indoor_temp_c.mean())
 v,_,_=build(b,c,e,'validation',source='legacy',snapshots=s)
 with pytest.raises(ValueError):fit_b_scaler(v)

def test_api_partial_update_and_simultaneous_conflict_unknown():
 b,c,e=source();mask=e.reg_date.eq(b.reg_date[4]+pd.Timedelta(seconds=5))&e.channel.eq('1');partial=e.loc[~mask]
 x,a,_=build(b,c,partial,'train');assert a.loc[4,'eligible'];assert a.loc[4,'retained_channels']==1
 extra=e.loc[mask].copy();extra.idx+=10000;extra.ratio='90';_,a,_=build(b,c,pd.concat([e,extra]),'train');assert a.loc[4,'conflicting_command'];assert np.isnan(a.loc[5,'previous_vent_pct'])

def test_legacy_midnight_longer_than_288_and_zero():
 b,c,e,s=legacy(310);b.reg_date=pd.date_range('2025-01-01 12:00',periods=len(b),freq='5min');s.reg_date=b.reg_date
 s.loc[10,['cont_skyl_vol','cont_skyr_vol']]=0
 x,a,_=build(b,c,e,'train',source='legacy',snapshots=s)
 assert a.loc[11,'previous_vent_pct']==0
 assert not a.loc[b.reg_date.eq(pd.Timestamp('2025-01-01 23:55')),'eligible'].any()
 assert a.loc[300,'previous_vent_pct']==.2
