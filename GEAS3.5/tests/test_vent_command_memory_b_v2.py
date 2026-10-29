import json
import numpy as np
import pandas as pd
import pytest
from test_vent_command_dataset_b_v1 import fixture
from geas35.rl.vent_command_memory_b_v2 import build,MAP,BINARY,OBS,fit_b_scaler,fingerprint

def source(n=10):
 b,_=fixture(n);c=pd.DataFrame({'reg_date':b.reg_date,'idx':range(n),'iot_data_idx':97});events=[]
 for ch,col in MAP.items():c[col]=20 if ch in (1,2) else 0
 for i,t in enumerate(b.reg_date):
  for ch,col in MAP.items():events.append({'idx':i*20+ch,'n_idx':'1','channel':str(ch),'reg_date':t+pd.Timedelta(seconds=5),'ratio':str(c.loc[i,col]),'operation':'OFF' if ch in BINARY else 'OPEN','duration':''})
 return b,c,pd.DataFrame(events)

def test_invariant_and_previous_memory():
 b,c,e=source();x,_,_=build(b,c,e,'train');c.loc[4,['pred_ltw','pred_rtw']]=70;e.loc[e.reg_date.eq(b.reg_date[4]+pd.Timedelta(seconds=5))&e.channel.isin(['1','2']),'ratio']='70';y,_,_=build(b,c,e,'train')
 xx=x.set_index('reg_date').loc[b.reg_date[4]];yy=y.set_index('reg_date').loc[b.reg_date[4]];np.testing.assert_allclose(xx[OBS].to_numpy(float),yy[OBS].to_numpy(float));assert yy.vent_pct==.7;assert yy.next_obs_prev_vent_pct==.7

def test_partial_and_zero_update_keep_known_memory():
 b,c,e=source();e=e.loc[~((e.reg_date.eq(b.reg_date[4]+pd.Timedelta(seconds=5)))&e.channel.eq('1'))];e=e.loc[~e.reg_date.eq(b.reg_date[5]+pd.Timedelta(seconds=5))]
 x,a,_=build(b,c,e,'train');assert a.loc[4,'retained_channels']==1;assert a.loc[5,'retained_channels']==11;assert a.loc[5,'eligible'];assert a.loc[5,'current_vent_pct']==.2

def test_override_quarantine_and_memory():
 b,c,e=source();extra=e.iloc[0].copy();extra['idx']=999;extra['reg_date']=b.reg_date[4]+pd.Timedelta(minutes=2);extra['ratio']='90';e=pd.concat([e,pd.DataFrame([extra])]);x,a,ea=build(b,c,e,'train');assert a.loc[4,'intervening_override'];assert not a.loc[4,'eligible'];assert a.loc[5,'previous_vent_pct']==pytest.approx(.55);assert 'manual_remote_or_unlinked' in set(ea.classification)

def test_ambiguous_duplicates_and_equal_time_conflict():
 b,c,e=source();extra=e.iloc[44].copy();extra['idx']=999;e=pd.concat([e,pd.DataFrame([extra])]);_,a,_=build(b,c,e,'train');assert a.loc[4,'ambiguous'];assert not a.loc[4,'eligible']

def test_boundary_no_leak_target_scaler_and_preserved_features():
 b,c,e=source();b.loc[5:,'episode_id']='new';x,a,_=build(b,c,e,'train');assert a.loc[5,'command_memory_unknown'];assert not a.loc[4,'eligible'];assert not a.loc[5,'eligible'];assert not a.loc[6,'eligible']
 b,c,e=source();x,a,ea=build(b,c,e,'train');y,aa,ee=build(b,c,e,'train');assert fingerprint(x)==fingerprint(y);assert fingerprint(a)==fingerprint(aa);assert fingerprint(ea)==fingerprint(ee)
 keep=[k for k in OBS if not k.startswith('obs_prev_') and k!='obs_derived_vent_loss_proxy'];np.testing.assert_allclose(x[keep],b.set_index('reg_date').loc[x.reg_date,keep],rtol=1e-6,atol=1e-8)
 assert (x.target_timestamp-x.reg_date).eq(pd.Timedelta(minutes=5)).all();fit_b_scaler(x)
 v,_,_=build(b,c,e,'validation')
 with pytest.raises(ValueError):fit_b_scaler(v)
 with pytest.raises(ValueError):build(b,c,e,'test')

def test_empty_transmissions_do_not_substitute_calculated_values():
 b,c,e=source();x,a,ea=build(b,c,e.iloc[:0],'train');assert len(x)==0;assert a.cycle_present.all();assert not a.linked.any();assert a.command_memory_unknown.all();assert ea.empty

def test_off_cycle_matching_payload_not_auto_and_timed_commands_quarantined():
 b,c,e=source();mask=e.reg_date.eq(b.reg_date[4]+pd.Timedelta(seconds=5))&e.channel.eq('1');e.loc[mask,'reg_date']+=pd.Timedelta(minutes=2)
 _,a,ea=build(b,c,e,'train');assert not a.loc[4,'eligible'];assert a.loc[4,'intervening_override'];assert ea.loc[ea.idx.eq(81),'classification'].iloc[0]=='manual_remote_or_unlinked'
 b,c,e=source();mask=e.reg_date.eq(b.reg_date[4]+pd.Timedelta(seconds=5))&e.channel.eq('1');e.loc[mask,'duration']='400'
 _,a,_=build(b,c,e,'train');assert not a.loc[4,'eligible']
