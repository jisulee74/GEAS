from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from geas35.rl.vent_command_dataset_b_v1 import assemble_b_frame,fit_b_scaler,window_counts,fingerprint,OBS,ACTIONS,CHANNELS
from geas35.features.derived import add_mdp_derived_features,vpd_kpa,dewpoint_c

def fixture(n=10):
 f=pd.DataFrame(0.,index=range(n),columns=OBS)
 f['reg_date']=pd.date_range('2025-01-01 12:00',periods=n,freq='5min')
 for k,v in [('crop','melon'),('series_id','s'),('segment_id','g'),('episode_id','e')]:f[k]=v
 f['obs_indoor_temp_c']=24.+np.arange(n)/10;f['obs_indoor_humidity_pct']=60.;f['obs_indoor_co2_ppm']=400.;f['obs_outdoor_temp_c']=15.;f['obs_outdoor_light']=100.;f['obs_current_target_temp_c']=22.;f['obs_current_target_temp_min_c']=21.;f['obs_current_target_temp_max_c']=23.
 f=add_mdp_derived_features(f);f['obs_current_vpd_kpa']=vpd_kpa(f.obs_indoor_temp_c,f.obs_indoor_humidity_pct);f['obs_current_dewpoint_c']=dewpoint_c(f.obs_indoor_temp_c,f.obs_indoor_humidity_pct);f['obs_current_condensation_margin_c']=f.obs_indoor_temp_c-f.obs_current_dewpoint_c
 c=f[['crop','series_id','segment_id','episode_id','reg_date']].copy();c['device_id']='97';c['source_table']='api';c['source_confidence']='verified';c['control_cycle_id']=[str(i) for i in range(n)];c['mapping_verified']=True
 c['sent_at_min']=c.reg_date+pd.Timedelta(seconds=10);c['sent_at_max']=c.reg_date+pd.Timedelta(seconds=20)
 c['event_ids']=[json.dumps([f'{i}-{j}' for j in CHANNELS]) for i in range(n)];c['channels']=json.dumps(CHANNELS)
 for a in ACTIONS:c[a]=.2 if a=='vent_pct' else 0.
 c['left_vent']=.2;c['right_vent']=.2
 return f,c

def test_current_action_invariant_and_next_carry():
 f,c=fixture();a,_=assemble_b_frame(f,c,split='train');c.loc[4,['vent_pct','left_vent','right_vent']]=.8;b,_=assemble_b_frame(f,c,split='train')
 aa=a.set_index('reg_date').loc[f.reg_date[4]];bb=b.set_index('reg_date').loc[f.reg_date[4]]
 np.testing.assert_allclose(aa[OBS].to_numpy(float),bb[OBS].to_numpy(float));assert bb.vent_pct==.8
 assert bb.next_obs_prev_vent_pct==.8
 assert aa.target_next_indoor_temp_c==bb.target_next_indoor_temp_c

def test_previous_changes_proxy_and_reward_history():
 f,c=fixture();c.loc[3,['vent_pct','left_vent','right_vent']]=.6;b,_=assemble_b_frame(f,c,split='train');r=b.set_index('reg_date').loc[f.reg_date[4]]
 assert r.obs_prev_vent_pct==.6
 assert r.obs_derived_vent_loss_proxy==pytest.approx(.6*(24.4-15)*.9)
 assert np.isfinite(r.reward)

@pytest.mark.parametrize('kind',['gap','episode','source','device','unknown','late','ambiguous'])
def test_no_boundary_or_unknown_leak(kind):
 f,c=fixture()
 if kind=='gap':
  f.loc[4:,'reg_date']+=pd.Timedelta(minutes=5);c.loc[4:,'reg_date']+=pd.Timedelta(minutes=5);c.loc[4:,'sent_at_min']+=pd.Timedelta(minutes=5);c.loc[4:,'sent_at_max']+=pd.Timedelta(minutes=5)
 if kind=='episode':f.loc[4:,'episode_id']='new';c.loc[4:,'episode_id']='new'
 if kind=='source':c.loc[4:,'source_table']='new'
 if kind=='device':c.loc[4:,'device_id']='other'
 if kind=='unknown':c=c.drop(index=3)
 if kind=='late':c.loc[3,'sent_at_max']=f.reg_date[4]+pd.Timedelta(seconds=1)
 if kind=='ambiguous':c=pd.concat([c,c.iloc[[3]]],ignore_index=True)
 b,a=assemble_b_frame(f,c,split='train');assert not bool(a.eligible.iloc[4]);assert pd.isna(a.previous_vent_pct.iloc[4]);assert not b.reg_date.eq(f.reg_date[4]).any()

def test_missing_provenance_never_produces_samples_or_scaler():
 f,_=fixture();b,a=assemble_b_frame(f,None,split='train');assert b.empty;assert a.reason.str.contains('missing_cycle').all()
 with pytest.raises(ValueError,match='blocked'):fit_b_scaler(b)

def test_exact_targets_windows_determinism_and_train_only_fit():
 f,c=fixture();b,_=assemble_b_frame(f,c,split='train');other,_=assemble_b_frame(f,c,split='train');assert fingerprint(b)==fingerprint(other);assert len(b)==7;assert window_counts(b)=={'3':5,'6':2,'12':0}
 assert (b.target_timestamp-b.reg_date).eq(pd.Timedelta(minutes=5)).all()
 for _,r in b.iterrows():assert r.target_next_indoor_temp_c==pytest.approx(f.set_index('reg_date').loc[r.target_timestamp].obs_indoor_temp_c)
 scaler=fit_b_scaler(b);assert scaler.means['obs_indoor_temp_c']==pytest.approx(b.obs_indoor_temp_c.mean())
 before=dict(scaler.means);v=b.copy();v['obs_indoor_temp_c']=1000.;scaler.transform_frame(v);assert before==scaler.means
 with pytest.raises(ValueError,match='Train-only'):fit_b_scaler(v,split='validation')
 with pytest.raises(ValueError,match='Test locked'):assemble_b_frame(f,c,split='test')

def test_solar_alias_degree_condensation_unchanged():
 f,c=fixture();b,_=assemble_b_frame(f,c,split='train');truth=f.set_index('reg_date').loc[b.reg_date]
 cols=[x for x in OBS if x.startswith('obs_derived_') and x!='obs_derived_vent_loss_proxy']+['obs_current_vpd_kpa','obs_current_dewpoint_c','obs_current_condensation_margin_c']
 np.testing.assert_allclose(b[cols],truth[cols],rtol=1e-6,atol=1e-8)

@pytest.mark.parametrize('field,value',[('vent_pct',50.),('left_vent',np.nan),('fan_run',.3),('mapping_verified',False),('event_ids','[]')])
def test_invalid_units_or_provenance_excluded(field,value):
 f,c=fixture();c.loc[3,field]=value;b,a=assemble_b_frame(f,c,split='train');assert not a.eligible.iloc[3] and not a.eligible.iloc[4]

def test_reused_event_and_duplicate_state_rejected():
 f,c=fixture();c.loc[4,'event_ids']=c.loc[3,'event_ids']
 b,a=assemble_b_frame(f,c,split='train')
 assert not a.loc[[3,4],'current_command_valid'].any()
 f,c=fixture();f=pd.concat([f.iloc[:5],f.iloc[[4]],f.iloc[5:]],ignore_index=True)
 b,a=assemble_b_frame(f,c,split='train')
 assert not b.reg_date.eq(f.reg_date.iloc[3]).any()
 assert not a.loc[a.reg_date.eq(f.reg_date.iloc[4]),'eligible'].any()

def test_reward_uses_both_previous_actions():
 from geas35.rl.mdp_v1 import compute_mdp_v1_reward
 f,c=fixture();c.loc[2,['vent_pct','left_vent','right_vent']]=.9
 b,_=assemble_b_frame(f,c,split='train');r=b.set_index('reg_date').loc[f.reg_date[4]]
 next_row=f.iloc[5].copy()
 for col in OBS:next_row[col]=r['next_'+col]
 expected,_=compute_mdp_v1_reward(next_row,c.iloc[4][ACTIONS].to_dict(),prev_action=c.iloc[3][ACTIONS].to_dict(),prev_prev_action=c.iloc[2][ACTIONS].to_dict())
 assert r.reward==pytest.approx(expected)

def test_validation_rows_cannot_be_fit_with_default_argument():
 f,c=fixture();b,_=assemble_b_frame(f,c,split='validation')
 with pytest.raises(ValueError,match='Train split provenance'):fit_b_scaler(b)
