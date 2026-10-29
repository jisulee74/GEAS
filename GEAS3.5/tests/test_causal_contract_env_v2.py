import numpy as np
import pandas as pd
import pytest
from test_recorded_light_env_v1 import IdentityScaler,row
from geas35.rl.causal_contract_env_v2 import CausalContractStateUpdater,load_causal_contract_env,DEGREES
from geas35.features.derived import add_mdp_derived_features,dewpoint_c

def state(t):
 r=row(t);r[DEGREES[0]]=20.;r[DEGREES[1]]=30.;r['obs_current_vpd_kpa']=99.;r['obs_current_dewpoint_c']=99.;r['obs_current_condensation_margin_c']=99.;r['obs_derived_condensation_margin_pred_10m']=0.;return r

def step(u,current,t,temp=25.,rh=55.):
 n=state(t);n[DEGREES[0]]=999999.;n[DEGREES[1]]=999999.
 return u.advance(current_scaled=current,recorded_next_scaled=n,prediction_physical={'obs_indoor_temp_c':temp,'obs_indoor_humidity_pct':rh},action={'vent_pct':.2})

@pytest.mark.parametrize('t,end,temp,heat,cool',[
 ('2025-01-01 12:00','2025-01-01 12:05',20.,30.,30.),
 ('2025-01-01 12:00','2025-01-01 12:05',25.,20.,45.),
 ('2025-01-01 23:55','2025-01-02 00:00',25.,0.,15.),
 ('2025-01-01 12:00','2025-01-01 12:15',20.,50.,30.)])
def test_carry_midday_midnight_gap_and_no_future_carry(t,end,temp,heat,cool):
 a=state(t);u=CausalContractStateUpdater(IdentityScaler());u.reset(a);b=step(u,a,end,temp)
 assert b[DEGREES[0]]==heat and b[DEGREES[1]]==cool
 assert b.obs_current_dewpoint_c==pytest.approx(dewpoint_c(pd.Series([temp]),pd.Series([55.])).iloc[0])
 assert b.obs_current_condensation_margin_c==pytest.approx(temp-b.obs_current_dewpoint_c)
 assert b.obs_current_vpd_kpa!=99.

def test_missing_fails_before_mutation():
 a=state('2025-01-01');u=CausalContractStateUpdater(IdentityScaler());u.reset(a)
 with pytest.raises(ValueError,match='Finite predicted'):step(u,a,'2025-01-01 00:05',np.nan)
 pd.testing.assert_series_equal(u.current_physical,a,check_names=False)

def test_prehistory_is_past_contiguous_and_same_episode():
 rows=pd.DataFrame([state(t) for t in pd.date_range('2025-01-01 11:30',periods=9,freq='5min')]);rows['episode_id']='a';a=rows.iloc[6].copy()
 rows.loc[3,'episode_id']='other';rows.loc[7,'obs_indoor_temp_c']=9999.
 u=CausalContractStateUpdater(IdentityScaler(),rows);u.reset(a)
 assert len(u._history)==3 and all(r.reg_date<=a.reg_date for r in u._history)
 b=step(u,a,'2025-01-01 12:05');assert b.obs_indoor_temp_c==25.

def test_long_history_degree_does_not_truncate():
 a=state('2025-01-01 00:00');u=CausalContractStateUpdater(IdentityScaler());u.reset(a)
 # Keep intervals small to exceed 288 rows without crossing the date boundary.
 for i in range(1,301):a=step(u,a,pd.Timestamp('2025-01-01')+pd.Timedelta(minutes=i))
 assert len(u._history)==288
 assert a[DEGREES[1]]==pytest.approx(30+3*300)

def test_condensation_seed_matches_full_causal_builder():
 history=pd.DataFrame([state(t) for t in pd.date_range('2025-01-01 12:00',periods=8,freq='5min')])
 history['obs_indoor_temp_c']=np.arange(8)*.1+20.
 initial=history.iloc[6];u=CausalContractStateUpdater(IdentityScaler(),history);u.reset(initial)
 out=step(u,initial,history.iloc[7].reg_date,float(history.iloc[7].obs_indoor_temp_c),60.)
 expected=add_mdp_derived_features(history).iloc[-1].obs_derived_condensation_margin_pred_10m
 assert out.obs_derived_condensation_margin_pred_10m==pytest.approx(expected)

def test_test_locked():
 with pytest.raises(ValueError,match='Test locked'):load_causal_contract_env(project_root='.',crop='melon',split='test',protocol_manifest_path='x')

def test_five_minute_long_rollout_crosses_midnight():
 a=state('2025-01-01');u=CausalContractStateUpdater(IdentityScaler());u.reset(a)
 for i in range(1,301):a=step(u,a,pd.Timestamp('2025-01-01')+pd.Timedelta(minutes=5*i))
 assert len(u._history)==288 and a[DEGREES[1]]==pytest.approx(195.)
 assert a[DEGREES[0]]==0.

def test_condensation_gap_resets_fit_but_degrees_integrate_elapsed_time():
 hist=pd.DataFrame([state(t) for t in pd.date_range('2025-01-01 12:00',periods=7,freq='5min')]);hist['obs_indoor_temp_c']=np.arange(7)+20.
 a=hist.iloc[-1];u=CausalContractStateUpdater(IdentityScaler(),hist);u.reset(a)
 b=step(u,a,'2025-01-01 12:45',25.,55.)
 assert b.obs_derived_condensation_margin_pred_10m==pytest.approx(b.obs_current_condensation_margin_c)
 assert b[DEGREES[1]]==75.

def test_missing_target_or_carry_rejected_before_mutation():
 a=state('2025-01-01');a[DEGREES[0]]=np.nan;u=CausalContractStateUpdater(IdentityScaler());u.reset(a)
 with pytest.raises(ValueError,match='carry'):step(u,a,'2025-01-01 00:05')
 assert len(u._history)==1
