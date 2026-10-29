from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geas35.rl.recorded_light_env_v1 import RecordedLightStateUpdater,SOLAR_FEATURES,load_recorded_light_env
from geas35.models.transition.causal_state import CausalStateUpdater
from geas35.features.derived import add_mdp_derived_features

class IdentityScaler:
    def inverse_transform_frame(self,frame):return frame.copy()
    def transform_frame(self,frame):return frame.copy()

def row(timestamp,solar=0.,dli=0.):
    values={'reg_date':pd.Timestamp(timestamp),'obs_indoor_temp_c':20.,'obs_indoor_humidity_pct':60.,
        'obs_outdoor_temp_c':15.,'obs_outdoor_light':100.,'obs_current_target_temp_c':22.,
        'vent_pct':0.,'obs_derived_light_sum':solar,'obs_derived_dli_mol_m2':dli,
        'obs_derived_clear_sky_sum':50.,'obs_derived_light_sum_ratio':solar/50.,'obs_derived_light_eta_minutes':5.}
    return pd.Series(values)

def advance(updater,current,next_row):
    return updater.advance(current_scaled=current,recorded_next_scaled=next_row,
        prediction_physical={'obs_indoor_temp_c':21.},action={'vent_pct':.2})

@pytest.mark.parametrize('start,end,solar,dli',[
    ('2025-01-01 12:00','2025-01-01 12:05',0.,2.),
    ('2025-01-01 12:00','2025-01-01 12:05',40.,8.),
    ('2025-01-01 23:55','2025-01-02 00:00',0.,0.),
])
def test_current_timestamp_features_preserved_with_prediction(start,end,solar,dli):
    a=row(start,30.,6.);b=row(end,solar,dli);u=RecordedLightStateUpdater(IdentityScaler());u.reset(a)
    before=b.copy();out=advance(u,a,b)
    for c in SOLAR_FEATURES:
        assert out[c]==b[c] and u.current_physical[c]==b[c] and u.history_frame.iloc[-1][c]==b[c]
    assert out['obs_indoor_temp_c']==21. and out['vent_pct']==.2
    pd.testing.assert_series_equal(before,b)

def test_legacy_zero_vs_integrated_regression():
    a=row('2025-01-01 12:00');b=row('2025-01-01 12:05')
    old=CausalStateUpdater(IdentityScaler());old.reset(a)
    new=RecordedLightStateUpdater(IdentityScaler());new.reset(a)
    assert advance(old,a,b)['obs_derived_light_sum']==pytest.approx(100*300/10000)
    assert advance(new,a,b)['obs_derived_light_sum']==0.

def test_missing_contract_fails_without_mutating_history():
    a=row('2025-01-01');b=row('2025-01-01 00:05');b['obs_derived_dli_mol_m2']=np.nan
    u=RecordedLightStateUpdater(IdentityScaler());u.reset(a)
    with pytest.raises(ValueError,match='finite'):advance(u,a,b)
    assert len(u._history)==1

def test_test_is_locked_before_loading():
    with pytest.raises(ValueError,match='Test locked'):
        load_recorded_light_env(project_root='.',crop='melon',split='test',protocol_manifest_path='missing')

def test_measured_zero_and_missing_fallback_are_distinct():
    frame=pd.DataFrame([row('2025-01-01 12:00'),row('2025-01-01 12:05')]);frame['out_light_sum']=[0.,0.]
    assert add_mdp_derived_features(frame).iloc[-1]['obs_derived_light_sum']==0.
    frame['out_light_sum']=np.nan
    value=add_mdp_derived_features(frame).iloc[-1]
    assert value['obs_derived_light_sum']==pytest.approx(3.)
    assert value['obs_derived_dli_mol_m2']==pytest.approx(100*300*2.02/1e6)
