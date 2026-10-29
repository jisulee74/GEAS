"""Opt-in causal repair candidate. Vent timestamp semantics remain unresolved."""
from pathlib import Path
import numpy as np
import pandas as pd
from geas35.rl.recorded_light_env_v1 import RecordedLightStateUpdater,load_recorded_light_env
from geas35.rl.mdp_v1 import prepare_mdp_v1_frame
from geas35.features.derived import vpd_kpa,dewpoint_c
VERSION='geas35.rl.causal_contract.v2_candidate'
CONTRACT={'solar':'recorded_light_contract.v1: contemporaneous exogenous solar only',
 'aliases':'recompute current VPD/dewpoint/margin from predicted T/RH every advance',
 'degrees':'initial recorded carry; predicted endpoint increment; calendar-day reset with incoming elapsed interval',
 'prehistory':'strictly before reset, same crop/series/segment/episode, contiguous 5min, trailing 60min QC; condensation builder retains 30min/date boundaries',
 'vent':'UNRESOLVED: legacy previous executed action retained; no new vent semantics asserted',
 'admission':'diagnostic candidate only; not approved for Step16-19 or learning'}
DEGREES=('obs_derived_heating_degree_minutes','obs_derived_cooling_degree_minutes')

class CausalContractStateUpdater(RecordedLightStateUpdater):
    def __init__(self,scaler,prehistory=None):
        super().__init__(scaler)
        self.prehistory=prehistory if prehistory is not None else pd.DataFrame()

    def reset(self,initial_scaled):
        super().reset(initial_scaled)
        initial=self.current_physical;t=pd.Timestamp(initial.reg_date)
        if self.prehistory.empty:return
        past=self.prehistory
        mask=(past.reg_date<t)&(past.reg_date>=t-pd.Timedelta(minutes=60))
        for c in ('crop','series_id','segment_id','episode_id'):
            if c in initial and c in past:mask &= past[c].astype(str).eq(str(initial[c]))
        selected=past.loc[mask].sort_values('reg_date')
        rows=[];expected=t-pd.Timedelta(minutes=5)
        for _,row in selected.iloc[::-1].iterrows():
            if pd.Timestamp(row.reg_date)!=expected:break
            rows.append(row.copy());expected-=pd.Timedelta(minutes=5)
        self._history=[*reversed(rows),initial]

    def advance(self,*,current_scaled,recorded_next_scaled,prediction_physical,action):
        before=self.current_physical;recorded=self._inverse(recorded_next_scaled)
        t0=pd.Timestamp(before.reg_date);t1=pd.Timestamp(recorded.reg_date)
        if pd.isna(t1) or pd.isna(t0) or t1<=t0:raise ValueError('Finite increasing timestamp required')
        for c in ('obs_indoor_temp_c','obs_indoor_humidity_pct'):
            if c not in prediction_physical or not np.isfinite(float(prediction_physical[c])):
                raise ValueError('Finite predicted T/RH required')
        target=float(recorded.get('obs_current_target_temp_c',before['obs_current_target_temp_c']))
        if not np.isfinite(target):raise ValueError('Finite target required')
        if not all(np.isfinite(float(before[c])) for c in DEGREES):raise ValueError('Finite degree carry required')
        # All non-cumulative derived windows are <=60 minutes. Solar is provided
        # exogenously and degrees have independent carry. Keep the audit history
        # but avoid re-evaluating its irrelevant prefix on every step.
        saved_history=self._history
        cutoff=t1-pd.Timedelta(minutes=60)
        self._history=[r for r in saved_history if pd.Timestamp(r.reg_date)>=cutoff]
        if not self._history:self._history=[saved_history[-1]]
        try:
            super().advance(current_scaled=current_scaled,recorded_next_scaled=recorded_next_scaled,
                            prediction_physical=prediction_physical,action=action)
            newest=self._history[-1]
        except Exception:
            self._history=saved_history
            raise
        self._history=[*saved_history,newest][-self.history_limit:]
        row=self._history[-1];temp=float(row.obs_indoor_temp_c);rh=float(row.obs_indoor_humidity_pct)
        ts=pd.Series([temp]);hs=pd.Series([rh])
        row['obs_current_vpd_kpa']=float(vpd_kpa(ts,hs).iloc[0])
        row['obs_current_dewpoint_c']=float(dewpoint_c(ts,hs).iloc[0])
        row['obs_current_condensation_margin_c']=temp-row['obs_current_dewpoint_c']
        dt=(t1-t0).total_seconds()/60.;same=t1.date()==t0.date()
        row[DEGREES[0]]=(float(before[DEGREES[0]]) if same else 0.)+max(target-temp,0.)*dt
        row[DEGREES[1]]=(float(before[DEGREES[1]]) if same else 0.)+max(temp-target,0.)*dt
        return self.scaler.transform_frame(self.current_physical.to_frame().T).iloc[0]


def load_causal_contract_env(*,project_root,crop,split,protocol_manifest_path):
    if split not in ('train','validation'):raise ValueError('Test locked; Train/Validation only')
    env=load_recorded_light_env(project_root=project_root,crop=crop,split=split,protocol_manifest_path=protocol_manifest_path)
    path=Path(project_root)/f'offline_dataset_preparation/datasets/03_quality_controlled/{crop}/{split}.parquet'
    history=prepare_mdp_v1_frame(pd.read_parquet(path))
    env.state_updater=CausalContractStateUpdater(env.scaler,history)
    env.diagnostic_environment_version=VERSION
    env.causal_contract=dict(CONTRACT)
    return env
