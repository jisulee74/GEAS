"""Opt-in HLV15 QC. No fit, model load, or legacy artifact mutation."""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

VERSION = 'HLV_WITH_15MIN_LIMIT_ELSE_INVALID'
VARS = ('out_light', 'out_windsp')
DEPENDENT = ('obs_outdoor_light','obs_outdoor_wind_speed','obs_is_daytime',
 'obs_current_target_temp_c','obs_current_target_temp_min_c','obs_current_target_temp_max_c',
 'obs_derived_dli_mol_m2','obs_derived_heating_degree_minutes','obs_derived_cooling_degree_minutes',
 'obs_derived_light_sum','obs_derived_light_sum_ratio','obs_derived_vent_loss_proxy')

def finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False

def first(row, names, default=np.nan):
    for c in names:
        if finite(row.get(c)): return float(row[c])
    return default

@dataclass
class HLV15:
    identity: object = None
    previous_time: object = None
    evidence_time: object = None
    evidence_value: float = np.nan

    def step(self, timestamp, identity, raw, raw_valid, *, provenance=True, discontinuity=False):
        t = pd.Timestamp(timestamp)
        invalid_time = pd.isna(t) or (self.previous_time is not None and t <= self.previous_time)
        reset = self.identity != identity or discontinuity or not provenance or invalid_time
        if reset: self.evidence_time, self.evidence_value = None, np.nan
        self.identity = identity
        # Keep a timestamp high-water mark: reversal cannot make an old row new.
        if not pd.isna(t) and (self.previous_time is None or t > self.previous_time):
            self.previous_time = t
        result = dict(method='INVALID', value=np.nan, evidence_timestamp=pd.NaT, age_minutes=np.nan,
                      identity_reset=bool(reset), reason='no_valid_past_raw', raw_value=float(raw) if finite(raw) else np.nan, raw_valid=bool(raw_valid))
        if invalid_time or not provenance or identity is None:
            result['reason'] = 'duplicate_or_reversed_timestamp' if invalid_time else 'unproven_identity_or_provenance'
            return result
        if raw_valid and finite(raw):
            self.evidence_time, self.evidence_value = t, float(raw)
            result.update(method='RAW',value=float(raw),evidence_timestamp=t,age_minutes=0.,reason='')
            return result
        if self.evidence_time is not None:
            age = (t-self.evidence_time).total_seconds()/60
            result.update(evidence_timestamp=self.evidence_time,age_minutes=age)
            if 0 < age <= 15:
                result.update(method='HLV',value=self.evidence_value,reason='')
            else: result['reason']='raw_evidence_older_than_15min'
        return result


def source_replay(raw, qc):
    """Input is pre-QC split stream; QC missing/rule flags are row-local only.
    No imputation-method/event registry is used to trigger the detector.
    A source segment is bound to its observed iot id using past rows only.
    Inserted rows have no measurement id and can never seed raw evidence.
    """
    assert len(raw)==len(qc)
    machines={v:HLV15() for v in VARS};records=[];binding={}
    rr=raw.to_dict('records');qq=qc.to_dict('records')
    for i,(r,q) in enumerate(zip(rr,qq)):
        key=tuple(str(r.get(c,'')) for c in ('crop','series_id','segment_id','source_segment_id','source_file_path'))
        known=all(x not in ('','None','nan','<NA>') for x in key)
        source_id=str(r.get('iot_data_idx',''))
        if finite(r.get('iot_data_idx')): source_id=str(int(float(r['iot_data_idx'])))
        if finite(r.get('idx')) and finite(r.get('iot_data_idx')):
            binding[key]=source_id
        device=binding.get(key)
        provenance=known and device is not None
        identity=(*key,device) if provenance else None
        for v in VARS:
            # QC AI is restricted to indoor variables in the frozen pipeline.
            # ffill is upstream/current metadata, not a future imputation result.
            valid=finite(r.get(v)) and finite(r.get('idx')) and finite(r.get('iot_data_idx'))
            for c in (v+'_ffill_flag',):
                valid &= finite(r.get(c)) and float(r[c])==0
            for c in (v+'_missing_flag',v+'_rule_outlier_flag'):
                valid &= finite(q.get(c)) and float(q[c])==0
            for suffix in ('_ai_outlier_flag','_ai_imputed_flag','_imputed_flag','_interpolated_flag','_reconstruction_flag'):
                if v+suffix in r: valid &= finite(r[v+suffix]) and float(r[v+suffix])==0
            discontinuity=any(bool(r.get(c,False)) for c in ('physical_discontinuity','sensor_discontinuity'))
            result=machines[v].step(r['reg_date'],(*identity,v) if identity else None,r.get(v),valid,
                                    provenance=provenance,discontinuity=discontinuity)
            records.append(dict(position=i,variable=v,reg_date=r['reg_date'],crop=r['crop'],series_id=r['series_id'],
                                segment_id=r['segment_id'],source_identity=str((*identity,v)) if identity else None,
                                raw_event_id=str(r.get('idx')),**result))
    return pd.DataFrame(records)


def derived_replay(base, reconstruction):
    """Fresh sequential accumulators, fixed current overrides; independent features retained."""
    out=base.copy(); light=reconstruction.loc[reconstruction.variable.eq('out_light'),'value'].to_numpy(float)
    wind=reconstruction.loc[reconstruction.variable.eq('out_windsp'),'value'].to_numpy(float)
    assert len(light)==len(base)
    result=[];previous=None;day=None;dli=heat=cool=energy=0.;identity=None
    for i,r in enumerate(base.to_dict('records')):
        t=pd.Timestamp(r['reg_date']);ident=tuple(r.get(c) for c in ('crop','series_id','segment_id'))
        dt=0. if previous is None else max((t-previous).total_seconds()/60,0.)
        same_day=day==t.date()
        if not same_day: dli=heat=cool=energy=0.
        elif ident!=identity: dli=heat=cool=energy=np.nan  # unknown same-day physical carry
        daily_dt=dt if same_day else 0.
        l=float(light[i]);w=float(wind[i]);daytime=float(l>=10) if finite(l) else np.nan
        target=first(r,['current_target_temp_c','target_temp_c'])
        if not finite(target) and finite(daytime):
            target=first(r,['obs_target_day_temp_c' if daytime else 'obs_target_night_temp_c'])
        vals=dict(obs_outdoor_light=l,obs_outdoor_wind_speed=w,obs_is_daytime=daytime,obs_current_target_temp_c=target)
        for suffix,sgn in [('min',-1),('max',1)]:
            value=first(r,[f'current_target_temp_{suffix}_c',f'target_temp_{suffix}_c',f'current_temp_{suffix}_c'])
            if not finite(value) and finite(daytime):
                period='day' if daytime else 'night'
                value=first(r,[f'target_{period}_temp_{suffix}_c',f'{period}_target_temp_{suffix}_c',f'{period}_temp_{suffix}_c'])
            if not finite(value):value=target+sgn if finite(target) else np.nan
            vals[f'obs_current_target_temp_{suffix}_c']=value
        if finite(l):
            dli+=max(l,0)*daily_dt*60*2.02/1e6;energy+=max(l,0)*daily_dt*60/1e4
        else:dli=energy=np.nan
        temp=first(r,['obs_indoor_temp_c'])
        if finite(temp) and finite(target):
            heat+=max(target-temp,0)*dt;cool+=max(temp-target,0)*dt
        else:heat=cool=np.nan
        measured=first(r,['out_light_sum','light_sum_jcm2','measured_sum_jcm2'])
        total=measured if finite(measured) else energy
        clear=first(r,['obs_derived_clear_sky_sum'])
        ratio=total/clear if finite(total) and finite(clear) and clear>0 else (0. if finite(total) else np.nan)
        vals.update(obs_derived_dli_mol_m2=dli,obs_derived_heating_degree_minutes=heat,
                    obs_derived_cooling_degree_minutes=cool,obs_derived_light_sum=total,obs_derived_light_sum_ratio=ratio)
        result.append(vals);previous=t;day=t.date();identity=ident
    for c in result[0] if result else []:out[c]=[r[c] for r in result]
    out['out_light']=light;out['out_windsp']=wind
    # B vent-loss is rebuilt by the official B admission function using previous commands.
    return out
