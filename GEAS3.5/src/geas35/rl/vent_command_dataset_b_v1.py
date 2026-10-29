"""Fail-closed Phase 0 builder. No Test, training model, or support fitting.

Input commands are cycle-linked, device-verified, seven-channel transmission
records normalized by an audited adapter, NOT nearest-rounded action rows.
"""
from __future__ import annotations
import hashlib,json
import numpy as np
import pandas as pd
from geas35.rl.mdp_v1 import (MDP_V1_OBSERVATION_COLUMNS,MDP_V1_ACTION_COLUMNS,
    MDP_V1_TRANSITION_TARGET_COLUMNS,fit_mdp_v1_observation_scaler,compute_mdp_v1_reward)
VERSION='vent_command_carry_b_v1'
KEYS=['crop','series_id','segment_id','episode_id','reg_date']
ACTIONS=list(MDP_V1_ACTION_COLUMNS)
OBS=list(MDP_V1_OBSERVATION_COLUMNS)
CHANNELS=[1,2,4,5,9,10,13]
PROVENANCE=['device_id','source_table','source_confidence','control_cycle_id','sent_at_min','sent_at_max','event_ids','channels','mapping_verified','left_vent','right_vent']

def fingerprint(frame):
    return hashlib.sha256(frame.to_json(orient='split',date_format='iso',double_precision=15).encode()).hexdigest()

def _text_list(value):
    try:return json.loads(value) if isinstance(value,str) else list(value)
    except (ValueError,TypeError):return []

def assemble_b_frame(base:pd.DataFrame,commands:pd.DataFrame|None,*,split:str):
    if split not in ('train','validation'):raise ValueError('Test locked')
    frame=base.copy().reset_index(drop=True)
    if frame[KEYS].isna().any().any():raise ValueError('Missing state boundary key')
    frame['reg_date']=pd.to_datetime(frame.reg_date)
    if not frame.reg_date.is_monotonic_increasing:raise ValueError('Unsorted state timestamps')
    n=len(frame);audit=frame[KEYS].copy();audit['row_id']=[hashlib.sha256(f'{split}|{v}'.encode()).hexdigest() for v in frame[KEYS].astype(str).agg('|'.join,axis=1)]
    audit['split']=split;audit['source_table']=np.where(frame.reg_date<=pd.Timestamp('2026-01-22 09:05'),'data_silla_enc','send_ds_control_api_history');audit['source_inferred_from_legacy_cutoff']=True
    cmd=pd.DataFrame(index=frame.index,columns=PROVENANCE+ACTIONS,dtype=object)
    reason=pd.Series('missing_cycle_linked_transmission',index=frame.index,dtype=object)
    if commands is not None and len(commands):
        missing=set(KEYS+PROVENANCE+ACTIONS)-set(commands)
        if missing:raise ValueError(f'Incomplete command schema: {sorted(missing)}')
        c=commands.copy();c['reg_date']=pd.to_datetime(c.reg_date)
        if 'split' in c and not c['split'].eq(split).all():raise ValueError('Cross-split command input')
        duplicate=c.duplicated(KEYS,keep=False)
        duplicates=set(map(tuple,c.loc[duplicate,KEYS].itertuples(index=False,name=None)))
        event_keys = []
        for _, event_row in c.iterrows():
            event_keys.extend((str(event_row.source_table), str(event_row.device_id), str(e)) for e in _text_list(event_row.event_ids))
        from collections import Counter
        reused_events = {k for k, count in Counter(event_keys).items() if count > 1}
        c=c.loc[~duplicate].set_index(KEYS)
        for i,key in enumerate(frame[KEYS].itertuples(index=False,name=None)):
            if key in duplicates:reason.iloc[i]='ambiguous_command';continue
            if key not in c.index:continue
            row=c.loc[key];cmd.loc[i]=row[PROVENANCE+ACTIONS].to_numpy();reason.iloc[i]=''
            if row.mapping_verified not in (True,np.bool_(True)) or any(pd.isna(row[k]) or not str(row[k]).strip() for k in ('device_id','source_table','source_confidence','control_cycle_id')):reason.iloc[i]='unverified_identity_or_cycle'
            ids=_text_list(row.event_ids);channels=_text_list(row.channels)
            if len(ids)!=7 or len(set(map(str,ids)))!=7 or set(map(str,channels))!=set(map(str,CHANNELS)) or len(channels)!=7:reason.iloc[i]='invalid_event_or_channel_provenance'
            if any((str(row.source_table),str(row.device_id),str(e)) in reused_events for e in ids):reason.iloc[i]='reused_transmission_event'
            lo=pd.to_datetime(row.sent_at_min,errors='coerce');hi=pd.to_datetime(row.sent_at_max,errors='coerce');t=frame.reg_date.iloc[i]
            if pd.isna(lo) or pd.isna(hi) or not (t<=lo<=hi<t+pd.Timedelta(minutes=5)):reason.iloc[i]='unavailable_or_late_transmission'
            vals=pd.to_numeric(row[ACTIONS],errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(vals).all() or (vals<0).any() or (vals>1).any() or not np.isin(vals[3:],[0.,1.]).all():reason.iloc[i]='invalid_command_unit_or_value'
            sides=pd.to_numeric(row[['left_vent','right_vent']],errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(sides).all() or (sides<0).any() or (sides>1).any() or not np.isclose(np.mean(sides),vals[0],rtol=1e-6,atol=1e-8):reason.iloc[i]='invalid_left_right_vent'
    valid=reason.eq('');same=frame.reg_date.diff().eq(pd.Timedelta(minutes=5))
    for k in KEYS[:-1]:same &= frame[k].eq(frame[k].shift())
    for k in ['device_id','source_table','source_confidence']:same &= cmd[k].notna()&cmd[k].eq(cmd[k].shift())
    duplicate_state=frame.duplicated(KEYS,keep=False);valid &= ~duplicate_state;reason.loc[duplicate_state]='duplicate_state'
    prev=valid.shift(fill_value=False)&same
    prevprev=prev & prev.shift(fill_value=False)
    actions=cmd[ACTIONS].apply(pd.to_numeric,errors='coerce')
    # Keep observation availability separate from current command validity.
    observations=frame[OBS].copy()
    for a in ACTIONS:observations['obs_prev_'+a]=actions[a].shift().where(prev)
    old_vent=observations.obs_derived_vent_loss_proxy.copy()
    factor=(frame.obs_indoor_temp_c-frame.obs_outdoor_temp_c).clip(lower=0)*(1-(frame.obs_outdoor_light.clip(lower=0)/1000).clip(0,1))
    observations['obs_derived_vent_loss_proxy']=actions.vent_pct.shift().where(prev)*factor
    same_next=same.shift(-1,fill_value=False)&frame.reg_date.dt.date.eq(frame.reg_date.shift(-1).dt.date)
    finite=pd.Series(np.isfinite(observations.to_numpy(dtype=float)).all(axis=1),index=frame.index)
    eligible=valid&prev&prevprev&same_next&finite&finite.shift(-1,fill_value=False)&~duplicate_state.shift(-1,fill_value=False)
    reasons=[]
    for i in range(n):
        flags=[]
        if reason.iloc[i]:flags.append(reason.iloc[i])
        if not prev.iloc[i]:flags.append('unknown_previous_command_or_boundary')
        if not prevprev.iloc[i]:flags.append('unknown_previous_previous_command_or_boundary')
        if i+1<n and duplicate_state.iloc[i+1]:flags.append('ambiguous_next_state')
        if not same_next.iloc[i]:flags.append('invalid_next_timestamp_or_boundary')
        if not finite.iloc[i] or not (bool(finite.iloc[i+1]) if i+1<n else False):flags.append('nonfinite_current_or_next_observation')
        reasons.append('|'.join(flags))
    result=pd.concat([audit[['row_id','split',*KEYS]],observations,actions],axis=1)
    for target,col in zip(MDP_V1_TRANSITION_TARGET_COLUMNS,['obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm']):result[target]=frame[col].shift(-1)
    result['target_timestamp']=frame.reg_date.shift(-1)
    for c in OBS:result['next_'+c]=observations[c].shift(-1)
    result['reward']=np.nan
    for i in np.flatnonzero(eligible):
        physical=frame.iloc[i+1].copy()
        for c in OBS:physical[c]=observations.iloc[i+1][c]
        value,_=compute_mdp_v1_reward(physical,actions.iloc[i].to_dict(),prev_action=actions.iloc[i-1].to_dict(),prev_prev_action=actions.iloc[i-2].to_dict());result.loc[i,'reward']=value
    result['rl_valid_transition']=eligible.astype(int);result['done']=(~eligible.shift(-1,fill_value=False)).astype(int)
    result['mdp_v1_rollout_id']=(~eligible.shift(fill_value=False)).cumsum()
    for lag,prefix in [(0,'current'),(1,'previous'),(2,'previous_previous')]:
        shifted=cmd.shift(lag)
        carry_ok=valid if lag==0 else prev if lag==1 else prevprev
        if lag:shifted=shifted.where(carry_ok,other=None)
        for c in PROVENANCE+ACTIONS:audit[prefix+'_'+c]=shifted[c]
        audit[prefix+'_sensor_timestamp']=frame.reg_date.shift(lag).where(carry_ok)
    audit['current_command_valid']=valid;audit['previous_command_valid']=prev;audit['previous_previous_command_valid']=prevprev
    audit['eligible']=eligible;audit['reason']=reasons;audit['vent_loss_A']=old_vent;audit['vent_loss_B']=observations.obs_derived_vent_loss_proxy
    audit['source_table']=cmd.source_table.where(cmd.source_table.notna(),audit.source_table)
    audit['source_inferred_from_legacy_cutoff']=cmd.source_table.isna()
    result=result.loc[eligible].reset_index(drop=True)
    if len(result):
        assert np.isfinite(result[OBS+ACTIONS+list(MDP_V1_TRANSITION_TARGET_COLUMNS)+['reward']].to_numpy(dtype=float)).all()
        assert (result.target_timestamp-result.reg_date).eq(pd.Timedelta(minutes=5)).all()
    return result,audit

def fit_b_scaler(train:pd.DataFrame,*,split='train'):
    if split!='train':raise ValueError('Scaler fit is Train-only')
    if 'split' not in train or not train['split'].eq('train').all():raise ValueError('Scaler fit rows must carry Train split provenance')
    if train.empty:raise ValueError('No verified eligible Train rows; scaler fit blocked')
    if not train.rl_valid_transition.eq(1).all():raise ValueError('Ineligible fit rows')
    prepared = train.copy()
    prepared["mdp_v1_valid_transition"] = True
    return fit_mdp_v1_observation_scaler(prepared,observation_columns=OBS)

def window_counts(frame):
    counts={str(h):0 for h in (3,6,12)}
    for _,g in frame.groupby('mdp_v1_rollout_id',sort=False):
        for h in (3,6,12):counts[str(h)]+=max(0,len(g)-h+1)
    return counts
