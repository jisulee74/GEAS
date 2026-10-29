"""B command memory, external contract v2. No model/support fitting.
Cycle matching is exact sensor/copy timestamp plus unique per-channel payload
agreement within [t,t+5m). Multiple candidates are quarantined, never selected.
"""
import json
import numpy as np
import pandas as pd
from geas35.rl.vent_command_dataset_b_v1 import OBS,ACTIONS,KEYS,fingerprint,fit_b_scaler,window_counts
from geas35.rl.mdp_v1 import compute_mdp_v1_reward
MAP={1:'pred_ltw',2:'pred_rtw',4:'pred_pc1',5:'pred_pc2',7:'pred_cp1',8:'pred_cp2',9:'pred_heater',10:'pred_cooler',11:'pred_tw1',12:'pred_tw2',13:'pred_fan'}
BINARY={7,8,9,10,13};NEEDED=[1,2,4,5,9,10,13]

def value(ch,ratio,operation=None):
 try:v=float(ratio)
 except (ValueError,TypeError):return np.nan
 if not np.isfinite(v) or not 0<=v<=100:return np.nan
 if ch in BINARY:
  if operation is not None:
   if operation not in ('ON','OFF'):return np.nan
   if v not in (0,1,100) or (operation=='OFF' and v!=0) or (operation=='ON' and v==0):return np.nan
  return float(v>0) if v in (0,1,100) else np.nan
 if operation is not None and operation!='OPEN':return np.nan
 return v/100

def memory_json(mem):
 return json.dumps({ch:{**item,'value':item['value'] if np.isfinite(item['value']) else None} for ch,item in mem.items()},sort_keys=True,allow_nan=False)

def vector(mem):
 vals={ch:mem.get(ch,{}).get('value',np.nan) for ch in NEEDED}
 return dict(zip(ACTIONS,[(vals[1]+vals[2])/2,vals[4],vals[5],vals[9],vals[10],vals[13]]))

def restore(base,cycles,events):
 b=base.reset_index(drop=True);ev=events.copy();ev['reg_date']=pd.to_datetime(ev.reg_date);ev['channel']=pd.to_numeric(ev.channel,errors='coerce');ev=ev.sort_values(['reg_date','idx']).reset_index(drop=True)
 events_list=ev.to_dict('records');times=ev.reg_date.to_numpy();classes=['outside_sensor_interval']*len(ev);sensor_times=[pd.NaT]*len(ev)
 groups={}
 for row in cycles.to_dict('records'):groups.setdefault(pd.Timestamp(row['reg_date']),[]).append(row)
 for e in events_list:
  e['_channel']=int(e['channel']) if pd.notna(e['channel']) else -1;e['_value']=value(e['_channel'],e['ratio'],e['operation']);e['_minute']=pd.Timestamp(e['reg_date']).floor('min')
 mem={};previous=None;prevbefore={};records=[]
 for row in b.to_dict('records'):
  t=pd.Timestamp(row['reg_date']);end=t+pd.Timedelta(minutes=5)
  continuous=previous is not None and t-pd.Timestamp(previous['reg_date'])==pd.Timedelta(minutes=5) and all(row[k]==previous[k] for k in KEYS[:-1])
  if not continuous:mem={};prevbefore={}
  before=dict(mem);p=vector(before);pp=vector(prevbefore)
  lo=int(times.searchsorted(t.to_datetime64()));hi=int(times.searchsorted(end.to_datetime64()));indices=range(lo,hi);cycle=groups.get(t)
  bychannel={};bytime={}
  for j in indices:
   e=events_list[j];bychannel.setdefault(e['_channel'],[]).append(j);bytime.setdefault((e['reg_date'],e['_channel']),[]).append(j)
  linked=False;ambiguous=False;matched=set();expected={};cid=None
  if cycle is not None and len(cycle)==1:
   cr=cycle[0];cid=int(cr['idx']);expected={ch:value(ch,cr[col]) for ch,col in MAP.items()}
   for ch in MAP:
    candidates=[]
    for j in bychannel.get(ch,[]):
     e=events_list[j]
     if e['_minute']==t and str(e['duration']).strip() in ('','0','None','nan') and np.isfinite(expected[ch]) and np.isfinite(e['_value']) and abs(e['_value']-expected[ch])<=1e-8:candidates.append(j)
    if len(candidates)>1:ambiguous=True
    elif len(candidates)==1:matched.add(candidates[0])
   linked=not ambiguous
  elif cycle is not None:ambiguous=True
  auto=dict(before)
  for j in sorted(matched):
   e=events_list[j];auto[e['_channel']]={'value':e['_value'],'event_id':int(e['idx']),'sent_at':str(e['reg_date']),'source':'send_ds_control_api_history'}
  if linked:linked=all(ch in auto and np.isfinite(expected[ch]) and np.isfinite(auto[ch]['value']) and abs(auto[ch]['value']-expected[ch])<=1e-8 for ch in MAP)
  unmatched=set(indices)-matched if linked else set(indices);override=bool(unmatched) if linked else hi>lo
  retained=sum(ch in before and np.isfinite(before[ch]['value']) and ch not in bychannel for ch in MAP)
  for j in indices:
   sensor_times[j]=t;classes[j]='automatic_payload_linked' if linked and j in matched else 'ambiguous_automatic_candidate' if ambiguous else 'manual_remote_or_unlinked'
  for (at,ch),js in sorted(bytime.items()):
   vals=[events_list[j]['_value'] for j in js];known=all(np.isfinite(v) for v in vals) and len(set(vals))==1
   mem[ch]={'value':vals[0] if known else np.nan,'event_id':','.join(str(events_list[j]['idx']) for j in js),'sent_at':str(at),'source':'send_ds_control_api_history'}
  current=vector(auto) if linked else {k:np.nan for k in ACTIONS}
  rec={'reg_date':t,'cycle_id':cid,'cycle_present':cycle is not None,'linked':linked,'ambiguous':ambiguous,'intervening_override':linked and override,'unlinked_event_interval':not linked and hi>lo,'retained_channels':retained,'event_count':hi-lo,'memory_before':memory_json(before),'memory_after':memory_json(mem),'automatic_memory':memory_json(auto if linked else {})}
  for prefix,vec in [('previous_',p),('previous_previous_',pp),('current_',current)]:rec.update({prefix+k:v for k,v in vec.items()})
  records.append(rec);prevbefore=before;previous=row
 ev['classification']=classes;ev['sensor_timestamp']=pd.to_datetime(sensor_times)
 return pd.DataFrame(records),ev

def build(base,cycles,events,split):
 if split not in ('train','validation'):raise ValueError('Test locked')
 base=base.reset_index(drop=True).copy();base.reg_date=pd.to_datetime(base.reg_date)
 if base[KEYS].isna().any().any() or not base.reg_date.is_monotonic_increasing or base.duplicated(KEYS).any():raise ValueError('Invalid sensor registry')
 cycles=cycles.copy();cycles.reg_date=pd.to_datetime(cycles.reg_date)
 a,event_audit=restore(base,cycles,events);obs=base[OBS].copy()
 for k in ACTIONS:obs['obs_prev_'+k]=a['previous_'+k]
 factor=(base.obs_indoor_temp_c-base.obs_outdoor_temp_c).clip(lower=0)*(1-(base.obs_outdoor_light.clip(lower=0)/1000).clip(0,1))
 obs['obs_derived_vent_loss_proxy']=a.previous_vent_pct*factor
 same=base.reg_date.shift(-1).sub(base.reg_date).eq(pd.Timedelta(minutes=5))
 for k in KEYS[:-1]:same &=base[k].eq(base[k].shift(-1))
 same &=base.reg_date.dt.date.eq(base.reg_date.shift(-1).dt.date)
 unknown=~np.isfinite(a[['previous_'+k for k in ACTIONS]+['previous_previous_'+k for k in ACTIONS]].to_numpy(float)).all(axis=1)
 finite=pd.Series(np.isfinite(obs.to_numpy(float)).all(axis=1));clean=a.linked&~a.intervening_override
 # Reward history remains a sequence of uninterrupted automatic cycles.
 history_clean=clean.shift(1,fill_value=False)&clean.shift(2,fill_value=False)
 eligible=clean&history_clean&~unknown&same&finite&finite.shift(-1,fill_value=False)
 a['eligible']=eligible;a['command_memory_unknown']=unknown;a['history_clean']=history_clean;a['vent_loss_A']=base.obs_derived_vent_loss_proxy;a['vent_loss_B']=obs.obs_derived_vent_loss_proxy
 reasons=[]
 for i in range(len(a)):
  flags=[]
  for ok,label in [(a.cycle_present[i],'missing_automatic_cycle'),(a.linked[i],'unlinked_or_ambiguous_transmission'),(not a.intervening_override[i],'intervening_override'),(history_clean[i],'unclean_previous_two_cycles'),(not unknown[i],'command_memory_unknown'),(same[i],'invalid_target_boundary'),(finite[i] and (finite[i+1] if i+1<len(a) else False),'nonfinite_observation')]:
   if not ok:flags.append(label)
  reasons.append('|'.join(flags))
 a['reason']=reasons
 import hashlib
 ids=[hashlib.sha256(f'{split}|{x}'.encode()).hexdigest() for x in base[KEYS].astype(str).agg('|'.join,axis=1)]
 result=base[KEYS].copy();result['row_id']=ids;result['split']=split
 result=pd.concat([result,obs,a[['current_'+k for k in ACTIONS]].rename(columns={'current_'+k:k for k in ACTIONS})],axis=1)
 for k in OBS:result['next_'+k]=obs[k].shift(-1)
 for name,col in [('temp_c','obs_indoor_temp_c'),('humidity_pct','obs_indoor_humidity_pct'),('co2_ppm','obs_indoor_co2_ppm')]:result['target_next_indoor_'+name]=base[col].shift(-1)
 result['target_timestamp']=base.reg_date.shift(-1);result['reward']=np.nan
 for i in np.flatnonzero(eligible):
  nxt=base.iloc[i+1].copy();nxt[OBS]=obs.iloc[i+1]
  result.loc[i,'reward']=compute_mdp_v1_reward(nxt,{k:a.loc[i,'current_'+k] for k in ACTIONS},prev_action={k:a.loc[i,'previous_'+k] for k in ACTIONS},prev_prev_action={k:a.loc[i,'previous_previous_'+k] for k in ACTIONS})[0]
 result['rl_valid_transition']=eligible.astype(int);result['mdp_v1_rollout_id']=(~eligible.shift(fill_value=False)).cumsum();result['done']=(~eligible.shift(-1,fill_value=False)).astype(int)
 a['row_id']=ids
 for k in KEYS[:-1]:a[k]=base[k]
 result=result.loc[eligible].reset_index(drop=True)
 numeric=OBS+ACTIONS+['next_'+k for k in OBS]+['target_next_indoor_temp_c','target_next_indoor_humidity_pct','target_next_indoor_co2_ppm','reward']
 result[numeric]=result[numeric].astype('float64')
 return result,a,event_audit
