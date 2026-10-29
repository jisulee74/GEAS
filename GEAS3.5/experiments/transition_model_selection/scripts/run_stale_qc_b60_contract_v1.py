from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
from stale_qc_b60_recovery_contract_v1 import Recovery
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/stale_qc_b60_contract_v1';S=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_shadow_b_v1';P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(D/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
inputs={}
def read(p):inputs[str(p.relative_to(ROOT))]=sha(p);return pd.read_parquet(p)
D.mkdir(exist_ok=False)
write('contract_manifest.json',{'status':'project-decided candidate; rebuild gate pending','signature':'B: raw CO2 sensor1/sensor2/T1/RH1/light exact numeric repeat','cutoff_minutes':60,'justification':'user project decision: conservative operational/QC criterion without external ground truth; not a physical sensor-failure truth or model-score-selected threshold','official_candidate_mode':'causal-forward','retrospective':'audit only; never official admission','supervisor_metadata_only':True,'replacement_values':False,'recovery_candidates':[2,3,6],'recovery_definition':'latched on first B60 stale; needs N consecutive finite, continuous raw observations whose signature changes from immediately previous raw row and differs from latched frozen signature; first change counts as1; repeated unchanged row resets progress; gap/source/crop/series/segment/raw-invalid resets progress but does not clear latch; split starts with no unobserved latch assumed','recovery_delay':'N qualifying rows span5*(N-1)min from first qualifying change, in addition to delay from stale onset; elapsed measured from known data only','metrics_not_used':True,'no_official_dataset_model_changes':True})
write('protocol_lock.json',{'sha256':sha(D/'contract_manifest.json'),'recovery_code_sha256':sha(Path(__file__).with_name('stale_qc_b60_recovery_contract_v1.py'))})
# Official gate decisions and model/scaler hashes are tracked without loading models.
for manifest in ('input_manifest.json','metric_input_manifest.json'):
 for p,h in json.loads((S/manifest).read_text()).items():assert sha(ROOT/p)==h;inputs[p]=h
summary=[];events=[];impacts=[]
for crop in ('melon','cucumber','strawberry'):
 for split in ('train','validation'):
  f=read(S/f'{crop}_{split}_row_sidecar.parquet');reg=read(P/crop/f'{split}_registry.parquet');windows=read(P/crop/f'{split}_windows.parquet')
  out=f[['crop','split','series_id','segment_id','episode_id','reg_date','raw_idx','raw_source','B_eligible','obs_indoor_co2_ppm','B_60_causal_forward']].copy()
  for n in (2,3,6):
   machine=Recovery(n);values=[];onset=None;firstchange=None;lastrelease=None;relapse60=0
   for row in f.itertuples(index=False):
    vals=tuple(float(getattr(row,'raw_'+c)) for c in ('in_co2','in_co2_2','in_temp','in_hum','out_light'))
    valid=bool(row.B_input_valid);identity=(row.crop,row.series_id,row.segment_id,row.raw_source);base=bool(row.B_60_causal_forward)
    was=machine.latched;res=machine.update(row.reg_date,identity,vals,valid,base);values.append(res)
    if res['entered']:
     onset=row.reg_date;firstchange=None
     if lastrelease is not None and (row.reg_date-lastrelease).total_seconds()<=3600:relapse60+=1
    if was and res['recovery_count']==1:firstchange=row.reg_date
    if res['released']:
     events.append({'crop':crop,'split':split,'steps':n,'stale_onset':onset,'qualifying_sequence_start':firstchange,'release':row.reg_date,'minutes_from_onset':(row.reg_date-onset).total_seconds()/60,'minutes_from_first_change':(row.reg_date-firstchange).total_seconds()/60,'source':row.raw_source})
     lastrelease=row.reg_date;firstchange=None
   z=pd.DataFrame(values);out[f'latched_{n}']=z.supervisor_blocked;out[f'input_valid_{n}']=z.input_valid;out[f'recovery_progress_{n}']=z.recovery_count
   summary.append({'crop':crop,'split':split,'steps':n,'stale_entries':int(z.entered.sum()),'releases':int(z.released.sum()),'latched_rows':int(z.supervisor_blocked.sum()),'additional_rows_beyond_B60':int((z.supervisor_blocked&~f.B_60_causal_forward).sum()),'latched_B_eligible':int((z.supervisor_blocked&f.B_eligible).sum()),'latched_high_CO2':int((z.supervisor_blocked&f.obs_indoor_co2_ppm.ge(1500)).sum()),'stale_reentries_within60min_of_release':relapse60,'unresolved_at_split_end':bool(machine.latched)})
  keys=['series_id','segment_id','reg_date'];fi=out.set_index(keys);ci=pd.MultiIndex.from_frame(reg[keys]);ni=pd.MultiIndex.from_frame(reg[['series_id','segment_id','target_timestamp']].rename(columns={'target_timestamp':'reg_date'}));positions=pd.Series(np.arange(len(reg)),index=reg.row_id)
  for label in ['B_60_causal_forward','latched_2','latched_3','latched_6']:
   current=fi.loc[ci,label].to_numpy(bool);nextflag=fi.loc[ni,label].to_numpy(bool);hit=current|nextflag
   rr={'crop':crop,'split':split,'mask':label,'original_rows':len(reg),'excluded_transitions':int(hit.sum()),'expected_transitions':int((~hit).sum()),'eligible_row_overlap':int(current.sum())}
   for h in (3,6,12):
    w=windows.loc[windows.horizon.eq(h)];st=positions.loc[w.start_row_id].to_numpy();pos=st[:,None]+np.arange(h)[None,:];assert np.array_equal(reg.iloc[st+h-1].target_timestamp.to_numpy(),w.end_target_timestamp.to_numpy());wh=current[st]|nextflag[pos].any(axis=1);rr[f'excluded_windows_{h}']=int(wh.sum());rr[f'expected_windows_{h}']=int((~wh).sum())
   impacts.append(rr)
  # Longer confirmation never releases earlier in this definition on the same stream.
  assert (~out.latched_2|out.latched_3).all();assert (~out.latched_3|out.latched_6).all()
  out.to_parquet(D/f'{crop}_{split}_recovery_sidecar.parquet',index=False)
pd.DataFrame(summary).to_csv(D/'recovery_summary.csv',index=False);pd.DataFrame(events).to_csv(D/'recovery_events.csv',index=False);pd.DataFrame(impacts).to_csv(D/'admission_impact.csv',index=False)
base=pd.read_csv(S/'impact_summary.csv');base=base.loc[base.signature.eq('B')&base.cutoff_minutes.eq(60)&base['mode'].eq('causal_forward')];base.to_csv(D/'B60_final_impact.csv',index=False)
for p,h in inputs.items():assert sha(ROOT/p)==h
write('input_manifest.json',inputs)
print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(impacts).loc[lambda x:x['mask'].eq('B_60_causal_forward')].to_string(index=False))
