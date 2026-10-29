from pathlib import Path
import sys,json,hashlib,dataclasses
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.mdp_v1 import prepare_mdp_v1_frame
from geas35.rl.vent_command_memory_b_v2 import *
from geas35.rl.causal_contract_env_v2 import CausalContractStateUpdater
D=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v2'
class Identity:
 def transform_frame(self,f):return f.copy()
 def inverse_transform_frame(self,f):return f.copy()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,default=str,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 out=D/'phase0_verified';out.mkdir(exist_ok=False);summary=[];inputs=[]
 for crop in ('melon','cucumber','strawberry'):
  dest=out/crop;dest.mkdir();frames={}
  for split in ('train','validation'):
   qp=ROOT/f'offline_dataset_preparation/datasets/03_quality_controlled/{crop}/{split}.parquet';cp=D/'raw'/f'{crop}_{split}_optimal_monitoring_copy.parquet';ep=D/'raw'/f'{crop}_{split}_send_ds_control_api_history.parquet'
   for p in (qp,cp,ep):inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
   base=prepare_mdp_v1_frame(pd.read_parquet(qp));cycles=pd.read_parquet(cp);events=pd.read_parquet(ep)
   b,a,ea=build(base,cycles,events,split);print(crop,split,'built',len(b),flush=True)
   b.to_parquet(dest/f'{split}_build_checkpoint.parquet',index=False)
   a.to_parquet(dest/f'{split}_audit_checkpoint.parquet',index=False)
   # Deterministic repeated event restoration (no second reward computation).
   aa,ee=restore(base,cycles,events);assert fingerprint(ea)==fingerprint(ee)
   for col in aa:assert fingerprint(a[[col]])==fingerprint(aa[[col]])
   keep=[x for x in OBS if not x.startswith('obs_prev_') and x!='obs_derived_vent_loss_proxy']
   if len(b):np.testing.assert_allclose(b[keep].to_numpy(dtype=float),base.set_index('reg_date').loc[b.reg_date,keep].to_numpy(dtype=float),rtol=1e-6,atol=1e-8)
   for c in ('idx','iot_data_idx','source_segment_id','source_file_path'):
    if c in base:a['sensor_'+c]=base[c].to_numpy()
   b.to_parquet(dest/f'{split}_physical.parquet',index=False);a.to_parquet(dest/f'{split}_provenance.parquet',index=False);ea.to_parquet(dest/f'{split}_events.parquet',index=False)
   b[['row_id',*KEYS,'target_timestamp']].to_parquet(dest/f'{split}_transition_registry.parquet',index=False)
   windows=[]
   for rid,g in b.groupby('mdp_v1_rollout_id'):
    for h in (3,6,12):
     for j in range(max(0,len(g)-h+1)):windows.append({'rollout_id':int(rid),'horizon':h,'start_row_id':g.iloc[j].row_id,'end_target_timestamp':g.iloc[j+h-1].target_timestamp})
   pd.DataFrame(windows,columns=['rollout_id','horizon','start_row_id','end_target_timestamp']).to_parquet(dest/f'{split}_windows.parquet',index=False)
   # Actual teacher-forced state advancement: up to 24 distributed eligible transitions.
   errors=[];index_by_time={t:i for i,t in enumerate(base.reg_date)}
   for pos in np.unique(np.linspace(0,len(b)-1,min(24,len(b))).astype(int)) if len(b) else []:
    row=b.iloc[pos];i=index_by_time[row.reg_date];current=base.iloc[i].copy();current[OBS]=row[OBS];nxt=base.iloc[i+1].copy()
    for col in OBS:nxt[col]=row['next_'+col]
    hist=base.iloc[max(0,i-12):i].copy();u=CausalContractStateUpdater(Identity(),hist);u.reset(current)
    pred={k:float(nxt[k]) for k in ('obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm')}
    got=u.advance(current_scaled=current,recorded_next_scaled=nxt,prediction_physical=pred,action={k:float(row[k]) for k in ACTIONS})
    for col in OBS:
     if not np.isclose(float(got[col]),float(nxt[col]),rtol=1e-6,atol=1e-8):errors.append({'timestamp':str(row.reg_date),'feature':col,'expected':float(nxt[col]),'actual':float(got[col])})
   write(dest/f'{split}_teacher_forcing.json',{'transitions':min(24,len(b)),'errors':errors,'scope':'distributed sample; full dataset feature preservation checked separately'})
   reasons=a.loc[~a.eligible,'reason'].str.split('|',regex=False).explode().value_counts().to_dict()
   change=None
   if len(b):
    ab=a.loc[a.eligible];change={'mean_A':float(ab.vent_loss_A.mean()),'mean_B':float(ab.vent_loss_B.mean()),'mean_abs_change':float((ab.vent_loss_A-ab.vent_loss_B).abs().mean()),'max_abs_change':float((ab.vent_loss_A-ab.vent_loss_B).abs().max())}
   stat={'crop':crop,'split':split,'qc_rows':len(base),'cycle_rows':len(cycles),'event_rows':len(events),'eligible':len(b),'excluded':int((~a.eligible).sum()),'exclusion_reasons':reasons,'cycle_present_rows':int(a.cycle_present.sum()),'linked_cycles':int(a.linked.sum()),'link_rate_all_sensor_rows':float(a.linked.mean()),'link_rate_cycle_present':float(a.loc[a.cycle_present,'linked'].mean()) if a.cycle_present.any() else None,'intervening_override':int(a.intervening_override.sum()),'unlinked_event_intervals':int(a.unlinked_event_interval.sum()),'ambiguous_cycles':int(a.ambiguous.sum()),'partial_retained_rows':int(a.retained_channels.gt(0).sum()),'retained_channel_updates':int(a.retained_channels.sum()),'unknown_excluded':int((a.command_memory_unknown&~a.eligible).sum()),'windows':window_counts(b),'teacher_forcing_transitions':min(24,len(b)),'teacher_forcing_errors':len(errors),'feature_distribution_change':change,'deterministic_restore':True}
   summary.append(stat);frames[split]=b;print(json.dumps(stat,default=str),flush=True)
  train=frames['train']
  if len(train):
   scaler=fit_b_scaler(train);write(dest/'observation_scaler.json',dataclasses.asdict(scaler));train[['row_id','split',*KEYS]].to_parquet(dest/'scaler_fit_registry.parquet',index=False)
   for split,b in frames.items():
    scaled=scaler.transform_frame(b)
    for col in scaler.columns:scaled['next_'+col]=(b['next_'+col]-scaler.means[col])/scaler.scales[col]
    assert np.isfinite(scaled[OBS+['next_'+k for k in OBS]+['reward']].to_numpy(float)).all();scaled.to_parquet(dest/f'{split}_scaled.parquet',index=False)
  for s in summary:
   if s['crop']==crop:s['scaler_created']=len(train)>0
 for p in inputs:assert sha(ROOT/p['path'])==p['sha256']
 write(out/'manifest.json',{'version':'vent_command_carry_b_v2','phase0':'BLOCKED' if any(s['eligible']==0 or s['teacher_forcing_errors'] for s in summary) else 'PENDING_FULL_GATE_REVIEW','phase1_authorized':False,'inputs':inputs,'summary':summary,'test_accessed':False,'model_support_ppo_hpo_fit':False,'outputs':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in out.rglob('*') if p.is_file()]})
if __name__=='__main__':main()
