"""Frozen Phase 0 v3 build; Train/Validation only; no model or support fit."""
from pathlib import Path
import sys,json,hashlib,dataclasses,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.mdp_v1 import prepare_mdp_v1_frame
from geas35.rl.vent_command_memory_b_v3 import *
D=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4';OLD=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str,allow_nan=False)+'\n')
def process(crop):
 dest=D/crop;dest.mkdir(exist_ok=False);frames={};stats=[];inputs=[]
 for split in ('train','validation'):
  qp=ROOT/f'offline_dataset_preparation/datasets/03_quality_controlled/{crop}/{split}.parquet';qc=pd.read_parquet(qp);base=prepare_mdp_v1_frame(qc);base.reg_date=pd.to_datetime(base.reg_date);base.to_parquet(dest/f'{split}_prepared_qc.parquet',index=False)
  cp=OLD/'raw'/f'{crop}_{split}_optimal_monitoring_copy.parquet';ep=OLD/'raw'/f'{crop}_{split}_send_ds_control_api_history.parquet';cy=pd.read_parquet(cp);ev=pd.read_parquet(ep)
  for p in (qp,cp,ep):inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
  source='api' if crop=='cucumber' else 'legacy';snap=None
  if source=='legacy':
   parts=[]
   for series,g in qc.groupby('series_id'):
    p=ROOT/f'offline_dataset_preparation/datasets/00_raw/{crop}/{series}.parquet'
    # Predicate pushdown: only frozen Train/Validation sensor timestamps; no Test rows.
    times=pd.to_datetime(g.reg_date).tolist();raw=pd.read_parquet(p,filters=[('reg_date','in',times)],columns=['idx','iot_data_idx','reg_date','series_id']+LEGACY_COLS)
    raw=raw.loc[raw.reg_date.isin(times)&raw.series_id.eq(series)&raw.iot_data_idx.astype(str).eq('97')];parts.append(raw);inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'read_filter':'exact '+split+' sensor timestamp membership, series '+str(series)})
   snap=pd.concat(parts,ignore_index=True);snap.to_parquet(dest/f'{split}_legacy_raw_snapshots.parquet',index=False)
  b,a,events=build(base,cy,ev,split,source=source,snapshots=snap);print(crop,split,'eligible',len(b),flush=True)
  b.to_parquet(dest/f'{split}_physical.parquet',index=False);a.to_parquet(dest/f'{split}_provenance.parquet',index=False);events.to_parquet(dest/f'{split}_events.parquet',index=False)
  aa,ee=restore(base,cy,ev,source=source,snapshots=snap);assert fingerprint(events)==fingerprint(ee)
  for col in aa:assert fingerprint(aa[[col]])==fingerprint(a[[col]])
  if source=='api':
   off,_=restore(base,cy,ev,collapse=False);offmask=admission(base,off)[-1];collapse_before=int(offmask.sum());collapse_recovered=int((a.eligible&~offmask).sum())
   off['eligible']=offmask;off.to_parquet(dest/f'{split}_collapse_off_provenance.parquet',index=False)
  else:collapse_before=0;collapse_recovered=0
  obs,same,unknown,finite,hist,eligible=admission(base,a)
  # All rows: construction invariant from changing only current command, never previous.
  mutated=a.copy();mutated['current_vent_pct']=1-pd.to_numeric(mutated.current_vent_pct);mobs=admission(base,mutated)[0]
  np.testing.assert_allclose(obs.to_numpy(float),mobs.to_numpy(float),equal_nan=True,rtol=1e-6,atol=1e-8)
  changed=a.copy();changed['previous_vent_pct']=.37;changedobs=admission(base,changed)[0]
  factor=(base.obs_indoor_temp_c-base.obs_outdoor_temp_c).clip(lower=0)*(1-(base.obs_outdoor_light.clip(lower=0)/1000).clip(0,1))
  np.testing.assert_allclose(changedobs.obs_derived_vent_loss_proxy.to_numpy(float),(.37*factor).to_numpy(float),equal_nan=True,rtol=1e-6,atol=1e-8)
  keep=[k for k in OBS if not k.startswith('obs_prev_') and k!='obs_derived_vent_loss_proxy'];np.testing.assert_allclose(obs[keep].to_numpy(float),base[keep].to_numpy(float),equal_nan=True,rtol=1e-6,atol=1e-8)
  if len(b):
   assert (b.target_timestamp-b.reg_date).eq(pd.Timedelta(minutes=5)).all()
   assert np.isfinite(b[OBS+ACTIONS+['next_'+k for k in OBS]+['reward']].to_numpy(float)).all()
   truth=base.set_index('reg_date').loc[b.target_timestamp]
   np.testing.assert_allclose(b[['target_next_indoor_temp_c','target_next_indoor_humidity_pct','target_next_indoor_co2_ppm']].to_numpy(float),truth[['obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm']].to_numpy(float))
  b[['row_id','source_confidence',*KEYS,'target_timestamp','mdp_v1_rollout_id']].to_parquet(dest/f'{split}_registry.parquet',index=False)
  windows=[]
  for rid,g in b.groupby('mdp_v1_rollout_id'):
   ids=g.row_id.to_numpy();ts=g.target_timestamp.to_numpy()
   for h in (3,6,12):
    windows.extend({'rollout_id':int(rid),'horizon':h,'start_row_id':ids[j],'end_target_timestamp':ts[j+h-1]} for j in range(max(0,len(g)-h+1)))
  pd.DataFrame(windows,columns=['rollout_id','horizon','start_row_id','end_target_timestamp']).to_parquet(dest/f'{split}_windows.parquet',index=False)
  ab=a.loc[a.eligible];diff=ab.vent_loss_B-ab.vent_loss_A
  stat={'crop':crop,'split':split,'source':source,'source_confidence':'api_transmitted_command' if source=='api' else 'legacy_command_snapshot_assumption','qc_rows':len(base),'eligible':len(b),'excluded':len(a)-len(b),'exclusion_reasons':a.loc[~a.eligible,'reason'].str.split('|',regex=False).explode().value_counts().to_dict(),'collapse_off_eligible':collapse_before,'collapse_recovered':collapse_recovered,'collapsed_raw_events':int(a.collapsed_events.sum()),'legacy_recovered':len(b) if source=='legacy' else 0,'override':int(a.intervening_override.sum()),'conflict':int(a.conflicting_command.sum()),'ambiguous':int(a.ambiguous.sum()),'unknown':int(a.command_memory_unknown.sum()),'boundary_legacy_rows':int(a.legacy_boundary_excluded.sum()) if source=='legacy' else None,'missing_invalid_legacy_snapshot_rows':int(a.legacy_snapshot_unknown.sum()) if source=='legacy' else None,'windows':window_counts(b),'eligible_episodes':int(b.episode_id.nunique()),'A_mean':float(ab.vent_loss_A.mean()) if len(b) else None,'B_mean':float(ab.vent_loss_B.mean()) if len(b) else None,'mean_abs_change':float(diff.abs().mean()) if len(b) else None,'max_abs_change':float(diff.abs().max()) if len(b) else None,'deterministic_restore':True,'all_row_causal_invariant':True,'all_row_preserved_features':True}
  old=pd.read_parquet(OLD/'phase0_verified'/crop/f'{split}_provenance.parquet',columns=['row_id','eligible']);oldids=set(old.loc[old.eligible,'row_id']);stat['v2_eligible']=len(oldids);stat['recovered_vs_v2']=int((~b.row_id.isin(oldids)).sum());stat['lost_vs_v2']=len(oldids-set(b.row_id))
  stats.append(stat);frames[split]=b;write(dest/f'{split}_summary.json',stat);print(json.dumps(stat,default=str),flush=True)
 if len(frames['train']):
  train=frames['train'];sc=fit_b_scaler(train);write(dest/'observation_scaler.json',dataclasses.asdict(sc));train[['row_id','split',*KEYS]].to_parquet(dest/'scaler_fit_registry.parquet',index=False)
  for col in sc.columns:assert np.isclose(sc.means[col],train[col].mean(),rtol=1e-12,atol=1e-12)
  for split,b in frames.items():
   x=sc.transform_frame(b)
   for col in sc.columns:x['next_'+col]=(b['next_'+col]-sc.means[col])/sc.scales[col]
   assert np.isfinite(x[OBS+['next_'+k for k in OBS]].to_numpy(float)).all();x.to_parquet(dest/f'{split}_scaled.parquet',index=False)
 for entry in inputs:assert sha(ROOT/entry['path'])==entry['sha256']
 write(dest/'build_manifest.json',{'crop':crop,'summary':stats,'inputs':inputs,'scaler_created':len(frames['train'])>0,'test_accessed':False,'fit':'observation scaler only','outputs':[{'path':str(p.relative_to(D)),'sha256':sha(p)} for p in sorted(dest.glob('*')) if p.is_file()]})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--crop',choices=['melon','cucumber','strawberry'],required=True);process(ap.parse_args().crop)
