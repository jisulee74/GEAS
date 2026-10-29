"""Offline-only admission rebuild from immutable B v4 physical rows, B60/R6 supervisor."""
from pathlib import Path
import sys,json,hashlib,dataclasses,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.vent_command_memory_b_v3 import OBS,ACTIONS,KEYS,fit_b_scaler,fingerprint
OLD=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4';REC=ROOT/'experiments/transition_model_selection/artifacts/stale_qc_b60_contract_v1';D=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def assemble(b,f):
 keys=['crop','series_id','segment_id','reg_date'];index=f.set_index(keys);ci=pd.MultiIndex.from_frame(b[keys]);ni=pd.MultiIndex.from_frame(b[['crop','series_id','segment_id','target_timestamp']].rename(columns={'target_timestamp':'reg_date'}));assert ci.isin(index.index).all() and ni.isin(index.index).all()
 raw=index.loc[ci,'B_60_causal_forward'].to_numpy(bool)|index.loc[ni,'B_60_causal_forward'].to_numpy(bool)
 block=index.loc[ci,'latched_6'].to_numpy(bool)|index.loc[ni,'latched_6'].to_numpy(bool);assert (~raw|block).all()
 out=b.loc[~block].copy().reset_index(drop=True)
 same=(out.reg_date-out.reg_date.shift()).eq(pd.Timedelta(minutes=5))&out.mdp_v1_rollout_id.eq(out.mdp_v1_rollout_id.shift())
 out['mdp_v1_rollout_id']=(~same).cumsum().astype(int);out['done']=(~same.shift(-1,fill_value=False)).astype(int)
 w=[]
 for rid,g in out.groupby('mdp_v1_rollout_id',sort=False):
  for h in (3,6,12):
   for j in range(max(0,len(g)-h+1)):w.append({'rollout_id':int(rid),'horizon':h,'start_row_id':g.row_id.iloc[j],'end_target_timestamp':g.target_timestamp.iloc[j+h-1]})
 return out,pd.DataFrame(w,columns=['rollout_id','horizon','start_row_id','end_target_timestamp']),raw,block
if __name__=='__main__':
 D.mkdir(exist_ok=False)
 gate=json.loads((OLD/'gate_protocol.json').read_text());gate.update(version=D.name,authority='user project decision: offline and deployment gates separated',stale_contract='B60 causal-forward + 6 qualifying changing observations recovery latch; operational assumption, not sensor failure truth',online_deployment='BLOCKED independently; does not block offline research',model_feature_change=False)
 write(D/'gate_protocol.json',gate);write(D/'gate_protocol_lock.json',{'sha256':sha(D/'gate_protocol.json')})
 contract=json.loads((REC/'contract_manifest.json').read_text());contract.update(version=D.name,offline_contract='official project research QC choice',recovery_steps=6,recovery_justification='conservative project QC choice, not physically optimal',offline_research_gate='pending new Phase0 verification',online_deployment_gate='BLOCKED; unverified actuator-safe handling',history_preservation='existing B physical/sensor-history features and their causal initial history retained; admission only; stale flag not observation; raw suspect history not imputed/recomputed differently',supervisor_latch='persists across physical gaps; qualifying count resets; raw repeated-run history resets; split initial prior latch unknown')
 write(D/'runtime_contract.json',contract);schema=json.loads((OLD/'schema.json').read_text());schema['version']=D.name;write(D/'schema.json',schema)
 inputs={};stats=[]
 for crop in ('melon','cucumber','strawberry'):
  dest=D/crop;dest.mkdir();frames={}
  for split in ('train','validation'):
   for name in (f'{split}_physical.parquet',f'{split}_prepared_qc.parquet',f'{split}_provenance.parquet',f'{split}_windows.parquet'):
    inputs[str((OLD/crop/name).relative_to(ROOT))]=sha(OLD/crop/name)
   rp=REC/f'{crop}_{split}_recovery_sidecar.parquet';inputs[str(rp.relative_to(ROOT))]=sha(rp)
   b=pd.read_parquet(OLD/crop/f'{split}_physical.parquet');f=pd.read_parquet(rp);out,w,raw,block=assemble(b,f);out2,w2,_,_=assemble(b,f)
   assert fingerprint(out)==fingerprint(out2) and fingerprint(w)==fingerprint(w2)
   # Preserve all physical values/actions/targets/rewards; only rollout boundary metadata changes.
   cols=[c for c in b if c not in ('done','mdp_v1_rollout_id')];pd.testing.assert_frame_equal(out[cols].reset_index(drop=True),b.loc[~block,cols].reset_index(drop=True))
   assert np.isfinite(out[OBS+ACTIONS+['next_'+c for c in OBS]+['reward']].to_numpy(float)).all()
   oldw=pd.read_parquet(OLD/crop/f'{split}_windows.parquet');assert set(zip(w.start_row_id,w.horizon,w.end_target_timestamp))<=set(zip(oldw.start_row_id,oldw.horizon,oldw.end_target_timestamp))
   out.to_parquet(dest/f'{split}_physical.parquet',index=False);w.to_parquet(dest/f'{split}_windows.parquet',index=False);out[['row_id','source_confidence',*KEYS,'target_timestamp','mdp_v1_rollout_id']].to_parquet(dest/f'{split}_registry.parquet',index=False)
   a=pd.read_parquet(OLD/crop/f'{split}_provenance.parquet');a['parent_eligible']=a.eligible;a['eligible']=a.row_id.isin(out.row_id);hit=a.parent_eligible&~a.eligible;stale_ids=set(b.loc[raw,'row_id']);a.loc[hit,'reason']=a.loc[hit,'row_id'].map(lambda x:'stale_B60_current_or_target' if x in stale_ids else 'recovery_pending_6_current_or_target');a.to_parquet(dest/f'{split}_provenance.parquet',index=False)
   for name in (f'{split}_prepared_qc.parquet',f'{split}_events.parquet',f'{split}_legacy_raw_snapshots.parquet'):
    src=OLD/crop/name
    if src.exists():inputs[str(src.relative_to(ROOT))]=sha(src);shutil.copyfile(src,dest/name)
   f.to_parquet(dest/f'{split}_quality_sidecar.parquet',index=False)
   row={'crop':crop,'split':split,'old_eligible':len(b),'stale_excluded':int(raw.sum()),'recovery_pending_only_excluded':int((block&~raw).sum()),'final_eligible':len(out),'windows':{str(h):int(w.horizon.eq(h).sum()) for h in (3,6,12)},'source_counts':out.source_confidence.value_counts().to_dict(),'high_CO2':int(out.obs_indoor_co2_ppm.ge(1500).sum()),'high_CO2_high_vent':int((out.obs_indoor_co2_ppm.ge(1500)&out.vent_pct.ge(.75)).sum()),'deterministic_rebuild':True,'physical_features_unchanged':True}
   stats.append(row);write(dest/f'{split}_summary.json',row);frames[split]=out
  scaler=fit_b_scaler(frames['train']);again=fit_b_scaler(frames['train']);assert dataclasses.asdict(scaler)==dataclasses.asdict(again)
  write(dest/'observation_scaler.json',dataclasses.asdict(scaler));frames['train'][['row_id','split',*KEYS]].to_parquet(dest/'scaler_fit_registry.parquet',index=False)
  assert frames['train'].split.eq('train').all();assert set(frames['train'].row_id).isdisjoint(frames['validation'].row_id)
  for c in scaler.columns:np.testing.assert_allclose(scaler.means[c],frames['train'][c].mean(),rtol=1e-12,atol=1e-12)
  for split,b in frames.items():
   x=scaler.transform_frame(b)
   for c in scaler.columns:x['next_'+c]=(b['next_'+c]-scaler.means[c])/scaler.scales[c]
   assert np.isfinite(x[OBS+['next_'+c for c in OBS]].to_numpy(float)).all();np.testing.assert_allclose(scaler.inverse_transform_frame(x)[OBS].to_numpy(float),b[OBS].to_numpy(float),rtol=1e-6,atol=1e-8);x.to_parquet(dest/f'{split}_scaled.parquet',index=False)
  print('BUILT',crop,flush=True)
 for p,h in inputs.items():assert sha(ROOT/p)==h
 write(D/'input_manifest.json',inputs);write(D/'build_summary.json',stats)
