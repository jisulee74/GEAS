"""HLV15 Phase0 only. Explicit input allowlist; fit delayed until integrity gate."""
from pathlib import Path
import sys,json,hashlib,platform,dataclasses,importlib.util,os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.past_only_qc_hlv15_v1 import source_replay,derived_replay,DEPENDENT,VARS
from geas35.rl.vent_command_memory_b_v3 import OBS,ACTIONS,KEYS,admission,fingerprint,fit_b_scaler
from geas35.rl.mdp_v1 import compute_mdp_v1_reward
B=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1'
O=B.parent/'past_only_qc_hlv15_b60_r6_v1'
LOCK=B.parent/'past_only_qc_hlv15_contract_lock_v1'
EXPECTED='635d4f35b2969fb4a7746651fc16b0519ab63fcb54a842c0919e6573544d7164'
CROPS=('melon','cucumber','strawberry');SPLITS=('train','validation')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')
def windows(b):
 rec=[]
 for rid,g in b.groupby('mdp_v1_rollout_id',sort=False):
  for h in (3,6,12):
   for i in range(max(0,len(g)-h+1)):rec.append(dict(rollout_id=int(rid),horizon=h,start_row_id=g.row_id.iloc[i],end_target_timestamp=g.target_timestamp.iloc[i+h-1]))
 return pd.DataFrame(rec,columns=['rollout_id','horizon','start_row_id','end_target_timestamp'])
def counts(w):return {str(h):int(w.horizon.eq(h).sum()) for h in (3,6,12)}
def assemble(base,a,f,split):
 obs,same,unknown,finite,hist,eligible=admission(base,a)
 keys=['crop','series_id','segment_id','reg_date']
 pd.testing.assert_frame_equal(base[keys].reset_index(drop=True),f[keys].reset_index(drop=True),check_dtype=False)
 stale=f.B_60_causal_forward.to_numpy(bool);pending=f.latched_6.to_numpy(bool)
 nxt=np.r_[pending[1:],True];blocked=pending|nxt
 eligible=eligible&~blocked
 result=base[KEYS].copy();result['row_id']=a.row_id;result['split']=split;result['source_confidence']=a.source_confidence
 result=pd.concat([result,obs,a[['current_'+k for k in ACTIONS]].rename(columns={'current_'+k:k for k in ACTIONS})],axis=1)
 for k in OBS:result['next_'+k]=obs[k].shift(-1)
 for name,col in [('temp_c','obs_indoor_temp_c'),('humidity_pct','obs_indoor_humidity_pct'),('co2_ppm','obs_indoor_co2_ppm')]:result['target_next_indoor_'+name]=base[col].shift(-1)
 result['target_timestamp']=base.reg_date.shift(-1);result['reward']=np.nan
 br=base.to_dict('records');ob=obs.to_dict('records');ar=a.to_dict('records')
 rewards=np.full(len(base),np.nan)
 for i in np.flatnonzero(eligible):
  rewards[i]=compute_mdp_v1_reward({**br[i+1],**ob[i+1]},
   {k:ar[i]['current_'+k] for k in ACTIONS},prev_action={k:ar[i]['previous_'+k] for k in ACTIONS},
   prev_prev_action={k:ar[i]['previous_previous_'+k] for k in ACTIONS})[0]
 result['reward']=rewards;result['rl_valid_transition']=eligible.astype(int)
 out=result.loc[eligible].copy().reset_index(drop=True)
 same_run=out.reg_date.diff().eq(pd.Timedelta(minutes=5))
 for k in KEYS[:-1]:same_run &=out[k].eq(out[k].shift())
 out['mdp_v1_rollout_id']=(~same_run).cumsum();out['done']=(~same_run.shift(-1,fill_value=False)).astype(int)
 numeric=OBS+ACTIONS+['next_'+k for k in OBS]+['reward']
 out[numeric]=out[numeric].astype(float)
 prov=a.copy();prov['historical_eligible']=prov.eligible;prov['eligible']=eligible
 prov['hlv_current_invalid']=~finite
 prov['hlv_target_invalid']=~finite.shift(-1,fill_value=False)
 prov['stale_current_or_target']=stale|np.r_[stale[1:],False]
 prov['recovery_pending_current_or_target']=blocked&~prov.stale_current_or_target
 reasons=[]
 for i,r in enumerate(ar):
  flags=[]
  if not r['linked'] or r['intervening_override'] or not hist.iloc[i] or unknown[i] or not same.iloc[i]:flags.append('other_existing_B_command_or_boundary_exclusion')
  if not finite.iloc[i]:flags.append('current_observation_invalid')
  if i+1==len(base) or not finite.iloc[i+1]:flags.append('target_observation_invalid')
  if prov.stale_current_or_target.iloc[i]:flags.append('stale_B60')
  elif blocked[i]:flags.append('recovery_pending_R6')
  reasons.append('|'.join(flags))
 prov['reason']=reasons
 return out,prov,obs

class Inputs:
 def __init__(self,paths):
  self.allowed={p.resolve() for p in paths};self.access=[]
  for p in self.allowed:
   assert 'test' not in p.name.lower() and p.is_file(),str(p)
  self.hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(self.allowed)}
 def read(self,p):
  assert p.resolve() in self.allowed,'Not allowlisted: '+str(p)
  self.access.append(str(p.relative_to(ROOT)));wr(O/'access_ledger.json',self.access)
  return pd.read_parquet(p) if p.suffix=='.parquet' else json.loads(p.read_text())
 def unchanged(self):return all(sha(ROOT/p)==h for p,h in self.hashes.items())

def main():
 assert sha(LOCK/'preregistration_lock_ko.md')==EXPECTED,'Lock hash mismatch'
 assert (LOCK/'protocol.sha256').read_text().split()[0]==EXPECTED
 O.mkdir(exist_ok=False)
 paths=[B/p for p in ('schema.json','runtime_contract.json','gate_protocol.json')]
 for crop in CROPS:
  for split in SPLITS:
   paths+=[B/crop/f'{split}_{s}.parquet' for s in ('prepared_qc','physical','provenance','windows','quality_sidecar','events')]
   paths+=[ROOT/f'offline_dataset_preparation/datasets/02_split/{crop}/{split}.parquet']
   paths+=[B.parent/'vent_command_carry_b_v2/raw'/f'{crop}_{split}_{table}.parquet' for table in ('optimal_monitoring_copy','send_ds_control_api_history')]
   if crop!='cucumber':paths+=[B/crop/f'{split}_legacy_raw_snapshots.parquet']
 inputs=Inputs(paths)
 codepaths=[Path(__file__),ROOT/'src/geas35/rl/past_only_qc_hlv15_v1.py',ROOT/'src/geas35/rl/vent_command_memory_b_v3.py',ROOT/'src/geas35/rl/mdp_v1.py',ROOT/'src/geas35/features/derived.py',ROOT/'src/geas35/rl/sensor_history_env_v4.py',ROOT/'src/geas35/rl/causal_contract_env_v2.py',ROOT/'src/geas35/rl/recorded_light_env_v1.py',ROOT/'experiments/rl_policy_training/scripts/verify_stale_qc_b60_r6_v1.py',ROOT/'tests/test_past_only_qc_hlv15_v1.py']
 import pyarrow,sklearn
 manifest=dict(version=O.name,lock_sha256=EXPECTED,code_hashes={str(p.relative_to(ROOT)):sha(p) for p in codepaths},inputs=inputs.hashes,
  versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pyarrow=pyarrow.__version__,sklearn=sklearn.__version__),
  input_allowlist=sorted(inputs.hashes),test_access='DENIED; exact six split paths only; no raw multi-split files read',
  source_binding='strict-past observed iot_data_idx + variable + source_segment_id + crop/series/segment/source_file; inserted rows never raw evidence; unknown identity invalid',
  metadata_causality='wind/light not AI QC variables; missing/rule flags are row-local existing bounds; pre-QC ffill flag never raw evidence',
  raw_supervisor='immutable raw B60/R6 sidecar; reconstruction never replaces its signature',
  historical='historical metric preserved / causal availability gate superseded by past-only QC rebuild',
  restrictions=['no Transition/model fit','no support','no Step14-19','no domain-v2/C/E','no Test','no HPO/PPO'],status='RUNNING')
 wr(O/'input_manifest.json',manifest)
 schema=inputs.read(B/'schema.json');assert schema['observation_columns']==OBS
 schema['version']=O.name;wr(O/'schema.json',schema)
 gate=inputs.read(B/'gate_protocol.json');gate['version']=O.name;gate['hlv_contract']=EXPECTED;wr(O/'gate_protocol.json',gate)
 contract=inputs.read(B/'runtime_contract.json');contract['version']=O.name;contract['hlv_contract']=EXPECTED;wr(O/'runtime_contract.json',contract)
 summaries=[];stats=[];checks=[];allframes={}
 for crop in CROPS:
  dest=O/crop;dest.mkdir()
  for split in SPLITS:
   print('REPLAY',crop,split,flush=True)
   base=inputs.read(B/crop/f'{split}_prepared_qc.parquet').reset_index(drop=True)
   raw=inputs.read(ROOT/f'offline_dataset_preparation/datasets/02_split/{crop}/{split}.parquet').reset_index(drop=True)
   keys=['crop','series_id','segment_id','reg_date'];pd.testing.assert_frame_equal(base[keys],raw[keys],check_dtype=False)
   old=inputs.read(B/crop/f'{split}_physical.parquet');oldw=inputs.read(B/crop/f'{split}_windows.parquet')
   a=inputs.read(B/crop/f'{split}_provenance.parquet');f=inputs.read(B/crop/f'{split}_quality_sidecar.parquet')
   rec=source_replay(raw,base);rec2=source_replay(raw,base);assert fingerprint(rec)==fingerprint(rec2)
   for n in sorted(set([1,min(50,len(raw)),len(raw)//2])):
    pre=source_replay(raw.iloc[:n],base.iloc[:n]);pd.testing.assert_frame_equal(pre,rec.iloc[:n*2].reset_index(drop=True))
    changed=raw.copy();changed['out_light']=pd.to_numeric(changed.out_light,errors='coerce').astype(float);changed['out_windsp']=pd.to_numeric(changed.out_windsp,errors='coerce').astype(float);changed.loc[n:,'out_light']=999999.;changed.loc[n:,'out_windsp']=999999.
    alt=source_replay(changed,base);pd.testing.assert_frame_equal(alt.iloc[:2*n],rec.iloc[:2*n])
   new=derived_replay(base,rec);again=derived_replay(base,rec2);assert fingerprint(new)==fingerprint(again)
   untouched=[c for c in OBS if c not in DEPENDENT];pd.testing.assert_frame_equal(new[untouched],base[untouched])
   obs=admission(new,a)[0];mut=a.copy()
   for c in ACTIONS:mut['current_'+c]=1-pd.to_numeric(mut['current_'+c])
   np.testing.assert_allclose(obs.to_numpy(float),admission(new,mut)[0].to_numpy(float),equal_nan=True,rtol=1e-6,atol=1e-8)
   out,prov,_=assemble(new,a,f,split);out2,prov2,_=assemble(again,a,f,split)
   assert fingerprint(out)==fingerprint(out2) and fingerprint(prov)==fingerprint(prov2)
   w=windows(out);assert np.isfinite(out[OBS+ACTIONS+['next_'+c for c in OBS]+['reward']].to_numpy(float)).all()
   assert out.target_timestamp.sub(out.reg_date).eq(pd.Timedelta(minutes=5)).all()
   for c in ACTIONS:np.testing.assert_allclose(out['next_obs_prev_'+c],out[c],rtol=1e-6,atol=1e-8)
   rec.to_parquet(dest/f'{split}_reconstruction.parquet',index=False)
   new.to_parquet(dest/f'{split}_prepared_qc.parquet',index=False);out.to_parquet(dest/f'{split}_physical.parquet',index=False)
   prov.to_parquet(dest/f'{split}_provenance.parquet',index=False);w.to_parquet(dest/f'{split}_windows.parquet',index=False)
   f.to_parquet(dest/f'{split}_quality_sidecar.parquet',index=False)
   for suffix in ('events','legacy_raw_snapshots'):
    p=B/crop/f'{split}_{suffix}.parquet'
    if p.resolve() in inputs.allowed:inputs.read(p).to_parquet(dest/p.name,index=False)
   out[['row_id','source_confidence',*KEYS,'target_timestamp','mdp_v1_rollout_id']].to_parquet(dest/f'{split}_registry.parquet',index=False)
   direct=rec.assign(invalid=rec.method.eq('INVALID')).groupby('position',sort=False).invalid.any().to_numpy()
   derived_cols=[c for c in DEPENDENT if c not in ('obs_outdoor_light','obs_outdoor_wind_speed','obs_derived_vent_loss_proxy')]
   prop=(~np.isfinite(new[derived_cols].to_numpy(float)).all(axis=1))&~direct
   prov['direct_source_invalid']=direct;prov['propagated_invalid']=prop;prov.to_parquet(dest/f'{split}_provenance.parquet',index=False)
   rawhit=rec.assign(hlv=rec.method.eq('HLV')).groupby('position',sort=False).hlv.any().to_numpy()
   oldids=set(old.row_id);keep=prov.row_id.isin(oldids).to_numpy();selected=prov.eligible.to_numpy()
   high=out.obs_indoor_co2_ppm.ge(1500)
   stat=dict(crop=crop,split=split,qc_rows=len(base),old_eligible=len(old),new_eligible=len(out),old_windows=counts(oldw),new_windows=counts(w),
    hlv_still_eligible=int((rawhit&selected).sum()),direct_INVALID_rows=int(direct.sum()),propagated_INVALID_rows=int(prop.sum()),
    old_eligible_direct_excluded=int((keep&direct&~selected).sum()),old_eligible_propagated_excluded=int((keep&prop&~selected).sum()),
    old_eligible_target_invalid=int((keep&prov.hlv_target_invalid.to_numpy()&~selected).sum()),
    old_eligible_lost=len(oldids-set(out.row_id)),newly_eligible=len(set(out.row_id)-oldids),
    stale_exclusion=int(prov.stale_current_or_target.sum()),recovery_exclusion=int(prov.recovery_pending_current_or_target.sum()),
    exclusion_reasons=prov.loc[~prov.eligible,'reason'].str.split('|',regex=False).explode().value_counts().to_dict(),
    high_CO2=int(high.sum()),high_CO2_previous_vent=int((high&out.obs_prev_vent_pct.ge(.75)).sum()),high_CO2_current_vent=int((high&out.vent_pct.ge(.75)).sum()),
    source_provenance=out.source_confidence.value_counts().to_dict())
   summaries.append(stat);wr(dest/f'{split}_summary.json',stat)
   for v,g in rec.groupby('variable'):
    ages=g.loc[g.method.eq('HLV'),'age_minutes']
    stats.append(dict(crop=crop,split=split,variable=v,raw_valid=int(g.method.eq('RAW').sum()),HLV=int(g.method.eq('HLV').sum()),INVALID=int(g.method.eq('INVALID').sum()),age_p50=float(ages.quantile(.5)) if len(ages) else None,age_p90=float(ages.quantile(.9)) if len(ages) else None,age_max=float(ages.max()) if len(ages) else None,exact15=int(ages.eq(15).sum()),identity_resets=int(g.identity_reset.sum()),invalid_reasons=g.loc[g.method.eq('INVALID'),'reason'].value_counts().to_dict()))
   rec.groupby(['crop','series_id','segment_id','source_identity','variable','method'],dropna=False).size().reset_index(name='rows').to_csv(dest/f'{split}_source_counts.csv',index=False)
   checks.append(dict(crop=crop,split=split,deterministic=True,prefix_suffix=True,future_endpoint_reads=0,independent_feature_mismatches=0,current_action_mismatches=0,next_carry_mismatches=0,stale_sidecar_unchanged=fingerprint(f)==fingerprint(pd.read_parquet(dest/f'{split}_quality_sidecar.parquet'))))
   allframes[crop,split]=out
   wr(O/'population_summary.json',summaries);wr(O/'reconstruction_summary.json',stats);wr(O/'integrity_checks.json',checks)
   print('ADMISSION',crop,split,len(old),'->',len(out),counts(w),flush=True)
 # No scaler before all causal checks and usable-population requirements.
 blockers=[]
 if any(not s['new_eligible'] or not all(s['new_windows'].values()) for s in summaries):blockers.append('usable_population_or_windows_missing')
 if not blockers:
  spec=importlib.util.spec_from_file_location('hlv_phase0_teacher',ROOT/'experiments/rl_policy_training/scripts/verify_stale_qc_b60_r6_v1.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);v.D=O
  for crop in CROPS:
   try:v.verify(crop)
   except Exception as exc:blockers.append(crop+':teacher_verification_exception:'+repr(exc));continue
   for split in SPLITS:
    summary=json.loads((O/crop/'verification'/f'{split}_summary.json').read_text())
    if not summary['pass']:blockers.append(crop+'/'+split+':teacher_forcing_mismatch')
 if not blockers:
  for crop in CROPS:
   train=allframes[crop,'train'];val=allframes[crop,'validation']
   assert train.split.eq('train').all() and set(train.row_id).isdisjoint(val.row_id)
   scaler=fit_b_scaler(train);again=fit_b_scaler(train);assert dataclasses.asdict(scaler)==dataclasses.asdict(again)
   wr(O/crop/'observation_scaler.json',dataclasses.asdict(scaler));train[['row_id','split',*KEYS]].to_parquet(O/crop/'scaler_fit_registry.parquet',index=False)
   for c in scaler.columns:np.testing.assert_allclose(scaler.means[c],train[c].mean(),rtol=1e-12,atol=1e-12)
   for split in SPLITS:
    b=allframes[crop,split];x=scaler.transform_frame(b)
    for c in scaler.columns:x['next_'+c]=(b['next_'+c]-scaler.means[c])/scaler.scales[c]
    np.testing.assert_allclose(scaler.inverse_transform_frame(x)[OBS],b[OBS],rtol=1e-6,atol=1e-8)
    assert np.isfinite(x[OBS+['next_'+c for c in OBS]].to_numpy(float)).all()
    x.to_parquet(O/crop/f'{split}_scaled.parquet',index=False)
 assert inputs.unchanged() and sha(LOCK/'preregistration_lock_ko.md')==EXPECTED
 wr(O/'gate.json',dict(status='BLOCKED' if blockers else 'PASS',blockers=blockers,scaler_created=not blockers,lock_unchanged=True,historical_inputs_unchanged=True,test_accessed=False,model_fit=False))
 wr(O/'output_hashes.json',{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='output_hashes.json'})
 print('FINAL',blockers or 'PASS',flush=True)
if __name__=='__main__':main()
