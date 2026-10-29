"""Read-only admission ledger and row-local metadata verification. No fitting."""
from pathlib import Path
import sys,json,hashlib,importlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'))
from geas35.rl.past_only_qc_hlv15_v1 import VARS
O=R/'experiments/rl_policy_training/artifacts/past_only_qc_hlv15_b60_r6_v1';B=O.parent/'vent_command_carry_b60_r6_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def main():
 domain=R/'offline_dataset_preparation/configs/qc/smartfarm_korea_domain_ranges.csv'
 wr('supplemental_audit_binding.json',dict(code_sha256=sha(Path(__file__)),rule_path=str(domain.relative_to(R)),rule_sha256=sha(domain),scope='same six Train/Validation inputs, output admission ledgers; no Test'))
 quality=importlib.import_module('geas35.preprocessing.3_missing_outliers_handling')
 summary=[];metadata=[]
 for crop in ('melon','cucumber','strawberry'):
  for split in ('train','validation'):
   d=O/crop;raw=pd.read_parquet(R/f'offline_dataset_preparation/datasets/02_split/{crop}/{split}.parquet')
   oldq=pd.read_parquet(B/crop/f'{split}_prepared_qc.parquet')
   local=quality.apply_default_rule_based_outlier_flags(raw[list(VARS)],csv_path=domain)
   for v in VARS:
    mismatch=int((local[v+'_rule_outlier_flag'].to_numpy()!=oldq[v+'_rule_outlier_flag'].to_numpy()).sum())
    missing=int((pd.to_numeric(raw[v],errors='coerce').isna().to_numpy()!=oldq[v+'_missing_flag'].astype(bool).to_numpy()).sum())
    metadata.append(dict(crop=crop,split=split,variable=v,rule_mismatch=mismatch,missing_mismatch=missing))
   b=pd.read_parquet(B/crop/f'{split}_physical.parquet');ow=pd.read_parquet(B/crop/f'{split}_windows.parquet');nw=pd.read_parquet(d/f'{split}_windows.parquet');p=pd.read_parquet(d/f'{split}_provenance.parquet')
   index={rid:i for i,rid in enumerate(p.row_id)};oldpos={rid:i for i,rid in enumerate(b.row_id)}
   newkeys=set(zip(nw.start_row_id,nw.horizon,nw.end_target_timestamp));parts=[]
   aligned=p.set_index('row_id').loc[b.row_id]
   flags=dict(direct_invalid=aligned.direct_source_invalid.to_numpy(bool),propagated_invalid=aligned.propagated_invalid.to_numpy(bool),target_invalid=aligned.hlv_target_invalid.to_numpy(bool),stale_or_recovery=(aligned.stale_current_or_target|aligned.recovery_pending_current_or_target).to_numpy(bool))
   for h,g in ow.groupby('horizon'):
    part=g[['start_row_id','horizon','end_target_timestamp']].copy();starts=g.start_row_id.map(oldpos).to_numpy(int)
    part['retained']=[key in newkeys for key in zip(g.start_row_id,g.horizon,g.end_target_timestamp)]
    for name,flag in flags.items():
     cs=np.r_[0,np.cumsum(flag)];part[name]=(cs[starts+int(h)]-cs[starts])>0
    parts.append(part)
   side=pd.concat(parts,ignore_index=True);side.to_parquet(d/f'{split}_old_window_admission.parquet',index=False)
   for h,g in side.groupby('horizon'):
    lost=g.loc[~g.retained];summary.append(dict(crop=crop,split=split,horizon=int(h),old=len(g),retained=int(g.retained.sum()),removed=len(lost),new_total=int(nw.horizon.eq(h).sum()),new_windows=int(nw.horizon.eq(h).sum())-int(g.retained.sum()),direct_invalid=int(lost.direct_invalid.sum()),propagated_invalid=int(lost.propagated_invalid.sum()),target_invalid=int(lost.target_invalid.sum()),stale_or_recovery=int(lost.stale_or_recovery.sum())))
 wr('row_local_metadata_audit.json',metadata);wr('window_admission_summary.json',summary)
 assert not any(r['rule_mismatch'] or r['missing_mismatch'] for r in metadata),metadata
 print('row-local metadata matches all six inputs; window reason ledger complete')
if __name__=='__main__':main()
