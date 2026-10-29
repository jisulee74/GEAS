"""Build quarantined B candidates; approve only with verified command provenance.
Usage: --output NEW_DIR [--ledger-dir DIR]; never reads Test or fits models/support.
Each ledger must use the normalized, independently audited cycle schema.
"""
from pathlib import Path
import argparse, dataclasses, hashlib, json, sys
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.mdp_v1 import prepare_mdp_v1_frame
from geas35.rl.vent_command_dataset_b_v1 import *

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--ledger-dir',type=Path);args=ap.parse_args()
 args.output.mkdir(parents=True,exist_ok=False)
 manifest={'version':VERSION,'status':'running','test_accessed':False,'model_fit':False,'support_fit':False,'inputs':[],'schema':{'observations':OBS,'actions':ACTIONS,'units':'physical observations; normalized commands [0,1]; binary HVAC'},'contract':{'observation_vent':'previous verified transmitted command','transition_action':'current verified transmitted command','reward_history':'previous and previous_previous commands','next_carry':'current command','missing':'exclude; never zero/current/interpolate','history':'QC before valid-transition filtering; preserve solar, aliases, degree and condensation definitions','timezone':'source-local naive timestamps; unresolved timezone provenance blocks production adapter'},'rows':[]}
 summaries=[];source_stats=[];reasons=[];scalers=[]
 for crop in ('melon','cucumber','strawberry'):
  frames={}
  for split in ('train','validation'):
   p=ROOT/f'offline_dataset_preparation/datasets/03_quality_controlled/{crop}/{split}.parquet';raw=pd.read_parquet(p);before=sha(p)
   base=prepare_mdp_v1_frame(raw);commands=None
   if args.ledger_dir:
    lp=args.ledger_dir/f'{crop}_{split}.parquet'
    if lp.exists():commands=pd.read_parquet(lp);manifest['inputs'].append({'path':str(lp),'sha256':sha(lp),'kind':'normalized_verified_ledger'})
   b,a=assemble_b_frame(base,commands,split=split)
   b2,a2=assemble_b_frame(base,commands,split=split)
   assert fingerprint(b)==fingerprint(b2) and fingerprint(a)==fingerprint(a2)
   assert before==sha(p)
   # Preserve raw sensor provenance without converting it into command proof.
   for c in ('idx','iot_data_idx','source_segment_id','source_file_path','source_interval_minutes','crop_cycle_id'):
    if c in raw:a['sensor_'+c]=raw[c].reset_index(drop=True)
   out=args.output/crop;out.mkdir(exist_ok=True)
   b.to_parquet(out/f'{split}_physical.parquet',index=False);a.to_parquet(out/f'{split}_provenance.parquet',index=False)
   b[['row_id',*KEYS]].to_parquet(out/f'{split}_eligible_registry.parquet',index=False)
   old=ROOT/f'offline_dataset_preparation/datasets/5_rl_dataset/{crop}/{split}.parquet'
   import pyarrow.parquet as pq
   old_count=pq.ParquetFile(old).metadata.num_rows if old.exists() else None
   stat={'crop':crop,'split':split,'qc_rows':len(base),'old_rl_rows':old_count,'eligible':len(b),'excluded':len(a)-len(b),'eligible_fraction':len(b)/len(a) if len(a) else 0,'action_history_boundary_or_unknown':int((~a.previous_previous_command_valid).sum()),'windows_stride1':window_counts(b),'deterministic_rebuild':True,'candidate_semantic_sha256':fingerprint(b),'provenance_semantic_sha256':fingerprint(a),'feature_change_A_to_B':None}
   if len(b):
    ab=a.loc[a.eligible,['vent_loss_A','vent_loss_B']];stat['feature_change_A_to_B']={'mean_A':float(ab.vent_loss_A.mean()),'mean_B':float(ab.vent_loss_B.mean()),'mean_abs_difference':float((ab.vent_loss_A-ab.vent_loss_B).abs().mean())}
   for source,g in a.groupby('source_table',dropna=False):source_stats.append({'crop':crop,'split':split,'source':source,'rows':len(g),'eligible':int(g.eligible.sum()),'eligible_fraction':float(g.eligible.mean()),'source_label_inferred_rows':int(g.source_inferred_from_legacy_cutoff.sum())})
   for reason,count in a.loc[~a.eligible,'reason'].str.split('|',regex=False).explode().value_counts().items():reasons.append({'crop':crop,'split':split,'reason':reason,'rows':int(count)})
   summaries.append(stat);frames[split]=b
   manifest['inputs'].append({'path':str(p.relative_to(ROOT)),'sha256':before,'kind':'frozen_QC','split':split});print(crop,split,len(base),len(b),'eligible',flush=True)
  if len(frames['train']):
   scaler=fit_b_scaler(frames['train']);payload=dataclasses.asdict(scaler);write_json(out/'observation_scaler.json',payload)
   frames['train'][['row_id',*KEYS]].to_parquet(out/'scaler_fit_registry.parquet',index=False)
   for split,b in frames.items():
    scaled=scaler.transform_frame(b)
    for c in scaler.columns:scaled['next_'+c]=(b['next_'+c]-scaler.means[c])/scaler.scales[c]
    assert np.isfinite(scaled[OBS+['next_'+c for c in OBS]].to_numpy(float)).all()
    scaled.to_parquet(out/f'{split}_scaled.parquet',index=False)
   scalers.append({'crop':crop,'status':'fitted_train_only','fit_rows':len(frames['train']),'scaler_sha256':sha(out/'observation_scaler.json')})
  else:scalers.append({'crop':crop,'status':'blocked_no_verified_train_rows','fit_rows':0})
 manifest['rows']=summaries;manifest['scalers']=scalers
 # Non-empty data and synthetic tests alone cannot approve an unaudited adapter.
 manifest['status']='blocked_command_provenance' if any(x['eligible']==0 for x in summaries) else 'pending_independent_source_adapter_and_full_gate_review'
 manifest['phase1_authorized']=False
 manifest['outputs']=[{'path':str(p.relative_to(args.output)),'sha256':sha(p)} for p in sorted(args.output.rglob('*')) if p.is_file()]
 for p in [Path(__file__),ROOT/'src/geas35/rl/vent_command_dataset_b_v1.py',ROOT/'tests/test_vent_command_dataset_b_v1.py']:manifest.setdefault('code',[]).append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in summaries]).to_csv(args.output/'row_summary.csv',index=False)
 pd.DataFrame(source_stats).to_csv(args.output/'source_summary.csv',index=False);pd.DataFrame(reasons).to_csv(args.output/'exclusion_reasons.csv',index=False)
 write_json(args.output/'manifest.json',manifest)
if __name__=='__main__':main()
