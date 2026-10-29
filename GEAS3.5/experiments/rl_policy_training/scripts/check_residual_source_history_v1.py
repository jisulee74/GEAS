"""Compare source-QC derived history with frozen rows, Train/Validation only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,gzip
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.mdp_v1 import prepare_mdp_v1_frame
from geas35.experiments.rl.step19_5_diagnostics import ART,write_json,sha256
D=ROOT/ART/'step19_5_residual_ood_v1';details=[json.loads(l) for l in gzip.open(D/'cucumber_feature_trace.jsonl.gz','rt')];results=[]
for split in ('validation','train'):
 source=ROOT/f'offline_dataset_preparation/datasets/03_quality_controlled/cucumber/{split}.parquet';raw=pd.read_parquet(source);f=prepare_mdp_v1_frame(raw);c='obs_derived_condensation_margin_pred_10m';errors=[];missing=0
 # Cucumber has a single series in these frozen splits. Reject ambiguous timestamps.
 assert not f.reg_date.duplicated().any()
 lookup=f.set_index('reg_date')
 for d in details:
  if d['split']!=split:continue
  t=pd.Timestamp(d['timestamp'])
  if t not in lookup.index:missing+=1;continue
  errors.append(abs(float(lookup.loc[t,c])-d['features'][c]['recorded']))
 result={'crop':'cucumber','split':split,'source_sha256':sha256(source),'source_rows':len(f),'compared_rows':len(errors),'missing_timestamps':missing,'max_abs_error':max(errors),'above_1e_6':sum(e>1e-6 for e in errors)};results.append(result);print(result,flush=True)
write_json(D/'source_history_evidence.json',results)
