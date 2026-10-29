"""Trace Validation timestamps only through source preparation stages; no Test reads."""
from pathlib import Path
import pandas as pd,numpy as np,json,pyarrow.parquet as pq
r=Path(__file__).resolve().parents[3];ds=r/'offline_dataset_preparation/datasets';out=r/'experiments/rl_policy_training/artifacts/step19_5_light_contract_v1';rows=[]
for crop in ['melon','cucumber']:
 val=pd.read_parquet(ds/'5_rl_dataset'/crop/'validation.parquet');dates=pd.to_datetime(val.reg_date);lo,hi=dates.min(),dates.max()
 for stage in ['00_raw','01_resampled','02_split','03_quality_controlled']:
  paths=list((ds/stage/crop).glob('*.parquet')) if stage in ['00_raw','01_resampled'] else [ds/stage/crop/'validation.parquet']
  stage_frames=[]
  for p in sorted(paths):
   names=pq.read_schema(p).names;cols=[c for c in ['reg_date','series_id','out_light','out_light_sum','out_light_sum_estimation_method','is_resampled_row'] if c in names]
   frame=pd.read_parquet(p,columns=cols,filters=[('reg_date','>=',lo),('reg_date','<=',hi)])
   if not len(frame):continue
   # Only Validation timestamps; no Test partition is read.
   frame=frame[frame.reg_date.isin(dates)]
   for col in ['out_light','out_light_sum']:
    if col not in frame:continue
    x=pd.to_numeric(frame[col],errors='coerce');rows.append(dict(crop=crop,stage=stage,path=str(p),column=col,n=len(x),missing=int(x.isna().sum()),zero=int(x.eq(0).sum()),minimum=x.min(),maximum=x.max()))
   stage_frames.append(frame)
  pd.concat(stage_frames,ignore_index=True).to_csv(out/f'{crop}_{stage}_validation_light.csv',index=False)
pd.DataFrame(rows).to_csv(out/'source_flow_statistics.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
