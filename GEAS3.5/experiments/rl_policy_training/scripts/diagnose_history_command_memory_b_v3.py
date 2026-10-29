from pathlib import Path
import pandas as pd,json,numpy as np
ROOT=Path(__file__).resolve().parents[3]
r=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v3'
b=pd.read_parquet(r/'strawberry/train_prepared_qc.parquet')
errs=json.loads((r/'strawberry/verification/train_teacher_errors.json').read_text())
rows=[];traces=[]
for ts in sorted({e['timestamp'] for e in errs}):
 t=pd.Timestamp(ts);n=t+pd.Timedelta(minutes=5);s=b.loc[b.reg_date.eq(t)].iloc[0];x=b.loc[(b.reg_date>n-pd.Timedelta(minutes=60))&(b.reg_date<=n)].copy();same=np.ones(len(x),bool)
 for k in ('crop','series_id','segment_id','episode_id'):same &= x[k].astype(str).eq(str(s[k])).to_numpy()
 low=x.obs_derived_vpd_kpa.lt(.5)
 rows.append(dict(timestamp=ts,next_timestamp=str(n),full_low=float(low.sum()*5),same_episode_low=float(low[same].sum()*5),full_rows=len(x),same_episode_rows=int(same.sum()),excluded_prior_episodes=x.loc[~same,'episode_id'].astype(str).unique().tolist(),current_episode=str(s.episode_id),recorded=float(b.loc[b.reg_date.eq(n),'obs_derived_low_vpd_minutes_1h'].iloc[0])))
 x['case_timestamp']=ts;x['same_episode']=same;traces.append(x[['case_timestamp','reg_date','crop','series_id','segment_id','episode_id','obs_derived_vpd_kpa','obs_derived_low_vpd_minutes_1h','same_episode']])
for row in rows:
 assert row['full_low']==row['recorded']
 observed=next(e['actual'] for e in errs if e['timestamp']==row['timestamp'] and e['feature']=='obs_derived_low_vpd_minutes_1h')
 assert row['same_episode_low']==observed
print(json.dumps(rows,indent=2))
(r/'strawberry/verification/history_boundary_diagnosis.json').write_text(json.dumps(rows,indent=2)+'\n')
pd.concat(traces).to_csv(r/'strawberry/verification/history_boundary_trace.csv.gz',index=False,compression='gzip')
