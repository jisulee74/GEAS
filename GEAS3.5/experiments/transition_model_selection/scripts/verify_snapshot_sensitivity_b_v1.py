"""Independent boundary checks and supplemental descriptive audit."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_sensitivity_b_v1'
spec=importlib.util.spec_from_file_location('diag',Path(__file__).with_name('diagnose_snapshot_sensitivity_b_v1.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def write(n,x):(D/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
# Synthetic raw-only contract checks: seven points means elapsed30; episodes do not break runs.
def frame(n=9):
 x=pd.DataFrame({'reg_date':pd.date_range('2000-01-01',periods=n,freq='5min'),'series_id':'s','segment_id':'g','episode_id':['a']*4+['b']*(n-4)})
 for c in m.SIG['B']:x[c]=0.0
 return x
x=frame();tag,r=m.tag_runs(x,x,'B');assert r.rows.tolist()==[9] and r.duration_minutes.iloc[0]==40
x2=x.copy();x2.loc[4:,'reg_date']+=pd.Timedelta(minutes=5);_,r=m.tag_runs(x2,x2,'B');assert r.rows.tolist()==[4,5]
for c in ('series_id','segment_id'):
 y=x.copy();y.loc[4:,c]='other';_,r=m.tag_runs(y,y,'B');assert sorted(r.rows)==[4,5]
y=x.copy();y.loc[4,'in_co2']=np.nan;_,r=m.tag_runs(y,y,'A');assert sorted(r.rows)==[1,4,4]
y=x.copy();y.loc[4:,'in_temp']=1;_,ra=m.tag_runs(y,y,'A');_,rb=m.tag_runs(y,y,'B');assert ra.rows.tolist()==[9] and rb.rows.tolist()==[4,5]
x=frame(7);tag,r=m.tag_runs(x,x,'B');assert (tag.duration_minutes>=30).all() and not (tag.duration_minutes>=60).any()
# Model metrics/date/episode irrelevant to run labels; timestamp itself is only continuity.
y=x.copy();y['prediction_error']=np.arange(7)*1e9;y['episode_id']='different';y.reg_date+=pd.Timedelta(days=365);tag2,_=m.tag_runs(y,y,'B');np.testing.assert_array_equal(tag.duration_minutes,tag2.duration_minutes)
# Refine non-detected category: raw-absent rows are unknown, not evidence of non-staleness.
f=pd.read_parquet(D/'strawberry_train_row_tags.parquet');high=f.obs_indoor_co2_ppm.ge(1500);old=high&~f.in_A;weather=f.obs_outdoor_light.isna()|f.obs_outdoor_wind_speed.isna();rows=[]
for s in m.SIG:
 for c in m.CUTS:
  hit=f[f'{s}_{c}'];row={'signature':s,'cutoff_minutes':c,'A_excluded_high':int(old.sum()),'repeated_and_weather':int((old&hit&weather).sum()),'observed_raw_not_flagged_and_weather':int((old&~hit&f.raw_present&weather).sum()),'raw_absent_weather_repeat_unknown':int((old&~f.raw_present&weather).sum()),'other':int((old&~weather).sum())};assert sum(row[k] for k in ['repeated_and_weather','observed_raw_not_flagged_and_weather','raw_absent_weather_repeat_unknown','other'])==2235;rows.append(row)
pd.DataFrame(rows).to_csv(D/'high_train_attribution_with_unknown.csv',index=False)
# Attach sensor range/context to all detected runs, without selecting by any model outcome.
run=pd.read_csv(D/'raw_runs.csv.gz',parse_dates=['start','end']);summ=[]
for (crop,split),rr in run.groupby(['crop','split']):
 f=pd.read_parquet(D/f'{crop}_{split}_row_tags.parquet')
 for _,r in rr.loc[rr.valid_signature&rr.duration_minutes.ge(30)].iterrows():
  g=f.loc[f[r.signature+'_run_id'].eq(r.run_id)]
  summ.append({**r.to_dict(),'representative_high_rows':int(g.obs_indoor_co2_ppm.ge(1500).sum()),'raw_co2_1_min':g.raw_co2_1.min(),'raw_co2_1_max':g.raw_co2_1.max(),'sensor_difference_min':g.sensor_abs_difference.min(),'sensor_difference_max':g.sensor_abs_difference.max(),'B_eligible_rows':int(g.eligible.sum())})
pd.DataFrame(summ).to_csv(D/'detected_runs_context.csv',index=False)
# How much fixed squared error lies on the diagnostic mask? Same windows, no new inference.
a=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1/strawberry';mask=pd.read_parquet(D/'diagnostic_window_masks.parquet');parts=[]
for model in ('lightgbm','persistence'):
 p=np.load(a/model/'rollout/predictions.npz')
 for h in (3,6,12):
  ix=np.flatnonzero(p['max_horizon']>=h);sse=((p['predictions'][ix,:h,2]-p['truth'][ix,:h,2])**2).sum(axis=1)
  for rule,g in mask.loc[mask.horizon_minutes.eq(h*5)].groupby('rule',sort=False):
   np.testing.assert_array_equal(g.start_row_id,p['start_row_id'][ix]);hit=g.masked.to_numpy(bool);parts.append({'model':model,'rule':rule,'horizon_minutes':h*5,'masked_windows':int(hit.sum()),'all_windows':len(hit),'masked_window_fraction':float(hit.mean()),'masked_sse_share':float(sse[hit].sum()/sse.sum())})
pd.DataFrame(parts).to_csv(D/'masked_error_concentration.csv',index=False)
write('boundary_regression_checks.json',{'elapsed_duration_cutoff':True,'zero_is_valid_repeat':True,'gap_breaks':True,'series_segment_breaks':True,'episode_does_not_break':True,'NaN_cannot_repeat':True,'signature_A_B_different':True,'date_episode_prediction_error_invariant':True,'raw_absent_classified_unknown':True,'same_window_ids_for_SSE':True})
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(summ).loc[lambda x:x.crop.eq('strawberry'),['split','signature','start','end','duration_minutes','rows','raw_co2_1_min','sensor_difference_min','B_eligible_rows']].to_string(index=False))
