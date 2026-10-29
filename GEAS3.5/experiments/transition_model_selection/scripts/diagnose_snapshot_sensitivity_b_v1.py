"""Read-only exact raw snapshot sensitivity. No model load, fit, or dataset rebuild."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_sensitivity_b_v1'
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4'
M=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1'
AUD=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_source_audit_b_v1'
SIG={'A':['in_co2','in_co2_2'],'B':['in_co2','in_co2_2','in_temp','in_hum','out_light']}
CUTS=(30,60,120,180);TARGETS=['obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm']
inputs={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def track(p):inputs[str(p.relative_to(ROOT))]=sha(p);return p
def read(p,**kw):return pd.read_parquet(track(p),**kw)
def write(n,x):(D/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
def csv(n,f):f.to_csv(D/n,index=False)
def tag_runs(q,raw,sig):
 cols=SIG[sig];r=raw.sort_values(['series_id','segment_id','reg_date']).copy().reset_index(drop=True)
 vals=r[cols].apply(pd.to_numeric,errors='coerce');valid=np.isfinite(vals.to_numpy(float)).all(axis=1)
 same=r.reg_date.diff().eq(pd.Timedelta(minutes=5))&r.series_id.eq(r.series_id.shift())&r.segment_id.eq(r.segment_id.shift())
 same &= vals.eq(vals.shift()).all(axis=1)&valid&pd.Series(valid).shift(fill_value=False)
 r['run_id']=(~same).cumsum();r['valid_signature']=valid
 g=r.groupby('run_id',sort=False)
 runs=g.agg(start=('reg_date','min'),end=('reg_date','max'),rows=('reg_date','size'),series_id=('series_id','first'),segment_id=('segment_id','first'),valid_signature=('valid_signature','all'))
 runs['duration_minutes']=(runs.end-runs.start).dt.total_seconds()/60
 r['duration_minutes']=r.run_id.map(runs.duration_minutes).where(valid,0)
 keys=['series_id','segment_id','reg_date'];lookup=r.set_index(keys)
 qi=pd.MultiIndex.from_frame(q[keys]);out=lookup[['run_id','duration_minutes','valid_signature']].reindex(qi).reset_index(drop=True)
 out['raw_present']=pd.Series(qi.isin(lookup.index));out['duration_minutes']=out.duration_minutes.fillna(0)
 return out,runs.reset_index()
def main():
 D.mkdir(exist_ok=False)
 protocol={'signatures':SIG,'duration_cutoffs_minutes':CUTS,'definition':'finite numeric exact equality; consecutive original raw rows exactly 5 minutes apart; same crop/series/segment and split; never use resampled/ffilled rows or episode/model/date to define runs','duration':'last raw timestamp minus first; inclusive >= cutoff; 30min requires >=7 raw rows','mask':'retrospective complete-run mask; one-step excludes if current or next target state flagged; multi-step excludes entire window if any initial/intermediate/endpoint recorded state flagged; same retained population for both models','normalization':'original full eligible Validation target std fixed; original all-target score .10 one + .10 15m + .25 30m + .45 60m + .10 mean horizon drift; CO2-only analog separately labeled','disagreement':'absolute sensor1-sensor2 difference bins 0/50/100/250/500/1000 ppm; descriptive only, no new QC threshold','controls':'all crops Train/Validation; no Test, fit, inference, official edits or approval; no cutoff selection from metrics; physical normal/stale ground truth unavailable','coverage_categories':'within already A-excluded: repeated first; nonrepeated with wind/light nonfinite second; other third; overlapping margins also reported'}
 write('protocol.json',protocol);write('protocol_lock.json',{'sha256':sha(D/'protocol.json')})
 # Pin official decisions and existing weights/scalers; no model object is loaded.
 for p in [P/'final_decision_ko.md',M/'final_decision_ko.md',M/'final_gate.json',M/'protocol.json',AUD/'final_decision_ko.md']:
  track(p)
 for crop in ('melon','cucumber','strawberry'):track(P/crop/'observation_scaler.json')
 for model in ('lightgbm','persistence'):track(M/f'strawberry/{model}/fit/model.pkl')
 summaries=[];allruns=[];coverage=[];disagree=[];disdist=[];short=[];lookup={};detections=[]
 for crop in ('melon','cucumber','strawberry'):
  for split in ('train','validation'):
   q=read(P/crop/f'{split}_prepared_qc.parquet');a=read(P/crop/f'{split}_provenance.parquet');assert q.reg_date.equals(a.reg_date)
   keys=['series_id','reg_date'];parts=[]
   for series,g in q.groupby('series_id'):
    path=ROOT/f'offline_dataset_preparation/datasets/00_raw/{crop}/{series}.parquet'
    r=read(path,filters=[('reg_date','in',g.reg_date.tolist())],columns=['idx','reg_date','series_id',*SIG['B']]);assert r.reg_date.isin(g.reg_date).all();assert not r.duplicated(keys).any()
    r=r.merge(g[keys+['segment_id']],on=keys,validate='one_to_one');parts.append(r)
   raw=pd.concat(parts,ignore_index=True);raw[SIG['B']]=raw[SIG['B']].apply(pd.to_numeric,errors='coerce')
   f=q[['crop','reg_date','series_id','segment_id','episode_id','obs_indoor_co2_ppm','obs_outdoor_light','obs_outdoor_wind_speed']].copy();f['split']=split;f['eligible']=a.eligible.to_numpy()
   z=raw.set_index(keys).reindex(pd.MultiIndex.from_frame(q[keys]));f['raw_idx']=z.idx.to_numpy();f['raw_co2_1']=z.in_co2.to_numpy();f['raw_co2_2']=z.in_co2_2.to_numpy();f['sensor_abs_difference']=(f.raw_co2_1-f.raw_co2_2).abs()
   if crop=='strawberry' and split=='train':
    membership=read(ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_diagnosis_b_v1/A_B_train_membership.parquet');assert q.reg_date.equals(membership.reg_date);f['in_A']=membership.in_A_train.to_numpy();f['B_reason']=a.reason.to_numpy()
   for sig in SIG:
    tags,runs=tag_runs(q,raw,sig);f[sig+'_duration_minutes']=tags.duration_minutes;f[sig+'_run_id']=tags.run_id;f['raw_present']=tags.raw_present
    runs['crop']=crop;runs['split']=split;runs['signature']=sig;allruns.append(runs)
    good=runs.loc[runs.valid_signature&runs.rows.ge(2)]
    for label,mask in [('lt30',good.duration_minutes.lt(30)),('30_to60',good.duration_minutes.ge(30)&good.duration_minutes.lt(60)),('60_to120',good.duration_minutes.ge(60)&good.duration_minutes.lt(120)),('120_to180',good.duration_minutes.ge(120)&good.duration_minutes.lt(180)),('ge180',good.duration_minutes.ge(180))]:
     gg=good.loc[mask];short.append({'crop':crop,'split':split,'signature':sig,'duration_bin':label,'runs':len(gg),'rows':int(gg.rows.sum())})
    for cut in CUTS:
     mask=tags.raw_present&tags.valid_signature.fillna(False)&tags.duration_minutes.ge(cut);f[f'{sig}_{cut}']=mask
     summaries.append({'crop':crop,'split':split,'signature':sig,'cutoff_minutes':cut,'runs':int((runs.valid_signature&runs.duration_minutes.ge(cut)).sum()),'flagged_rows':int(mask.sum()),'QC_rows':len(q),'raw_rows':len(raw),'fraction_QC':float(mask.mean()),'fraction_raw':float(mask.sum()/len(raw)),'flagged_high_QC_co2':int((mask&q.obs_indoor_co2_ppm.ge(1500)).sum()),'flagged_high_raw_sensor1':int((mask&f.raw_co2_1.ge(1500)).sum()),'flagged_B_eligible':int((mask&f.eligible).sum())})
     for threshold in (50,100,250,500,1000):
      large=f.sensor_abs_difference.ge(threshold);disagree.append({'crop':crop,'split':split,'signature':sig,'cutoff_minutes':cut,'difference_ge_ppm':threshold,'finite_sensor_pair_rows':int(f.sensor_abs_difference.notna().sum()),'large_difference_rows':int(large.sum()),'overlap_flagged_rows':int((large&mask).sum())})
     if crop=='strawberry' and split=='train':
      high=f.obs_indoor_co2_ppm.ge(1500);old=high&~f.in_A;weather=f.obs_outdoor_light.isna()|f.obs_outdoor_wind_speed.isna()
      coverage.append({'signature':sig,'cutoff_minutes':cut,'QC_high':int(high.sum()),'A_excluded_high':int(old.sum()),'A_excluded_repeated':int((old&mask).sum()),'A_excluded_not_repeated_weather_QC':int((old&~mask&weather).sum()),'A_excluded_other':int((old&~mask&~weather).sum()),'A_excluded_repeated_and_weather_QC':int((old&mask&weather).sum()),'B_added_exclusion_repeated':int((high&f.in_A&~f.eligible&mask).sum()),'B_remaining_repeated':int((high&f.eligible&mask).sum()),'raw_absent_high':int((high&~tags.raw_present).sum())})
   bins=pd.cut(f.sensor_abs_difference,[-np.inf,50,100,250,500,1000,np.inf],right=False).astype(str)
   for group,n in bins.value_counts().items():disdist.append({'crop':crop,'split':split,'abs_difference_bin':group,'rows':int(n)})
   f.to_parquet(D/f'{crop}_{split}_row_tags.parquet',index=False);lookup[(crop,split)]=f
   print('TAGGED',crop,split,flush=True)
 csv('duration_sensitivity.csv',pd.DataFrame(summaries));csv('raw_runs.csv.gz',pd.concat(allruns,ignore_index=True));csv('short_repeat_reference.csv',pd.DataFrame(short));csv('high_train_attribution.csv',pd.DataFrame(coverage));csv('sensor_disagreement_overlap.csv',pd.DataFrame(disagree));csv('sensor_disagreement_distribution.csv',pd.DataFrame(disdist))
 high=lookup['strawberry','train'];csv('train_high_2267_tags.csv',high.loc[high.obs_indoor_co2_ppm.ge(1500)])
 # Dates enter only this post-detection audit table, never candidate definitions.
 for split,start,end in [('validation','2025-11-10 18:45','2025-11-11 09:35'),('train','2024-10-01 00:45','2024-10-02 10:15'),('train','2024-10-20 03:15','2024-10-21 10:20')]:
  f=lookup['strawberry',split];part=f.loc[f.reg_date.between(start,end)]
  for sig in SIG:
   for cut in CUTS:detections.append({'split':split,'audit_start':start,'audit_end':end,'signature':sig,'cutoff_minutes':cut,'rows':len(part),'detected_rows':int(part[f'{sig}_{cut}'].sum()),'eligible_rows':int(part.eligible.sum()),'eligible_detected':int((part.eligible&part[f'{sig}_{cut}']).sum())})
 csv('known_intervals_detection.csv',pd.DataFrame(detections))
 # Existing frozen predictions only. No call to predict / fit.
 b=read(P/'strawberry/validation_physical.parquet');reg=read(P/'strawberry/validation_windows.parquet');f=lookup['strawberry','validation'];keys=['series_id','segment_id','reg_date'];fi=f.set_index(keys);curidx=pd.MultiIndex.from_frame(b[keys]);nxt=b[['series_id','segment_id','target_timestamp']].rename(columns={'target_timestamp':'reg_date'});nxtidx=pd.MultiIndex.from_frame(nxt)
 assert curidx.isin(fi.index).all() and nxtidx.isin(fi.index).all()
 arrays={};one={};meta={}
 for name in ('lightgbm','persistence'):
  arrays[name]=np.load(track(M/f'strawberry/{name}/rollout/predictions.npz'));one[name]=read(M/f'strawberry/{name}/fit/validation_predictions.parquet')[TARGETS].to_numpy(float);meta[name]=json.loads(track(M/f'strawberry/{name}/fit/completed.json').read_text())
 m=arrays['lightgbm'];p=arrays['persistence'];np.testing.assert_array_equal(m['start_row_id'],p['start_row_id']);np.testing.assert_allclose(m['truth'],p['truth'],equal_nan=True)
 start=pd.Series(np.arange(len(b)),index=b.row_id).loc[m['start_row_id']].to_numpy();scales=np.array([meta['lightgbm']['scales'][t] for t in TARGETS]);truth_one=b[[t.replace('obs_','target_next_',1) for t in TARGETS]].to_numpy(float)
 masks=[];results=[];scores=[]
 for rule in ['unmasked']+[f'{s}_{c}' for s in SIG for c in CUTS]:
  current=np.zeros(len(b),bool) if rule=='unmasked' else fi.loc[curidx,rule].to_numpy(bool);nextflag=np.zeros(len(b),bool) if rule=='unmasked' else fi.loc[nxtidx,rule].to_numpy(bool)
  keep_one=~(current|nextflag);keeps={}
  for h in (3,6,12):
   ix=np.flatnonzero(m['max_horizon']>=h);positions=start[ix,None]+np.arange(h)[None,:]
   assert len(ix)==reg.horizon.eq(h).sum()
   hit=current[start[ix]]|nextflag[positions].any(axis=1);keeps[h]=(ix,~hit)
   masks.append(pd.DataFrame({'rule':rule,'horizon_minutes':h*5,'start_row_id':m['start_row_id'][ix],'masked':hit}))
  for name in arrays:
   err=one[name][keep_one]-truth_one[keep_one];rmse=np.sqrt(np.mean(err**2,axis=0));comp={'one':float((rmse/scales).mean())};co={'one':float(rmse[2]/scales[2])};dr=[];dc=[]
   results.append({'rule':rule,'model':name,'horizon_minutes':5,'kept':int(keep_one.sum()),'total':len(b),'co2_rmse':rmse[2],'all_target_nrmse':comp['one']})
   for h,(ix,keep) in keeps.items():
    e=arrays[name]['predictions'][ix,:h][keep]-m['truth'][ix,:h][keep];rm=np.sqrt(np.mean(e**2,axis=(0,1)));t=np.arange(h)-np.arange(h).mean();drift=(np.abs(e/scales)*t[None,:,None]).sum(axis=1)/(t*t).sum();dr.append(float(drift.mean()));dc.append(float(drift[:,2].mean()));comp[h]=float((rm/scales).mean());co[h]=float(rm[2]/scales[2])
    results.append({'rule':rule,'model':name,'horizon_minutes':h*5,'kept':int(keep.sum()),'total':len(ix),'co2_rmse':rm[2],'all_target_nrmse':comp[h],'co2_endpoint_rmse':float(np.sqrt(np.mean(e[:,-1,2]**2))),'all_target_drift':dr[-1],'co2_drift':dc[-1]})
   score=.1*comp['one']+.1*comp[3]+.25*comp[6]+.45*comp[12]+.1*np.mean(dr)
   cs=.1*co['one']+.1*co[3]+.25*co[6]+.45*co[12]+.1*np.mean(dc)
   scores.append({'rule':rule,'model':name,'all_target_weighted_score':score,'CO2_only_weighted_analog_not_official':cs})
   if rule=='unmasked':
    saved=json.loads(track(M/f'strawberry/{name}/rollout/completed.json').read_text());np.testing.assert_allclose(score,saved['score'],rtol=1e-6,atol=1e-8)
    for row in results[-4:]:
     if row['horizon_minutes']>5:np.testing.assert_allclose(row['co2_rmse'],saved['rollout'][str(row['horizon_minutes'])+'min']['targets']['obs_indoor_co2_ppm']['rmse'],rtol=1e-6,atol=1e-8)
 rr=pd.DataFrame(results);baseline=rr.loc[rr.model.eq('persistence'),['rule','horizon_minutes','co2_rmse']].rename(columns={'co2_rmse':'persistence_co2_rmse'});rr=rr.merge(baseline,on=['rule','horizon_minutes'],validate='many_to_one');rr['CO2_skill_vs_persistence']=1-rr.co2_rmse/rr.persistence_co2_rmse
 csv('metric_sensitivity.csv',rr);csv('weighted_score_sensitivity.csv',pd.DataFrame(scores));pd.concat(masks,ignore_index=True).to_parquet(D/'diagnostic_window_masks.parquet',index=False)
 # Mask nesting, B subset of A, nonfinite and source-boundary invariants.
 for f in lookup.values():
  for c in CUTS:assert (~f[f'B_{c}']|f[f'A_{c}']).all();assert (~f[f'A_{c}']|f.raw_present).all()
  for s in SIG:
   for lo,hi in zip(CUTS[:-1],CUTS[1:]):assert (~f[f'{s}_{hi}']|f[f'{s}_{lo}']).all()
 assert np.isfinite(rr[['co2_rmse','all_target_nrmse']]).all().all()
 write('input_manifest.json',inputs)
 for path,h in inputs.items():assert sha(ROOT/path)==h
 write('verification.json',{'all_input_hashes_unchanged':True,'official_unmasked_scores_and_rollout_rmse_reproduced':True,'signature_B_subset_of_A':True,'duration_masks_nested':True,'flagged_rows_have_original_raw_rows':True,'no_Test_fit_inference_regeneration':True,'official_Phase0':'PASS unchanged','official_Phase1_2':'BLOCKED unchanged','physical_staleness':'C / undetermined unchanged'})
 print('COMPLETE',D,flush=True)
if __name__=='__main__':main()
