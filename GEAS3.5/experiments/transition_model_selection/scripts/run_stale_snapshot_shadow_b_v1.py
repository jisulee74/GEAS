"""Frozen Train/Validation shadow sidecars; never runs model or production pipeline."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
from stale_snapshot_detector_shadow_v1 import Detector,SIGNATURES,CUTOFFS
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_shadow_b_v1';P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4';M=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1'
inputs={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def track(p):inputs[str(p.relative_to(ROOT))]=sha(p);return p
def read(p,**kwargs):return pd.read_parquet(track(p),**kwargs)
def write(n,x):(D/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def csv(n,x):x.to_csv(D/n,index=False)
def main():
 D.mkdir(exist_ok=False)
 protocol={'version':'stale_snapshot.shadow.v1','signatures':SIGNATURES,'cutoffs':CUTOFFS,'stream_identity':'crop + series + segment + raw table data_silla_enc + raw source_file_path + source_segment_id','break':'exact5min mismatch, identity/source change, raw absence, nonfinite/unparseable signature or explicit raw invalid; duplicates fail audit','invalid_scope':'raw layer has no sensor health invalid flag; numeric finite validity only; no QC AI/model flags or new physical thresholds used','continuity':'day/episode boundary ignored; splits processed separately; missing pre-split history left-censored','online':'current and prior inputs only; elapsed since first identical raw; stale starts at elapsed>=cutoff; first changed valid value clears candidate immediately; raw invalid is unknown not healthy','offline_modes':['causal_forward','confirmed_run_retrospective'],'retrospective':'offline-only annotation; decision available_at=start+cutoff; never sent to online state; not authorized for official Validation','impact':'one-step affected if t or target t+5 flagged; full window affected if any state t0..th flagged','metrics':'stored strawberry LightGBM/Persistence predictions only; fixed full Validation scales and original all-target weights; diagnostic only','normality':'every flagged run remains undetermined physically; no positive/negative ground truth','official_cutoff':None,'official_policy':None,'scope':'no Test/fit/predict/HPO/features/dataset/scaler/support/official steps/official artifact edits'}
 write('protocol.json',protocol);write('protocol_lock.json',{'sha256':sha(D/'protocol.json'),'detector_sha256':sha(Path(__file__).with_name('stale_snapshot_detector_shadow_v1.py'))})
 for p in [P/'final_decision_ko.md',M/'final_decision_ko.md',M/'final_gate.json']:track(p)
 for crop in ('melon','cucumber','strawberry'):track(P/crop/'observation_scaler.json')
 for name in ('lightgbm','persistence'):track(M/f'strawberry/{name}/fit/model.pkl')
 summaries=[];allruns=[];allwindows=[];alltrans=[];problem=[];boundary=[]
 for crop in ('melon','cucumber','strawberry'):
  for split in ('train','validation'):
   q=read(P/crop/f'{split}_prepared_qc.parquet');a=read(P/crop/f'{split}_provenance.parquet');reg=read(P/crop/f'{split}_registry.parquet');windows=read(P/crop/f'{split}_windows.parquet');assert q.reg_date.equals(a.reg_date)
   parts=[]
   for series,g in q.groupby('series_id'):
    path=ROOT/f'offline_dataset_preparation/datasets/00_raw/{crop}/{series}.parquet'
    raw=read(path,filters=[('reg_date','in',g.reg_date.tolist())],columns=['idx','reg_date','series_id','source_segment_id','source_file_path',*SIGNATURES['B']]);assert not raw.duplicated(['series_id','reg_date']).any()
    parts.append(raw)
   raw=pd.concat(parts,ignore_index=True);raw['raw_source']='data_silla_enc|'+raw.source_file_path.astype(str)+'|'+raw.source_segment_id.astype(str)
   raw=raw.drop(columns=['source_segment_id','source_file_path']).rename(columns={'idx':'raw_idx',**{c:'raw_'+c for c in SIGNATURES['B']}})
   f=q[['crop','series_id','segment_id','episode_id','reg_date','obs_indoor_co2_ppm']].merge(raw,on=['series_id','reg_date'],how='left',validate='one_to_one');f['B_eligible']=a.eligible.to_numpy();f['split']=split;assert f.reg_date.equals(q.reg_date)
   # Raw absence is fed to detector explicitly, rather than skipped.
   for sig in SIGNATURES:
    detector=Detector(sig);trace=[]
    for row in f.itertuples(index=False):
     trace.append(detector.update(timestamp=row.reg_date.to_pydatetime(),crop=row.crop,series=row.series_id,segment=row.segment_id,source=row.raw_source if pd.notna(row.raw_source) else None,raw_present=pd.notna(row.raw_idx),raw_values={c:getattr(row,'raw_'+c) for c in SIGNATURES[sig]}))
    t=pd.DataFrame(trace);f[sig+'_run_id']=t.run_id;f[sig+'_run_start']=t.run_start;f[sig+'_elapsed_minutes']=t.elapsed_minutes;f[sig+'_input_valid']=t.input_valid;f[sig+'_reason']=t.reason
    r=f.loc[t.input_valid].groupby(sig+'_run_id',sort=False).agg(start=('reg_date','first'),end=('reg_date','last'),rows=('reg_date','size'),source=('raw_source','first'),series=('series_id','first'),segment=('segment_id','first'),duration_minutes=(sig+'_elapsed_minutes','max'),first_raw_idx=('raw_idx','first'),last_raw_idx=('raw_idx','last')).reset_index().rename(columns={sig+'_run_id':'run_id'})
    assert np.allclose(r.duration_minutes,(r.rows-1)*5)
    duration=f[sig+'_run_id'].map(r.set_index('run_id').duration_minutes)
    for cutoff in CUTOFFS:
     fw=t[f'stale_{cutoff}'].to_numpy(bool);retro=duration.ge(cutoff)&t.input_valid
     f[f'{sig}_{cutoff}_causal_forward']=fw;f[f'{sig}_{cutoff}_confirmed_run_retrospective']=retro
     assert (~fw|retro).all()
     for _,run in r.loc[r.duration_minutes.ge(cutoff)].iterrows():
      first=run.start+pd.Timedelta(minutes=cutoff);g=f.loc[f[sig+'_run_id'].eq(run.run_id)];online=g.loc[g[f'{sig}_{cutoff}_causal_forward'],'reg_date'];assert online.iloc[0]==first
      allruns.append({'crop':crop,'split':split,'signature':sig,'cutoff_minutes':cutoff,**run.to_dict(),'first_online_detection':first,'decision_available_at':first,'delay_from_run_start_minutes':cutoff,'causal_rows':len(online),'retrospective_rows':len(g),'normality':'undetermined','left_censored_possible':run.start==f.reg_date.iloc[0],'right_censored_possible':run.end==f.reg_date.iloc[-1]})
    # Causal history continuity at episode/day boundaries is audited independently.
    cross=(f.episode_id.ne(f.episode_id.shift())|f.reg_date.dt.date.ne(f.reg_date.shift().dt.date))&t.reason.eq('continue')
    boundary.append({'crop':crop,'split':split,'signature':sig,'continued_episode_day_edges':int(cross.sum()),'break_reasons':t.reason.value_counts().to_dict()})
   f.to_parquet(D/f'{crop}_{split}_row_sidecar.parquet',index=False)
   keys=['series_id','segment_id','reg_date'];fi=f.set_index(keys);ci=pd.MultiIndex.from_frame(reg[keys]);ni=pd.MultiIndex.from_frame(reg[['series_id','segment_id','target_timestamp']].rename(columns={'target_timestamp':'reg_date'}));assert ci.isin(fi.index).all() and ni.isin(fi.index).all();assert len(reg)==int(f.B_eligible.sum())
   positions=pd.Series(np.arange(len(reg)),index=reg.row_id)
   for sig in SIGNATURES:
    for cutoff in CUTOFFS:
     for mode in ('causal_forward','confirmed_run_retrospective'):
      col=f'{sig}_{cutoff}_{mode}';hit=f[col];current=fi.loc[ci,col].to_numpy(bool);nxt=fi.loc[ni,col].to_numpy(bool);thit=current|nxt
      alltrans.append(pd.DataFrame({'crop':crop,'split':split,'signature':sig,'cutoff_minutes':cutoff,'mode':mode,'row_id':reg.row_id,'masked':thit}))
      result={'crop':crop,'split':split,'signature':sig,'cutoff_minutes':cutoff,'mode':mode,'stale_runs':int(f.loc[hit,sig+'_run_id'].nunique()),'normality_undetermined_runs':int(f.loc[hit,sig+'_run_id'].nunique()),'raw_rows':int(f.raw_idx.notna().sum()),'QC_rows':len(f),'masked_raw_rows':int(hit.sum()),'causal_after_detection_rows':int(f[f'{sig}_{cutoff}_causal_forward'].sum()),'retrospective_whole_run_rows':int(f[f'{sig}_{cutoff}_confirmed_run_retrospective'].sum()),'B_eligible_overlap':int((hit&f.B_eligible).sum()),'high_CO2_overlap':int((hit&f.obs_indoor_co2_ppm.ge(1500)).sum()),'affected_transitions':int(thit.sum()),'total_transitions':len(reg)}
      for h in (3,6,12):
       w=windows.loc[windows.horizon.eq(h)];st=positions.loc[w.start_row_id].to_numpy();ix=st[:,None]+np.arange(h)[None,:];assert (ix<len(reg)).all();np.testing.assert_array_equal(reg.iloc[st+h-1].target_timestamp.to_numpy(),w.end_target_timestamp.to_numpy());assert (reg.iloc[st].reg_date.to_numpy()+np.timedelta64(h*5,'m')==w.end_target_timestamp.to_numpy()).all()
       wh=current[st]|nxt[ix].any(axis=1);result[f'affected_windows_{h}']=int(wh.sum());result[f'total_windows_{h}']=len(w)
       allwindows.append(pd.DataFrame({'crop':crop,'split':split,'signature':sig,'cutoff_minutes':cutoff,'mode':mode,'horizon':h,'start_row_id':w.start_row_id.to_numpy(),'masked':wh}))
      summaries.append(result)
   print('SHADOW',crop,split,flush=True)
 csv('impact_summary.csv',pd.DataFrame(summaries));csv('detected_run_provenance.csv',pd.DataFrame(allruns));write('boundary_summary.json',boundary)
 pd.concat(alltrans,ignore_index=True).to_parquet(D/'transition_masks.parquet',index=False);pd.concat(allwindows,ignore_index=True).to_parquet(D/'window_masks.parquet',index=False)
 runs=pd.DataFrame(allruns)
 for split,start,end in [('validation','2025-11-10 18:45','2025-11-11 09:35'),('train','2024-10-01 00:45','2024-10-02 10:15'),('train','2024-10-20 03:15','2024-10-21 10:20')]:
  z=runs.loc[runs.crop.eq('strawberry')&runs.split.eq(split)&runs.start.le(pd.Timestamp(end))&runs.end.ge(pd.Timestamp(start))].copy();z['audit_interval_start']=start;z['audit_interval_end']=end;problem.append(z)
 csv('known_interval_detection.csv',pd.concat(problem,ignore_index=True));csv('melon_train_run_provenance.csv',runs.loc[runs.crop.eq('melon')&runs.split.eq('train')])
 write('input_manifest.json',inputs)
 for p,h in inputs.items():assert sha(ROOT/p)==h
 write('verification.json',{'raw_elapsed_equals_actual_contiguous_time':True,'first_detection_equals_start_plus_cutoff':True,'causal_mask_subset_retrospective':True,'registry_targets_and_horizons_exact':True,'original_hashes_unchanged':True,'no_Test_fit_predict_or_regeneration':True})
if __name__=='__main__':main()
