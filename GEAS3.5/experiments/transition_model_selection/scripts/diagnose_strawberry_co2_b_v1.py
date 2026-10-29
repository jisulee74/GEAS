"""Read-only diagnostics of frozen B strawberry CO2; never fits any model."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]
A=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1'
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry'
D=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_diagnosis_b_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str,allow_nan=True)+'\n')
def nummean(v):
 a=pd.to_numeric(v,errors='coerce').to_numpy(dtype=float,na_value=np.nan);a=a[np.isfinite(a)];return float(a.mean()) if len(a) else float('nan')
def bins(x,edges):return pd.cut(pd.Series(x),edges,right=False,include_lowest=True).astype(str).to_numpy()
def strata(frame,vent):
 t=pd.to_datetime(frame.reg_date)
 return pd.DataFrame({'hour':t.dt.hour.astype(str).to_numpy(),'day_night':np.where(frame.obs_is_daytime.to_numpy()>0,'day','night'),'date':t.dt.date.astype(str).to_numpy(),'dat':bins(frame.obs_growth_stage_dat,[-np.inf,0,30,60,90,120,150,180,240,365,np.inf]),'initial_co2':bins(frame.obs_indoor_co2_ppm,[-np.inf,400,600,800,1000,1500,np.inf]),'light':bins(frame.obs_outdoor_light,[-np.inf,0,10,200,500,np.inf]),'vent':bins(vent,[-np.inf,0,.01,.25,.5,.75,1.000001,np.inf])})
def aggregate(w,keys):
 g=w.groupby(keys,dropna=False,observed=True);out=g.agg(windows=('window_id','size'),steps=('steps','sum'),model_sse=('model_sse','sum'),persistence_sse=('persistence_sse','sum'),teacher_sse=('teacher_sse','sum'),model_error_sum=('model_error_sum','sum'),persistence_error_sum=('persistence_error_sum','sum'),endpoint_sse=('endpoint_sse','sum')).reset_index()
 out['model_rmse']=np.sqrt(out.model_sse/out.steps);out['persistence_rmse']=np.sqrt(out.persistence_sse/out.steps);out['teacher_rmse']=np.sqrt(out.teacher_sse/out.steps);out['skill']=1-out.model_rmse/out.persistence_rmse;out['bias']=out.model_error_sum/out.steps;out['endpoint_rmse']=np.sqrt(out.endpoint_sse/out.windows);out['excess_sse']=out.model_sse-out.persistence_sse
 return out
if __name__=='__main__':
 D.mkdir(exist_ok=False)
 protocol={'scope':'strawberry only; existing fitted LightGBM/Persistence and frozen Train/Validation; no model inference required for main teacher forcing','horizons':[3,6,12],'teacher_forced':'saved full eligible one-step predictions at recorded state; constant-start Persistence for free reference; per-step recorded-state Persistence also reported','groups':'fixed physical CO2 bins 400/600/800/1000/1500; light 0/10/200/500; vent 0/.01/.25/.5/.75/1; DAT 30-day to180 then240/365; delta 5/20/50/100 ppm','coverage':'QC Train eligible vs excluded, unknown commands never filled, exact 5min same crop/series/segment/episode delta only','support':'existing 10 percentage-point vent change and minimum100 descriptive count; no support artifact/calibration','causality':'overlapping windows are correlated; stratification descriptive, no exclusions or criterion changes'}
 write(D/'protocol.json',protocol)
 paths=[A/'protocol.json',A/'final_gate.json',A/'artifact_integrity.json']
 for name in ('lightgbm','persistence'):
  paths += [A/f'strawberry/{name}/rollout/predictions.npz',A/f'strawberry/{name}/rollout/completed.json',A/f'strawberry/{name}/fit/validation_predictions.parquet',A/f'strawberry/{name}/fit/completed.json',A/f'strawberry/{name}/fit/model.pkl']
 for split in ('train','validation'):paths += [P/f'{split}_{s}.parquet' for s in ('physical','prepared_qc','provenance','windows')]
 inputs=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in paths];write(D/'input_manifest.json',inputs)
 b=pd.read_parquet(P/'validation_physical.parquet');registry=pd.read_parquet(P/'validation_windows.parquet')
 m=np.load(A/'strawberry/lightgbm/rollout/predictions.npz');p=np.load(A/'strawberry/persistence/rollout/predictions.npz');np.testing.assert_array_equal(m['start_row_id'],p['start_row_id']);np.testing.assert_array_equal(m['max_horizon'],p['max_horizon']);np.testing.assert_allclose(m['truth'],p['truth'],equal_nan=True)
 start=pd.Series(np.arange(len(b)),index=b.row_id).loc[m['start_row_id']].to_numpy();initial=b.iloc[start].reset_index(drop=True);groups=strata(initial,initial.vent_pct);pred=m['predictions'][:,:,2];truth=m['truth'][:,:,2];persist=p['predictions'][:,:,2];tf_all=pd.read_parquet(A/'strawberry/lightgbm/fit/validation_predictions.parquet').obs_indoor_co2_ppm.to_numpy();target=b.target_next_indoor_co2_ppm.to_numpy();current=b.obs_indoor_co2_ppm.to_numpy()
 window_rows=[];traces=[]
 for h in (3,6,12):
  ix=np.flatnonzero(m['max_horizon']>=h);positions=start[ix,None]+np.arange(h)[None,:];true=truth[ix,:h];pr=pred[ix,:h];pe=persist[ix,:h];tf=tf_all[positions];err=pr-true;perr=pe-true;te=tf-true;tfdelta=tf-current[positions];cum=initial.obs_indoor_co2_ppm.to_numpy()[ix,None]+np.cumsum(tfdelta,axis=1)
  np.testing.assert_allclose(true,target[positions]);np.testing.assert_allclose(pe,initial.obs_indoor_co2_ppm.to_numpy()[ix,None]+np.zeros_like(pe));np.testing.assert_allclose(cum-true,np.cumsum(te,axis=1),rtol=1e-6,atol=1e-8)
  w=groups.iloc[ix].copy().reset_index(drop=True);w['window_id']=m['start_row_id'][ix];w['horizon']=h*5;w['episode']=initial.iloc[ix].episode_id.to_numpy();w['steps']=h;w['model_sse']=(err**2).sum(axis=1);w['persistence_sse']=(perr**2).sum(axis=1);w['teacher_sse']=(te**2).sum(axis=1);w['model_error_sum']=err.sum(axis=1);w['persistence_error_sum']=perr.sum(axis=1);w['endpoint_sse']=err[:,-1]**2;w['endpoint_bias']=err[:,-1];w['true_endpoint_change']=true[:,-1]-initial.iloc[ix].obs_indoor_co2_ppm.to_numpy();w['change_magnitude']=bins(np.abs(w.true_endpoint_change),[0,5,20,50,100,np.inf]);w['series_id']=initial.iloc[ix].series_id.to_numpy();window_rows.append(w)
  trace=pd.DataFrame({'horizon_minutes':h*5,'window_id':np.repeat(m['start_row_id'][ix],h),'episode':np.repeat(w.episode.to_numpy(),h),'step':np.tile(np.arange(1,h+1),len(ix)),'timestamp':b.iloc[positions.ravel()].target_timestamp.to_numpy(),'initial_co2':np.repeat(initial.iloc[ix].obs_indoor_co2_ppm.to_numpy(),h),'recorded_current_co2':current[positions].ravel(),'truth_co2':true.ravel(),'lightgbm_co2':pr.ravel(),'persistence_co2':pe.ravel(),'teacher_forced_co2':tf.ravel(),'model_error':err.ravel(),'model_abs_error':np.abs(err).ravel(),'persistence_error':perr.ravel(),'persistence_abs_error':np.abs(perr).ravel(),'teacher_error':te.ravel(),'teacher_abs_error':np.abs(te).ravel(),'summed_teacher_delta_co2':cum.ravel(),'summed_teacher_error':(cum-true).ravel(),'feedback_residual':(pr-cum).ravel(),'is_endpoint':np.tile(np.arange(h)==h-1,len(ix))});traces.append(trace)
  assert len(ix)==registry.horizon.eq(h).sum()
 w=pd.concat(window_rows,ignore_index=True);trace=pd.concat(traces,ignore_index=True);w.to_parquet(D/'window_decomposition.parquet',index=False);trace.to_parquet(D/'step_trace.parquet',index=False);trace.to_csv(D/'step_trace.csv.gz',index=False,compression='gzip')
 totals=aggregate(w,['horizon']);totals.to_csv(D/'horizon_metrics.csv',index=False)
 ep=aggregate(w,['horizon','episode']);ep.to_csv(D/'episode_metrics.csv',index=False)
 strat=[]
 for col in ('hour','day_night','date','dat','initial_co2','light','vent','change_magnitude'):
  g=aggregate(w,['horizon',col]).rename(columns={col:'group'});g['dimension']=col;strat.append(g)
 pd.concat(strat,ignore_index=True).to_csv(D/'group_metrics.csv',index=False)
 concentration=[]
 for h,g in ep.groupby('horizon'):
  for measure in ('model_sse','excess_sse'):
   ranked=g.sort_values(measure,ascending=False)
   for k in (1,3,5,10):
    concentration.append({'horizon':int(h),'rank_by':measure,'top_k':k,'episodes':ranked.head(k).episode.tolist(),'model_sse_share':float(ranked.head(k).model_sse.sum()/g.model_sse.sum()),'windows_share':float(ranked.head(k).windows.sum()/g.windows.sum()),'net_excess_sse_share':float(ranked.head(k).excess_sse.sum()/g.excess_sse.sum())})
 write(D/'episode_concentration.json',concentration)
 step=[]
 for (h,j),g in trace.groupby(['horizon_minutes','step']):
  row={'horizon':int(h),'step':int(j),'windows':len(g)}
  for key in ('model_error','persistence_error','teacher_error','summed_teacher_error','feedback_residual'):
   v=g[key].to_numpy();row[key+'_bias']=float(v.mean());row[key+'_rmse']=float(np.sqrt(np.mean(v*v)))
  row['teacher_persistence_rmse']=float(np.sqrt(np.mean((g.recorded_current_co2-g.truth_co2)**2)));step.append(row)
 pd.DataFrame(step).to_csv(D/'step_metrics.csv',index=False)
 # Coverage/exclusion: exact sensor time pairs, not a shift across missing intervals.
 dist=[];counts=[];action_rows=[];population=[]
 for split in ('train','validation'):
  q=pd.read_parquet(P/f'{split}_prepared_qc.parquet');a=pd.read_parquet(P/f'{split}_provenance.parquet');assert len(q)==len(a);np.testing.assert_array_equal(q.reg_date,a.reg_date)
  valid=(q.reg_date.shift(-1)-q.reg_date).eq(pd.Timedelta(minutes=5))
  for k in ('crop','series_id','segment_id','episode_id'):valid &= q[k].eq(q[k].shift(-1))
  delta=(q.obs_indoor_co2_ppm.shift(-1)-q.obs_indoor_co2_ppm).where(valid)
  previous_delta=(q.obs_indoor_co2_ppm-q.obs_indoor_co2_ppm.shift(1)).where(valid.shift(1,fill_value=False))
  vent=pd.to_numeric(a.current_vent_pct,errors='coerce');prev=pd.to_numeric(a.previous_vent_pct,errors='coerce');changes=vent-prev
  sg=strata(q,vent);sg['delta_co2']=bins(np.abs(delta),[0,5,20,50,100,np.inf]);sg['split']=split;sg['population']=np.where(a.eligible,'eligible','excluded')
  population.append(sg)
  fields={'co2':q.obs_indoor_co2_ppm,'delta_co2':delta,'light':q.obs_outdoor_light,'temperature':q.obs_indoor_temp_c,'humidity':q.obs_indoor_humidity_pct,'vent_command':vent,'vent_change':changes,'dat':q.obs_growth_stage_dat}
  for pop,mask in [('eligible',a.eligible),('excluded',~a.eligible)]:
   for name,v in fields.items():
    ar=v.loc[mask].to_numpy(float);ar=ar[np.isfinite(ar)];dist.append({'split':split,'population':pop,'feature':name,'rows':int(mask.sum()),'finite_rows':len(ar),'unknown_rows':int(mask.sum())-len(ar),'mean':float(np.mean(ar)) if len(ar) else None,'std':float(np.std(ar)) if len(ar) else None,'q10':float(np.quantile(ar,.1)) if len(ar) else None,'median':float(np.median(ar)) if len(ar) else None,'q90':float(np.quantile(ar,.9)) if len(ar) else None,'min':float(np.min(ar)) if len(ar) else None,'max':float(np.max(ar)) if len(ar) else None})
  for col in ('hour','day_night','date','dat','initial_co2','light','vent','delta_co2'):
   ct=sg.groupby(['population',col],dropna=False).size().reset_index(name='rows').rename(columns={col:'group'});ct['dimension']=col;ct['split']=split;counts.append(ct)
  action=pd.DataFrame({'timestamp':q.reg_date,'episode':q.episode_id,'split':split,'eligible':a.eligible,'vent':vent,'previous_vent':prev,'vent_change':changes,'delta_co2':delta,'previous_delta_co2':previous_delta,'co2':q.obs_indoor_co2_ppm,'hour':sg.hour,'day_night':sg.day_night,'dat':sg.dat,'initial_co2':sg.initial_co2,'light':sg.light})
  action['change_class']=np.select([changes.isna(),changes.ge(.1),changes.le(-.1),changes.abs().gt(1e-12)],['unknown','open_ge10pp','close_ge10pp','small_change'],default='unchanged');action_rows.append(action)
 pd.DataFrame(dist).to_csv(D/'population_distribution.csv',index=False);counts=pd.concat(counts,ignore_index=True);counts.to_csv(D/'population_group_counts.csv',index=False)
 actions=pd.concat(action_rows,ignore_index=True);actions.to_parquet(D/'action_response_rows.parquet',index=False)
 action_summary=[]
 for dimension in ('all','hour','day_night','dat','initial_co2','light'):
  z=actions.copy();z['group']='all' if dimension=='all' else z[dimension]
  for key,g in z.groupby(['split','eligible','group','change_class']):
   ar=g.delta_co2.dropna().astype(float);action_summary.append({'split':key[0],'eligible':bool(key[1]),'dimension':dimension,'group':key[2],'change_class':key[3],'rows':len(g),'finite_delta_rows':len(ar),'mean_delta_co2':float(ar.mean()),'mean_abs_delta_co2':float(ar.abs().mean()),'std_delta_co2':float(ar.std(ddof=0)),'mean_command':nummean(g.vent),'mean_change':nummean(g.vent_change),'mean_previous_delta_co2':nummean(g.previous_delta_co2),'finite_previous_delta_rows':int(g.previous_delta_co2.notna().sum())})
 pd.DataFrame(action_summary).to_csv(D/'action_response_groups.csv',index=False)
 # Join error strata with train counts, retaining empty support cells.
 joined=[]
 for g in strat:
  dim=g.dimension.iloc[0];tr=counts.loc[counts.split.eq('train')&counts.dimension.eq(dim)]
  if not len(tr):continue
  tr=tr.pivot_table(index='group',columns='population',values='rows',aggfunc='sum',fill_value=0).reset_index().rename(columns={'eligible':'train_eligible_rows','excluded':'train_excluded_rows'})
  z=g.merge(tr,on='group',how='left').fillna({'train_eligible_rows':0,'train_excluded_rows':0});z['train_eligible_fraction']=z.train_eligible_rows/(z.train_eligible_rows+z.train_excluded_rows);joined.append(z)
 pd.concat(joined,ignore_index=True).to_csv(D/'coverage_error_overlap.csv',index=False)
 # Independent aggregate reproduction of previously frozen CO2 results.
 old=json.loads((A/'strawberry/lightgbm/rollout/completed.json').read_text())
 for h in (15,30,60):
  got=totals.loc[totals.horizon.eq(h)].iloc[0];expected=old['rollout'][f'{h}min']['targets']['obs_indoor_co2_ppm']['rmse'];np.testing.assert_allclose(got.model_rmse,expected,rtol=1e-6,atol=1e-8)
 for x in inputs:assert sha(ROOT/x['path'])==x['sha256']
 write(D/'verification.json',{'all_registry_windows':len(w),'trace_steps':len(trace),'horizon_counts':w.groupby('horizon').size().to_dict(),'aggregate_matches_frozen_metrics':True,'teacher_error_sum_identity':True,'all_inputs_unchanged':True,'fit_called':False,'Test_accessed':False})
 print(totals.to_string(index=False));print('done',D)
