"""Full fixed-window B evaluation. No safety projection or truncated rollout."""
import json,pickle,time
from types import SimpleNamespace
import numpy as np,pandas as pd
from geas35.models.transition.command_carry_b_v1 import *
from geas35.models.transition.batch_sensor_rollout_b_v1 import BatchSensorUpdater
from geas35.models.transition.hyperparameters import validation_rollout_weighted_score
from geas35.rl.sensor_history_env_v4 import SensorHistoryStateUpdater
class Identity:
 def transform_frame(self,f):return f.copy()
 def inverse_transform_frame(self,f):return f.copy()

def metrics(pred,truth,scales):
 e=pred-truth;abs_e=np.abs(e);h=e.shape[1];s=np.asarray([scales[c] for c in TARGETS]);normalized=e/s
 finite=np.isfinite(pred);bad=~finite | (pred<np.array([-60,0,0])) | (pred>np.array([80,100,10000]))
 out={'window_count':len(e),'trajectory_mae':float(np.mean(abs_e)),'trajectory_rmse':float(np.sqrt(np.mean(e**2))),'trajectory_nrmse':float(np.mean(np.sqrt(np.mean(e**2,axis=(0,1)))/s)),'nan_inf_count':int((~finite).sum()),'physical_violation_rate':float(bad.mean()),'safety_gate_passed':float(not bad.any()),'final_step_mae':float(abs_e[:,-1].mean())}
 x=np.arange(h)-np.arange(h).mean();drift=(np.abs(normalized)*x[None,:,None]).sum(axis=1)/(x*x).sum();out['drift_slope_nmae']=float(drift.mean());windows=np.sqrt(np.mean(normalized**2,axis=1)).mean(axis=1)
 for q in ('q90','cvar90'):
  cut=np.quantile(windows,.9);out['window_trajectory_nrmse_'+q]=float(cut if q=='q90' else np.mean(windows[windows>=cut]))
 out['window_trajectory_nrmse_mean']=float(windows.mean());out['window_trajectory_nrmse_std']=float(windows.std())
 targets={}
 for j,c in enumerate(TARGETS):
  er=e[:,:,j];a=np.abs(er);y=truth[:,:,j];den=((y-y.mean())**2).sum();rmse=np.sqrt(np.mean(er**2));cut=np.quantile(a,.9)
  targets[c]={'mae':float(a.mean()),'rmse':float(rmse),'r2':float(1-(er**2).sum()/den) if den else float('nan'),'nrmse':float(rmse/s[j]),'q90':float(cut),'cvar90':float(a[a>=cut].mean()),'endpoint_rmse':float(np.sqrt(np.mean(er[:,-1]**2))),'endpoint_mae':float(a[:,-1].mean()),'endpoint_signed_drift':float(er[:,-1].mean()),'drift_slope_nmae':float(drift[:,j].mean()),'physical_violation_rate':float(bad[:,:,j].mean()),'nan_inf_count':int((~finite[:,:,j]).sum())}
  for key in ('mae','rmse','r2','nrmse'):out[c+'_trajectory_'+key]=targets[c][key]
 out['targets']=targets;return out

def evaluate(crop,label,repeat=False):
 pro=protocol();dest=D/crop/label;fit_dir=dest/('repeat' if repeat else 'fit');fit=json.loads((fit_dir/'completed.json').read_text());out=dest/('repeat_rollout' if repeat else 'rollout');out.mkdir(exist_ok=True)
 if (out/'completed.json').exists():
  done=json.loads((out/'completed.json').read_text());assert done['protocol_sha256']==sha(D/'protocol.json');return done
 parity=json.loads((D/'batch_runtime_parity.json').read_text());assert parity['passed'] and parity['module_sha256']==sha(ROOT/'src/geas35/models/transition/batch_sensor_rollout_b_v1.py')
 with (fit_dir/'model.pkl').open('rb') as f:model=pickle.load(f)
 assert sha(fit_dir/'model.pkl')==fit['model_sha256']
 base=pd.read_parquet(P/crop/'validation_prepared_qc.parquet');b=pd.read_parquet(P/crop/'validation_physical.parquet');windows=pd.read_parquet(P/crop/'validation_windows.parquet');lookup=pd.Series(np.arange(len(b)),index=b.row_id)
 ids=windows.groupby('start_row_id',sort=False).horizon.max();starts=lookup.loc[ids.index].to_numpy();maxh=ids.to_numpy();qpos=pd.Index(base.reg_date).get_indexer(b.iloc[starts].reg_date);assert (qpos>=0).all()
 for h in (3,6,12):
  w=windows.loc[windows.horizon.eq(h)];ii=lookup.loc[w.start_row_id].to_numpy();assert np.array_equal(b.iloc[ii+h-1].target_timestamp.to_numpy(),w.end_target_timestamp.to_numpy());assert np.all(b.iloc[ii].mdp_v1_rollout_id.to_numpy()==b.iloc[ii+h-1].mdp_v1_rollout_id.to_numpy())
 current=b.iloc[starts][OBS].to_numpy(float);batch=BatchSensorUpdater(base,qpos,current);predictions=np.full((len(starts),12,3),np.nan);truth=np.full_like(predictions,np.nan)
 means=np.array([model.observation_scaler.means.get(c,0.) for c in OBS]);scales=np.array([model.observation_scaler.scales.get(c,1.) for c in OBS]);start_time=time.perf_counter()
 for step in range(12):
  ix=np.flatnonzero(maxh>step);trans=b.iloc[starts[ix]+step];recorded=trans[['next_'+c for c in OBS]].to_numpy(float);action=trans[ACTIONS].to_numpy(float);x=np.column_stack([(batch.state[ix]-means)/scales,action]);valid=np.isfinite(x).all(axis=1);pr=np.full((len(ix),3),np.nan)
  if valid.any():pr[valid]=model.predict(pd.DataFrame(x[valid],columns=OBS+ACTIONS)).next_observation.to_numpy(float)
  predictions[ix,step]=pr;truth[ix,step]=recorded[:,:3]
  with np.errstate(all='ignore'):batch.advance(ix,pr,recorded,action,trans.target_timestamp.to_numpy(dtype='datetime64[ns]').astype('int64'))
 # Actual-model path parity on deterministic first/middle/last full windows.
 candidates=np.flatnonzero(maxh==12);checks=[]
 for k in candidates[np.unique(np.linspace(0,len(candidates)-1,3).astype(int))]:
  i=qpos[k];initial=base.iloc[i].copy();initial[OBS]=b.iloc[starts[k]][OBS];u=SensorHistoryStateUpdater(Identity(),base.iloc[max(0,i-12):i]);u.reset(initial);maxerr=0.;checked=0
  for step in range(12):
   trans=b.iloc[starts[k]+step];action=trans[ACTIONS].to_numpy(float);x=np.r_[(u.current_physical[OBS].to_numpy(float)-means)/scales,action]
   if not np.isfinite(x).all():break
   pr=model.predict(pd.DataFrame([x],columns=OBS+ACTIONS)).next_observation.to_numpy(float)[0]
   np.testing.assert_allclose(pr,predictions[k,step],rtol=1e-6,atol=1e-8,equal_nan=True);maxerr=max(maxerr,float(np.nanmax(np.abs(pr-predictions[k,step]))));checked+=1
   if not np.isfinite(pr).all():break
   nxt=base.iloc[i+step+1].copy();nxt[OBS]=trans[['next_'+c for c in OBS]].to_numpy(float)
   u.advance(current_scaled=u.current_physical,recorded_next_scaled=nxt,prediction_physical=dict(zip(TARGETS,pr)),action=dict(zip(ACTIONS,action)))
  checks.append({'start_row_id':str(ids.index[k]),'steps_compared':checked,'max_abs_error':maxerr})
 multi={}
 for h in (3,6,12):
  ix=np.flatnonzero(maxh>=h);assert len(ix)==int(windows.horizon.eq(h).sum());multi[f'{5*h}min']=metrics(predictions[ix,:h],truth[ix,:h],fit['scales'])
 one=SimpleNamespace(aggregate_metrics=fit['one_step']['aggregate_metrics']);score,components=validation_rollout_weighted_score(one,multi,objective_weights=pro['weights'])
 finite_one=all(np.isfinite(v) for v in one.aggregate_metrics.values() if isinstance(v,(float,int)))
 eligible=fit['name']!='persistence' and finite_one and np.isfinite(score) and fit['action_response']['eligible'] and all(v['safety_gate_passed']==1 for v in multi.values())
 np.savez_compressed(out/'predictions.npz',start_row_id=np.array(ids.index,dtype=str),max_horizon=maxh,predictions=predictions,truth=truth)
 result={'crop':crop,'label':label,'name':fit['name'],'seed':fit['seed'],'score':score,'components':components,'rollout':multi,'eligible':bool(eligible),'model_path_parity':checks,'seconds':time.perf_counter()-start_time,'protocol_sha256':sha(D/'protocol.json'),'model_sha256':fit['model_sha256'],'window_registry_sha256':sha(P/crop/'validation_windows.parquet'),'full_registry_evaluated':True,'test_accessed':False}
 write(out/'completed.json',result);print('ROLLOUT COMPLETE',crop,label,flush=True);return result
