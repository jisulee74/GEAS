"""Frozen 4CH candidate execution. Canonical Train/Validation only; no downstream."""
from pathlib import Path
import sys,os,json,hashlib,platform,importlib.metadata,argparse,time,pickle,traceback,subprocess
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'))
B=R/'experiments/rl_policy_training/artifacts';PH=B/'transition_4ch_phase0_canonical_v1';PR=B/'transition_4ch_migration_prereg_v1';D=B/'transition_4ch_candidate_reselection_v1'
EXPECTED='/NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/GEAS/GEAS3.5/.venv/bin/python'
PRO=json.loads((PR/'migration_preregistration.json').read_text());P=PH/'dataset'/PRO['next_dataset_namespace']
assert sys.executable==EXPECTED and platform.python_version()=='3.12.3','Canonical Python required'
VERSIONS={n:importlib.metadata.version(n) for n in PRO['dependencies']}
assert VERSIONS==PRO['dependencies'],'Dependency drift'
import numpy as np
import pandas as pd
from types import SimpleNamespace
from geas35.models.transition.four_channel_phase0 import FourChannelRuntime,TARGETS,sha,scaler_from
from geas35.models.transition.four_channel_batch import FourChannelBatchUpdater,OBS,ACTIONS
from geas35.models.transition.registry import build_transition_model
from geas35.models.transition.base import TransitionDataset,TransitionTargetFormulation
from geas35.models.transition.evaluator import evaluate_transition_predictions
from geas35.models.transition.action_response import evaluate_action_response
from geas35.models.transition.hyperparameters import validation_rollout_weighted_score
ORDER=('persistence','linear_regression','linear_svr','knn','extra_trees','lightgbm','xgboost','mlp')
CROPS=('melon','cucumber','strawberry')
LOW=np.array([-60,-60,0,0]);HIGH=np.array([80,80,100,100])

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)

def code_paths():
 names=['base.py','registry.py','sklearn_models.py','boosting_models.py','evaluator.py','hyperparameters.py','action_response.py','four_channel_phase0.py','four_channel_batch.py','causal_state.py']
 return [Path(__file__)]+[R/'src/geas35/models/transition'/n for n in names]+[R/'src/geas35/rl'/n for n in ['mdp_v1.py','sensor_history_env_v4.py','causal_contract_env_v2.py','recorded_light_env_v1.py']]+[R/'src/geas35/features/derived.py']

def lock(full=False):
 x=json.loads((D/'execution_binding.json').read_text());assert sha(D/'execution_binding.json')==json.loads((D/'execution_lock.json').read_text())['sha256']
 assert sha(PR/'migration_preregistration.json')==x['preregistration_sha256']
 for p,h in x['code_hashes'].items():assert sha(R/p)==h,('code changed',p)
 if full:
  for p,h in x['input_hashes'].items():assert sha(R/p)==h,('input changed',p)
 return x

def freeze():
 assert not D.exists(),'Do not overwrite execution'
 phase=json.loads((PH/'phase0_manifest.json').read_text());assert phase['status']=='TRANSITION 4CH PHASE0 — PASS / FIXED-CANDIDATE RESELECTION READY'
 assert phase['gates']['dependency_lock'] and phase['runtime_environment']['sys_executable']==EXPECTED
 assert phase['fitting_authority']==str(P)
 for p,h in phase['artifact_hashes'].items():assert sha(P/p)==h,p
 for p,h in PRO['parent_metric_code_hashes'].items():assert sha(R/p)==h,p
 for crop in CROPS:assert set(PRO['candidates'][crop])==set(ORDER)
 assert OBS==PRO['schema']['observation_columns'] and TARGETS==OBS[:4] and len(OBS+ACTIONS)==62
 D.mkdir()
 inputs={str((P/p).relative_to(R)):h for p,h in phase['artifact_hashes'].items()}
 for p in [PH/'phase0_manifest.json',PH/'interpreter_gate.json',PR/'migration_preregistration.json',PR/'preregistration_lock.json']:inputs[str(p.relative_to(R))]=sha(p)
 binding={'sys_executable':sys.executable,'python':platform.python_version(),'dependencies':VERSIONS,'phase0_authority':str(P),'logical_model_namespace':PRO['next_model_namespace'],'preregistration_sha256':sha(PR/'migration_preregistration.json'),'input_hashes':inputs,'code_hashes':{str(p.relative_to(R)):sha(p) for p in code_paths()},'candidates':PRO['candidates'],'metric':PRO['metric'],'selection':PRO['selection'],'order':ORDER,'scope':'Train/Validation only, no Test/P0/live/support/Step16/PPO/HPO','implementation_policy':{'full_windows':'each exact frozen 3/6/12 window; no sampling for scoring','action_response':'unchanged helper minimum_support=100; metadata separates exact contiguous new runs, no gap crossing','invalid_prediction':'stop invalid free path; retain invalid prediction, later predictions NaN and candidate gate fails; never clip','batch':'four-channel port; mandatory scalar equivalence before first fit and per candidate model paths','validation_fit':'None; MLP internal Train early stopping only','prior_winners':'post-selection reference only'},'created_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
 write(D/'execution_binding.json',binding);write(D/'execution_lock.json',{'sha256':sha(D/'execution_binding.json')})
 print('EXECUTION LOCKED BEFORE FIT',flush=True)

def frames(crop,split):
 if crop not in CROPS or split not in ('train','validation'):raise ValueError('Fixed Train/Validation only')
 b=pd.read_parquet(P/crop/f'{split}_physical.parquet');assert len(b)==PRO['admission']['expected_pair_counts'][f'{crop}/{split}'];return b

def dataset(crop,split):
 b=frames(crop,split);sc=scaler_from(json.loads((P/crop/'observation_scaler.json').read_text()))
 s=pd.read_parquet(P/crop/f'{split}_scaled.parquet');np.testing.assert_array_equal(b.row_id,s.row_id)
 md=b[['row_id','crop','reg_date','episode_id','segment_id']].copy()
 br=b.reg_date.ne(b.target_timestamp.shift())|b.mdp_v1_rollout_id.ne(b.mdp_v1_rollout_id.shift())
 md['segment_id']=md.segment_id.astype(str)+'|4chrun='+br.cumsum().astype(str)
 y=b[['next_'+c for c in TARGETS]].copy();y.columns=TARGETS
 return TransitionDataset(s[OBS+ACTIONS],y,tuple(OBS+ACTIONS),tuple(TARGETS),md,tuple(OBS),tuple(ACTIONS)),sc

def setup(crop,split='validation'):
 b=frames(crop,split);base=pd.read_parquet(P/crop/f'{split}_sensor_prehistory.parquet');w=pd.read_parquet(P/crop/f'{split}_windows.parquet');w=w[w.horizon.isin([3,6,12])]
 ids=w.groupby('start_row_id',sort=False).horizon.max();lookup=pd.Series(np.arange(len(b)),index=b.row_id);starts=lookup.loc[ids.index].to_numpy();mh=ids.to_numpy()
 keys=['crop','series_id','segment_id','episode_id','reg_date'];index=pd.MultiIndex.from_frame(base[keys]);assert index.is_unique
 qp=index.get_indexer(pd.MultiIndex.from_frame(b.iloc[starts][keys]));assert (qp>=0).all()
 for h in (3,6,12):
  ww=w[w.horizon.eq(h)];ss=lookup.loc[ww.start_row_id].to_numpy()
  assert set(ww.start_row_id)==set(ids.index[mh>=h])
  assert np.array_equal(b.iloc[ss+h-1].target_timestamp.to_numpy(),ww.end_target_timestamp.to_numpy())
  for k in range(h):
   assert np.array_equal(b.iloc[ss+k].reg_date.to_numpy(),b.iloc[ss].reg_date.to_numpy()+np.timedelta64(5*k,'m'))
   for key in ['crop','series_id','segment_id','episode_id','source_confidence']:assert np.array_equal(b.iloc[ss+k][key],b.iloc[ss][key])
 return b,base,ids,starts,mh,qp

def scalar(crop,base):
 return FourChannelRuntime(P/'schema.json',P/crop/'observation_scaler.json',schema_hash=sha(P/'schema.json'),scaler_hash=sha(P/crop/'observation_scaler.json'),prehistory=base)

def target(row):
 t=row.copy();t[OBS]=row[['next_'+c for c in OBS]].to_numpy(float);t['reg_date']=row.target_timestamp;return t

def parity():
 lock(True);checks=[]
 for crop in CROPS:
  for split in ('train','validation'):
   b,base,ids,starts,mh,qp=setup(crop,split)
   batch=FourChannelBatchUpdater(base,qp,b.iloc[starts][OBS].to_numpy(float))
   # Entire reset population of scored windows: recorded frozen state must equal reset recurrence.
   np.testing.assert_allclose(batch.state,b.iloc[starts][OBS].to_numpy(float),rtol=1e-6,atol=1e-8)
   choices=np.flatnonzero(mh==12);ks=choices[np.unique(np.linspace(0,len(choices)-1,3).astype(int))]
   maxerr=0.;n=0
   for k in ks:
    rt=scalar(crop,base);rt.reset(b.iloc[starts[k]])
    for step in range(12):
     row=b.iloc[starts[k]+step];t=target(row);action={a:float(row[a]) for a in ACTIONS}
     pr=rt.next_from_residual([.001,.002,-.001,-.002]);nv=rt.advance(prediction=pr,action=action,exogenous=rt.exogenous(t))
     bv=batch.advance(np.array([k]),np.array([list(pr.values())]),t[OBS].to_numpy(float)[None,:],np.array([list(action.values())]),np.array([pd.Timestamp(t.reg_date).value]))[0]
     np.testing.assert_allclose(bv,nv[OBS].to_numpy(float),rtol=1e-6,atol=1e-8);maxerr=max(maxerr,float(np.max(np.abs(bv-nv[OBS].to_numpy(float)))));n+=1
   checks.append({'crop':crop,'split':split,'all_window_resets':len(starts),'scalar_steps':n,'max_abs_difference':maxerr})
   print('BATCH PARITY',checks[-1],flush=True)
 write(D/'batch_scalar_parity.json',{'passed':True,'checks':checks,'module_sha256':sha(R/'src/geas35/models/transition/four_channel_batch.py'),'execution_lock_sha256':sha(D/'execution_binding.json')})

def invalid(x):return (~np.isfinite(x))|(x<LOW)|(x>HIGH)

def metrics(pred,truth,scales):
 e=pred-truth;a=np.abs(e);h=e.shape[1];s=np.array([scales[c] for c in TARGETS]);norm=e/s;bad=invalid(pred)
 x=np.arange(h)-np.arange(h).mean();drift=(np.abs(norm)*x[None,:,None]).sum(axis=1)/(x*x).sum() if h>1 else np.zeros((len(e),4))
 window=np.sqrt(np.mean(norm**2,axis=1)).mean(axis=1);cut=np.quantile(window,.9)
 out={'window_count':len(e),'trajectory_mae':float(a.mean()),'trajectory_rmse':float(np.sqrt(np.mean(e**2))),'trajectory_nrmse':float((np.sqrt(np.mean(e**2,axis=(0,1)))/s).mean()),'nan_inf_count':int((~np.isfinite(pred)).sum()),'physical_violation_rate':float(bad.mean()),'safety_gate_passed':not bad.any(),'final_step_mae':float(a[:,-1].mean()),'drift_slope_nmae':float(drift.mean()),'window_trajectory_nrmse_q90':float(cut),'window_trajectory_nrmse_cvar90':float(window[window>=cut].mean()) if np.isfinite(cut) else float('nan'),'window_trajectory_nrmse_mean':float(window.mean()),'window_trajectory_nrmse_std':float(window.std()),'targets':{}}
 for j,c in enumerate(TARGETS):
  er=e[:,:,j];aa=a[:,:,j];y=truth[:,:,j];den=((y-y.mean())**2).sum();rmse=np.sqrt(np.mean(er**2));cut=np.quantile(aa,.9)
  out['targets'][c]={'mae':float(aa.mean()),'rmse':float(rmse),'r2':float(1-(er**2).sum()/den) if den else float('nan'),'nrmse':float(rmse/s[j]),'q90':float(cut),'cvar90':float(aa[aa>=cut].mean()) if np.isfinite(cut) else float('nan'),'endpoint_rmse':float(np.sqrt(np.mean(er[:,-1]**2))),'endpoint_mae':float(aa[:,-1].mean()),'endpoint_signed_drift':float(er[:,-1].mean()),'drift_slope_nmae':float(drift[:,j].mean()),'physical_violation_rate':float(bad[:,:,j].mean()),'nan_inf_count':int((~np.isfinite(pred[:,:,j])).sum())}
 return out

def fit(crop,name,seed,repeat=False):
 lock();params=json.loads(json.dumps(PRO['candidates'][crop][name]['params']));label=name+(f'_seed_{seed}' if name=='mlp' else '')
 if name=='mlp':params['random_state']=seed
 dest=D/crop/label/('repeat' if repeat else 'fit');dest.mkdir(parents=True,exist_ok=True)
 if (dest/'completed.json').exists():
  v=json.loads((dest/'completed.json').read_text());assert v['model_sha256']==sha(dest/'model.pkl');assert v['execution_sha256']==sha(D/'execution_binding.json');return v
 tr,sc=dataset(crop,'train');va,_=dataset(crop,'validation');scales=json.loads((P/crop/'evaluation_normalization.json').read_text())['std_ddof0']
 np.testing.assert_allclose([va.y[c].std(ddof=0) for c in TARGETS],list(scales[c] for c in TARGETS),rtol=0,atol=0)
 model=build_transition_model(name,params=params)
 if name=='persistence':model.observation_scaler=sc
 else:model.configure_target_formulation(TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA,observation_scaler=sc)
 start=time.monotonic();model.fit(tr,validation_dataset=None);elapsed=time.monotonic()-start
 pr=model.predict(va).next_observation;one=evaluate_transition_predictions(va.y,pr,target_columns=TARGETS,normalization_scales=scales,target_groups={'indoor_four_channels':TARGETS}).to_artifact()
 action=evaluate_action_response(model,va).to_artifact()
 with (dest/'model.pkl').open('xb') as f:pickle.dump(model,f)
 with (dest/'model.pkl').open('rb') as f:reloaded=pickle.load(f)
 np.testing.assert_allclose(reloaded.predict(va).next_observation.to_numpy(float),pr.to_numpy(float),rtol=1e-6,atol=1e-8)
 assert list(model.input_columns_)==OBS+ACTIONS and list(model.target_columns_)==TARGETS
 pr.to_parquet(dest/'validation_predictions.parquet',index=False)
 extra=metrics(pr.to_numpy(float)[:,None,:],va.y.to_numpy(float)[:,None,:],scales)
 result={'crop':crop,'label':label,'name':name,'seed':params.get('random_state'),'params':params,'fit_seconds':elapsed,'train_rows':len(tr.x),'validation_rows':len(va.x),'one_step':one,'one_step_diagnostics':extra,'action_response':action,'scales':scales,'one_step_finite_physical_pass':not invalid(pr.to_numpy(float)).any(),'reload_pass':True,'model_sha256':sha(dest/'model.pkl'),'schema_sha256':sha(P/'schema.json'),'scaler_sha256':sha(P/crop/'observation_scaler.json'),'train_registry_sha256':sha(P/crop/'train_registry.parquet'),'train_scaled_sha256':sha(P/crop/'train_scaled.parquet'),'execution_sha256':sha(D/'execution_binding.json'),'from_scratch':True,'validation_fit_argument':None}
 write(dest/'completed.json',result);print('FIT',crop,label,round(elapsed,2),flush=True);return result

def evaluate(crop,label,repeat=False):
 lock();par=json.loads((D/'batch_scalar_parity.json').read_text());assert par['passed'] and par['module_sha256']==sha(R/'src/geas35/models/transition/four_channel_batch.py')
 dest=D/crop/label;fd=dest/('repeat' if repeat else 'fit');out=dest/('repeat_rollout' if repeat else 'rollout');out.mkdir(exist_ok=True)
 fitinfo=json.loads((fd/'completed.json').read_text());assert sha(fd/'model.pkl')==fitinfo['model_sha256']
 if (out/'completed.json').exists():
  done=json.loads((out/'completed.json').read_text());assert done['execution_sha256']==sha(D/'execution_binding.json');return done
 with (fd/'model.pkl').open('rb') as f:model=pickle.load(f)
 b,base,ids,starts,mh,qp=setup(crop);batch=FourChannelBatchUpdater(base,qp,b.iloc[starts][OBS].to_numpy(float));pr=np.full((len(starts),12,4),np.nan);truth=np.full_like(pr,np.nan);dead=np.zeros(len(starts),bool)
 means=np.array([model.observation_scaler.means.get(c,0.) for c in OBS]);scale=np.array([model.observation_scaler.scales.get(c,1.) for c in OBS]);begin=time.monotonic()
 for step in range(12):
  allix=np.flatnonzero(mh>step);tr=b.iloc[starts[allix]+step];truth[allix,step]=tr[['next_'+c for c in TARGETS]].to_numpy(float)
  ix=np.flatnonzero((mh>step)&~dead);tr=b.iloc[starts[ix]+step];action=tr[ACTIONS].to_numpy(float);xx=np.column_stack([(batch.state[ix]-means)/scale,action]);ok=np.isfinite(xx).all(axis=1);p=np.full((len(ix),4),np.nan)
  if ok.any():p[ok]=model.predict(pd.DataFrame(xx[ok],columns=OBS+ACTIONS)).next_observation.to_numpy(float)
  pr[ix,step]=p;good=~invalid(p).any(axis=1);dead[ix[~good]]=True
  if good.any():batch.advance(ix[good],p[good],tr[['next_'+c for c in OBS]].to_numpy(float)[good],action[good],tr.target_timestamp.to_numpy(dtype='datetime64[ns]').astype('int64')[good])
 # Independent scalar model predictions and full56 state parity, not just output equality.
 ks=np.flatnonzero(mh==12);ks=ks[np.unique(np.linspace(0,len(ks)-1,3).astype(int))];checks=[]
 for k in ks:
  rt=scalar(crop,base);rt.reset(b.iloc[starts[k]]);ba=FourChannelBatchUpdater(base,np.array([qp[k]]),b.iloc[[starts[k]]][OBS].to_numpy(float));maxerr=0.;n=0
  for step in range(12):
   row=b.iloc[starts[k]+step];a={c:float(row[c]) for c in ACTIONS};pred=model.predict(rt.model_input(a).to_frame().T).next_observation.iloc[0].to_numpy(float)
   np.testing.assert_allclose(pred,pr[k,step],rtol=1e-6,atol=1e-8,equal_nan=True);n+=1
   if invalid(pred).any():break
   t=target(row);nv=rt.advance(prediction=dict(zip(TARGETS,pred)),action=a,exogenous=rt.exogenous(t));bv=ba.advance(np.array([0]),pred[None,:],t[OBS].to_numpy(float)[None,:],np.array([list(a.values())]),np.array([pd.Timestamp(t.reg_date).value]))[0]
   np.testing.assert_allclose(bv,nv[OBS].to_numpy(float),rtol=1e-6,atol=1e-8);maxerr=max(maxerr,float(np.max(np.abs(bv-nv[OBS].to_numpy(float)))))
  checks.append({'start_row_id':str(ids.index[k]),'steps_compared':n,'max_abs_state_difference':maxerr})
 multi={f'{h*5}min':metrics(pr[mh>=h,:h],truth[mh>=h,:h],fitinfo['scales']) for h in (3,6,12)}
 score,components=validation_rollout_weighted_score(SimpleNamespace(aggregate_metrics=fitinfo['one_step']['aggregate_metrics']),multi,objective_weights=PRO['metric']['horizon_weights'])
 finite_one=all(isinstance(v,(int,float)) and np.isfinite(v) for v in fitinfo['one_step']['aggregate_metrics'].values())
 gates={'finite_one_step_metrics':finite_one,'one_step_finite_physical':fitinfo['one_step_finite_physical_pass'],'all_horizons_finite_physical':all(m['safety_gate_passed'] for m in multi.values()),'action_response':fitinfo['action_response']['eligible'],'finite_score':bool(np.isfinite(score)),'model_scalar_batch_parity':True,'reload':fitinfo['reload_pass']}
 eligible=fitinfo['name']!='persistence' and all(gates.values())
 np.savez_compressed(out/'predictions.npz',start_row_id=np.array(ids.index,dtype=str),max_horizon=mh,predictions=pr,truth=truth)
 result={'crop':crop,'label':label,'name':fitinfo['name'],'seed':fitinfo['seed'],'score':score,'components':components,'rollout':multi,'eligible':eligible,'gates':gates,'model_path_parity':checks,'model_sha256':fitinfo['model_sha256'],'execution_sha256':sha(D/'execution_binding.json'),'window_registry_sha256':sha(P/crop/'validation_windows.parquet'),'stopped_invalid_paths':int(dead.sum()),'full_registry_evaluated':True,'seconds':time.monotonic()-begin}
 write(out/'completed.json',result);print('EVALUATED',crop,label,result['score'],eligible,flush=True);return result

def select(crop):
 rows=[];failures=[]
 for name in ORDER:
  choices=[]
  for seed in PRO['candidates'][crop][name]['seeds']:
   label=name+(f'_seed_{seed}' if name=='mlp' else '');file=D/crop/label/'rollout/completed.json'
   if file.exists():
    v=json.loads(file.read_text())
    if v['eligible']:choices.append(v)
   else:
    assert (D/crop/label/'failure.json').exists(),('incomplete',crop,label);failures.append(label)
  if choices:
   choices.sort(key=lambda x:(x['score'],x['seed'] or 0));rows.append(choices[len(choices)//2] if name=='mlp' else choices[0])
 basefile=D/crop/'persistence/rollout/completed.json';base=json.loads(basefile.read_text()) if basefile.exists() else None
 rows.sort(key=lambda x:(x['score'],x['label']));leader=rows[0] if rows else None
 result={'crop':crop,'status':'BLOCKED','selected':None,'leader':leader['label'] if leader else None,'eligible_ranking':[{'label':r['label'],'score':r['score']} for r in rows],'persistence_score':base['score'] if base else None,'failures':failures,'execution_sha256':sha(D/'execution_binding.json')}
 if base is None or base['score'] is None:result['reason']='Persistence evaluation failed'
 elif leader is None:result['reason']='No eligible learned candidate'
 elif not leader['score']<base['score']:result['reason']='Strict Persistence improvement guard failed'
 else:
  label=leader['label'];name=leader['name'];seed=leader['seed'] if name=='mlp' else None
  try:
   rep=fit(crop,name,seed,True);rr=evaluate(crop,label,True)
   p=D/crop/label;a=pd.read_parquet(p/'fit/validation_predictions.parquet').to_numpy(float);b=pd.read_parquet(p/'repeat/validation_predictions.parquet').to_numpy(float)
   np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-8)
   with np.load(p/'rollout/predictions.npz') as aa,np.load(p/'repeat_rollout/predictions.npz') as bb:
    np.testing.assert_array_equal(aa['start_row_id'],bb['start_row_id']);np.testing.assert_array_equal(aa['max_horizon'],bb['max_horizon']);np.testing.assert_allclose(aa['predictions'],bb['predictions'],rtol=1e-6,atol=1e-8)
    maxroll=float(np.max(np.abs(aa['predictions']-bb['predictions'])))
   assert rr['eligible'] and np.isclose(rr['score'],leader['score'],rtol=1e-6,atol=1e-8)
   result.update(status='PASS',selected=label,algorithm=name,score=leader['score'],margin_vs_persistence=base['score']-leader['score'],gates={**leader['gates'],'strict_persistence_improvement':True,'fresh_same_seed_repeat':True},model_artifact=str((p/'fit/model.pkl').relative_to(R)),model_sha256=leader['model_sha256'],repeat_model_sha256=rep['model_sha256'],reproducibility={'seed':rep['seed'],'one_step_max_abs_diff':float(np.max(np.abs(a-b))),'rollout_max_abs_diff':maxroll,'reload':True,'rtol':1e-6,'atol':1e-8},schema_sha256=sha(P/'schema.json'),scaler_sha256=sha(P/crop/'observation_scaler.json'),train_registry_sha256=sha(P/crop/'train_registry.parquet'),window_registry_sha256=sha(P/crop/'validation_windows.parquet'),evaluation_code_sha256=sha(Path(__file__)),dependencies=VERSIONS,reason='Frozen score minimum; all gates + strict Persistence guard + repeat/reload passed; pending handoff/support/Step16')
  except Exception as e:result.update(reason='Selected repeat/reload failed',repeat_error=repr(e));traceback.print_exc()
 write(D/crop/'selected_model_manifest.json',result);return result

def run(crop):
 lock(True);assert json.loads((D/'batch_scalar_parity.json').read_text())['passed']
 job=D/crop/'job.lock';job.parent.mkdir(exist_ok=True);fd=os.open(job,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
 try:
  for name in ORDER:
   for seed in PRO['candidates'][crop][name]['seeds']:
    label=name+(f'_seed_{seed}' if name=='mlp' else '')
    if (D/crop/label/'failure.json').exists():continue
    try:fit(crop,name,seed);evaluate(crop,label)
    except Exception as exc:
     traceback.print_exc();write(D/crop/label/'failure.json',{'type':type(exc).__name__,'message':str(exc),'execution_sha256':sha(D/'execution_binding.json'),'retained_failure_no_retuning':True});print('FAILED',crop,label,repr(exc),flush=True)
  s=select(crop);lock(True);print('SELECTED',crop,s['status'],s['selected'],flush=True)
 finally:job.unlink()

def report():
 lock(True);sels={c:json.loads((D/c/'selected_model_manifest.json').read_text()) for c in CROPS};rows=[]
 for c in CROPS:
  for name in ORDER:
   for seed in PRO['candidates'][c][name]['seeds']:
    label=name+(f'_seed_{seed}' if name=='mlp' else '');p=D/c/label
    row={'crop':c,'label':label,'name':name}
    for key,filename in [('fit','fit/completed.json'),('evaluation','rollout/completed.json'),('failure','failure.json')]:
     if (p/filename).exists():row[key]=json.loads((p/filename).read_text())
    rows.append(row)
 blocked=[c for c,v in sels.items() if v['status']!='PASS']
 status='TRANSITION 4CH RESELECTION — PASS / NEW MODELS SELECTED' if not blocked else ('TRANSITION 4CH RESELECTION — BLOCKED' if len(blocked)==3 else 'TRANSITION 4CH RESELECTION — PARTIAL / '+', '.join(blocked)+' BLOCKED')
 write(D/'candidate_scores.json',{'status':status,'candidates':rows,'selected':sels,'execution_sha256':sha(D/'execution_binding.json'),'restrictions_observed':True})
 lines=['# Transition 4CH fixed-candidate reselection','',status,'','Canonical Python 3.12.3 + frozen dependency lock. Canonical Phase0 only, fresh Train fits; no Validation fitting, Test/P0/live data, HPO, support, Step16 or PPO.','', '| crop | selected | score | Persistence | margin | gate |','|---|---|---:|---:|---:|---|']
 for c,v in sels.items():lines.append(f"| {c} | {v.get('selected')} | {v.get('score')} | {v.get('persistence_score')} | {v.get('margin_vs_persistence')} | {v['status']} |")
 lines+=['','## Every fixed candidate / seed','', '| crop | label | score | eligible | 1-step | 15min | 30min | 60min | drift |','|---|---|---:|---|---:|---:|---:|---:|---:|']
 for row in rows:
  ev=row.get('evaluation',{});co=ev.get('components',{});lines.append('| '+' | '.join(str(x) for x in [row['crop'],row['label'],ev.get('score'),ev.get('eligible',False),*[co.get(k) for k in ['one_step','15min','30min','60min','drift']]])+' |')
 lines+=['','Per-channel raw MAE/RMSE/R²/nRMSE, endpoint, signed drift, p90/CVaR90, every horizon, physical/finite/action-response gates and failures are retained in candidate_scores.json and each candidate fit/rollout completed.json. All windows are scored; scalar checks are first/middle/last 12-step paths per candidate, not a scoring subsample.','', '## Historical winners (reference only)','', '| crop | old representative T/RH/CO₂ | new T1/T2/RH1/RH2 |','|---|---|---|']
 for c,old in [('melon','XGBoost'),('cucumber','ExtraTrees'),('strawberry','LightGBM')]:lines.append(f"| {c} | {old} | {sels[c].get('algorithm','BLOCKED')} |")
 lines+=['','The prediction problem and target normalization changed; old/new aggregate scores are not numerically equivalent. No preference was given to old winners.','', '## Binding and limitations','']
 for c,v in sels.items():lines += [f"- {c}: {v['status']}; {v.get('reason')}; model SHA256 `{v.get('model_sha256')}`."]
 lines+=['','Melon/strawberry retain legacy_command_snapshot_assumption; cucumber uses api_transmitted_command. Phase0 admission and historical context covariates remain fixed. Selected outputs do not replace any runtime/production handoff.','', 'Next step only after all crops PASS: new 4CH handoff/support/Step16 planning. Nothing downstream was executed.']
 (D/'final_transition_4ch_reselection.md').write_text('\n'.join(lines)+'\n');print(status,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['freeze','parity','run','report','all'],required=True);ap.add_argument('--crop',choices=CROPS);args=ap.parse_args()
 if args.stage=='freeze':freeze()
 elif args.stage=='parity':parity()
 elif args.stage=='run':run(args.crop)
 elif args.stage=='report':report()
 else:
  freeze();parity()
  jobs=[]
  for c in CROPS:
   log=(D/f'{c}_execution.log').open('w');jobs.append((subprocess.Popen([EXPECTED,'-B',str(Path(__file__)),'--stage','run','--crop',c],stdout=log,stderr=subprocess.STDOUT),log))
  codes=[]
  for job,log in jobs:codes.append(job.wait());log.close()
  assert not any(codes),('worker failure',codes)
  report()
