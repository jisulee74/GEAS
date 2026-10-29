"""Frozen B transition inputs and fresh-fit jobs. No Test or HPO entry point."""
from pathlib import Path
import json,hashlib,pickle,time,os
import numpy as np
import pandas as pd
from geas35.rl.vent_command_memory_b_v3 import OBS,ACTIONS
from geas35.rl.mdp_v1 import MdpV1ObservationScaler
from geas35.models.transition.base import TransitionDataset,TransitionTargetFormulation
from geas35.models.transition.registry import build_transition_model
from geas35.models.transition.evaluator import evaluate_transition_model_one_step
from geas35.models.transition.hyperparameters import transition_target_normalization_scales
from geas35.models.transition.action_response import evaluate_action_response
from geas35.models.transition.rollout import _physical_violation_rate
TARGETS=tuple(OBS[:3])
ROOT=Path(__file__).resolve().parents[4]
D=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1'
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str,allow_nan=True)+'\n');tmp.replace(p)
def protocol():
 f=D/'protocol.json';assert sha(f)==json.loads((D/'protocol_lock.json').read_text())['sha256'];return json.loads(f.read_text())
def load(crop,split):
 if crop not in ('melon','cucumber','strawberry') or split not in ('train','validation'):raise ValueError('Only fixed crop Train/Validation')
 sc=MdpV1ObservationScaler(**json.loads((P/crop/'observation_scaler.json').read_text()))
 frame=pd.read_parquet(P/crop/f'{split}_scaled.parquet');metadata=frame[['row_id','crop','reg_date','episode_id','segment_id','mdp_v1_rollout_id']].copy()
 # Existing action-response compares adjacent eligible rows; separate discontinuous runs.
 metadata['segment_id']=metadata.segment_id.astype(str)+'|run='+metadata.mdp_v1_rollout_id.astype(str)
 y=pd.DataFrame({t:frame[t.replace('obs_','target_next_',1)].to_numpy(float) for t in TARGETS})
 ds=TransitionDataset(frame[OBS+ACTIONS].reset_index(drop=True),y,tuple(OBS+ACTIONS),TARGETS,metadata,tuple(OBS),tuple(ACTIONS))
 return ds,sc

def fresh_fit(crop,name,seed=None,repeat=False):
 pro=protocol();spec=pro['candidates'][crop][name];params=json.loads(json.dumps(spec['params']))
 if seed is not None:params['random_state']=int(seed)
 label=name+(f'_seed_{seed}' if name=='mlp' else '')
 dest=D/crop/label/('repeat' if repeat else 'fit');dest.mkdir(parents=True,exist_ok=True)
 done=dest/'completed.json'
 if done.exists():
  x=json.loads(done.read_text());assert x['protocol_sha256']==sha(D/'protocol.json') and x['model_sha256']==sha(dest/'model.pkl');return x
 lock=dest/'job.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
 try:
  train,sc=load(crop,'train');val,_=load(crop,'validation');scales=transition_target_normalization_scales(val.y,TARGETS)
  model=build_transition_model(name,params=params)
  if name=='persistence':model.observation_scaler=sc
  else:model.configure_target_formulation(TransitionTargetFormulation.RESIDUAL_PHYSICAL_DELTA,observation_scaler=sc)
  t=time.perf_counter();model.fit(train,validation_dataset=None);elapsed=time.perf_counter()-t
  pred=model.predict(val).next_observation;one=evaluate_transition_model_one_step(model,val,normalization_scales=scales)
  action=evaluate_action_response(model,val)
  with (dest/'model.pkl.tmp').open('wb') as f:pickle.dump(model,f)
  (dest/'model.pkl.tmp').replace(dest/'model.pkl')
  with (dest/'model.pkl').open('rb') as f:loaded=pickle.load(f)
  np.testing.assert_allclose(loaded.predict(val).next_observation.to_numpy(float),pred.to_numpy(float),rtol=1e-6,atol=1e-8)
  pred.to_parquet(dest/'validation_predictions.parquet',index=False)
  result={'crop':crop,'name':name,'seed':params.get('random_state'),'params':params,'train_rows':len(train.x),'validation_rows':len(val.x),'fit_seconds':elapsed,'one_step':one.to_artifact(),'action_response':action.to_artifact(),'scales':scales,'one_step_physical_violation_rate':_physical_violation_rate(pred),'one_step_nan_inf':int((~np.isfinite(pred.to_numpy(float))).sum()),'reload_pass':True,'protocol_sha256':sha(D/'protocol.json'),'model_sha256':sha(dest/'model.pkl'),'scaler_sha256':sha(P/crop/'observation_scaler.json'),'from_scratch':True,'test_accessed':False}
  write(done,result);print('FIT COMPLETE',crop,label,flush=True);return result
 except Exception as e:
  write(dest/'failure.json',{'type':type(e).__name__,'message':str(e),'protocol_sha256':sha(D/'protocol.json')});raise
 finally:lock.unlink(missing_ok=True)
