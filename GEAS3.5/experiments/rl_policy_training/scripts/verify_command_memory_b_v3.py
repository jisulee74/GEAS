"""Non-learning gate: real episode resets, longest rollouts, source invariants."""
from pathlib import Path
import sys,json,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.vent_command_memory_b_v3 import *
from geas35.rl.causal_contract_env_v2 import CausalContractStateUpdater
from geas35.rl.mdp_v1 import compute_mdp_v1_reward
D=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v3';OLD=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v2'
class Identity:
 def transform_frame(self,f):return f.copy()
 def inverse_transform_frame(self,f):return f.copy()
def write(p,x):p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n')
def verify(crop):
 dest=D/crop;out=dest/'verification';out.mkdir(exist_ok=False)
 for split in ('train','validation'):
  b=pd.read_parquet(dest/f'{split}_physical.parquet');base=pd.read_parquet(dest/f'{split}_prepared_qc.parquet');a=pd.read_parquet(dest/f'{split}_provenance.parquet');times={t:i for i,t in enumerate(base.reg_date)}
  tests=[];errors=[];traces=[];maxhist=0
  if len(b):
   # Every eligible original episode gets an actual causal reset/advance.
   for _,g in b.groupby(['crop','series_id','segment_id','episode_id'],sort=False):tests.append(('episode_reset',g.iloc[:1]))
   groups=list(b.groupby('mdp_v1_rollout_id',sort=False));_,longest=max(groups,key=lambda x:len(x[1]));tests.append(('longest_rollout',longest));tests.append(('midrun_reset',longest.iloc[len(longest)//2:]))
  for kind,g in tests:
   first=g.iloc[0];idx=times[first.reg_date];initial=base.iloc[idx].copy();initial[OBS]=first[OBS]
   past=base.iloc[max(0,idx-12):idx].copy();u=CausalContractStateUpdater(Identity(),past);u.reset(initial);current=initial
   for row in g.to_dict('records'):
    i=times[row['reg_date']];nxt=base.iloc[i+1].copy()
    for col in OBS:nxt[col]=row['next_'+col]
    pred={k:float(nxt[k]) for k in ('obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm')};action={k:float(row[k]) for k in ACTIONS}
    got=u.advance(current_scaled=current,recorded_next_scaled=nxt,prediction_physical=pred,action=action);maxhist=max(maxhist,len(u._history))
    actual=got[OBS].to_numpy(float);truth=nxt[OBS].to_numpy(float);ok=np.isclose(actual,truth,rtol=1e-6,atol=1e-8)
    for j in np.flatnonzero(~ok):errors.append({'test':kind,'timestamp':str(row['reg_date']),'feature':OBS[j],'actual':float(actual[j]) if np.isfinite(actual[j]) else None,'expected':float(truth[j]) if np.isfinite(truth[j]) else None})
    ar=a.iloc[i];rr=compute_mdp_v1_reward(got,action,prev_action={k:float(ar['previous_'+k]) for k in ACTIONS},prev_prev_action={k:float(ar['previous_previous_'+k]) for k in ACTIONS})[0]
    if not np.isclose(rr,row['reward'],rtol=1e-6,atol=1e-8):errors.append({'test':kind,'timestamp':str(row['reg_date']),'feature':'reward','actual':rr,'expected':row['reward']})
    traces.append({'test':kind,'timestamp':row['reg_date'],'next_timestamp':row['target_timestamp'],'episode_id':row['episode_id'],'history_length':len(u._history),'max_abs_feature_error':float(np.max(np.abs(actual-truth))),'passed':bool(ok.all())});current=got
  pd.DataFrame(traces).to_csv(out/f'{split}_teacher_trace.csv.gz',index=False,compression='gzip');write(out/f'{split}_teacher_errors.json',errors)
  # Input-level counterfactuals with actual events/snapshots; not only assembled columns.
  checks=[];source='api' if crop=='cucumber' else 'legacy'
  cy=pd.read_parquet(OLD/'raw'/f'{crop}_{split}_optimal_monitoring_copy.parquet');ev=pd.read_parquet(OLD/'raw'/f'{crop}_{split}_send_ds_control_api_history.parquet');snap=pd.read_parquet(dest/f'{split}_legacy_raw_snapshots.parquet') if source=='legacy' else None
  for pos in np.unique(np.linspace(0,len(b)-1,min(3,len(b))).astype(int)) if len(b) else []:
   t=b.iloc[pos].reg_date;i=times[t];lo=max(0,i-14);hi=min(len(base),i+4);sub=base.iloc[lo:hi].reset_index(drop=True);t0=sub.reg_date.min();t1=sub.reg_date.max()+pd.Timedelta(minutes=5)
   cc=cy.loc[cy.reg_date.ge(t0)&cy.reg_date.lt(t1)].copy();ee=ev.loc[ev.reg_date.ge(t0)&ev.reg_date.lt(t1)].copy();ss=snap.loc[snap.reg_date.ge(t0)&snap.reg_date.lt(t1)].copy() if snap is not None else None
   x,ax,_=build(sub,cc,ee,split,source=source,snapshots=ss)
   if not x.reg_date.eq(t).any():raise AssertionError('Actual invariant target lost on bounded reset')
   if source=='legacy':
    old=float(ss.loc[ss.reg_date.eq(t),'cont_skyl_vol'].iloc[0]);new=100. if old<50 else 0.;ss.loc[ss.reg_date.eq(t),'cont_skyl_vol']=new
   else:
    old=float(cc.loc[cc.reg_date.eq(t),'pred_ltw'].iloc[0]);new=100. if old<50 else 0.;cc.loc[cc.reg_date.eq(t),'pred_ltw']=new
    mask=ee.reg_date.ge(t)&ee.reg_date.lt(t+pd.Timedelta(minutes=1))&ee.channel.astype(str).eq('1');ee.loc[mask,'ratio']=str(new)
   y,ay,_=build(sub,cc,ee,split,source=source,snapshots=ss);xx=x.set_index('reg_date').loc[t];yy=y.set_index('reg_date').loc[t]
   np.testing.assert_allclose(xx[OBS].to_numpy(float),yy[OBS].to_numpy(float),rtol=1e-6,atol=1e-8)
   assert xx.vent_pct!=yy.vent_pct;assert yy.next_obs_prev_vent_pct==yy.vent_pct;assert yy.target_next_indoor_temp_c==xx.target_next_indoor_temp_c
   # Repeat complete bounded rebuild including reward/targets.
   y2,ay2,_=build(sub,cc,ee,split,source=source,snapshots=ss);assert fingerprint(y)==fingerprint(y2) and fingerprint(ay)==fingerprint(ay2)
   checks.append({'timestamp':str(t),'raw_input_current_command_changed':True,'current_observation_invariant':True,'next_carry_and_target_correct':True,'deterministic_full_rebuild':True})
  write(out/f'{split}_actual_invariants.json',checks)
  summary={'crop':crop,'split':split,'eligible_episodes':int(b.episode_id.nunique()),'episode_resets':sum(k=='episode_reset' for k,_ in tests),'longest_run_steps':max([len(g) for k,g in tests if k=='longest_rollout'],default=0),'midrun_reset_steps':max([len(g) for k,g in tests if k=='midrun_reset'],default=0),'teacher_steps':len(traces),'teacher_feature_reward_errors':len(errors),'max_history_rows':maxhist,'actual_raw_invariants':len(checks),'pass':len(b)>0 and not errors and len(checks)>0}
  write(out/f'{split}_summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--crop',choices=['melon','cucumber','strawberry'],required=True);verify(ap.parse_args().crop)
