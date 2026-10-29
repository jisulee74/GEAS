"""Recompute trace scores/rewards/fallback and classify residual endpoints, no training."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,gzip
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.model_driven_env import load_model_driven_env_from_step15
from geas35.rl.mdp_v1 import compute_mdp_v1_reward,project_mdp_v1_action_constraints
from geas35.features.derived import vpd_kpa,dewpoint_c
from geas35.experiments.rl.step19_5_diagnostics import ART,OUT,read_json,write_json,sha256,verify_inputs
D=ROOT/ART/'step19_5_causal_contract_v2'
def lines(p):return [json.loads(l) for l in gzip.open(p,'rt')]
def sync(r):
 r=r.copy();t=pd.Series([r.obs_indoor_temp_c]);h=pd.Series([r.obs_indoor_humidity_pct]);r['obs_current_vpd_kpa']=float(vpd_kpa(t,h).iloc[0]);r['obs_current_dewpoint_c']=float(dewpoint_c(t,h).iloc[0]);r['obs_current_condensation_margin_c']=r.obs_indoor_temp_c-r.obs_current_dewpoint_c;return r

def main():
 assert read_json(D/'verification_summary.json')['status']=='complete'
 summaries=[];endpoints=[];parity=0
 for crop in ('melon','cucumber'):
  for split in ('validation','train'):
   env=load_model_driven_env_from_step15(project_root=ROOT,crop=crop,split=split,protocol_manifest_path=ROOT/ART/'step15_v1_4/rl_protocol_manifest.json');cols=list(env.observation_columns);actions=list(env.action_columns);phy=env.scaler.inverse_transform_frame(env.frame.copy());count=state=joint=rewardchecks=fallbackchecks=0;rewarderr=aliaserr=0.;trajectory_errors={c:[] for c in ('obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm')};paired_old={c:[] for c in trajectory_errors};paired_new={c:[] for c in trajectory_errors}
   for p in sorted((D/f'{crop}_{split}/episodes').glob('*.jsonl.gz')):
    records=lines(p);solar=lines(ROOT/ART/'step19_5_light_contract_v1'/p.relative_to(D));pos=records[0]['source_position'];prev=env.frame.iloc[pos][actions].astype(float).to_dict();prevprev=prev.copy()
    states=pd.DataFrame([r['scaled_state'] for r in records],columns=cols);acts=pd.DataFrame([r['info']['proposed_action'] for r in records]);scores=env.support.scores(states,acts)
    for j,r in enumerate(records):
     info=r['info'];pos=r['source_position'];current=pd.Series(r['physical_state'],index=cols);correct=sync(current)
     for c in trajectory_errors:
      err=float(current[c]-phy.iloc[pos][c]);trajectory_errors[c].append(err)
      if j<len(solar):paired_new[c].append(err);paired_old[c].append(float(solar[j]['physical_state'][cols.index(c)]-phy.iloc[pos][c]))
     for c in ('obs_current_vpd_kpa','obs_current_dewpoint_c','obs_current_condensation_margin_c'):
      error=abs(current[c]-correct[c]);aliaserr=max(aliaserr,error);np.testing.assert_allclose(current[c],correct[c],rtol=1e-6,atol=1e-8)
     for key in ('state','joint'):np.testing.assert_allclose(scores[key][j],info[key+'_ood_score'],rtol=1e-6,atol=1e-8)
     state+=int(scores['state'][j]>env.support.config['thresholds']['state_severe']);joint+=int(scores['joint'][j]>env.support.config['thresholds']['joint_severe']);count+=1
     if info.get('transition_prediction_physical'):
      nextrow=phy.iloc[pos+1].copy()
      for c,v in info['transition_prediction_physical'].items():nextrow[c]=v
      nextrow=sync(nextrow);action=info['executed_action'];reward,_=compute_mdp_v1_reward(nextrow,action,prev_action=prev,prev_prev_action=prevprev,config=env.mdp_config)
      error=abs(reward-info['base_reward']);rewarderr=max(rewarderr,error);np.testing.assert_allclose(reward,info['base_reward'],rtol=1e-6,atol=1e-8);rewardchecks+=1
      if info['safety_fallback_applied']:
       env._current_physical=current;env._current_scaled=env.frame.iloc[pos].copy();env._current_scaled.loc[cols]=np.asarray(r['scaled_state'],dtype=float)
       expected,_=project_mdp_v1_action_constraints(env._safe_fallback_action(),current,env.mdp_config)
       np.testing.assert_allclose([expected[c] for c in actions],[action[c] for c in actions],rtol=1e-6,atol=1e-8);fallbackchecks+=1
      prevprev=prev;prev=action
    last=records[-1]
    if last['info'].get('termination_reason')=='severe_state_ood':
     pos=last['source_position'];actual=pd.Series(last['physical_state'],index=cols);recorded_score=env.support.scores(env.frame.iloc[[pos]][cols],env.frame.iloc[[pos]][actions])['state'][0]
     endpoints.append({'crop':crop,'split':split,'episode_index':last['episode_index'],'step':last['step'],'timestamp':str(phy.iloc[pos].reg_date),'rollout_score':last['info']['state_ood_score'],'recorded_score':float(recorded_score),'recorded_already_severe':bool(recorded_score>env.support.config['thresholds']['state_severe']),'indoor_difference':{c:float(actual[c]-phy.iloc[pos][c]) for c in ('obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm')},'distance':last.get('state_distance'),'prior_actions_modified':sum(bool(r.get('recorded_action_modified')) for r in records),'low_vpd_runtime':float(actual.obs_derived_low_vpd_minutes_1h),'low_vpd_recorded':float(phy.iloc[pos].obs_derived_low_vpd_minutes_1h)})
    # Complete episodes saved before the window optimization must be identical.
    old=ROOT/ART/'step19_5_causal_contract_v2_incomplete_attempt1'/p.relative_to(D)
    if old.exists() and old.with_suffix('').with_suffix('.json').exists():
     before=lines(old);assert len(before)==len(records)
     for a,b in zip(before,records):
      np.testing.assert_allclose(a['physical_state'],b['physical_state'],rtol=1e-6,atol=1e-8);np.testing.assert_allclose(a['reward'],b['reward'],rtol=1e-6,atol=1e-8)
     parity+=1
   frozen=env.frame.loc[env.frame.rl_valid_transition.astype(bool)];frozen_scores=env.support.scores(frozen[cols],frozen[actions])
   recorded_counts={k:int(np.sum(v>env.support.config['thresholds'][k+'_severe'])) for k,v in frozen_scores.items()}
   summaries.append({'recorded_valid_rows':len(frozen),'recorded_severe_counts':recorded_counts,'crop':crop,'split':split,'executed':count,'state_severe':state,'joint_severe_proposed':joint,'base_reward_checks':rewardchecks,'fallback_checks':fallbackchecks,'max_base_reward_error':rewarderr,'max_alias_formula_error':aliaserr,'trajectory_error':{c:{'new_all_mae':float(np.mean(np.abs(v))),'common_prefix_old_mae':float(np.mean(np.abs(paired_old[c]))),'common_prefix_new_mae':float(np.mean(np.abs(paired_new[c]))),'new_all_max_abs':float(np.max(np.abs(v)))} for c,v in trajectory_errors.items()}})
 for record in read_json(D/'verification_manifest.json')['files']:assert sha256(ROOT/record['path'])==record['sha256']
 verify_inputs(ROOT,read_json(ROOT/OUT/'input_manifest.json'))
 write_json(D/'consistency_audit.json',{'summaries':summaries,'window_optimization_parity_episodes':parity,'original_inputs_unchanged':True});write_json(D/'residual_endpoints.json',endpoints)
 print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
