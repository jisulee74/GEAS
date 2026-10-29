"""Reproduce frozen reference selection; diagnostic unsampled-distance comparison only."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.recorded_light_env_v1 import load_recorded_light_env
from geas35.experiments.rl.step19_5_diagnostics import ART,OUT,read_json,write_json,verify_inputs
DEST=ROOT/ART/'step19_5_residual_ood_v1'
def main():
 original=read_json(ROOT/OUT/'input_manifest.json');verify_inputs(ROOT,original);results=[]
 for crop in ('melon','cucumber'):
  envs={s:load_recorded_light_env(project_root=ROOT,crop=crop,split=s,protocol_manifest_path=ROOT/ART/'step15_v1_4/rl_protocol_manifest.json') for s in ('train','validation')}
  env=envs['train'];support=env.support;cfg=support.config;cols=list(env.observation_columns)
  train=env.frame.loc[env.frame.rl_valid_transition.astype(bool)].reset_index(drop=True);val=envs['validation'].frame.loc[envs['validation'].frame.rl_valid_transition.astype(bool)]
  episodes=np.array(sorted(train.episode_id.astype(str).unique()));rng=np.random.default_rng(cfg['seed']);rng.shuffle(episodes)
  mask=train.episode_id.astype(str).isin(set(episodes[:cfg['calibration_episode_count']])).to_numpy()
  full=np.flatnonzero(~mask);cal=np.flatnonzero(mask);ref=full
  if len(ref)>cfg['reference_rows']:ref=np.sort(rng.choice(ref,cfg['reference_rows'],replace=False))
  if len(cal)>cfg['calibration_rows']:cal=np.sort(rng.choice(cal,cfg['calibration_rows'],replace=False))
  np.testing.assert_array_equal(train.iloc[ref][cols].to_numpy(dtype=float),support.reference_state_raw)
  np.testing.assert_array_equal(train.iloc[ref][list(env.action_columns)].to_numpy(dtype=float),support.reference_action_raw)
  th=cfg['thresholds']['state_severe'];w=support.state_weights
  def score(x,nn):return nn.kneighbors(x[cols].to_numpy(dtype=float)*w)[0][:,-1]
  baseline=score(val,support._state_nn);cal_scores=score(train.iloc[cal],support._state_nn)
  nn=NearestNeighbors(n_neighbors=support.k,n_jobs=1).fit(train.iloc[full][cols].to_numpy(dtype=float)*w)
  comparison=score(val,nn)
  result={'crop':crop,'reference_selection_exact_match':True,'reference_pool_rows':len(full),'stored_reference_rows':len(ref),'calibration_rows':len(cal),'calibration_severe_rate':float(np.mean(cal_scores>th)),'validation_rows':len(val),'validation_severe_stored_reference':int(np.sum(baseline>th)),'validation_severe_unsampled_pool_same_threshold':int(np.sum(comparison>th)),'interpretation':'Diagnostic distance sensitivity, not a new calibrated support or permission to change threshold.'}
  # Quantify marginal shifts vs all Train, separately from reference sampling.
  trp=env.scaler.inverse_transform_frame(train.copy());vp=env.scaler.inverse_transform_frame(val.copy());rows=[]
  for c in cols:
   lo,hi=trp[c].min(),trp[c].max();rows.append({'feature':c,'train_min':float(lo),'train_max':float(hi),'validation_min':float(vp[c].min()),'validation_max':float(vp[c].max()),'validation_outside_train':int(((vp[c]<lo-1e-8)|(vp[c]>hi+1e-8)).sum())})
  pd.DataFrame(rows).to_csv(DEST/f'{crop}_full_train_range_comparison.csv',index=False)
  results.append(result);print(result,flush=True)
 verify_inputs(ROOT,original);write_json(DEST/'support_sampling_evidence.json',{'results':results,'original_inputs_unchanged':True,'test_accessed':False})
if __name__=='__main__':main()
