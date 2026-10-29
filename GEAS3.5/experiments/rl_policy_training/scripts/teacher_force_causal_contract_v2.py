"""Full frozen-episode teacher forcing, no model prediction or training."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,gzip
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.causal_contract_env_v2 import load_causal_contract_env,VERSION
from geas35.experiments.rl.step19_5_diagnostics import ART,OUT,read_json,write_json,atomic_episode,digest,sha256,lock,verify_inputs
D=ROOT/ART/'step19_5_causal_contract_v2'
def main():
 with lock(D/'teacher.lock'):
  manifest=read_json(ROOT/OUT/'input_manifest.json');verify_inputs(ROOT,manifest);allresults=[]
  for crop in ('melon','cucumber'):
   for split in ('validation','train'):
    env=load_causal_contract_env(project_root=ROOT,crop=crop,split=split,protocol_manifest_path=ROOT/ART/'step15_v1_4/rl_protocol_manifest.json');physical=env.scaler.inverse_transform_frame(env.frame.copy());_,targets=env.model._require_fitted()
    features=[c for c in env.observation_columns if c.startswith('obs_derived_') or c in ('obs_current_vpd_kpa','obs_current_dewpoint_c','obs_current_condensation_margin_c')]
    for ordinal,index in enumerate(manifest['registries'][crop][split]['indices']):
     ep=env.episodes[index];path=D/f'{crop}_{split}/teacher/{ordinal:04d}.json'
     signature=digest({'version':VERSION,'script':sha256(Path(__file__)),'environment':sha256(ROOT/'src/geas35/rl/causal_contract_env_v2.py'),'crop':crop,'split':split,'episode':index,'source':sha256(ROOT/f'offline_dataset_preparation/datasets/03_quality_controlled/{crop}/{split}.parquet')})
     def operation(stream):
      up=env.state_updater;current=env.frame.iloc[ep['start']];up.reset(current);maxerr={c:0. for c in features};first={};n=0;prehistory=len(up._history)-1
      for pos in range(ep['start'],ep['final_observation']+1):
       now=up.current_physical;truth=physical.iloc[pos];step=pos-ep['start'];values={}
       for c in features:
        a=float(now[c]);b=float(truth[c]);err=abs(a-b);maxerr[c]=max(maxerr[c],err);values[c]={'runtime':a,'recorded':b}
        if c not in first and not np.isclose(a,b,rtol=1e-6,atol=1e-8):first[c]={'step':step,'timestamp':str(truth.reg_date),'runtime':a,'recorded':b}
       stream.write(json.dumps({'step':step,'timestamp':str(truth.reg_date),'values':values})+'\n');n+=1
       if pos<ep['final_observation']:
        nextrow=physical.iloc[pos+1];current=up.advance(current_scaled=current,recorded_next_scaled=env.frame.iloc[pos+1],prediction_physical={c:float(nextrow[c]) for c in targets},action=env.frame.iloc[pos][list(env.action_columns)].astype(float).to_dict())
      return {'crop':crop,'split':split,'episode_index':index,'steps_observed':n,'prehistory_rows':prehistory,'max_abs_errors':maxerr,'first_divergence':first,'expected_equal_pass':not any(c!='obs_derived_vent_loss_proxy' for c in first),'vent_excluded_reason':'timestamp contract unresolved'}
     result=atomic_episode(path,signature,operation);allresults.append(result)
     print(crop,split,index,'teacher',result['expected_equal_pass'],list(result['first_divergence']),flush=True)
     write_json(D/'teacher_status.json',{'status':'running','completed':len(allresults)})
  verify_inputs(ROOT,manifest);write_json(D/'teacher_summary.json',{'version':VERSION,'episodes':allresults,'all_expected_equal_pass':all(r['expected_equal_pass'] for r in allresults),'original_inputs_unchanged':True});write_json(D/'teacher_status.json',{'status':'complete','completed':len(allresults)})
if __name__=='__main__':main()
