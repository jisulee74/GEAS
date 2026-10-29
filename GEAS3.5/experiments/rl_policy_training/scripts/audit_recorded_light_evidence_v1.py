"""Recheck frozen recorded states and summarize residual rollout OOD; no fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import sys,json,gzip
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.recorded_light_env_v1 import load_recorded_light_env
from geas35.experiments.rl.step19_5_diagnostics import ART,OUT,read_json,write_json,verify_inputs
DEST=ROOT/ART/'step19_5_light_contract_v1'
def main():
 manifest=read_json(ROOT/OUT/'input_manifest.json');verify_inputs(ROOT,manifest);result=[]
 for crop in ('melon','cucumber'):
  env=load_recorded_light_env(project_root=ROOT,crop=crop,split='validation',protocol_manifest_path=ROOT/ART/'step15_v1_4/rl_protocol_manifest.json')
  legacy=ROOT/OUT/'jobs'/f'{crop}__validation__recorded-one-step__representative_0__42__524288__0'
  n=severe=0;maximum=0.;top=Counter();examples=[]
  for ordinal,index in enumerate(manifest['registries'][crop]['validation']['indices']):
   ep=env.episodes[index];frame=env.frame.iloc[ep['start']:ep['final_observation']]
   scores=env.support.scores(frame.loc[:,list(env.observation_columns)],frame.loc[:,list(env.action_columns)])
   old=[json.loads(line) for line in gzip.open(legacy/'episodes'/f'{ordinal:04d}.jsonl.gz','rt')]
   assert len(old)==len(frame)
   for kind in ('state','joint'):
    before=np.array([r[kind+'_ood_score'] for r in old]);after=scores[kind]
    np.testing.assert_allclose(after,before,rtol=1e-6,atol=1e-8)
    maximum=max(maximum,float(np.max(np.abs(after-before))))
   n+=len(frame);severe+=int(np.sum(scores['state']>env.support.config['thresholds']['state_severe']))
   trace=[json.loads(line) for line in gzip.open(DEST/f'{crop}_validation/episodes/{ordinal:04d}.jsonl.gz','rt')]
   for row in trace:
    if row['info'].get('termination_reason')=='severe_state_ood':
     components=row['state_distance']['squared_distance_components'];rank=sorted(components,key=components.get,reverse=True)
     top[rank[0]]+=1;examples.append({'episode_index':index,'step':row['step'],'top_components':{k:components[k] for k in rank[:5]},'light_sum_component':components['obs_derived_light_sum']})
  result.append({'crop':crop,'recorded_steps':n,'recorded_state_severe_steps':severe,'recorded_state_severe_rate':severe/n,'old_new_max_score_error':maximum,'residual_top_component_counts':dict(top),'residual_examples':examples})
 verify_inputs(ROOT,manifest)
 write_json(DEST/'recorded_state_and_residual_evidence.json',{'results':result,'original_inputs_unchanged':True,'test_accessed':False})
 print(json.dumps([{k:v for k,v in r.items() if k!='residual_examples'} for r in result],indent=2))
if __name__=='__main__':main()
