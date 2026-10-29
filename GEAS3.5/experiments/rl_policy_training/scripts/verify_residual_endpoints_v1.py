"""Verify first crossing and both endpoint distance decompositions."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,gzip
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.recorded_light_env_v1 import load_recorded_light_env
from geas35.experiments.rl.step19_5_diagnostics import ART,OUT,read_json,write_json,verify_inputs,sha256
D=ROOT/ART/'step19_5_residual_ood_v1';allrows=[];checks=0
for crop in ('melon','cucumber'):
 for split in ('validation','train'):
  env=load_recorded_light_env(project_root=ROOT,crop=crop,split=split,protocol_manifest_path=ROOT/ART/'step15_v1_4/rl_protocol_manifest.json');cols=list(env.observation_columns);w=env.support.state_weights;threshold=env.support.config['thresholds']['state_severe']
  registry=read_json(ROOT/OUT/'input_manifest.json')['registries'][crop][split]['indices']
  for ep in json.loads((D/f'{crop}_summary.json').read_text()):
   if ep['split']!=split:continue
   ordinal=registry.index(ep['episode_index']);path=ROOT/ART/f'step19_5_light_contract_v1/{crop}_{split}/episodes/{ordinal:04d}.jsonl.gz';trace=[json.loads(l) for l in gzip.open(path,'rt')]
   scores=[r['info']['state_ood_score'] for r in trace];assert all(x<=threshold for x in scores[:-1]);assert scores[-1]>threshold
   for row in trace[-2:]:
    weighted=np.array(row['scaled_state'],dtype=float)*w;dist,idx=env.support._state_nn.kneighbors(weighted[None,:]);component=np.square(weighted-env.support._state_reference[idx[0,-1]])
    np.testing.assert_allclose(component.sum(),dist[0,-1]**2,rtol=1e-6,atol=1e-8);np.testing.assert_allclose(dist[0,-1],row['info']['state_ood_score'],rtol=1e-6,atol=1e-8)
    allrows.append({'crop':crop,'split':split,'episode_index':ep['episode_index'],'step':row['step'],'state_score':float(dist[0,-1]),'threshold':threshold,'is_terminal':row is trace[-1],'squared_distance_components':dict(zip(cols,map(float,component)))})
   checks+=1
write_json(D/'endpoint_distance_components.json',allrows)
verify_inputs(ROOT,read_json(ROOT/OUT/'input_manifest.json'))
write_json(D/'verification.json',{'verified_residual_episodes':checks,'verified_pre_and_terminal_rows':len(allrows),'first_crossing_verified':True,'distance_sums_and_trace_scores_match':True,'original_inputs_unchanged':True,'test_accessed':False})
print('Verified',checks,'episodes;',len(allrows),'endpoint rows',flush=True)
