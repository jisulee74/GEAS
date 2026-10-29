from pathlib import Path
import os,sys,json,pickle
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from geas35.rl.vent_command_memory_b_v3 import OBS,ACTIONS
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry';A=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1/strawberry/lightgbm';D=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_diagnosis_b_v1'
with (A/'fit/model.pkl').open('rb') as f:model=pickle.load(f)
b=pd.read_parquet(P/'validation_physical.parquet');x=pd.read_parquet(P/'validation_scaled.parquet')[OBS+ACTIONS];mask=b.obs_indoor_co2_ppm.ge(1500);est=model.estimators_['obs_indoor_co2_ppm'];leaf=est.booster_.predict(x.loc[mask],pred_leaf=True,num_threads=1);residual=est.predict(x.loc[mask]);dump=est.booster_.dump_model();counts=[]
for j,tree in enumerate(dump['tree_info']):
 target=int(leaf[0,j]);stack=[tree['tree_structure']]
 while stack:
  n=stack.pop()
  if 'leaf_index' in n:
   if n['leaf_index']==target:counts.append(int(n['leaf_count']))
  else:stack.extend([n['left_child'],n['right_child']])
tr=pd.read_parquet(P/'train_physical.parquet');result={'validation_high_rows':int(mask.sum()),'unique_high_leaf_paths':len(np.unique(leaf,axis=0)),'trees':leaf.shape[1],'high_predicted_delta_min':float(residual.min()),'high_predicted_delta_max':float(residual.max()),'high_predicted_delta_std':float(residual.std()),'min_child_samples':est.get_params()['min_child_samples'],'train_high_rows':int(tr.obs_indoor_co2_ppm.ge(1500).sum()),'high_path_leaf_training_count_min':min(counts),'high_path_leaf_training_count_max':max(counts),'fit_performed':False,'interpretation':'observed high Validation states share one leaf path, hence same physical residual; does not establish universal model-family limitation'}
(D/'high_co2_leaf_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
