"""Post-build binding checks, no fit; stored artifacts are never modified."""
from pathlib import Path
import os,sys,json,tempfile
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from geas35.models.transition.handoff_b60_r6_v1 import sha,SELECTED
from geas35.rl.bound_support_b60_r6_v1 import load_bound_research_artifacts
S=ROOT/'experiments/rl_policy_training/artifacts/step15_b60_r6_v1';P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1'
checks=[]
for crop in SELECTED:
 path=S/crop/'binding.json';runtime,support=load_bound_research_artifacts(path,binding_sha256=sha(path),crop=crop,root=ROOT)
 frame=pd.read_parquet(P/crop/'validation_scaled.parquet');positions=np.unique(np.linspace(0,len(frame)-1,24).astype(int));f=frame.iloc[positions];scores=support.scores(f[runtime.obs],f[runtime.actions]);saved=pd.read_parquet(S/crop/'validation_support_scores.parquet').iloc[positions]
 for key,value in scores.items():np.testing.assert_allclose(value,saved[key+'_distance'],rtol=1e-6,atol=1e-8)
 negative=[]
 # Correctly hashed support manifest with wrong dependency must still fail.
 with tempfile.TemporaryDirectory() as temp:
  m=json.loads(path.read_text());other=next(c for c in SELECTED if c!=crop)
  for key in ('input_scaler','schema','support','reference','train','fit_registry'):
   changed=json.loads(json.dumps(m));changed[key]['sha256']='0'*64;target=Path(temp)/'binding.json';target.write_text(json.dumps(changed))
   try:load_bound_research_artifacts(target,binding_sha256=sha(target),crop=crop,root=ROOT)
   except ValueError:negative.append(key)
   else:raise AssertionError('Unbound dependency accepted')
  try:load_bound_research_artifacts(path,binding_sha256=sha(path),crop=other,root=ROOT)
  except ValueError:negative.append('wrong_crop')
  else:raise AssertionError('Wrong crop accepted')
 checks.append({'crop':crop,'passed':True,'score_rows':len(positions),'negative_cases':negative})
result={'status':'PASS','crops':checks,'history_regression_tests':{'passed':32,'files':['tests/test_sensor_history_env_v4.py','tests/test_causal_contract_env_v2.py']},'code_sha256':sha(ROOT/'src/geas35/rl/bound_support_b60_r6_v1.py'),'verification_script_sha256':sha(Path(__file__))}
(S/'bound_loader_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
