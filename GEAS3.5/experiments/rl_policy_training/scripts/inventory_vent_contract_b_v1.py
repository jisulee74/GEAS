"""Impact inventory only; does not generate B training data or fit artifacts."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.models.transition.causal_state import load_frozen_observation_scaler
D=ROOT/'experiments/rl_policy_training/artifacts/step19_5_vent_contract_b_v1'
protocol=json.loads((ROOT/'experiments/rl_policy_training/artifacts/step15_v1_4/rl_protocol_manifest.json').read_text());stats=[];hashes=[]
for crop in ('melon','cucumber','strawberry'):
 s=ROOT/protocol['crops'][crop]['observation_scaler']['path'];scaler,h=load_frozen_observation_scaler(s);assert h==protocol['crops'][crop]['observation_scaler']['sha256']
 for split in ('train','validation'):
  p=ROOT/f'offline_dataset_preparation/datasets/5_rl_dataset/{crop}/{split}.parquet';f=scaler.inverse_transform_frame(pd.read_parquet(p));hashes.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  t=pd.to_datetime(f.reg_date);known=t.diff().eq(pd.Timedelta(minutes=5))
  for c in ('crop','series_id','segment_id','episode_id'):known &= f[c].eq(f[c].shift())
  a=f.vent_pct.astype(float);previous=a.shift();known &= np.isfinite(previous)&previous.between(0,1)
  two=known & known.shift(fill_value=False)
  oldprev=f.obs_prev_vent_pct.astype(float)
  stats.append({'crop':crop,'split':split,'frozen_rows':len(f),'previous_candidate_rows_in_filtered_frame':int(known.sum()),'previous_unknown_in_filtered_frame':int((~known).sum()),'two_previous_candidates':int(two.sum()),'known_prev_feature_disagreement':int((known & ~np.isclose(oldprev,previous,rtol=1e-6,atol=1e-8)).sum()),'boundary_old_prev_equals_current':int((~known & np.isclose(oldprev,a,rtol=1e-6,atol=1e-8)).sum()),'note':'Geometry only: not final B eligibility; source timestamps and pre-filter QC history must be audited.'})
pd.DataFrame(stats).to_csv(D/'carry_geometry_inventory.csv',index=False)
(D/'input_manifest.json').write_text(json.dumps({'files':hashes,'test_accessed':False,'B_dataset_generated':False},indent=2)+'\n')
print(pd.DataFrame(stats).drop(columns='note').to_string(index=False))
