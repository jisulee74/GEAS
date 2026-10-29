from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v4/strawberry';D=ROOT/'experiments/transition_model_selection/artifacts/strawberry_co2_diagnosis_b_v1';source=ROOT/'offline_dataset_preparation/datasets/5_rl_dataset/strawberry/train.parquet';keys=['reg_date','series_id','segment_id','episode_id']
a=pd.read_parquet(source,columns=keys+['rl_valid_transition','obs_indoor_co2_ppm']);assert a.rl_valid_transition.astype(bool).all();assert len(a)==90861
q=pd.read_parquet(P/'train_prepared_qc.parquet');p=pd.read_parquet(P/'train_provenance.parquet');old=pd.MultiIndex.from_frame(a[keys]);qi=pd.MultiIndex.from_frame(q[keys]);assert old.is_unique and qi.is_unique;assert old.isin(qi).all();mask=qi.isin(old);high=q.obs_indoor_co2_ppm.ge(1500).to_numpy();new=p.eligible.to_numpy(bool)
scaler_path=source.with_name('observation_scaler.json');sc=json.loads(scaler_path.read_text());physical=a.obs_indoor_co2_ppm.to_numpy(float)*sc['scale']['obs_indoor_co2_ppm']+sc['mean']['obs_indoor_co2_ppm']
recorded=q.set_index(keys).loc[old,'obs_indoor_co2_ppm'].to_numpy(float)
np.testing.assert_allclose(physical,recorded,rtol=1e-6,atol=1e-8)
result={'A_physical_co2_matches_QC':True,'A_scaler_sha256':hashlib.sha256(scaler_path.read_bytes()).hexdigest(),'old_A_train':len(a),'new_B_train':int(new.sum()),'overlap':int((mask&new).sum()),'old_A_lost_to_B':int((mask&~new).sum()),'new_B_not_in_A':int((~mask&new).sum()),'high_co2_A_train':int((mask&high).sum()),'high_co2_B_train':int((new&high).sum()),'high_co2_A_lost_to_B':int((mask&~new&high).sum()),'high_co2_QC_already_excluded_by_A':int((~mask&high).sum()),'reference_path':str(source.relative_to(ROOT)),'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'Test_accessed':False,'fit_performed':False}
(D/'A_B_train_membership.json').write_text(json.dumps(result,indent=2)+'\n')
x=q[keys+['obs_indoor_co2_ppm']].copy();x['in_A_train']=mask;x['in_B_train']=new;x['B_reason']=p.reason;x.to_parquet(D/'A_B_train_membership.parquet',index=False)
print(json.dumps(result,indent=2))
