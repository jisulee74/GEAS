from pathlib import Path
from datetime import datetime,timedelta
import json,hashlib
import numpy as np,pandas as pd
from stale_snapshot_detector_shadow_v1 import Detector,SIGNATURES,CUTOFFS
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_shadow_b_v1'
def event(i=0,**kw):
 x=dict(timestamp=datetime(2000,1,1,23,30)+timedelta(minutes=5*i),crop='c',series='s',segment='g',source='data_silla_enc|file|segment',raw_present=True,raw_values={c:0 for c in SIGNATURES['B']});x.update(kw);return x
for sig in SIGNATURES:
 d=Detector(sig);out=[d.update(**event(i)) for i in range(40)]
 for c in CUTOFFS:
  assert next(i for i,x in enumerate(out) if x[f'stale_{c}'])==c//5
  assert out[c//5]['elapsed_minutes']==c
 # Day crosses midnight; no episode argument exists in detector API.
 assert out[10]['elapsed_minutes']==50
 for field in ('crop','series','segment','source'):
  d=Detector(sig)
  for i in range(8):d.update(**event(i))
  z=d.update(**event(8,**{field:'other'}));assert z['elapsed_minutes']==0 and not z['stale_30']
 for altered in [event(8,raw_present=False),event(8,raw_valid=False),event(8,raw_values={**event()['raw_values'],'in_co2':np.nan}),event(8,raw_values={**event()['raw_values'],'in_co2':float('inf')}),event(8,source=None)]:
  d=Detector(sig)
  for i in range(8):d.update(**event(i))
  z=d.update(**altered);assert not z['input_valid'] and not z['stale_30'];z=d.update(**event(9));assert z['elapsed_minutes']==0
 d=Detector(sig)
 for i in range(8):d.update(**event(i))
 assert d.update(**event(9))['elapsed_minutes']==0
 assert d.update(**event(2))['elapsed_minutes']==0
 changed=event(10,raw_values={**event()['raw_values'],'in_co2':1});assert d.update(**changed)['elapsed_minutes']==0
 # Pure prefix invariance and serialize/reload without looking at future rows.
 d=Detector(sig);prefix=[d.update(**event(i)) for i in range(20)];saved=json.loads(json.dumps(d.dump()));restored=Detector.restore(saved)
 tail=[restored.update(**event(i)) for i in range(20,40)];assert prefix+tail==out
 changed_future=[event(i) if i<20 else event(i,raw_values={c:9999 for c in SIGNATURES['B']}) for i in range(40)]
 d=Detector(sig);mutated=[d.update(**x) for x in changed_future];assert mutated[:20]==prefix
# Replay every actual stream in two chunks, with state serialized at midpoint.
count=0
for path in sorted(D.glob('*_row_sidecar.parquet')):
 f=pd.read_parquet(path)
 for sig in SIGNATURES:
  d=Detector(sig)
  for i,r in enumerate(f.itertuples(index=False)):
   if i==len(f)//2:d=Detector.restore(json.loads(json.dumps(d.dump())))
   z=d.update(timestamp=r.reg_date.to_pydatetime(),crop=r.crop,series=r.series_id,segment=r.segment_id,source=r.raw_source if pd.notna(r.raw_source) else None,raw_present=pd.notna(r.raw_idx),raw_values={c:getattr(r,'raw_'+c) for c in SIGNATURES[sig]})
   for c in CUTOFFS:assert z[f'stale_{c}']==getattr(r,f'{sig}_{c}_causal_forward')
   expected=getattr(r,sig+'_elapsed_minutes');assert (pd.isna(expected) and z['elapsed_minutes'] is None) or expected==z['elapsed_minutes']
   count+=1
# Retrospective sidecar must reproduce prior raw sensitivity when extra source boundary has no impact.
prior=ROOT/'experiments/transition_model_selection/artifacts/stale_snapshot_sensitivity_b_v1';parity=[]
for path in sorted(D.glob('*_row_sidecar.parquet')):
 prefix=path.name.removesuffix('_row_sidecar.parquet');old=pd.read_parquet(prior/f'{prefix}_row_tags.parquet');f=pd.read_parquet(path);assert old.reg_date.equals(f.reg_date)
 for s in SIGNATURES:
  for c in CUTOFFS:parity.append({'stream':prefix,'signature':s,'cutoff':c,'different_rows':int((old[f'{s}_{c}'].to_numpy()!=f[f'{s}_{c}_confirmed_run_retrospective'].to_numpy()).sum())})
pd.DataFrame(parity).to_csv(D/'previous_sensitivity_mask_parity.csv',index=False)
(D/'causality_boundary_verification.json').write_text(json.dumps({'synthetic_checks':['exact elapsed thresholds','valid zero','midnight continuity','crop/series/segment/source reset','raw absence reset','explicit invalid/NaN/Inf reset','unknown source reset','gap/nonmonotonic reset','value change recovery','future suffix cannot affect prefix','JSON state restart parity'],'actual_replayed_updates':count,'actual_chunk_resume_exact':True,'future_information_used_by_online':False},indent=2)+'\n')
print('Causal and boundary checks passed:',count,'actual updates replayed.')
