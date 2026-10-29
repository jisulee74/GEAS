from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from geas35.models.transition.command_carry_b_v1 import P,D,OBS,ACTIONS,write,sha
from geas35.models.transition.batch_sensor_rollout_b_v1 import BatchSensorUpdater
from geas35.rl.sensor_history_env_v4 import SensorHistoryStateUpdater
class Identity:
 def transform_frame(self,f):return f.copy()
 def inverse_transform_frame(self,f):return f.copy()
rows=[]
for crop in ('melon','cucumber','strawberry'):
 for split in ('train','validation'):
  base=pd.read_parquet(P/crop/f'{split}_prepared_qc.parquet');b=pd.read_parquet(P/crop/f'{split}_physical.parquet');windows=pd.read_parquet(P/crop/f'{split}_windows.parquet');ids=windows.loc[windows.horizon.eq(12),'start_row_id'].to_numpy();chosen=ids[np.unique(np.linspace(0,len(ids)-1,8).astype(int))];starts=b.index[b.row_id.isin(chosen)].to_numpy()
  pos=pd.Index(base.reg_date).get_indexer(b.iloc[starts].reg_date)
  for mode in ('teacher','perturbed'):
   batch=BatchSensorUpdater(base,pos,b.iloc[starts][OBS].to_numpy(float));scalar=[]
   for j,i in enumerate(pos):
    initial=base.iloc[i].copy();initial[OBS]=b.iloc[starts[j]][OBS];u=SensorHistoryStateUpdater(Identity(),base.iloc[max(0,i-12):i]);u.reset(initial);scalar.append(u)
   maxerr=0.
   for step in range(12):
    trans=b.iloc[starts+step];truth=trans[['next_'+c for c in OBS]].to_numpy(float);actions=trans[ACTIONS].to_numpy(float);pred=truth[:,:3].copy()
    if mode=='perturbed':pred=batch.state[:,:3]+np.array([.13,-.17,1.7])
    got=batch.advance(np.arange(len(starts)),pred,truth,actions,trans.target_timestamp.to_numpy(dtype='datetime64[ns]').astype('int64'))
    for j,u in enumerate(scalar):
     nxt=base.iloc[pos[j]+step+1].copy();nxt[OBS]=truth[j]
     expected=u.advance(current_scaled=u.current_physical,recorded_next_scaled=nxt,prediction_physical=dict(zip(OBS[:3],pred[j])),action=dict(zip(ACTIONS,actions[j])))
     np.testing.assert_allclose(got[j],expected[OBS].to_numpy(float),rtol=1e-6,atol=1e-8);maxerr=max(maxerr,float(np.max(np.abs(got[j]-expected[OBS].to_numpy(float)))))
   rows.append({'crop':crop,'split':split,'mode':mode,'windows':len(starts),'steps':12*len(starts),'max_abs_error':maxerr,'pass':True});print(rows[-1],flush=True)
write(D/'batch_runtime_parity.json',{'rows':rows,'passed':all(x['pass'] for x in rows),'module_sha256':sha(ROOT/'src/geas35/models/transition/batch_sensor_rollout_b_v1.py'),'rtol':1e-6,'atol':1e-8})
