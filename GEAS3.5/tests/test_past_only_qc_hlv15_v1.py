import unittest
import numpy as np
import pandas as pd
from geas35.rl.past_only_qc_hlv15_v1 import HLV15,derived_replay

class HLVContract(unittest.TestCase):
 def test_age_chain(self):
  m=HLV15();t=pd.Timestamp('2024-01-01 10:00')
  a=[m.step(t+pd.Timedelta(minutes=i),'sensor',100 if i==0 else np.nan,i==0) for i in (0,5,10,15,20)]
  self.assertEqual([x['method'] for x in a],['RAW','HLV','HLV','HLV','INVALID'])
  self.assertTrue(all(x['evidence_timestamp']==t for x in a))
  self.assertEqual(m.step(t+pd.Timedelta(minutes=21),'sensor',120,True)['value'],120)
 def test_exact_age(self):
  for seconds,method in [(899,'HLV'),(900,'HLV'),(901,'INVALID')]:
   m=HLV15();t=pd.Timestamp('2024-01-01');m.step(t,'a',3,True)
   self.assertEqual(m.step(t+pd.Timedelta(seconds=seconds),'a',0,False)['method'],method)
 def test_boundaries(self):
  t=pd.Timestamp('2024-01-01')
  for kwargs in [dict(identity='b'),dict(provenance=False),dict(discontinuity=True)]:
   m=HLV15();m.step(t,'a',5,True);args=dict(timestamp=t+pd.Timedelta(minutes=5),identity='a',raw=0,raw_valid=False);args.update(kwargs)
   self.assertEqual(m.step(**args)['method'],'INVALID')
  for delta in [0,-5]:
   m=HLV15();m.step(t,'a',1,True)
   self.assertEqual(m.step(t+pd.Timedelta(minutes=delta),'a',2,True)['method'],'INVALID')
   self.assertEqual(m.step(t+pd.Timedelta(minutes=10),'a',0,False)['method'],'INVALID')
 def test_prefix_suffix(self):
  t=pd.Timestamp('2024-01-01')
  def replay(values):
   m=HLV15();return [m.step(t+pd.Timedelta(minutes=5*i),'a',v,np.isfinite(v)) for i,v in enumerate(values)]
  xs=[10,np.nan,np.nan,50,np.nan,80]
  full=replay(xs)
  for n in range(1,len(xs)+1):
   self.assertEqual(str(full[:n]),str(replay(xs[:n])))
   self.assertEqual(str(full[:n]),str(replay(xs[:n]+[999]*4)[:n]))
 def test_daily_and_branches(self):
  times=pd.to_datetime(['2024-01-01 23:40','2024-01-01 23:45','2024-01-01 23:50','2024-01-02 00:00','2024-01-02 00:05'])
  base=pd.DataFrame(dict(reg_date=times,crop='s',series_id='x',segment_id='a',obs_indoor_temp_c=20.,obs_target_day_temp_c=24.,obs_target_night_temp_c=12.,out_light_sum=0.,obs_derived_clear_sky_sum=10.,unrelated=7.))
  rec=pd.DataFrame([dict(variable=v,value=x) for v in ['out_light','out_windsp'] for x in ([100,np.nan,100,100,100] if v=='out_light' else [1]*5)])
  got=derived_replay(base,rec)
  self.assertTrue(got.obs_derived_dli_mol_m2.iloc[1:3].isna().all())
  self.assertEqual(got.obs_derived_dli_mol_m2.iloc[3],0.)
  self.assertTrue(got.obs_derived_heating_degree_minutes.iloc[1:3].isna().all())
  self.assertTrue(np.isfinite(got.obs_derived_heating_degree_minutes.iloc[3:]).all())
  self.assertTrue(got.obs_derived_light_sum.eq(0).all())
  self.assertTrue(got.unrelated.eq(7).all())
  base['current_target_temp_c']=23.;base['current_target_temp_min_c']=21.;base['current_target_temp_max_c']=25.
  override=derived_replay(base,rec)
  self.assertTrue(override.obs_current_target_temp_c.eq(23).all())
  self.assertTrue(override.obs_current_target_temp_min_c.eq(21).all())
  self.assertTrue(np.isfinite(override.obs_derived_heating_degree_minutes).all())
  base['out_light_sum']=np.nan
  fallback=derived_replay(base,rec)
  self.assertTrue(fallback.obs_derived_light_sum.iloc[1:3].isna().all())
  self.assertEqual(fallback.obs_derived_light_sum.iloc[3],0.)

if __name__=='__main__':unittest.main()
