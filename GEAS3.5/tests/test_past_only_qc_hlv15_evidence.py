import unittest
import numpy as np
import pandas as pd
from geas35.rl.past_only_qc_hlv15_v1 import HLV15,source_replay,derived_replay

class RawEvidence(unittest.TestCase):
 def frame(self):
  f=pd.DataFrame(dict(reg_date=pd.date_range('2024-01-01',periods=6,freq='5min'),crop='strawberry',series_id='s',segment_id='g',source_segment_id='iot97_seg1',source_file_path='source',iot_data_idx=97.,idx=np.arange(6,dtype=float),out_light=100.,out_windsp=3.,out_light_ffill_flag=0,out_windsp_ffill_flag=0))
  q=f.copy()
  for v in ('out_light','out_windsp'):
   q[v+'_missing_flag']=0;q[v+'_rule_outlier_flag']=0
  return f,q
 def test_reversal_cannot_reset_time_watermark(self):
  m=HLV15();t=pd.Timestamp('2024-01-01 10:00')
  m.step(t,'a',5,True)
  for minutes in (-10,-5,0):
   self.assertEqual(m.step(t+pd.Timedelta(minutes=minutes),'a',6,True)['method'],'INVALID')
  self.assertEqual(m.step(t+pd.Timedelta(minutes=5),'a',0,False)['method'],'INVALID')
 def test_imputation_evidence_excluded(self):
  for suffix in ('_imputed_flag','_ai_outlier_flag','_ai_imputed_flag','_interpolated_flag','_reconstruction_flag'):
   f,q=self.frame();f['out_light'+suffix]=1
   self.assertTrue(source_replay(f,q).query("variable == 'out_light'").method.eq('INVALID').all())
 def test_synthetic_never_evidence(self):
  f,q=self.frame();f.loc[1:4,'idx']=np.nan;f.loc[1:4,'iot_data_idx']=np.nan
  f.loc[1:4,'out_light']=999.;f.loc[1:4,'out_light_ffill_flag']=1
  r=source_replay(f,q);g=r[r.variable.eq('out_light')]
  self.assertEqual(g.method.tolist(),['RAW','HLV','HLV','HLV','INVALID','RAW'])
  self.assertEqual(g.value.iloc[1:4].tolist(),[100]*3)
  self.assertTrue(g.evidence_timestamp.iloc[:5].eq(f.reg_date.iloc[0]).all())
 def test_no_event_or_imputation_trigger(self):
  f,q=self.frame();q.loc[1:,'out_light_rule_outlier_flag']=1
  a=source_replay(f,q)
  q['out_light_imputation_method']='future interpolation';q['out_light_imputed_flag']=1
  pd.testing.assert_frame_equal(a,source_replay(f,q))
 def test_identity_unknown_and_crop_series_segment_source(self):
  f,q=self.frame();f.loc[0,'iot_data_idx']=np.nan
  self.assertEqual(source_replay(f,q).iloc[0].method,'INVALID')
  for c in ('crop','series_id','segment_id','source_segment_id','source_file_path'):
   f,q=self.frame();f.loc[1,c]='other';f.loc[1,'idx']=np.nan;f.loc[1,'iot_data_idx']=np.nan
   self.assertEqual(source_replay(f,q).iloc[2].method,'INVALID')
 def test_derived_prefix_and_future_change(self):
  f,q=self.frame();f['obs_indoor_temp_c']=20.;f['obs_target_day_temp_c']=24.;f['obs_target_night_temp_c']=12.;f['out_light_sum']=np.nan;f['obs_derived_clear_sky_sum']=10.
  r=source_replay(f,q);full=derived_replay(f,r)
  for n in (1,3,5):
   pd.testing.assert_frame_equal(full.iloc[:n],derived_replay(f.iloc[:n],r.iloc[:2*n]))
   suffix=r.copy();suffix.loc[suffix.position.ge(n),'value']=9000
   pd.testing.assert_frame_equal(full.iloc[:n],derived_replay(f,suffix).iloc[:n])

if __name__=='__main__':unittest.main()
