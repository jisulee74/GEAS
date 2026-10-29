"""Diagnostic-only causal detector. No production environment or QC integration."""
from dataclasses import dataclass,asdict
from datetime import datetime,timedelta
import math
SIGNATURES={'A':('in_co2','in_co2_2'),'B':('in_co2','in_co2_2','in_temp','in_hum','out_light')}
CUTOFFS=(30,60,120,180)
@dataclass
class Detector:
 signature:str
 previous_timestamp:datetime|None=None
 start_timestamp:datetime|None=None
 identity:tuple|None=None
 values:tuple|None=None
 run_id:int=0
 def dump(self):
  d=asdict(self)
  for k in ('previous_timestamp','start_timestamp'):
   if d[k] is not None:d[k]=d[k].isoformat()
  return d
 @classmethod
 def restore(cls,d):
  d=dict(d)
  for k in ('previous_timestamp','start_timestamp'):
   if d[k] is not None:d[k]=datetime.fromisoformat(d[k])
  for k in ('identity','values'):
   if d[k] is not None:d[k]=tuple(d[k])
  return cls(**d)
 def update(self,*,timestamp,crop,series,segment,source,raw_present,raw_values,raw_valid=True):
  identity=(crop,series,segment,source)
  try:values=tuple(float(raw_values[c]) for c in SIGNATURES[self.signature]);valid=all(math.isfinite(v) for v in values)
  except (ValueError,TypeError,KeyError):values=None;valid=False
  reason=None
  if not raw_present:reason='raw_absent'
  elif not raw_valid or not valid:reason='invalid_signature'
  elif any(v is None or str(v)=='' for v in identity):reason='unknown_source_or_identity'
  if reason:
   self.previous_timestamp=self.start_timestamp=self.identity=self.values=None
   return dict(run_id=None,run_start=None,elapsed_minutes=None,input_valid=False,reason=reason,**{f'stale_{c}':False for c in CUTOFFS})
  if self.previous_timestamp is None:reason='start_or_after_invalid'
  elif timestamp-self.previous_timestamp!=timedelta(minutes=5):reason='timestamp_gap_or_nonmonotonic'
  elif identity!=self.identity:reason='crop_series_segment_source_change'
  elif values!=self.values:reason='signature_changed'
  else:reason='continue'
  if reason!='continue':self.run_id+=1;self.start_timestamp=timestamp
  self.previous_timestamp=timestamp;self.identity=identity;self.values=values
  elapsed=(timestamp-self.start_timestamp).total_seconds()/60
  return dict(run_id=self.run_id,run_start=self.start_timestamp,elapsed_minutes=elapsed,input_valid=True,reason=reason,**{f'stale_{c}':elapsed>=c for c in CUTOFFS})
