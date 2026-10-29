"""Quality metadata only; no controller or production feature changes."""
from dataclasses import dataclass
@dataclass
class Recovery:
 steps:int
 latched:bool=False
 count:int=0
 previous:tuple|None=None
 previous_identity:tuple|None=None
 previous_time:object=None
 frozen:tuple|None=None
 def update(self,t,identity,values,valid,base_stale):
  continuous=self.previous_time is not None and (t-self.previous_time).total_seconds()==300 and identity==self.previous_identity
  changed=continuous and values!=self.previous
  entered=base_stale and not self.latched
  released=False
  if base_stale:
   self.latched=True;self.frozen=values;self.count=0
  elif self.latched:
   # Every qualifying row must change relative to the immediately prior valid raw row.
   # Missing/gap/source switch resets progress, never silently clears the latch.
   qualify=valid and changed and values!=self.frozen
   self.count=self.count+1 if qualify else 0
   if self.count>=self.steps:self.latched=False;released=True;self.count=0
  else:self.count=0
  self.previous=values if valid else None;self.previous_time=t if valid else None;self.previous_identity=identity if valid else None
  return {'supervisor_blocked':self.latched,'recovery_count':self.count,'entered':entered,'released':released,'input_valid':valid,'quality_blocked':self.latched or not valid}
