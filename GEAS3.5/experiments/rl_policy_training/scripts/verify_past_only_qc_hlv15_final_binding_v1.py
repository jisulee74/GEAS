"""Bind final boundary-hardened implementation; require identical full-stream output."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'))
from geas35.rl.past_only_qc_hlv15_v1 import source_replay,derived_replay
O=R/'experiments/rl_policy_training/artifacts/past_only_qc_hlv15_b60_r6_v1';B=O.parent/'vent_command_carry_b60_r6_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=json.loads((O/'input_manifest.json').read_text())
 finalpaths=[R/p for p in m['code_hashes']]+[Path(__file__),R/'tests/test_past_only_qc_hlv15_evidence.py']
 binding=dict(initial_code_hashes=m['code_hashes'],final_code_hashes={str(p.relative_to(R)):sha(p) for p in finalpaths},reason='boundary hardening: timestamp high-water mark after reversal; reject explicit imputation evidence; derived-invalid ledger excludes pre-existing command unknown; no actual population change permitted',results=[])
 (O/'final_implementation_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
 for crop in ('melon','cucumber','strawberry'):
  for split in ('train','validation'):
   raw=pd.read_parquet(R/f'offline_dataset_preparation/datasets/02_split/{crop}/{split}.parquet')
   base=pd.read_parquet(B/crop/f'{split}_prepared_qc.parquet')
   rec=source_replay(raw,base);saved=pd.read_parquet(O/crop/f'{split}_reconstruction.parquet')
   pd.testing.assert_frame_equal(rec,saved)
   new=derived_replay(base,rec);saved_new=pd.read_parquet(O/crop/f'{split}_prepared_qc.parquet')
   pd.testing.assert_frame_equal(new,saved_new)
   binding['results'].append(dict(crop=crop,split=split,reconstruction_mismatches=0,derived_mismatches=0,rows=len(base)))
   (O/'final_implementation_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
   print(crop,split,'final implementation exact parity',flush=True)
 binding['pass']=True
 (O/'final_implementation_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
if __name__=='__main__':main()
