"""Protocol-bound metadata-only preflight. No observation/action values read."""
from pathlib import Path
import sys,json,hashlib,datetime,platform,importlib.metadata
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[3]
O=R/'experiments/rl_policy_training/artifacts/strawberry_domain_v2_execution_v1'
L=R/'experiments/rl_policy_training/artifacts/strawberry_domain_v2_execution_lock_v1'
B=R/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str)+'\n')
def main():
 expected=(L/'protocol.sha256').read_text().split()[0];assert sha(L/'preregistration_lock_ko.md')==expected
 O.mkdir(exist_ok=False)
 write(O/'protocol_lock.json',{'sha256':expected,'path':str(L/'preregistration_lock_ko.md'),'verified':True})
 (O/'protocol.sha256').write_text((L/'protocol.sha256').read_text())
 paths=[B/'schema.json',B/'runtime_contract.json']+[B/'strawberry'/n for n in ['train_registry.parquet','train_physical.parquet','train_provenance.parquet','train_prepared_qc.parquet']]
 write(O/'input_allowlist.json',{'paths':[str(p) for p in paths],'denied':['validation','test','scaler statistics','model','support'],'E_feature_read_permitted':False})
 write(O/'preflight_code_lock.json',{'sha256':sha(Path(__file__)),'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow']},'python':platform.python_version(),'locked_before_metadata_reads':True})
 log=[];catalog={}
 for p in paths:
  assert 'validation' not in p.name and 'test' not in p.name
  if p.suffix=='.parquet':
   meta=pq.ParquetFile(p);catalog[p.name]={'rows':meta.metadata.num_rows,'row_groups':meta.metadata.num_row_groups,'columns':meta.schema_arrow.names,'types':{c:str(meta.schema_arrow.field(c).type) for c in meta.schema_arrow.names if any(x in c for x in ['time','date','source','history','lineage'])}};kind='footer_schema_only'
  else:catalog[p.name]=json.loads(p.read_text());kind='schema_contract_definitions'
  log.append({'path':str(p),'kind':kind,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'E_feature_access':False})
 write(O/'metadata_catalog.json',catalog);write(O/'access_ledger.json',log)
 write(O/'exposure_ledger.json',{'prior_full_Train_time_aggregates_known':True,'prior_Validation_shadow_known':True,'E_name':'sealed internal temporal evaluation after v2 lock','E_feature_access':0,'Validation_Test_access':0})
 print(json.dumps({k:({'rows':v['rows'],'lineage_columns':[c for c in v['columns'] if any(x in c for x in ['time','date','source','history','lineage','missing','imput','ffill'])]} if 'columns' in v else v) for k,v in catalog.items()},ensure_ascii=False,default=str))
if __name__=='__main__':main()
