"""Protocol stage: metadata registry and discovery-only sensor lineage audit."""
from pathlib import Path
import json,hashlib,datetime
import pandas as pd
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[3];O=R/'experiments/rl_policy_training/artifacts/strawberry_domain_v2_execution_v1';B=R/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1';Q=B/'strawberry'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
allowed=set(json.loads((O/'input_allowlist.json').read_text())['paths']);ledger=json.loads((O/'access_ledger.json').read_text())
def read(name,columns,filters=None,kind='metadata_only'):
 p=Q/name;assert str(p) in allowed and 'validation' not in name and 'test' not in name
 ledger.append({'path':str(p),'columns':columns,'filters':str(filters),'kind':kind,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'E_feature_access':False});put('access_ledger.json',ledger)
 return pd.read_parquet(p,columns=columns,filters=filters)
def main():
 assert not (O/'lineage_preflight_summary.json').exists()
 put('lineage_code_lock.json',{'sha256':sha(Path(__file__)),'source_qc_sha256':sha(R/'offline_dataset_preparation/scripts/03_control_quality.py'),'scope':'Train registry metadata and D-only quality/provenance; E feature forbidden'})
 r=read('train_registry.parquet',['row_id','crop','series_id','segment_id','episode_id','reg_date','target_timestamp','source_confidence'])
 for c in ['reg_date','target_timestamp']:r[c]=pd.to_datetime(r[c])
 start=pd.Timestamp('2024-09-01');dend=pd.Timestamp('2024-12-01');cend=pd.Timestamp('2025-01-01');end=pd.Timestamp('2025-04-01')
 def st(t):return 'D' if start<=t<dend else 'C' if dend<=t<cend else 'E' if cend<=t<end else 'outside'
 r['stage']=r.reg_date.map(st);r['reason']='';r['sensor_history_max_timestamp']=pd.NaT;r['command_history_max_timestamp']=pd.NaT
 # Registry metadata never exposes sensor values. Whole episode boundary quarantine.
 for _,g in r.groupby(['crop','series_id','segment_id','source_confidence','episode_id']):
  lo=g.reg_date.min();hi=g.target_timestamp.max()
  if any(lo<b<=hi for b in [start,dend,cend,end]):r.loc[g.index,'reason']='episode_crosses_stage_boundary'
 for i,x in r.iterrows():
  if x.stage!='outside' and st(x.target_timestamp)!=x.stage:r.loc[i,'reason']+='|target_outside_period'
 r['quarantine']=r.reason.str.len().gt(0);r.to_parquet(O/'split_metadata_registry.parquet',index=False)
 filt=[('reg_date','>=',start),('reg_date','<',dend)]
 cat=json.loads((O/'metadata_catalog.json').read_text());qc_cols=cat['train_prepared_qc.parquet']['columns']
 flags=[c for c in qc_cols if c.endswith('_imputation_method') or c.endswith('_imputed_flag')]
 qc=read('train_prepared_qc.parquet',['crop','series_id','segment_id','episode_id','reg_date']+flags,filt,'D_quality_lineage_only')
 qc.reg_date=pd.to_datetime(qc.reg_date)
 pr=read('train_provenance.parquet',['row_id','reg_date','legacy_previous_timestamp','legacy_previous_previous_timestamp','history_clean','eligible','source_confidence'],filt,'D_command_timestamp_lineage')
 d=r.loc[r.stage.eq('D')&~r.quarantine].copy();d=d.merge(qc,on=['crop','series_id','segment_id','episode_id','reg_date'],how='left',validate='one_to_one',indicator=True)
 assert d._merge.eq('both').all()
 methods=[c for c in flags if c.endswith('_imputation_method')]
 hits=[]
 for c in methods:
  mask=d[c].astype(str).str.contains('interpolation',regex=False,na=False)
  for _,x in d.loc[mask].iterrows():hits.append({'row_id':x.row_id,'episode_id':x.episode_id,'reg_date':x.reg_date,'column':c,'method':x[c],'expected_boundary_quarantine':False,'future_endpoint_required_by_code':True})
 h=pd.DataFrame(hits,columns=['row_id','episode_id','reg_date','column','method','expected_boundary_quarantine','future_endpoint_required_by_code']);h.to_csv(O/'D_future_interpolation_evidence.csv',index=False)
 counts={'eligible_Train':len(r),'stage_calendar':r.stage.value_counts().to_dict(),'boundary_quarantine':r.loc[r.quarantine].stage.value_counts().to_dict(),'post_boundary':r.loc[~r.quarantine].stage.value_counts().to_dict()}
 put('preliminary_denominator_ledger.json',counts)
 put('lineage_preflight_summary.json',{'D_post_boundary_rows':len(d),'future_interpolation_rows':int(h.row_id.nunique()),'future_interpolation_by_method':h.groupby(['column','method']).size().to_dict() if False else [{'column':c,'method':m,'rows':len(g)} for (c,m),g in h.groupby(['column','method'])],'E_feature_access':0,'C_feature_access':0,'Validation_Test_access':0,'interpretation':'interpolation method explicitly uses later right endpoint; not merely timestamp range crossing','proceed_to_geometry':h.empty})
 print((O/'lineage_preflight_summary.json').read_text())
if __name__=='__main__':main()
