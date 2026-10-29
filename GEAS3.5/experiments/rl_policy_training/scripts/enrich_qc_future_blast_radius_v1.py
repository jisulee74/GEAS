"""Evidence enrichment only; no replacements, model loading, or metrics."""
from pathlib import Path
import json,hashlib,gzip,datetime
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[3];O=R/'experiments/rl_policy_training/artifacts/qc_future_interpolation_blast_radius_v1';B=R/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1';U=R/'experiments/rl_policy_training/artifacts/step16_b60_r6_v1';T=R/'experiments/transition_model_selection/artifacts/command_carry_stale_b60_r6_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
logs=[];hashes={}
def rd(p,cols=None):
 assert 'test' not in p.name.lower();hashes[str(p.relative_to(R))]=sha(p);logs.append({'path':str(p.relative_to(R)),'columns':cols,'purpose':'lineage evidence enrichment only'})
 return pd.read_parquet(p,columns=cols) if p.suffix=='.parquet' else json.loads(p.read_text())
def main():
 write('enrichment_code_lock.json',{'sha256':sha(Path(__file__))})
 obs=rd(B/'schema.json')['observation_columns'];fs=[];extra=[];runinfo=[];allobs=[];look=[];usage=[]
 selected={'melon':'xgboost','cucumber':'extra_trees','strawberry':'knn'}
 for crop in selected:
  for split in ['train','validation']:
   tag=f'{crop}_{split}';q=rd(B/crop/f'{split}_prepared_qc.parquet');b=rd(B/crop/f'{split}_physical.parquet');q.reg_date=pd.to_datetime(q.reg_date);b.reg_date=pd.to_datetime(b.reg_date);b.target_timestamp=pd.to_datetime(b.target_timestamp)
   ed=pd.read_parquet(O/f'{tag}_feature_lineage.parquet')
   try:ev=pd.read_csv(O/f'{tag}_source_events.csv.gz')
   except pd.errors.EmptyDataError:ev=pd.DataFrame()
   side=pd.read_parquet(O/f'{tag}_transition_sidecar.parquet')
   keys=['crop','series_id','segment_id','reg_date'];idx=pd.MultiIndex.from_frame(q[keys]);cur=idx.get_indexer(pd.MultiIndex.from_frame(b[keys]));tar=idx.get_indexer(pd.MultiIndex.from_frame(b[keys[:-1]+['target_timestamp']].rename(columns={'target_timestamp':'reg_date'})))
   # Audit exact recorded integration, never alter any frame values or write datasets.
   sec=q.reg_date.groupby(q.reg_date.dt.date).diff().dt.total_seconds().fillna(0).clip(lower=0)
   dli=(pd.to_numeric(q.obs_outdoor_light).fillna(0).clip(lower=0)*sec*2.02).groupby(q.reg_date.dt.date).cumsum()/1e6
   dli_ok=np.isclose(dli.iloc[cur],b.obs_derived_dli_mol_m2,rtol=1e-6,atol=1e-8)
   extra.append({'crop':crop,'split':split,'DLI_recorded_integral_rows':len(b),'matches':int(dli_ok.sum())})
   vals=[]
   for role,positions in [('current',cur),('target',tar)]:
    mapping={}
    for j,k in enumerate(positions):mapping.setdefault(k,[]).append(j)
    for row in ed.itertuples():
     for j in mapping.get(row.recipient_position,[]):
      event=ev.iloc[row.event_id];feature=row.feature;value=float(b.iloc[j][('next_' if role=='target' else '')+feature]);base=float(q.iloc[row.recipient_position][feature]);expected=base
      if feature=='obs_derived_vent_loss_proxy':
       rr=b.iloc[j];prefix='next_' if role=='target' else '';expected=float(rr[prefix+'obs_prev_vent_pct'])*max(float(rr[prefix+'obs_indoor_temp_c'])-float(rr[prefix+'obs_outdoor_temp_c']),0)*(1-np.clip(float(rr[prefix+'obs_outdoor_light'])/1000,0,1))
      vals.append({'event_id':row.event_id,'row_id':b.iloc[j].row_id,'role':role,'current_timestamp':b.iloc[j].reg_date,'observation_timestamp':b.iloc[j].target_timestamp if role=='target' else b.iloc[j].reg_date,'source_timestamp':event.current_timestamp,'right_endpoint_timestamp':event.right_timestamp,'source_QC_value':event.qc_value,'feature':feature,'prepared_QC_value':base,'physical_observation_value':value,'B_expected_value':expected,'mapped_match':bool(np.isclose(value,expected,rtol=1e-6,atol=1e-8)),'direct':row.direct,'propagated':row.propagated,'lag_minutes':row.lag_minutes,'right_endpoint_after_recipient':row.right_endpoint_after_recipient})
   vv=pd.DataFrame(vals);vv.to_parquet(O/f'{tag}_observation_value_lineage.parquet',index=False)
   if len(vv):
    for (role,feature),g in vv.groupby(['role','feature']):fs.append({'crop':crop,'split':split,'role':role,'feature':feature,'unique_rows':g.row_id.nunique(),'direct_rows':g.loc[g.direct,'row_id'].nunique(),'propagated_rows':g.loc[g.propagated,'row_id'].nunique(),'future_at_current_rows':g.loc[g.right_endpoint_after_recipient,'row_id'].nunique(),'value_comparisons':len(g),'value_mismatches':int((~g.mapped_match).sum()),'max_lag_minutes':g.lag_minutes.max()})
   if len(ev):
    for (var,method),g in ev.groupby(['variable','method']):look.append({'crop':crop,'split':split,'variable':var,'method':method,'QC_feature_rows':len(g),'eligible_feature_rows':int(g.eligible_current.sum()),'lookahead_p50':g.lookahead_min.median(),'lookahead_p90':g.lookahead_min.quantile(.9),'lookahead_max':g.lookahead_min.max()})
   # Count actual imputed representatives; no model loaded or predicted.
   for kind in ['temp','hum','co2']:
    col=f'in_{kind}_representative_source'
    if col in q:
     for source,g in q.groupby(col,dropna=False):usage.append({'crop':crop,'split':split,'representative':kind,'source':str(source),'QC_rows':len(g),'eligible_rows':int(np.isin(g.index,cur).sum())})
   if split=='train':
    done=rd(T/crop/selected[crop]/'fit/completed.json');assert done['train_rows']==len(b)
   else:
    done=rd(T/crop/selected[crop]/'fit/completed.json');assert done['validation_rows']==len(b)
    lookup=side.set_index('row_id');blookup=b.set_index('row_id');featuremap={}
    if len(vv):
     for rid,g in vv.loc[vv.role.eq('current')].groupby('row_id'):featuremap[rid]=set(g.feature)
    traced=[]
    for path in sorted((U/crop/'episodes').glob('*.jsonl.gz')):
     hashes[str(path.relative_to(R))]=sha(path);logs.append({'path':str(path.relative_to(R)),'purpose':'saved trace observation membership only; rewards/prediction performance not evaluated'})
     with gzip.open(path,'rt') as file:
      for line in file:
       x=json.loads(line);rid=x['row_id'];assert rid in lookup.index;y=lookup.loc[rid];features=featuremap.get(rid,set())
       equal=[c for c in features if np.isclose(x['current_physical'][obs.index(c)],float(blookup.loc[rid,c]),rtol=1e-6,atol=1e-8)]
       traced.append({'row_id':rid,'episode_index':x['episode_index'],'step':x['step'],'initial':x['step']==0,'executed':x['executed'],'recorded_current_direct':bool(y.direct),'recorded_current_propagated':bool(y.propagated),'recorded_current_any':bool(y.current_any),'same_value_lineage_features':','.join(sorted(equal)),'recorded_lineage_features':','.join(sorted(features))})
    tr=pd.DataFrame(traced);tr.to_parquet(O/f'{crop}_Step16_saved_trace_lineage.parquet',index=False)
    for name,g in [('all_current_including_pretermination',tr),('initial',tr.loc[tr.initial]),('executed_current',tr.loc[tr.executed])]:runinfo.append({'crop':crop,'subset':name,'trace_rows':len(g),'recorded_direct':int(g.recorded_current_direct.sum()),'recorded_propagated':int(g.recorded_current_propagated.sum()),'recorded_any':int(g.recorded_current_any.sum()),'same_stored_feature_value_rows':int(g.same_value_lineage_features.ne('').sum())})
   print('ENRICHED',tag,flush=True)
 pd.DataFrame(fs).to_csv(O/'feature_summary.csv',index=False);pd.DataFrame(extra).to_csv(O/'DLI_integral_verification.csv',index=False);pd.DataFrame(look).to_csv(O/'variable_lookahead_summary.csv',index=False);pd.DataFrame(usage).to_csv(O/'representative_source_usage.csv',index=False);pd.DataFrame(runinfo).to_csv(O/'Step16_trace_membership.csv',index=False)
 for p,h in hashes.items():assert sha(R/p)==h
 write('enrichment_input_hashes.json',hashes);write('enrichment_access_ledger.json',logs)
if __name__=='__main__':main()
