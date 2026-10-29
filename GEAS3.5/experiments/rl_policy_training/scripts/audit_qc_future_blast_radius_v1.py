"""Read-only QC future-endpoint lineage audit; no fits, replacements or domain gates."""
from pathlib import Path
import os,json,hashlib,datetime,gzip
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3];B=R/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1';O=R/'experiments/rl_policy_training/artifacts/qc_future_interpolation_blast_radius_v1'
S=R/'experiments/rl_policy_training/artifacts/step15_b60_r6_v1';T=R/'experiments/transition_model_selection/artifacts/command_carry_stale_b60_r6_v1';U=R/'experiments/rl_policy_training/artifacts/step16_b60_r6_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
log=[];hashes={}
def read(p,cols=None):
 assert 'test' not in p.name.lower() and p.is_relative_to(R)
 assert any(p.is_relative_to(z) for z in [B,S,T,U])
 hashes[str(p.relative_to(R))]=sha(p);log.append({'path':str(p.relative_to(R)),'columns':cols,'purpose':'contamination lineage only; no domain-v2 evaluation','time':datetime.datetime.now(datetime.timezone.utc).isoformat()});write('access_ledger.json',log)
 return pd.read_parquet(p,columns=cols) if p.suffix=='.parquet' else pd.read_csv(p) if p.suffix=='.csv' else json.loads(p.read_text())
def main():
 O.mkdir(exist_ok=False)
 write('protocol.json',{'scope':'6 crop/split frozen populations; causal dependency lineage, not performance','Test':False,'domain_v2_C_E_gate':False,'exposure':'full strawberry Train source audit overlaps calendar C/E; QC values are exposed in this distinct audit, not domain E admission. Do not claim historically unseen E.','definitions':{'direct':'current observation depends on same-row interpolated source, including same-row derived dependencies','propagated':'current observation depends on earlier interpolated source','provenance':'dependency on offline future-filled source even after endpoint becomes available','still_future':'source right timestamp exceeds recipient timestamp','effect':'structural/active-branch dependency, not counterfactual numerical or performance change'},'classification':'A isolated=no substantial history or downstream fit/support involvement; B bounded but pipeline relevant=finite contained lineage yet official artifacts touched; C systemic=wide cross-crop/history spread requiring dataset-wide causal rebuild. Counts and caveats, no performance thresholds.'})
 write('code_lock.json',{'sha256':sha(Path(__file__)),'numpy':np.__version__,'pandas':pd.__version__})
 # Freeze previous domain-v2 and protocol outputs byte-wise, never interpret C/E metrics.
 original=[]
 for ns in ['strawberry_domain_v2_execution_v1','strawberry_domain_v2_execution_lock_v1']:
  for p in sorted((B.parent/ns).iterdir()):
   if p.is_file():original.append({'path':str(p.relative_to(R)),'sha256':sha(p)})
 write('v2_original_hashes.json',original)
 schema=read(B/'schema.json');obs=schema['observation_columns'];allsummary=[];lineages=[];windows=[];methods=[];groups=[];ai=[]
 for crop in ['melon','cucumber','strawberry']:
  for split in ['train','validation']:
   tag=f'{crop}_{split}';print('START',tag,flush=True)
   q=read(B/crop/f'{split}_prepared_qc.parquet');b=read(B/crop/f'{split}_physical.parquet');w=read(B/crop/f'{split}_windows.parquet')
   q=q.reset_index(drop=True);q.reg_date=pd.to_datetime(q.reg_date);b.reg_date=pd.to_datetime(b.reg_date);b.target_timestamp=pd.to_datetime(b.target_timestamp)
   assert q.reg_date.is_monotonic_increasing
   keys=['crop','series_id','segment_id','reg_date'];idx=pd.MultiIndex.from_frame(q[keys]);assert idx.is_unique
   cur=idx.get_indexer(pd.MultiIndex.from_frame(b[keys]));tar=idx.get_indexer(pd.MultiIndex.from_frame(b[keys[:-1]+['target_timestamp']].rename(columns={'target_timestamp':'reg_date'})));assert (cur>=0).all() and (tar>=0).all()
   mapcols={'out_temp':'obs_outdoor_temp_c','out_hum':'obs_outdoor_humidity_pct','out_windsp':'obs_outdoor_wind_speed','out_light':'obs_outdoor_light','out_light_sum':'obs_derived_light_sum','out_rain':'obs_rain_flag'}
   ev=[];runs=[];unknown=[]
   for mc in [c for c in q if c.endswith('_imputation_method')]:
    var=mc.removesuffix('_imputation_method')
    for method,g in q.groupby(mc,dropna=False):methods.append({'crop':crop,'split':split,'variable':var,'method':str(method),'QC_rows':len(g),'eligible_rows':int(np.isin(g.index,cur).sum())})
    for ep,g in q.groupby('episode_id',sort=False):
     positions=g.index.to_numpy();labels=g[mc].fillna('').astype(str).to_numpy();a=0
     while a<len(g):
      if 'interpolation' not in labels[a]:a+=1;continue
      end=a+1
      while end<len(g) and labels[end]==labels[a]:end+=1
      ids=positions[a:end];left=positions[a-1] if a>0 else None;right=positions[end] if end<len(g) else None
      if left is None or right is None:unknown.extend([{'position':int(i),'variable':var,'reason':'missing_endpoint_in_prepared_QC'} for i in ids]);a=end;continue
      lt=q.loc[left,'reg_date'];rt=q.loc[right,'reg_date'];lv=float(q.loc[left,var]);rv=float(q.loc[right,var]);runid=f'{tag}:{var}:{ids[0]}'
      rr={'run_id':runid,'variable':var,'episode_id':ep,'start':q.loc[ids[0],'reg_date'],'end':q.loc[ids[-1],'reg_date'],'gap_rows':len(ids),'past_timestamp':lt,'last_valid_past_value':lv,'right_timestamp':rt,'right_value':rv,'past_age_at_end_min':(q.loc[ids[-1],'reg_date']-lt).total_seconds()/60,'raw_value_status':'see per-row raw/missing/rule flags; no replacement'};runs.append(rr)
      for i in ids:
       t=q.loc[i,'reg_date'];f=(t-lt)/(rt-lt)
       expected=lv+f*(rv-lv)
       if 'circular' in labels[a]:expected=(lv+f*((rv-lv+180)%360-180))%360
       if var in ['out_light','out_light_sum']:expected=max(0,expected)
       stored=float(q.loc[i,var]);match=bool(np.isclose(expected,stored,rtol=1e-6,atol=1e-8));physpos=np.flatnonzero(cur==i)
       physical=float(b.iloc[physpos[0]][mapcols[var]]) if len(physpos) and var in mapcols else None
       ev.append({'event_id':len(ev),'run_id':runid,'position':int(i),'crop':crop,'split':split,'series_id':q.loc[i,'series_id'],'segment_id':q.loc[i,'segment_id'],'episode_id':ep,'current_timestamp':t,'variable':var,'method':labels[a],'left_timestamp':lt,'right_timestamp':rt,'lookahead_min':(rt-t).total_seconds()/60,'last_valid_past_value':lv,'left_age_min':(t-lt).total_seconds()/60,'qc_value':stored,'expected':expected,'formula_match':match,'physical_mapped_value':physical,'physical_match':None if physical is None else bool(np.isclose(stored,physical,rtol=1e-6,atol=1e-8)),'eligible_current':bool(len(physpos)),'raw_value':q.loc[i,var+'_raw_value'] if var+'_raw_value' in q else None,'missing_flag':q.loc[i,var+'_missing_flag'] if var+'_missing_flag' in q else None,'rule_outlier_flag':q.loc[i,var+'_rule_outlier_flag'] if var+'_rule_outlier_flag' in q else None})
      a=end
   e=pd.DataFrame(ev);e.to_csv(O/f'{tag}_source_events.csv.gz',index=False,compression='gzip');pd.DataFrame(runs).to_csv(O/f'{tag}_past_available_runs.csv',index=False);write(tag+'_unresolved_endpoints.json',unknown)
   assert not unknown,'Unresolved endpoints require explicit audit extension'
   # Dependency edges keep both structural ancestry and future-at-recipient state.
   edges=[];times=q.reg_date.to_numpy(dtype='datetime64[ns]');dates=q.reg_date.dt.date
   def explicit(i,cs):return any(c in q and pd.notna(q.loc[i,c]) for c in cs)
   for event in ev:
    i=event['position'];var=event['variable'];t=q.loc[i,'reg_date'];right=event['right_timestamp'];dest={}
    def add(js,features,kind):
     for j in js:
      for feature in features:
       if feature in obs:dest[(int(j),feature)]=kind
    if var in mapcols:add([i],[mapcols[var]],'direct')
    dayids=q.index[dates.eq(t.date())&(q.reg_date>=t)].to_numpy()
    # Dataset derived implementation integrates by calendar date; no new episode reset is invented.
    if var=='out_light':
     add([i],['obs_is_daytime','obs_derived_vent_loss_proxy'],'direct')
     same_day_before=q.index[dates.eq(t.date())&(q.reg_date<t)]
     if len(same_day_before):add(dayids,['obs_derived_dli_mol_m2'],'history')
     for j in dayids:
      if not explicit(j,['out_light_sum','light_sum_jcm2','measured_sum_jcm2']):
       if len(same_day_before):add([j],['obs_derived_light_sum','obs_derived_light_sum_ratio'],'history')
     targets=[]
     if not explicit(i,['current_target_temp_c','target_temp_c']):targets+=['obs_current_target_temp_c']
     for suffix in ['min','max']:
      if not explicit(i,[f'current_target_temp_{suffix}_c',f'target_temp_{suffix}_c',f'current_temp_{suffix}_c']):targets.append(f'obs_current_target_temp_{suffix}_c')
     add([i],targets,'direct')
     if 'obs_current_target_temp_c' in targets:add(dayids,['obs_derived_heating_degree_minutes','obs_derived_cooling_degree_minutes'],'history')
    elif var=='out_temp':add([i],['obs_derived_temp_diff_in_out_c','obs_derived_vent_loss_proxy'],'direct')
    elif var=='out_light_sum':add([i],['obs_derived_light_sum_ratio'],'direct')
    # Wind direction/rainfall/pressure have no selected 55 observation column.
    for (j,feature),kind in dest.items():
     edges.append({'event_id':event['event_id'],'source_position':i,'recipient_position':j,'feature':feature,'lag_minutes':(q.loc[j,'reg_date']-t).total_seconds()/60,'direct':j==i,'propagated':j>i,'right_endpoint_after_recipient':right>q.loc[j,'reg_date'],'source_variable':var})
   ed=pd.DataFrame(edges,columns=['event_id','source_position','recipient_position','feature','lag_minutes','direct','propagated','right_endpoint_after_recipient','source_variable']);ed.to_parquet(O/f'{tag}_feature_lineage.parquet',index=False)
   direct=np.zeros(len(q),bool);prop=direct.copy();strict=direct.copy()
   if len(ed):
    direct[ed.loc[ed.direct,'recipient_position'].unique()]=True;prop[ed.loc[ed.propagated,'recipient_position'].unique()]=True;strict[ed.loc[ed.right_endpoint_after_recipient,'recipient_position'].unique()]=True
   anymask=direct|prop
   side=b[['row_id','crop','series_id','segment_id','episode_id','reg_date','target_timestamp','source_confidence','mdp_v1_rollout_id']].copy()
   side['direct']=direct[cur];side['propagated']=prop[cur];side['propagated_only']=prop[cur]&~direct[cur];side['current_any']=anymask[cur];side['current_still_future']=strict[cur];side['target_any']=anymask[tar];side['target_still_future']=strict[tar];side['transition_any']=anymask[cur]|anymask[tar]
   side.to_parquet(O/f'{tag}_transition_sidecar.parquet',index=False)
   pos_by_id={rid:j for j,rid in enumerate(b.row_id)};win=[]
   for wr in w.itertuples():
    j=pos_by_id[wr.start_row_id];h=int(wr.horizon);loc=np.arange(j,j+h);assert len(loc)==h and b.iloc[loc].mdp_v1_rollout_id.nunique()==1 and b.iloc[loc[-1]].target_timestamp==wr.end_target_timestamp
    win.append({'start_row_id':wr.start_row_id,'horizon':h,'current_path_any':bool(side.current_any.iloc[loc].any()),'target_path_any':bool(side.target_any.iloc[loc].any()),'trajectory_any':bool(side.transition_any.iloc[loc].any()),'trajectory_still_future':bool((side.current_still_future.iloc[loc]|side.target_still_future.iloc[loc]).any())})
   win=pd.DataFrame(win);win.to_parquet(O/f'{tag}_window_lineage.parquet',index=False)
   for h,g in win.groupby('horizon'):windows.append({'crop':crop,'split':split,'horizon':int(h),'total':len(g),**{c:int(g[c].sum()) for c in ['current_path_any','target_path_any','trajectory_any','trajectory_still_future']}})
   for (source,series),g in side.groupby(['source_confidence','series_id']):groups.append({'crop':crop,'split':split,'source':source,'series':series,'total':len(g),**{c:int(g[c].sum()) for c in ['direct','propagated','propagated_only','current_any','current_still_future','target_any','transition_any']}})
   summary={'crop':crop,'split':split,'QC_rows':len(q),'eligible_rows':len(b),'QC_future_raw_rows':int(e.position.nunique()) if len(e) else 0,'QC_future_feature_rows':len(e),'eligible_future_raw_rows':int(e.loc[e.eligible_current,'position'].nunique()) if len(e) else 0,'formula_mismatch':int((~e.formula_match).sum()) if len(e) else 0,'eligible_direct':int(direct[cur].sum()),'eligible_propagated':int(prop[cur].sum()),'eligible_propagated_only':int((prop[cur]&~direct[cur]).sum()),'eligible_any':int(anymask[cur].sum()),'current_still_future':int(strict[cur].sum()),'target_any':int(anymask[tar].sum()),'transition_any':int((anymask[cur]|anymask[tar]).sum()),'max_propagation_minutes':float(ed.loc[ed.propagated,'lag_minutes'].max()) if ed.propagated.any() else 0,'max_eligible_propagation_minutes':float(ed.loc[ed.propagated&ed.recipient_position.isin(cur),'lag_minutes'].max()) if (ed.propagated&ed.recipient_position.isin(cur)).any() else 0,'lookahead_max':float(e.lookahead_min.max()) if len(e) else 0,'lookahead_p50':float(e.lookahead_min.median()) if len(e) else 0,'lookahead_p90':float(e.lookahead_min.quantile(.9)) if len(e) else 0};allsummary.append(summary)
   # Membership only, never load/fit model or support distances.
   def membership(name,ids,include_target=False):
    lookup=side.set_index('row_id');assert pd.Index(ids).isin(lookup.index).all();x=lookup.loc[list(ids)];lineages.append({'crop':crop,'split':split,'artifact':name,'rows':len(x),'direct':int(x.direct.sum()),'propagated':int(x.propagated.sum()),'current_any':int(x.current_any.sum()),'target_any':int(x.target_any.sum()),'transition_any':int(x.transition_any.sum())})
   if split=='train':
    fit=read(B/crop/'scaler_fit_registry.parquet',['row_id']);membership('scaler_fit',fit.row_id)
    membership('Transition_fit_all_B_Train_protocol_binding',b.row_id)
    sr=read(S/crop/'fit_row_registry.parquet',['row_id','role'])
    for role,g in sr.groupby('role'):membership('Step15_'+role,g.row_id)
   else:
    membership('Transition_Validation_one_step',b.row_id)
    ep=read(U/crop/'episode_summary.csv');starts=[];visited=[]
    for x in ep.itertuples():
     gg=side.loc[side.mdp_v1_rollout_id.astype(str).eq(str(x.rollout_id))];assert len(gg)>0
     starts.append(gg.row_id.iloc[0]);visited.extend(gg.row_id.iloc[:int(x.executed_steps)].tolist())
    membership('Step16_recorded_initial',starts);membership('Step16_recorded_at_executed_timestamps_NOT_simulated_states',visited)
   for c in [c for c in q if c.endswith('_imputed_flag') and c.startswith('in_')]:ai.append({'crop':crop,'split':split,'variable':c,'QC_imputed':int(pd.to_numeric(q[c],errors='coerce').fillna(0).ne(0).sum()),'eligible_imputed':int(pd.to_numeric(q.iloc[cur][c],errors='coerce').fillna(0).ne(0).sum())})
   pd.DataFrame(allsummary).to_csv(O/'population_summary.csv',index=False);pd.DataFrame(lineages).to_csv(O/'artifact_lineage.csv',index=False);pd.DataFrame(windows).to_csv(O/'window_summary.csv',index=False);pd.DataFrame(groups).to_csv(O/'source_series_summary.csv',index=False);pd.DataFrame(methods).to_csv(O/'method_usage.csv',index=False);pd.DataFrame(ai).to_csv(O/'AI_imputation_usage.csv',index=False)
   print('DONE',summary,flush=True)
 for p,h in hashes.items():assert sha(R/p)==h
 for x in original:assert sha(R/x['path'])==x['sha256']
 write('input_hashes.json',hashes);write('verification.json',{'input_hashes_unchanged':True,'domain_v2_outputs_unchanged':True,'Test_read':False,'no_performance_recalculation':True,'all_window_memberships_original_registry':True,'lineage_not_counterfactual_effect':True})
if __name__=='__main__':main()
