from pathlib import Path
import os,json,hashlib,datetime
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3];B=R/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1';A=B.parent/'qc_future_interpolation_blast_radius_v1';O=B.parent/'past_only_qc_shadow_v1';S=B.parent/'step15_b60_r6_v1'
CANDS=['HOLD_LAST_VALID','INVALID_QUARANTINE','PAST_TWO_POINT_EXTRAPOLATION']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str)+'\n')
logs=[];hashes={}
def rd(p):
 assert 'test' not in p.name.lower();hashes[str(p.relative_to(R))]=sha(p);logs.append(str(p.relative_to(R)));wr('access_ledger.json',logs)
 return pd.read_parquet(p) if p.suffix=='.parquet' else json.loads(p.read_text())
def first(row,names,default=np.nan):
 for c in names:
  if c in row and pd.notna(row[c]):return float(row[c])
 return default
def main():
 O.mkdir(exist_ok=False)
 wr('protocol.json',{'candidates':CANDS,'events':'immutable prior source-event registry only','evidence':'strictly previous same crop/series/segment/source_file_path; valid finite raw measurements, excludes interpolated/imputed/ffilled; no synthetic values reused as measurement','age_cutoff':None,'no_clamp_on_extrapolated_source':True,'invalid':'NaN and full frozen ancestry mask, never numeric fill','differences':'signed shadow-original; exact equality plus rtol1e-6 atol1e-8 meaningful change','performance_metrics':False,'domain_v2_gate':False,'physical_truth':'unknown','recommendation_order':['causal availability','simplicity','few new assumptions','invalid loss','online/offline parity']})
 wr('protocol_lock.json',{'sha256':sha(O/'protocol.json'),'code_sha256':sha(Path(__file__)),'numpy':np.__version__,'pandas':pd.__version__})
 schema=rd(B/'schema.json');obs=schema['observation_columns'];summ=[];wins=[];fits=[];sources=[];diffs=[];checks=[]
 for crop in ['melon','cucumber','strawberry']:
  for split in ['train','validation']:
   tag=f'{crop}_{split}';print(tag,flush=True);epath=A/f'{tag}_source_events.csv.gz';hashes[str(epath.relative_to(R))]=sha(epath)
   try:e=pd.read_csv(epath,parse_dates=['current_timestamp','right_timestamp','left_timestamp'])
   except pd.errors.EmptyDataError:e=pd.DataFrame()
   q=rd(B/crop/f'{split}_prepared_qc.parquet').reset_index(drop=True);b=rd(B/crop/f'{split}_physical.parquet');w=rd(B/crop/f'{split}_windows.parquet');edges=rd(A/f'{tag}_feature_lineage.parquet')
   q.reg_date=pd.to_datetime(q.reg_date);b.reg_date=pd.to_datetime(b.reg_date);b.target_timestamp=pd.to_datetime(b.target_timestamp)
   keys=['crop','series_id','segment_id','reg_date'];index=pd.MultiIndex.from_frame(q[keys]);cur=index.get_indexer(pd.MultiIndex.from_frame(b[keys]));tar=index.get_indexer(pd.MultiIndex.from_frame(b[keys[:-1]+['target_timestamp']].rename(columns={'target_timestamp':'reg_date'})))
   assert (cur>=0).all() and (tar>=0).all()
   evidence={};eventidx={}
   for var in ['out_light','out_windsp']:
    history={};events={} if e.empty else {int(x.position):x for x in e.loc[e.variable.eq(var)].itertuples()}
    for i,row in q.iterrows():
     identity=tuple(str(row.get(c,'<missing>')) for c in ['crop','series_id','segment_id','source_file_path']);past=history.setdefault(identity,[])
     if i in events:
      x=events[i];assert row.reg_date==x.current_timestamp
      evidence[x.event_id]=past[-2:].copy();eventidx[x.event_id]=i
     value=first(row,[var+'_raw_value',var]);valid=np.isfinite(value)
     for suffix in ['_missing_flag','_rule_outlier_flag','_ai_outlier_flag','_imputed_flag','_ffill_flag']:
      valid &= not bool(row.get(var+suffix,0)) if pd.notna(row.get(var+suffix,0)) else False
     valid &= 'interpolation' not in str(row.get(var+'_imputation_method',''))
     if valid:
      if past:assert past[-1][0]<=row.reg_date
      past.append((row.reg_date,float(value),i,identity))
   original=q[obs].to_numpy(float);dates=q.reg_date.dt.date.to_numpy();tt=q.reg_date
   ancestry=np.zeros((len(q),len(obs)),bool)
   for x in edges.itertuples():ancestry[x.recipient_position,obs.index(x.feature)]=True
   for cand in CANDS:
    shadows={};failed=[];records=[]
    for x in e.itertuples():
     past=evidence[x.event_id];assert all(z[0]<x.current_timestamp for z in past)
     value=np.nan
     if cand==CANDS[0] and past:value=past[-1][1]
     if cand==CANDS[2] and len(past)==2:
      t0,v0,_,_=past[0];t1,v1,_,_=past[1];assert t1>t0;value=v1+(v1-v0)*(x.current_timestamp-t1)/(t1-t0)
     if not np.isfinite(value):failed.append(x.event_id)
     shadows[(x.position,x.variable)]=value
     rec={'crop':crop,'split':split,'candidate':cand,'event_id':x.event_id,'run_id':x.run_id,'position':x.position,'variable':x.variable,'current_timestamp':x.current_timestamp,'original_QC':x.qc_value,'shadow_value':value,'reconstructable':bool(np.isfinite(value)),'past_timestamp':past[-1][0] if past else None,'past_value':past[-1][1] if past else None,'second_past_timestamp':past[-2][0] if len(past)>1 else None,'second_past_value':past[-2][1] if len(past)>1 else None,'last_valid_age_min':(x.current_timestamp-past[-1][0]).total_seconds()/60 if past else None,'gap_rows_since_valid':x.position-past[-1][2] if past else None,'right_endpoint_timestamp_audit_only':x.right_timestamp,'right_endpoint_value_used':False,'original_lookahead':x.lookahead_min,'same_source_identity':True,'signed_difference':value-x.qc_value,'absolute_difference':abs(value-x.qc_value),'exact_same':bool(value==x.qc_value)}
     records.append(rec);sources.append(rec)
    invalid=np.zeros_like(ancestry)
    for x in edges.loc[edges.event_id.isin(failed)].itertuples():invalid[x.recipient_position,obs.index(x.feature)]=True
    # Sequential per-timestamp delta reconstruction. No downstream future values enter this recurrence.
    def replay():
     out=original.copy();dd=dh=dc=0.;prev=None
     for i,row in q.iterrows():
      t=row.reg_date;dt=0 if prev is None else max((t-prev).total_seconds()/60,0);daily_dt=dt if i and dates[i]==dates[i-1] else 0
      if not i or dates[i]!=dates[i-1]:dd=dh=dc=0.
      oldlight=float(row.obs_outdoor_light);light=shadows.get((i,'out_light'),oldlight);wind=shadows.get((i,'out_windsp'),float(row.obs_outdoor_wind_speed))
      out[i,obs.index('obs_outdoor_light')]=light;out[i,obs.index('obs_outdoor_wind_speed')]=wind
      if np.isfinite(light):
       day=light>=10.;out[i,obs.index('obs_is_daytime')]=float(day)
       target=first(row,['current_target_temp_c','target_temp_c'],float(row.obs_target_day_temp_c if day else row.obs_target_night_temp_c))
       out[i,obs.index('obs_current_target_temp_c')]=target
       for suffix,sign in [('min',-1),('max',1)]:
        period='day' if day else 'night';pv=first(row,[f'target_{period}_temp_{suffix}_c',f'{period}_target_temp_{suffix}_c',f'{period}_temp_{suffix}_c'])
        v=first(row,[f'current_target_temp_{suffix}_c',f'target_temp_{suffix}_c',f'current_temp_{suffix}_c'],pv)
        out[i,obs.index(f'obs_current_target_temp_{suffix}_c')]=v if np.isfinite(v) else target+sign
       dd+=(max(light,0)-max(oldlight,0))*daily_dt*60*2.02/1e6
       temp=float(row.obs_indoor_temp_c);oldtarget=float(row.obs_current_target_temp_c)
       dh+=(max(target-temp,0)-max(oldtarget-temp,0))*dt;dc+=(max(temp-target,0)-max(temp-oldtarget,0))*dt
      out[i,obs.index('obs_derived_dli_mol_m2')]+=dd
      out[i,obs.index('obs_derived_heating_degree_minutes')]+=dh;out[i,obs.index('obs_derived_cooling_degree_minutes')]+=dc
      # Missing-light integral is never treated as numeric zero: invalid mask overrides every descendant.
      prev=t
     out[invalid]=np.nan
     return out
    out=replay();out2=replay();assert np.array_equal(out,out2,equal_nan=True)
    # Evaluate only original registry observations, retaining B previous-command vent definition.
    flags={};rowdiff=[]
    for role,positions in [('current',cur),('target',tar)]:
     prefix='' if role=='current' else 'next_';old=b[[prefix+c for c in obs]].to_numpy(float);new=out[positions].copy()
     # Nonaffected features must be original B physical values (QC carries pre-B actions).
     new=np.where(ancestry[positions],new,old)
     vi=obs.index('obs_derived_vent_loss_proxy');li=obs.index('obs_outdoor_light');pi=obs.index('obs_prev_vent_pct');ti=obs.index('obs_indoor_temp_c');oi=obs.index('obs_outdoor_temp_c')
     new[:,vi]=old[:,pi]*np.maximum(new[:,ti]-new[:,oi],0)*(1-np.clip(new[:,li]/1000,0,1));new[invalid[positions]]=np.nan
     changed=np.isfinite(new)&np.isfinite(old)&~np.isclose(new,old,rtol=1e-6,atol=1e-8);bad=~np.isfinite(new);anychanged=changed.any(axis=1);anybad=bad.any(axis=1)
     flags[role]=(anychanged,anybad)
     for j,k in zip(*np.where(ancestry[positions])):
      rowdiff.append({'row_id':b.iloc[j].row_id,'role':role,'timestamp':b.iloc[j].reg_date if role=='current' else b.iloc[j].target_timestamp,'feature':obs[k],'original':old[j,k],'shadow':new[j,k],'signed_difference':new[j,k]-old[j,k],'absolute_difference':abs(new[j,k]-old[j,k]),'exact_same':bool(new[j,k]==old[j,k]),'meaningful_changed':bool(changed[j,k]),'invalid':bool(bad[j,k])})
    pd.DataFrame(rowdiff).to_parquet(O/f'{tag}_{cand}_feature_comparison.parquet',index=False)
    directpos=set() if e.empty else set(e.position)
    direct=np.array([p in directpos for p in cur]);changed,bad=flags['current'];ances=ancestry[cur].any(axis=1)
    summary={'crop':crop,'split':split,'candidate':cand,'rows':len(b),'direct_changed_rows':int((changed&direct).sum()),'propagated_changed_rows':int((changed&~direct).sum()),'ancestry_only_unchanged_rows':int((ances&~changed&~bad).sum()),'invalid_current':int(bad.sum()),'invalid_propagated_rows':int((bad&~direct).sum()),'current_changed':int(changed.sum()),'target_changed':int(flags['target'][0].sum()),'target_invalid':int(flags['target'][1].sum())}
    affected=changed|bad|flags['target'][0]|flags['target'][1];summary['transition_affected']=int(affected.sum());summ.append(summary)
    side=b[['row_id','mdp_v1_rollout_id']].copy();side['current_changed']=changed;side['current_invalid']=bad;side['target_changed']=flags['target'][0];side['target_invalid']=flags['target'][1];side['transition_affected']=affected;side.to_parquet(O/f'{tag}_{cand}_transition_sidecar.parquet',index=False)
    ids={rid:j for j,rid in enumerate(b.row_id)}
    for h,g in w.groupby('horizon'):
     hit=0
     for x in g.itertuples():
      j=ids[x.start_row_id];loc=np.arange(j,j+int(h));assert b.iloc[loc].mdp_v1_rollout_id.nunique()==1 and b.iloc[loc[-1]].target_timestamp==x.end_target_timestamp;hit+=int(affected[loc].any())
     wins.append({'crop':crop,'split':split,'candidate':cand,'horizon':h,'windows':len(g),'affected':hit})
    if split=='train':
     regs={'scaler_fit':rd(B/crop/'scaler_fit_registry.parquet').row_id,'Transition_fit':b.row_id};sr=rd(S/crop/'fit_row_registry.parquet')
     for role,g in sr.groupby('role'):regs['Step15_'+role]=g.row_id
     for name,rr in regs.items():
      jj=[ids[r] for r in rr];fits.append({'crop':crop,'candidate':cand,'registry':name,'rows':len(jj),'changed_current_values':int(changed[jj].sum()),'invalid_current':int(bad[jj].sum()),'transition_readmission_required':int((bad|flags['target'][1])[jj].sum()),'value_or_target_affected':int(affected[jj].sum())})
    checks.append({'crop':crop,'split':split,'candidate':cand,'deterministic_replay':True,'future_endpoint_consumed':False,'identity_boundary_checks':True,'invalid_fill':False})
   pd.DataFrame(summ).to_csv(O/'population_summary.csv',index=False);pd.DataFrame(wins).to_csv(O/'window_summary.csv',index=False);pd.DataFrame(fits).to_csv(O/'membership_summary.csv',index=False);pd.DataFrame(sources).to_parquet(O/'source_reconstruction.parquet',index=False)
 for p,h in hashes.items():assert sha(R/p)==h
 wr('input_hashes.json',hashes);wr('verification.json',checks)
if __name__=='__main__':main()
