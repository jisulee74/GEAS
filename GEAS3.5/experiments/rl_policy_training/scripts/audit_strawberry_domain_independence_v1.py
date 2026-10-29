"""Train-only independent evidence audit; all frozen domain definitions unchanged."""
from pathlib import Path
import os,sys,json,hashlib,datetime
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[3]
D=R/'experiments/rl_policy_training/artifacts/strawberry_domain_independence_v1'
V=R/'experiments/rl_policy_training/artifacts/strawberry_research_domain_shadow_v1'
P=R/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1'
LEVEL=np.array(['IN_DOMAIN','LOW_SUPPORT','OUT_OF_DOMAIN'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')
def level(n,e):return np.where(n==0,2,np.where((n<50)|(e<2),1,0))
def verify_inputs():
 assert sha(D/'protocol.json')==read(D/'protocol_lock.json')['protocol_sha256']
 assert sha(V/'frozen_domain_rule.json')==read(D/'protocol_lock.json')['rule_sha256']
 for x in read(D/'original_v1_integrity.json'):assert sha(R/x['path'])==x['sha256'],x['path']
def run():
 verify_inputs();assert not (D/'summary.json').exists(),'Completed audit must not be overwritten'
 rule=read(V/'frozen_domain_rule.json');obs=rule['observation_columns'];actions=rule['actions']
 assert len(rule['state_cells'])==14 and len(rule['action_cells'])==6 and rule['minimum_rows']==50 and rule['minimum_episodes']==2
 trainpath=P/'strawberry/train_physical.parquet';trainhash=sha(trainpath)
 f=pd.read_parquet(trainpath,columns=['row_id','episode_id','series_id','segment_id','reg_date','source_confidence']+obs+actions)
 assert len(f)==44097 and f.row_id.is_unique
 t=pd.to_datetime(f.reg_date);assert t.notna().all()
 ep,epnames=pd.factorize(f.episode_id.astype(str),sort=True);ne=len(epnames);n=len(f)
 assert f.series_id.nunique()==1 and f.series_id.iloc[0]=='strawberry_geas2_0'
 specs={**rule['state_cells'],**rule['action_cells'],'DAT_30d':{'columns':['obs_growth_stage_dat']}}
 cells={}
 for name,spec in specs.items():
  if name=='DAT_30d':keys=np.floor(f.obs_growth_stage_dat.to_numpy()/30).astype(int).astype(str)
  else:
   a=[]
   for c in spec['columns']:
    x=f[c].to_numpy(float);edge=rule['edges'][c];a.append(np.where(x>edge[-1],len(edge)+1,np.searchsorted(edge,x,side='right')))
   keys=np.array(['|'.join(map(str,row)) for row in np.column_stack(a)])
  code,uniq=pd.factorize(keys,sort=True);nc=len(uniq)
  def counts(mask):
   rows=np.bincount(code[mask],minlength=nc);eps=(np.bincount(code[mask]*ne+ep[mask],minlength=nc*ne).reshape(nc,ne)>0).sum(axis=1)
   return rows,eps
  fulln,fulle=counts(np.ones(n,bool))
  old=rule['DAT_30d'] if name=='DAT_30d' else spec['occupancy']
  for i,k in enumerate(uniq):assert fulln[i]==old[k]['rows'] and fulle[i]==old[k]['episodes'],(name,k)
  cells[name]={'code':code,'keys':uniq,'nc':nc,'fulln':fulln,'fulle':fulle,'fulllevel':level(fulln,fulle),'kind':'action' if name in rule['action_cells'] else 'DAT' if name=='DAT_30d' else 'state'}
 state_names=list(rule['state_cells'])+['DAT_30d'];action_names=list(rule['action_cells'])
 def aggregate(names,levels):return np.maximum.reduce([levels[k] for k in names])
 full={k:c['fulllevel'][c['code']] for k,c in cells.items()}
 base_state=aggregate(state_names,full);base_action=aggregate(action_names,full)
 # All Train rows are within frozen marginal/source constraints; geometry is never refit.
 assert np.isfinite(f[obs].to_numpy(float)).all()
 for col,(lo,hi) in rule['ranges'].items():assert f[col].between(lo,hi).all()
 assert f.source_confidence.isin(rule['known_source_confidence']).all()
 assert np.bincount(base_state,minlength=3).tolist()==[43452,645,0]
 assert np.bincount(base_action,minlength=3).tolist()==[44075,22,0]
 month=t.dt.strftime('%Y-%m').to_numpy();mi=t.dt.month.to_numpy();yr=t.dt.year.to_numpy()
 seasons=np.select([np.isin(mi,[12,1,2]),np.isin(mi,[3,4,5]),np.isin(mi,[6,7,8])],['DJF','MAM','JJA'],default='SON')
 season=np.array([f'{y+(m==12)}-{s}' for y,m,s in zip(yr,mi,seasons)])
 cal=((t.dt.normalize()-pd.Timestamp('1970-01-01')).dt.days//14).to_numpy()
 dat=np.floor(f.obs_growth_stage_dat/30).astype(int).to_numpy()
 row_registry=f[['row_id','episode_id','series_id','segment_id','reg_date','source_confidence']].copy()
 row_registry['month']=month;row_registry['season']=season;row_registry['DAT30']=dat;row_registry['calendar14_id']=cal
 row_registry['full_state_level']=LEVEL[base_state];row_registry['full_action_level']=LEVEL[base_action]
 row_registry.to_parquet(D/'Train_fold_registry.parquet',index=False)
 folds=[]
 for method,labels in [('episode',ep),('month',month),('season',season),('DAT30',dat),('calendar14',cal),('forward_month',month)]:
  for key in sorted(np.unique(labels)):
   test=(labels==key);excluded_ep=np.unique(ep[test]);evidence=~np.isin(ep,excluded_ep)
   if method=='forward_month':evidence&=(t<pd.Timestamp(str(key)+'-01')).to_numpy()
   folds.append((method,str(epnames[key]) if method=='episode' else str(key),test,evidence,method=='forward_month' and key==sorted(np.unique(month))[0]))
 write(D/'execution_inputs.json',{'Train_path':str(trainpath.relative_to(R)),'Train_sha256':trainhash,'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(D/'protocol.json'),'fold_count':len(folds),'episodes':ne,'rows':n,'Validation_read':False,'Test_read':False,'geometry':'frozen full-Train edges/ranges; conditional occupancy audit only','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 results=[];details=[];rowparts=[];brute_checks=0
 for fi,(method,key,test,evidence,cold) in enumerate(folds):
  idx=np.flatnonzero(test);ev=np.flatnonzero(evidence)
  assert not np.intersect1d(ep[idx],ep[ev]).size
  assert not (test&evidence).any()
  if method=='forward_month' and len(ev):assert t.iloc[ev].max()<pd.Timestamp(key+'-01')<=t.iloc[idx].min()
  lv={};es=[]
  for name,c in cells.items():
   code=c['code'];nc=c['nc'];rn=np.bincount(code[evidence],minlength=nc)
   re=(np.bincount(code[evidence]*ne+ep[evidence],minlength=nc*ne).reshape(nc,ne)>0).sum(axis=1)
   hold=np.bincount(code[test],minlength=nc);levels=level(rn,re);lv[name]=levels[code[idx]]
   assert (rn<=c['fulln']).all() and (re<=c['fulle']).all()
   assert (levels>=c['fulllevel']).all()
   # Independent pandas reconstruction on first episode and last forward fold.
   if fi in (0,len(folds)-1):
    z=pd.DataFrame({'cell':code[evidence],'episode':ep[evidence]}).groupby('cell').episode.agg(['size','nunique']).reindex(range(nc),fill_value=0)
    np.testing.assert_array_equal(rn,z['size']);np.testing.assert_array_equal(re,z['nunique']);brute_checks+=1
   for j in np.flatnonzero(hold):
    fulllev=int(c['fulllevel'][j]);newlev=int(levels[j])
    details.append({'method':method,'fold':key,'cold_start':cold,'kind':c['kind'],'cell':name,'bin_key':c['keys'][j],'heldout_rows':int(hold[j]),'full_rows':int(c['fulln'][j]),'retained_rows':int(rn[j]),'retained_fraction':float(rn[j]/c['fulln'][j]),'full_episodes':int(c['fulle'][j]),'retained_episodes':int(re[j]),'full_level':fulllev,'heldout_level':newlev,'supported_to_sparse':fulllev==0 and newlev==1,'supported_to_unobserved':fulllev==0 and newlev==2,'sparse_to_unobserved':fulllev==1 and newlev==2,'rows50_cross':bool(c['fulln'][j]>=50 and rn[j]<50),'episodes2_cross':bool(c['fulle'][j]>=2 and re[j]<2)})
  state=aggregate(state_names,lv);action=aggregate(action_names,lv);combined=np.maximum(state,action)
  out=f.iloc[idx][['row_id','episode_id','reg_date']].copy();out['method']=method;out['fold']=key;out['cold_start']=cold;out['state_level']=state;out['action_level']=action;out['combined_level']=combined
  out['full_state_level']=base_state[idx];out['full_action_level']=base_action[idx];rowparts.append(out)
  res={'method':method,'fold':key,'cold_start':cold,'rows':len(idx),'heldout_episodes':len(np.unique(ep[idx])),'evidence_rows':len(ev),'evidence_episodes':len(np.unique(ep[ev])),'excluded_rows':n-len(ev),'purged_nonblock_rows':int((~evidence&~test).sum()),'start':str(t.iloc[idx].min()),'end':str(t.iloc[idx].max()),'evidence_last':str(t.iloc[ev].max()) if len(ev) else None,'DAT_min':float(f.obs_growth_stage_dat.iloc[idx].min()),'DAT_max':float(f.obs_growth_stage_dat.iloc[idx].max()),'full_IN':int((base_state[idx]==0).sum())}
  for kind,a in [('state',state),('action',action),('combined',combined)]:
   for j,label in enumerate(['IN','LOW','OUT']):res[kind+'_'+label]=int((a==j).sum());res[kind+'_'+label+'_rate']=float((a==j).mean())
  results.append(res)
  if fi%25==0:print('fold',fi+1,'/',len(folds),method,flush=True)
 result=pd.DataFrame(results);detail=pd.DataFrame(details);rows=pd.concat(rowparts,ignore_index=True)
 result.to_csv(D/'fold_summary.csv',index=False);detail.to_parquet(D/'fold_cell_evidence.parquet',index=False);rows.to_parquet(D/'heldout_row_sidecar.parquet',index=False)
 sums=[]
 for method,g in result.groupby('method'):
  for mature in ([False,True] if method=='forward_month' else [False]):
   x=g.loc[~g.cold_start] if mature else g
   d={'method':method,'mature_only':mature,'folds':len(x),'rows':int(x.rows.sum())}
   for kind in ['state','action','combined']:
    for label in ['IN','LOW','OUT']:d[kind+'_'+label+'_rate']=float(x[kind+'_'+label].sum()/x.rows.sum()) if len(x) else None
    d[kind+'_median_IN_rate']=float(x[kind+'_IN_rate'].median());d[kind+'_min_IN_rate']=float(x[kind+'_IN_rate'].min())
   d['state_drop_from_full_pp']=100*((base_state==0).mean()-d['state_IN_rate']);sums.append(d)
 sm=pd.DataFrame(sums);sm.to_csv(D/'method_summary.csv',index=False)
 stability=[]
 for (method,kind,name),g in detail.groupby(['method','kind','cell']):
  for mature in ([False,True] if method=='forward_month' else [False]):
   x=g.loc[~g.cold_start] if mature else g;w=x.heldout_rows
   item={'method':method,'mature_only':mature,'kind':kind,'cell':name,'fold_cell_combinations':len(x),'evaluated_rows':int(w.sum()),'full_supported_rows':int(w[x.full_level.eq(0)].sum()),'heldout_supported_rows':int(w[x.heldout_level.eq(0)].sum()),'heldout_sparse_rows':int(w[x.heldout_level.eq(1)].sum()),'heldout_unobserved_rows':int(w[x.heldout_level.eq(2)].sum()),'retained_fraction_weighted':float(np.average(x.retained_fraction,weights=w))}
   for flag in ['supported_to_sparse','supported_to_unobserved','sparse_to_unobserved','rows50_cross','episodes2_cross']:
    item[flag+'_cells']=int(x[flag].sum());item[flag+'_rows']=int(w[x[flag]].sum());item[flag+'_row_rate']=float(w[x[flag]].sum()/w.sum())
   item['supported_loss_row_rate']=(item['supported_to_sparse_rows']+item['supported_to_unobserved_rows'])/item['evaluated_rows'];stability.append(item)
 stable=pd.DataFrame(stability);stable.to_csv(D/'cell_stability.csv',index=False)
 stable.loc[stable.kind.eq('action')].to_csv(D/'action_pair_stability.csv',index=False)
 edge=rows.loc[rows.row_id.isin(f.loc[dat==6,'row_id'])];edge.to_csv(D/'DAT180_209_heldout_rows.csv',index=False)
 detail.loc[detail.cell.eq('DAT_30d')&detail.bin_key.eq('6')].to_csv(D/'DAT191_evidence.csv',index=False)
 # Deterministic reconstruction, no source data or rule updates.
 verify_inputs();assert sha(trainpath)==trainhash
 for method in result.method.unique():assert rows.loc[rows.method.eq(method)].row_id.nunique()==n and len(rows.loc[rows.method.eq(method)])==n
 def metric(method,field):return float(sm.loc[sm.method.eq(method)&sm.mature_only.eq(method=='forward_month'),field].iloc[0])
 a=(metric('episode','state_IN_rate')>=.9 and metric('episode','state_median_IN_rate')>=.9 and metric('episode','state_min_IN_rate')>=.5 and all(metric(k,'state_IN_rate')>=.7 and metric(k,'state_median_IN_rate')>=.7 and metric(k,'state_min_IN_rate')>=.2 for k in ['month','calendar14']) and metric('season','state_IN_rate')>=.5 and metric('forward_month','state_IN_rate')>=.5 and metric('episode','action_IN_rate')>=.95 and all(metric(k,'action_IN_rate')>=.8 for k in ['month','calendar14','forward_month']))
 b=(metric('episode','state_IN_rate')>=.7 and all(metric(k,'state_IN_rate')>=.4 for k in ['month','calendar14']) and metric('episode','action_IN_rate')>=.8)
 insufficient=result.loc[result.method.eq('forward_month')&~result.cold_start].empty
 judgment='B' if insufficient else 'A' if a else 'B' if b else 'C'
 write(D/'verification.json',{'frozen_v1_all_files_unchanged':True,'rule_and_protocol_hash_verified':True,'Train_hash_unchanged':True,'all_folds_disjoint_episode_evidence':True,'forward_occupancy_strictly_past':True,'global_frozen_geometry_is_not_past_only':True,'full_occupancy_reconstructed_exactly':True,'full_state_action_counts_match_v1':True,'independent_pandas_cell_checks':brute_checks,'monotonic_evidence_degradation':True,'each_method_covers_all_Train_rows_once':True,'Validation_read_for_evaluation':False,'Test_read':False,'official_mutations':False})
 write(D/'summary.json',{'judgment':judgment,'A_criteria_met':bool(a),'B_minimum_met':bool(b),'insufficient':insufficient,'full_Train_rows':n,'episodes':ne,'full_IN_rate':float((base_state==0).mean()),'folds':len(folds),'methods':sums,'strawberry_Step16':'BLOCKED','overall_Step17':'BLOCKED','online_deployment':'BLOCKED','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print('COMPLETE',judgment,flush=True);print(sm.to_string(index=False),flush=True)
if __name__=='__main__':run()
