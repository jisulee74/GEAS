"""Read-only diagnosis: frozen traces + recorded-state teacher forcing, no fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,gzip
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.recorded_light_env_v1 import load_recorded_light_env,RecordedLightStateUpdater,SOLAR_FEATURES
from geas35.features.derived import vpd_kpa,dewpoint_c
from geas35.experiments.rl.step19_5_diagnostics import ART,OUT,read_json,write_json,verify_inputs,sha256,lock
DEST=ROOT/ART/'step19_5_residual_ood_v1';BASE=ROOT/ART/'step19_5_light_contract_v1'
ALIASES=['obs_current_vpd_kpa','obs_current_dewpoint_c','obs_current_condensation_margin_c']
INDOOR=['obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm']
def aliases(frame):
 f=frame.copy();f[ALIASES[0]]=vpd_kpa(f[INDOOR[0]],f[INDOOR[1]]);f[ALIASES[1]]=dewpoint_c(f[INDOOR[0]],f[INDOOR[1]]);f[ALIASES[2]]=f[INDOOR[0]]-f[ALIASES[1]];return f

def run(crop):
 summary=[];div=[];endpoints=[];detail=[];dist=[];ranges=[]
 for split in ('validation','train'):
  env=load_recorded_light_env(project_root=ROOT,crop=crop,split=split,protocol_manifest_path=ROOT/ART/'step15_v1_4/rl_protocol_manifest.json')
  cols=list(env.observation_columns);actions=list(env.action_columns);phy=env.scaler.inverse_transform_frame(env.frame.copy());_,targets=env.model._require_fitted()
  features=[c for c in cols if c.startswith('obs_derived_') and c not in SOLAR_FEATURES]+ALIASES
  history=[c for c in features if c not in ALIASES]
  selected=read_json(ROOT/OUT/'input_manifest.json')['registries'][crop][split]['indices']
  severe_count=0
  for ordinal,index in enumerate(selected):
   epfile=BASE/f'{crop}_{split}/episodes/{ordinal:04d}.json';meta=read_json(epfile)
   if meta['termination_reason']!='severe_state_ood':continue
   severe_count+=1;tracepath=epfile.with_suffix('.jsonl.gz');assert sha256(tracepath)==meta['trace_sha256']
   trace=[json.loads(l) for l in gzip.open(tracepath,'rt')];up=RecordedLightStateUpdater(env.scaler)
   start=trace[0]['source_position'];up.reset(env.frame.iloc[start]);current=env.frame.iloc[start];seen=set();seen_teacher=set();max_teacher={c:0. for c in features}
   for j,row in enumerate(trace):
    pos=row['source_position'];record=phy.iloc[pos];roll=pd.Series(row['physical_state'],index=cols);teacher=up.current_physical
    formula=aliases(roll.to_frame().T).iloc[0]
    for c in features:
     err=float(teacher[c]-record[c]);max_teacher[c]=max(max_teacher[c],abs(err))
     for mode,val,used in [('rollout',roll[c],seen),('teacher',teacher[c],seen_teacher)]:
      if c not in used and not np.isclose(float(val),float(record[c]),rtol=1e-6,atol=1e-8):
       used.add(c);div.append(dict(crop=crop,split=split,episode_index=index,feature=c,mode=mode,step=j,timestamp=str(record.reg_date),recorded=float(record[c]),reset=float(phy.iloc[start][c]),value=float(val),day_boundary=j>0 and pd.Timestamp(record.reg_date).date()!=pd.Timestamp(phy.iloc[pos-1].reg_date).date(),history_rows=min(j+1,288),missing=bool(pd.isna(record[c])),recorded_zero=float(record[c])==0.))
    detail.append({'crop':crop,'split':split,'episode_index':index,'step':j,'timestamp':str(record.reg_date),'features':{c:{'recorded':float(record[c]),'reset':float(phy.iloc[start][c]),'rollout':float(roll[c]),'teacher':float(teacher[c])} for c in features},'indoor_difference':{c:float(roll[c]-record[c]) for c in INDOOR},'alias_formula_error':{c:float(roll[c]-formula[c]) for c in ALIASES}})
    if j>=len(trace)-2:
     variants={'rollout':roll,'recorded':record.reindex(cols),'teacher':teacher.reindex(cols),'recompute_current_aliases':formula.reindex(cols)}
     for name,group in [('replace_indoor_with_recorded',INDOOR),('replace_history_with_recorded',history)]:
      v=roll.copy();v.loc[group]=record.loc[group].to_numpy(dtype=float);variants[name]=v
     variants_frame=env.scaler.transform_frame(pd.DataFrame(variants).T)
     scores=env.support.scores(variants_frame,pd.DataFrame([env.frame.iloc[pos][actions].to_dict()]*len(variants),index=variants_frame.index))['state']
     endpoints.append({'crop':crop,'split':split,'episode_index':index,'step':j,'timestamp':str(record.reg_date),'is_terminal':j==len(trace)-1,'scores':dict(zip(variants,map(float,scores))),'threshold':env.support.config['thresholds']['state_severe'],'distance':row.get('state_distance'),'indoor_difference':{c:float(roll[c]-record[c]) for c in INDOOR}})
    if j+1<len(trace):
     nextpos=trace[j+1]['source_position'];action=env.frame.iloc[pos][actions].astype(float).to_dict()
     current=up.advance(current_scaled=current,recorded_next_scaled=env.frame.iloc[nextpos],prediction_physical={c:float(phy.iloc[nextpos][c]) for c in targets},action=action)
   summary.append({'crop':crop,'split':split,'episode_index':index,'steps':len(trace),'teacher_max_abs_error':max_teacher,'targets':list(targets),'history_limit_reached':len(trace)>288})
   print(crop,split,'episode',index,'done',flush=True)
  # Score the frozen states, including Train, to distinguish data/support coverage.
  valid=env.frame['rl_valid_transition'].astype(bool).to_numpy();f=env.frame.loc[valid];p=phy.loc[valid].copy()
  sc=env.support.scores(f[cols],f[actions])['state'];p['severe']=sc>env.support.config['thresholds']['state_severe'];p['score']=sc
  p['hour']=pd.to_datetime(p.reg_date).dt.hour;p['date']=pd.to_datetime(p.reg_date).dt.date.astype(str)
  p['wind_bin']=pd.cut(p.obs_outdoor_wind_speed,[-np.inf,0,1,2,4,8,np.inf]).astype(str)
  p['light_bin']=pd.cut(p.obs_outdoor_light,[-np.inf,0,100,300,600,1000,np.inf]).astype(str)
  for key in ('hour','date','series_id','wind_bin','light_bin'):
   for value,g in p.groupby(key,dropna=False):dist.append(dict(crop=crop,split=split,dimension=key,value=str(value),n=len(g),severe=int(g.severe.sum()),rate=float(g.severe.mean()),mean_score=float(g.score.mean())))
  dist.append(dict(crop=crop,split=split,dimension='all',value='all',n=len(p),severe=int(p.severe.sum()),rate=float(p.severe.mean()),mean_score=float(p.score.mean())))
  ref=env.scaler.inverse_transform_frame(pd.DataFrame(env.support.reference_state_raw,columns=cols))
  for c in cols:
   ranges.append(dict(crop=crop,split=split,feature=c,data_min=float(p[c].min()),data_max=float(p[c].max()),support_min=float(ref[c].min()),support_max=float(ref[c].max()),outside_reference_range=int(((p[c]<ref[c].min()-1e-8)|(p[c]>ref[c].max()+1e-8)).sum())))
  write_json(DEST/f'{crop}_{split}_support_config.json',dict(env.support.config))
 pd.DataFrame(div).to_csv(DEST/f'{crop}_first_divergence.csv',index=False)
 pd.DataFrame(dist).to_csv(DEST/f'{crop}_recorded_distribution.csv',index=False)
 pd.DataFrame(ranges).to_csv(DEST/f'{crop}_support_ranges.csv',index=False)
 write_json(DEST/f'{crop}_endpoints.json',endpoints);write_json(DEST/f'{crop}_summary.json',summary)
 with gzip.open(DEST/f'{crop}_feature_trace.jsonl.gz','wt') as stream:
  for r in detail:stream.write(json.dumps(r,allow_nan=False)+'\n')
 return {'crop':crop,'residual_episodes':len(summary),'terminal_count':sum(x['is_terminal'] for x in endpoints)}

def main():
 DEST.mkdir(parents=True,exist_ok=True)
 with lock(DEST/'run.lock'):
  manifest=read_json(ROOT/OUT/'input_manifest.json');verify_inputs(ROOT,manifest)
  p=DEST/'manifest.json';body={'script_sha256':sha256(Path(__file__)),'solar_manifest_sha256':sha256(BASE/'verification_manifest.json'),'original_manifest_sha256':sha256(ROOT/OUT/'input_manifest.json'),'new_training':False,'test_accessed':False}
  if p.exists() and read_json(p)!=body:raise ValueError('Inputs changed; use new output version')
  write_json(p,body)
  with ProcessPoolExecutor(max_workers=2) as pool:result=list(pool.map(run,('melon','cucumber')))
  verify_inputs(ROOT,manifest);write_json(DEST/'status.json',{'status':'complete','results':result,'original_inputs_unchanged':True})
if __name__=='__main__':main()
