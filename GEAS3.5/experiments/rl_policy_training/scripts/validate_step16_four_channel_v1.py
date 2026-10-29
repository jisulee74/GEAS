"""Reward-free 4CH sequential research validation; frozen parent action ordering."""
from pathlib import Path
import sys,os,json,importlib.util,argparse,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'))
import numpy as np
import pandas as pd
from geas35.models.transition.handoff_four_channel_v1 import contract,checked,CONTRACT_HASH
from geas35.models.transition.four_channel_phase0 import sha,TARGETS,physical_gate
from geas35.models.transition.four_channel_batch import FourChannelBatchUpdater,I
from geas35.rl.bound_support_four_channel_v1 import load_bound_support
from geas35.rl.mdp_v1 import MdpV1Config,project_mdp_v1_action_constraints
from unittest.mock import patch
B=R/'experiments/rl_policy_training/artifacts';D=B/'step16_hlv15_b60_r6_4ch_v1';S=B/'step15_hlv15_b60_r6_4ch_v1';H=R/'experiments/transition_model_selection/artifacts/step14_hlv15_b60_r6_4ch_v1';C=contract(R);OBS=C['schema']['observation_columns'];ACT=C['schema']['action_columns'];CFG=MdpV1Config();P=Path(json.loads((B/'transition_4ch_phase0_canonical_v1/phase0_manifest.json').read_text())['fitting_authority'])
def read(p):return json.loads(Path(p).read_text())
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
def close(a,b):np.testing.assert_allclose(np.asarray(a,float),np.asarray(b,float),rtol=1e-6,atol=1e-8)
def rec(p):return {'path':str(Path(p).relative_to(R)),'sha256':sha(p)}
def freeze():
 m=read(D/'step16_manifest.json');assert not (D/'protocol.json').exists()
 write(D/'preflight_manifest_preserved.json',m)
 if (D/'final_step16_4ch.md').exists():(D/'final_step16_4ch.md').rename(D/'preflight_decision_preserved.md')
 inputs={}
 for folder in [H,S]:
  for n,h in read(folder/'artifact_integrity.json')['files'].items():assert sha(folder/n)==h;inputs[str((folder/n).relative_to(R))]=h
 for n,h in read(B/'transition_4ch_phase0_canonical_v1/phase0_manifest.json')['artifact_hashes'].items():assert sha(P/n)==h;inputs[str((P/n).relative_to(R))]=h
 files=[Path(__file__),R/'src/geas35/rl/model_driven_env.py',R/'src/geas35/rl/mdp_v1.py',R/'src/geas35/models/transition/four_channel_batch.py',R/'src/geas35/models/transition/four_channel_phase0.py',R/'src/geas35/rl/support_v1.py',R/'src/geas35/rl/bound_support_four_channel_v1.py',R/'src/geas35/models/transition/handoff_four_channel_v1.py']
 pro={'version':'geas35.rl.step16.hlv15_b60_r6.4ch.v1','contract_sha256':CONTRACT_HASH,'environment':C['environment'],'inputs':inputs,'code':[rec(p) for p in files],'resolution':'사용자가 1번 선택: 요청서 순서 표현보다 기존 frozen 부모 action 적용 순서를 우선','order':['current state + recorded proposed joint scores','current state severe: terminate before prediction','if proposed joint warning: local projection','if proposed joint severe: frozen fallback + counter increment; else reset counter','weather cap on resulting action','Transition four-channel prediction','derived/sensor history and next command carry','next state score diagnostic','natural end first, then repeated joint>=3; current severe checked at next decision if another step exists'],'reward':'완전 비활성/null; legacy/Future-F/PPO reward 모두 호출 금지','practical_guard':m['parent_practical_rollout_guard'],'repetitions':2,'rtol':1e-6,'atol':1e-8,'scalar_validation':'매 run 첫 executed step 및 가장 긴8run의 모든 executed step, full56/carry/예측 비교','unsupported_empty_mask':'빈 mask가 필요한 projection/fallback에서 나타나면 unsupported_local_support_failure로 fail closed; 임의 허용 action 생성 금지','exposure_denominators':'state current는 모든 attempted decision, joint proposed는 모든 attempted decision, final/next는 executed transition, intervention은 executed steps','drift':'각 run reset 후5/15/30/60분에서 실제 실행된 경로만; action 변경 여부 별도; 생존 조건부 지표','physical':'T1/T2[-60,80],RH1/RH2[0,100]; invalid prediction stop without clipping','scope':'Validation only, no Train rollout, no reward/PPO/fit/Test/P0/live'}
 write(D/'protocol.json',pro);write(D/'protocol_lock.json',{'sha256':sha(D/'protocol.json')});m.update(status='RUNNING',phase='action_order_resolved_protocol_frozen',action_order_resolution=pro['resolution'],protocol_sha256=sha(D/'protocol.json'),blocker=None);write(D/'step16_manifest.json',m);print('동결 완료',flush=True)
def check():
 contract(R);p=read(D/'protocol.json');assert sha(D/'protocol.json')==read(D/'protocol_lock.json')['sha256']
 for f,h in p['inputs'].items():assert sha(R/f)==h,f
 for r in p['code']:checked(R,r)
 return p

def project(a,local):
 out=a.copy();empty=False
 for j,c in enumerate(ACT[:3]):out[j]=np.clip(out[j],*local['continuous'][c])
 for j,c in enumerate(ACT[3:],3):
  values=local['binary'][c]
  if not values:empty=True
  elif out[j] not in values:out[j]=min(values,key=lambda v:abs(v-out[j]))
 return out,empty

def fallback(state,local):
 t,rh=state[I[TARGETS[0]]],state[I[TARGETS[2]]];low=state[I['obs_current_target_temp_min_c']];high=state[I['obs_current_target_temp_max_c']];day=state[I['obs_is_daytime']]>.5;vpd=state[I['obs_current_vpd_kpa']]
 raw=np.array([.2 if t>high or rh>CFG.rh_max_pct else 0, float(day and t>high),float(not day and t<low),float(t<low),float(t>high),float(rh>CFG.rh_max_pct or vpd<CFG.vpd_min_kpa)])
 out,empty=project(raw,local);return raw,out,empty

def replay(crop,repeat):
 path=S/crop/'binding.json';rt,sup=load_bound_support(path,manifest_sha256=sha(path),crop=crop,root=R)
 b=pd.read_parquet(P/crop/'validation_physical.parquet');base=pd.read_parquet(P/crop/'validation_sensor_prehistory.parquet');startsflag=b.mdp_v1_rollout_id.ne(b.mdp_v1_rollout_id.shift())|b.reg_date.ne(b.target_timestamp.shift());groups=[g for _,g in b.groupby(startsflag.cumsum(),sort=False)];starts=np.array([g.index[0] for g in groups]);lengths=np.array([len(g) for g in groups]);n=len(groups)
 for g in groups:
  for key in ['crop','series_id','segment_id','episode_id','source_confidence']:assert g[key].nunique()==1
  assert (g.target_timestamp-g.reg_date).eq(pd.Timedelta(minutes=5)).all()
  if len(g)>1:assert np.array_equal(g.target_timestamp.iloc[:-1],g.reg_date.iloc[1:])
 keys=['crop','series_id','segment_id','episode_id','reg_date'];idx=pd.MultiIndex.from_frame(base[keys]);assert idx.is_unique;qp=idx.get_indexer(pd.MultiIndex.from_frame(b.iloc[starts][keys]));assert (qp>=0).all();batch=FourChannelBatchUpdater(base,qp,b.iloc[starts][OBS].to_numpy(float));close(batch.state,b.iloc[starts][OBS].to_numpy(float))
 scalar_ids=set(np.argsort(-lengths,kind='stable')[:8]);scalars={}
 # Separate recurrence objects only; model is already hash bound and immutable.
 from geas35.models.transition.four_channel_phase0 import FourChannelRuntime
 for k in range(n):
  sr=FourChannelRuntime(P/'schema.json',P/crop/'observation_scaler.json',schema_hash=sha(P/'schema.json'),scaler_hash=sha(P/crop/'observation_scaler.json'),prehistory=base);sr.reset(b.iloc[starts[k]]);close(sr.current[OBS],batch.state[k]);scalars[k]=sr
 means=np.array([rt.scaler.means.get(c,0.) for c in OBS]);scale=np.array([rt.scaler.scales.get(c,1.) for c in OBS]);recorded_scaled=rt.scaler.transform_frame(b[OBS]);recorded=sup.scores(recorded_scaled,b[ACT]);targetphys=b[['next_'+c for c in OBS]].copy();targetphys.columns=OBS;targetscaled=rt.scaler.transform_frame(targetphys);targetscore=sup.scores(targetscaled,b[ACT])['state'];thr=sup.config['thresholds'];done=np.zeros(n,bool);counter=np.zeros(n,int);executed=np.zeros(n,int);changed=np.zeros(n,bool);terms=['']*n;rows=[];scalar_checks=0;maxparity=0.;errors=[]
 for step in range(int(lengths.max())):
  ix=np.flatnonzero(~done & (lengths>step))
  if not len(ix):break
  pos=starts[ix]+step;frame=b.iloc[pos];state=batch.state[ix].copy();scaled=(state-means)/scale;acts=frame[ACT].to_numpy(float);scores=sup.scores(pd.DataFrame(scaled,columns=OBS),pd.DataFrame(acts,columns=ACT));locals_=sup.local_action_support(pd.DataFrame(scaled,columns=OBS));pending=[]
  for z,k in enumerate(ix):
   j=pos[z];a=acts[z];ss=float(scores['state'][z]);js=float(scores['joint'][z]);loc=locals_[z];severe=ss>thr['state_severe'];jw=js>thr['joint_warning'];jsev=js>thr['joint_severe'];local=a.copy();fbraw=None;fb=None;empty=False
   item={'run':int(k),'episode_id':str(frame.iloc[z].episode_id),'row_id':str(frame.iloc[z].row_id),'timestamp':str(frame.iloc[z].reg_date),'target_timestamp':str(frame.iloc[z].target_timestamp),'step':step,'reward':None,'current':state[z].tolist(),'proposed':a.tolist(),'current_state_distance':ss,'proposed_joint_distance':js,'current_state_warning':ss>thr['state_warning'],'current_state_severe':severe,'proposed_joint_warning':jw,'proposed_joint_severe':jsev,'recorded_state_distance':float(recorded['state'][j]),'recorded_joint_distance':float(recorded['joint'][j]),'recorded_state_severe':bool(recorded['state'][j]>thr['state_severe']),'recorded_joint_severe':bool(recorded['joint'][j]>thr['joint_severe']),'prior_action_changes':bool(changed[k]),'local_lower':[loc['continuous'][a][0] for a in ACT[:3]],'local_upper':[loc['continuous'][a][1] for a in ACT[:3]],'binary_masks':[loc['binary'][a] for a in ACT[3:]],'executed':False,'termination':None}
   if severe:terms[k]='severe_state_ood';done[k]=True;item['termination']=terms[k];rows.append(item);continue
   if jw:local,empty=project(a,loc)
   if jsev:fbraw,fb,fe=fallback(state[z],loc);empty|=fe;counter[k]+=1
   else:counter[k]=0
   if empty:terms[k]='unsupported_local_support_failure';done[k]=True;item['termination']=terms[k];rows.append(item);continue
   preweather=fb if jsev else local;weather,wi=project_mdp_v1_action_constraints(dict(zip(ACT,preweather)),dict(zip(OBS,state[z])),CFG);final=np.array([weather[a] for a in ACT]);item.update(local_projected=local.tolist(),fallback_raw=fbraw.tolist() if fbraw is not None else None,fallback_action=fb.tolist() if fb is not None else None,preweather_action=preweather.tolist(),weather_capped_action=final.tolist(),final_action=final.tolist(),fallback_used=bool(jsev),local_changed=(local!=a).tolist(),weather_changed=(final!=preweather).tolist(),weather_rule_active=bool(wi['weather_constraint_applied']),counter=int(counter[k]));pending.append((z,k,j,final,item))
  if not pending:continue
  zz=np.array([x[0] for x in pending]);kk=np.array([x[1] for x in pending]);jj=np.array([x[2] for x in pending]);final=np.array([x[3] for x in pending]);xx=np.column_stack([scaled[zz],final]);pred=rt.model.predict(pd.DataFrame(xx,columns=OBS+ACT)).next_observation.to_numpy(float)
  fs=sup.scores(pd.DataFrame(scaled[zz],columns=OBS),pd.DataFrame(final,columns=ACT))['joint'];valid=np.isfinite(pred).all(axis=1)&(pred>=np.array([-60,-60,0,0])).all(axis=1)&(pred<=np.array([80,80,100,100])).all(axis=1)
  good=np.flatnonzero(valid)
  nxt=np.full((len(pending),56),np.nan)
  if len(good):
   future=b.iloc[jj[good]][['next_'+c for c in OBS]].to_numpy(float);times=b.iloc[jj[good]].target_timestamp.to_numpy(dtype='datetime64[ns]').astype('int64');nxt[good]=batch.advance(kk[good],pred[good],future,final[good],times)
   close(nxt[good][:,[I['obs_prev_'+a] for a in ACT]],final[good]);close(nxt[good][:,:4],pred[good]);assert np.isfinite(nxt[good]).all()
   ns=sup.scores(pd.DataFrame((nxt[good]-means)/scale,columns=OBS),pd.DataFrame(final[good],columns=ACT))['state']
  nmap=dict(zip(good,ns if len(good) else []))
  for q,(z,k,j,a,item) in enumerate(pending):
   item.update(prediction=pred[q].tolist(),final_joint_distance=float(fs[q]),final_joint_warning=bool(fs[q]>thr['joint_warning']),final_joint_severe=bool(fs[q]>thr['joint_severe']))
   if not valid[q]:
    terms[k]='nan_inf' if not np.isfinite(pred[q]).all() else 'physical_violation';done[k]=True;item['termination']=terms[k];errors.append(item);rows.append(item);continue
   if step==0 or k in scalar_ids:
    sr=scalars[k];close(sr.current[OBS],state[z]);ai=dict(zip(ACT,a));inp=sr.model_input(ai);close(inp[OBS],scaled[z]);spred=rt.model.predict(inp.to_frame().T).next_observation.iloc[0];close(spred,pred[q]);tar=b.iloc[j].copy();tar[OBS]=b.iloc[j][['next_'+c for c in OBS]].to_numpy(float);tar['reg_date']=b.iloc[j].target_timestamp;exog=sr.exogenous(tar)
    assert not any(c in exog for c in TARGETS+['obs_indoor_temp_c','obs_indoor_humidity_pct'])
    sn=sr.advance(prediction=dict(zip(TARGETS,pred[q])),action=ai,exogenous=exog);close(sn[OBS],nxt[q]);assert sn.obs_indoor_temp_c==pred[q,0] and sn.obs_indoor_humidity_pct==pred[q,2];maxparity=max(maxparity,float(np.max(np.abs(sn[OBS].to_numpy(float)-nxt[q]))));scalar_checks+=1
   executed[k]+=1;changed[k]|=bool((a!=acts[z]).any());dist=float(nmap[q]);item.update(executed=True,next_state=nxt[q].tolist(),next_state_distance=dist,next_state_warning=dist>thr['state_warning'],next_state_severe=dist>thr['state_severe'],recorded_target_state_distance=float(targetscore[j]),recorded_target_severe=bool(targetscore[j]>thr['state_severe']),recorded_target=b.iloc[j][['next_'+c for c in TARGETS]].to_numpy(float).tolist(),action_changed=bool((a!=acts[z]).any()),path_action_changed=bool(changed[k]))
   if executed[k]>=lengths[k]:terms[k]='natural_completion';done[k]=True
   elif counter[k]>=3:terms[k]='repeated_severe_action_ood';done[k]=True
   item['termination']=terms[k] if done[k] else None;rows.append(item)
 assert done.all() and all(terms)
 episodes=[{'run':k,'episode_id':str(groups[k].iloc[0].episode_id),'series_id':str(groups[k].iloc[0].series_id),'source_confidence':str(groups[k].iloc[0].source_confidence),'start_timestamp':str(groups[k].iloc[0].reg_date),'scheduled_steps':int(lengths[k]),'executed_steps':int(executed[k]),'termination':terms[k]} for k in range(n)]
 return rows,episodes,{'scalar_steps_checked':scalar_checks,'max_full56_parity_difference':maxparity,'all_reset_points':n,'physical_failures':len(errors),'thresholds':thr}

def run(crop):
 check();out=D/crop;out.mkdir(exist_ok=False)
 # Reward entrypoint is guarded; no legacy environment is instantiated.
 with patch('geas35.rl.mdp_v1.compute_mdp_v1_reward',side_effect=AssertionError('보상 호출 금지')) as guard:
  a,ep,checks=replay(crop,0);b,ep2,checks2=replay(crop,1);assert not guard.called
 assert ep==ep2 and len(a)==len(b)
 numeric=['current','proposed','local_projected','fallback_raw','fallback_action','preweather_action','weather_capped_action','final_action','prediction','next_state','recorded_target']
 maxdiff=0.
 for x,y in zip(a,b):
  assert x.keys()==y.keys()
  for k in x:
   if k in numeric and x[k] is not None:
    close(x[k],y[k]);maxdiff=max(maxdiff,float(np.nanmax(np.abs(np.asarray(x[k])-np.asarray(y[k])))))
   elif isinstance(x[k],float):close(x[k],y[k])
   else:assert x[k]==y[k],(k,x[k],y[k])
 write(out/'replay_validation.json',{'PASS':True,'full_reload_repeats':2,'same_admission_actions_classification_termination':True,'max_prediction_state_action_difference':maxdiff,'reward_calls':0,**checks})
 pd.DataFrame(a).to_parquet(out/'step_trace.parquet',index=False);pd.DataFrame(b).to_parquet(out/'repeat_step_trace.parquet',index=False);pd.DataFrame(ep).to_csv(out/'episode_summary.csv',index=False)
 summarize(crop,a,ep,checks);check();print(crop,'완료',flush=True)

def summarize(crop,rows,ep,checks):
 out=D/crop;f=pd.DataFrame(rows);e=pd.DataFrame(ep);ok=f[f.executed];n=len(ok);sched=int(e.scheduled_steps.sum());ex=int(e.executed_steps.sum());guard=ex/sched>=.5 and float((e.executed_steps>=np.minimum(e.scheduled_steps,3)).mean())>=.5 and (not (e.scheduled_steps>=12).any() or (e.executed_steps>=12).any())
 term=e.termination.value_counts().to_dict();physical=term.get('physical_violation',0);nf=term.get('nan_inf',0);technical=not physical and not nf and not term.get('unsupported_local_support_failure',0) and not term.get('system_invalid',0)
 action=[]
 for j,c in enumerate(ACT):
  proposed=np.array(ok.proposed.to_list())[:,j];final=np.array(ok.final_action.to_list())[:,j];local=np.array(ok.local_projected.to_list())[:,j];pre=np.array(ok.preweather_action.to_list())[:,j];delta=np.abs(final-proposed);v={'action':c,'executed_denominator':n,'local_projection_rate':float(np.mean(local!=proposed)),'weather_cap_rate':float(np.mean(final!=pre)),'fallback_applied_rate':float(ok.fallback_used.mean()),'fallback_changed_action_rate':float(np.mean(ok.fallback_used.to_numpy()&(pre!=local))),'unchanged_rate':float(np.mean(final==proposed))}
  if j<3:
   for name,dd in [('final_change',delta),('local_projection_magnitude',np.abs(local-proposed))]:v.update({name+'_mean':float(dd.mean()),**{name+f'_p{q}':float(np.percentile(dd,q)) for q in [50,90,99,100]}})
  else:v.update(zero_to_one=int(((proposed==0)&(final==1)).sum()),one_to_zero=int(((proposed==1)&(final==0)).sum()))
  action.append(v)
 pd.DataFrame(action).to_csv(out/'intervention_by_action.csv',index=False)
 drift=[]
 for h in [1,3,6,12]:
  for category,ff in [('all_survivors',ok),('unchanged_action_path',ok[~ok.path_action_changed]),('changed_action_path',ok[ok.path_action_changed])]:
   sub=ff[ff.step.eq(h-1)]
   if sub.empty:continue
   err=np.array(sub.prediction.to_list())-np.array(sub.recorded_target.to_list())
   for j,c in enumerate(TARGETS):drift.append({'target':c,'minutes':h*5,'category':category,'n':len(sub),'mae':float(np.abs(err[:,j]).mean()),'rmse':float(np.sqrt(np.mean(err[:,j]**2))),'signed_drift':float(err[:,j].mean()),'p90_abs_error':float(np.percentile(np.abs(err[:,j]),90))})
 pd.DataFrame(drift).to_csv(out/'drift_diagnostics.csv',index=False)
 origins=[]
 for k,g in f.groupby('run',sort=True):
  earlier=False
  for _,x in g.iterrows():
   if x.current_state_severe:origins.append({'run':int(k),'step':int(x.step),'timestamp':x.timestamp,'location':'current_state','classification':'recorded_overlap' if x.recorded_state_severe else ('rollout_only_after_action_changes' if x.prior_action_changes else 'rollout_only_unchanged_actions'),'earlier_recorded_severe':earlier})
   if bool(x.executed) and bool(x.next_state_severe):origins.append({'run':int(k),'step':int(x.step+1),'timestamp':x.target_timestamp,'location':'next_state','classification':'recorded_overlap' if x.recorded_target_severe else ('rollout_only_after_action_changes' if x.path_action_changed else 'rollout_only_unchanged_actions'),'earlier_recorded_severe':earlier or bool(x.recorded_state_severe)})
   earlier|=bool(x.recorded_state_severe)
 pd.DataFrame(origins,columns=['run','step','timestamp','location','classification','earlier_recorded_severe']).to_csv(out/'ood_origins.csv',index=False)
 first=[]
 for k,g in f.groupby('run'):
  for level in ['warning','severe']:
   th=checks['thresholds']['state_'+level]
   hits=[(int(x.step+1),x.target_timestamp) for _,x in g[g.executed].iterrows() if x.next_state_distance>th and x.recorded_target_state_distance<=th]
   if hits:first.append({'run':int(k),'level':level,'first_step':hits[0][0],'timestamp':hits[0][1]})
 pd.DataFrame(first,columns=['run','level','first_step','timestamp']).to_csv(out/'first_rollout_only_ood.csv',index=False)
 exposure={}
 for key,ff in [('current_state',f),('proposed_joint',f),('final_joint',ok),('next_state',ok)]:exposure[key]={'denominator':len(ff),**{lev:float(ff[key+'_'+lev].mean()) for lev in ['warning','severe']}}
 result={'crop':crop,'status':'PASS' if technical and guard else 'BLOCKED','technical_gate':technical,'practical_guard':guard,'scheduled_runs':len(e),'scheduled_episodes':int(e.episode_id.nunique()),'executed_runs':int((e.executed_steps>0).sum()),'scheduled_steps':sched,'executed_steps':ex,'execution_rate':ex/sched,'fraction_reaching15_or_short_end':float((e.executed_steps>=np.minimum(e.scheduled_steps,3)).mean()),'length':{'mean':float(e.executed_steps.mean()),'median':float(e.executed_steps.median()),'p10':float(e.executed_steps.quantile(.1)),'p90':float(e.executed_steps.quantile(.9)),'max':int(e.executed_steps.max())},'terminations':{k:{'count':int(term.get(k,0)),'rate':float(term.get(k,0)/len(e))} for k in ['natural_completion','severe_state_ood','repeated_severe_action_ood','unsupported_local_support_failure','physical_violation','nan_inf','system_invalid','other']},'exposure':exposure,'projection_rate':float(np.mean([any(x) for x in ok.local_changed])),'fallback_rate':float(ok.fallback_used.mean()),'weather_cap_rate':float(np.mean([any(x) for x in ok.weather_changed])),'physical_violations':physical,'nan_inf':nf,'rollout_only_next_severe_rate':float(((ok.next_state_distance>checks['thresholds']['state_severe'])&(ok.recorded_target_state_distance<=checks['thresholds']['state_severe'])).mean()),'deterministic_replay':'PASS','reward':None,'provenance':C['selected'][crop]['command_source']}
 write(out/'result.json',result)

def report():
 check();results={c:read(D/c/'result.json') for c in C['selected']};blocked=[c for c,v in results.items() if v['status']!='PASS'];status='TRANSITION 4CH STEP16 — PASS / PPO-ENTRY REASSESSMENT READY' if not blocked else ('TRANSITION 4CH STEP16 — BLOCKED' if len(blocked)==3 else 'TRANSITION 4CH STEP16 — PARTIAL / '+', '.join(blocked)+' BLOCKED')
 m=read(D/'step16_manifest.json');m.update(status=status,phase='completed',Step16_completed=True,validation_rollout_started=True,crop_rollout_results=results,reward_enabled=False,original_inputs_unchanged=True);write(D/'step16_manifest.json',m)
 lines=['# 4CH Step16 순차 연구 환경 검증','',status,'','사용자가 확정한 부모 frozen 순서: proposed joint 평가 → warning 시 local projection / severe 시 fallback → 최종 weather cap → Transition. 다음 상태 support는 기록하되, 부모와 동일하게 다음 decision에서 state severe를 검사하며 natural end가 반복 joint 종료보다 우선한다. 보상은 모든 trace에서 null이고 계산하지 않았다.','', '| 작물 | 실행/예정 step | 실행률 | 정상 | state severe | joint 반복 severe | 판정 |','|---|---:|---:|---:|---:|---:|---|']
 for c,v in results.items():lines.append(f"| {c} | {v['executed_steps']}/{v['scheduled_steps']} | {v['execution_rate']:.2%} | {v['terminations']['natural_completion']['count']} | {v['terminations']['severe_state_ood']['count']} | {v['terminations']['repeated_severe_action_ood']['count']} | {v['status']} |")
 lines+=['','## 실제 실행 intervention','', '| 작물 | local projection | fallback | weather로 값 변경 | 물리 위반 | NaN/Inf | reload/replay |','|---|---:|---:|---:|---:|---:|---|']
 for c,v in results.items():lines.append(f"| {c} | {v['projection_rate']:.2%} | {v['fallback_rate']:.2%} | {v['weather_cap_rate']:.2%} | {v['physical_violations']} | {v['nan_inf']} | {v['deterministic_replay']} |")
 lines+=['','## OOD 노출','', '| 작물 | current state warning/severe | proposed joint warning/severe | rollout-only next severe |','|---|---:|---:|---:|']
 for c,v in results.items():
  s=v['exposure']['current_state'];j=v['exposure']['proposed_joint'];lines.append(f"| {c} | {s['warning']:.2%}/{s['severe']:.2%} | {j['warning']:.2%}/{j['severe']:.2%} | {v['rollout_only_next_severe_rate']:.2%} |")
 lines+=['','warning은 severe 포함. current/proposed 노출 분모는 미실행 종료 decision도 포함한 모든 attempted decision이며 final/next/intervention 분모는 executed steps다. recorded overlap과 action 변경 후 rollout-only를 모델 단독 오차로 합치지 않는다. OOD 최초 발생점과 이전 recorded severe 여부는 각 crop ood_origins.csv/first_rollout_only_ood.csv에 보존했다.','', '5/15/30/60분 drift는 실제 생존한 run의 조건부 진단이며 전체 Phase1/2 성능을 대체하지 않는다. action 변경 경로와 unchanged 경로를 분리했다. 물리 위반을 clipping으로 보정하지 않았고 support/model/scaler/threshold는 변경하지 않았다.','', '실행 가능성은 기존 50% 실행률, 50%의 run이15분/짧은run끝 도달, 예정된60분run이 있으면 실제1개 이상 도달 기준으로 평가했다. Future-F reward/Profile/QualityEvaluator/gamma/GAE와 legacy reward/PPO는 호출하지 않았다. Test/P0/live/GEAS2.5 작업 없음.','', 'melon/strawberry legacy_command_snapshot_assumption과 cucumber api_transmitted_command를 유지한다. online actuator-safe fallback은 별도 BLOCKED다.','', '다음 단계: 새 4CH 환경과 고정 Future-F cucumber reward의 PPO-entry 재평가.' if not blocked else '다음 단계: BLOCKED 작물의 recorded-domain OOD와 action 변경/rollout 이탈을 비학습 방식으로 먼저 분리한다. PPO 진입 보류.']
 (D/'final_step16_4ch.md').write_text('\n'.join(lines)+'\n');print(status,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run','report']);ap.add_argument('--crop',choices=list(C['selected']));a=ap.parse_args()
 try:
  if a.stage=='freeze':freeze()
  elif a.stage=='run':run(a.crop)
  else:report()
 except Exception as e:write(D/f'{a.stage}_{a.crop}_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
