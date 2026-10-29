"""Execute pinned 4CH handoff/support protocol. No fit, Step16 or live access."""
from pathlib import Path
import os,sys,json,shutil,tempfile,importlib.util,argparse,subprocess,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'))
import numpy as np
import pandas as pd
from unittest.mock import patch
from geas35.models.transition.handoff_four_channel_v1 import *
from geas35.models.transition.four_channel_batch import FourChannelBatchUpdater
from geas35.rl.support_v1 import SupportBuildConfig,build_support_artifact,StateActionSupportModel,conformal_quantile
from geas35.rl.bound_support_four_channel_v1 import load_bound_support
from geas35.models.transition.action_response import evaluate_action_response
B=R/'experiments/rl_policy_training/artifacts';H=R/'experiments/transition_model_selection/artifacts/step14_hlv15_b60_r6_4ch_v1';S=B/'step15_hlv15_b60_r6_4ch_v1';D=B/'transition_4ch_candidate_reselection_v1'
C=contract(R);OBS=C['schema']['observation_columns'];ACT=C['schema']['action_columns'];P=Path(json.loads((B/'transition_4ch_phase0_canonical_v1/phase0_manifest.json').read_text())['fitting_authority'])
sp=importlib.util.spec_from_file_location('reselection_read_helpers',R/'experiments/rl_policy_training/scripts/run_transition_4ch_reselection.py');core=importlib.util.module_from_spec(sp);sp.loader.exec_module(core)

def read(p):return json.loads(Path(p).read_text())
def write(p,x):core.write(p,x)
def rec(p):return {'path':str(Path(p).relative_to(R)),'sha256':sha(p)}
def close(a,b):np.testing.assert_allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=1e-6,atol=1e-8)
def code():return [Path(__file__),R/'src/geas35/models/transition/handoff_four_channel_v1.py',R/'src/geas35/rl/bound_support_four_channel_v1.py',R/'src/geas35/rl/support_v1.py']
def freeze():
 assert read(H/'execution_manifest.json')['contract_sha256']==CONTRACT_HASH
 core.lock(True)
 for record in C['parent_methodology_only']['code'].values():checked(R,record)
 assert not (H/'implementation_lock.json').exists()
 write(H/'implementation_lock.json',{'contract_sha256':CONTRACT_HASH,'code':[rec(p) for p in code()],'reselection_binding':rec(D/'execution_binding.json')})
 print('IMPLEMENTATION LOCKED',flush=True)
def check():
 contract(R);core.lock(True)
 for r in read(H/'implementation_lock.json')['code']:checked(R,r)

def negative_tests(out,crop):
 results=[]
 with tempfile.TemporaryDirectory() as t:
  p=Path(t)/crop;shutil.copytree(out,p);original=read(p/'manifest.json')
  def reject(label,pinned,requested=crop):
   with patch('geas35.models.transition.handoff_four_channel_v1.pickle.load',side_effect=AssertionError('Deserialized before rejection')) as mocked:
    try:BoundFourChannelRuntime(p,crop=requested,manifest_sha256=pinned,root=R)
    except ValueError:assert not mocked.called;results.append(label)
    else:raise AssertionError('Invalid binding accepted: '+label)
  pin=sha(p/'manifest.json');reject('wrong_crop',pin,next(c for c in C['selected'] if c!=crop));reject('manifest_hash','0'*64)
  for f in ['model.pkl','schema.json','observation_scaler.json']:
   q=p/f;raw=q.read_bytes();q.write_bytes(raw+b' ');reject('mutated_'+f,pin);q.write_bytes(raw)
  for label in ['legacy55_3','reordered_input','reordered_target','wrong_algorithm','path_substitution','runtime_adapter','dependency']:
   m=json.loads(json.dumps(original))
   if label=='legacy55_3':m['version']='geas35.step14.hlv15_b60_r6.v1'
   elif label=='reordered_input':m['schema_contract']['input_order'][0:2]=m['schema_contract']['input_order'][1::-1]
   elif label=='reordered_target':m['schema_contract']['target_order'].reverse()
   elif label=='wrong_algorithm':m['algorithm']='persistence'
   elif label=='path_substitution':m['bundle_paths']['model.pkl']='elsewhere/model.pkl'
   elif label=='runtime_adapter':m['authority']['runtime_adapter']['sha256']='0'*64
   else:m['dependencies']['python']='3.11'
   write(p/'manifest.json',m);reject(label,sha(p/'manifest.json'))
 return results

def handoff(crop):
 check();b=C['selected'][crop];out=H/crop;out.mkdir(exist_ok=False)
 for name,key in [('model.pkl','model'),('schema.json','schema'),('observation_scaler.json','observation_scaler')]:shutil.copyfile(checked(R,b[key]),out/name)
 m={'version':VERSION,'crop':crop,'algorithm':b['algorithm'],'contract_sha256':CONTRACT_HASH,'authority':C['authoritative'],'selected_binding':b,'schema_contract':C['schema'],'dependencies':C['environment'],'bundle_hashes':{p.name:sha(p) for p in out.iterdir()},'bundle_paths':{n:n for n in ['model.pkl','schema.json','observation_scaler.json']},'implementation_bindings':[rec(p) for p in code()],'activation':'research bundle only; no production/default handoff replacement'}
 write(out/'manifest.json',m);pin=sha(out/'manifest.json');rt=BoundFourChannelRuntime(out,crop=crop,manifest_sha256=pin,root=R)
 data,base,ids,starts,mh,qp=core.setup(crop);ds,_=core.dataset(crop,'validation');pred=rt.predict(data[OBS],data[ACT].to_numpy(float));ref=pd.read_parquet(D/crop/b['algorithm']/'fit/validation_predictions.parquet');close(pred,ref)
 raw=rt.model.predict(ds.x);close(raw.next_observation.to_numpy(float)-data[TARGETS].to_numpy(float),raw.raw_prediction.to_numpy(float))
 arr=np.load(D/crop/b['algorithm']/'rollout/predictions.npz');assert list(arr['start_row_id'])==list(ids.index.astype(str));choices=np.flatnonzero(mh==12);selected=choices[np.unique(np.linspace(0,len(choices)-1,8).astype(int))];trace=[]
 for k in selected:
  row=data.iloc[starts[k]];rt.reset(row,base);batch=FourChannelBatchUpdater(base,np.array([qp[k]]),row[OBS].to_numpy(float)[None,:])
  for step in range(12):
   row=data.iloc[starts[k]+step];a={c:float(row[c]) for c in ACT};target=core.target(row);before=rt.adapter.current[OBS].to_numpy(float).copy()
   p,n=rt.advance(a,rt.adapter.exogenous(target));close(p,arr['predictions'][k,step]);close(n[['obs_prev_'+c for c in ACT]],list(a.values()))
   bn=batch.advance(np.array([0]),p.to_numpy(float)[None,:],target[OBS].to_numpy(float)[None,:],np.array([list(a.values())]),np.array([pd.Timestamp(target.reg_date).value]))[0];close(n[OBS],bn)
   trace.append({'row_id':str(row.row_id),'start_row_id':str(ids.index[k]),'step':step,'current':before.tolist(),'action':list(a.values()),'prediction':p.tolist(),'next':n[OBS].to_numpy(float).tolist(),'prediction_max_abs_diff':float(np.max(np.abs(p.to_numpy(float)-arr['predictions'][k,step]))),'state_max_abs_diff':float(np.max(np.abs(n[OBS].to_numpy(float)-bn)))})
 pd.DataFrame(trace).to_parquet(out/'runtime_parity_trace.parquet',index=False)
 resets=[]
 for split in ['train','validation']:
  f=core.frames(crop,split);history=pd.read_parquet(P/crop/f'{split}_sensor_prehistory.parquet');start=f.mdp_v1_rollout_id.ne(f.mdp_v1_rollout_id.shift())|f.reg_date.ne(f.target_timestamp.shift());groups=[g for _,g in f.groupby(start.cumsum(),sort=False)];maxerr=0.;steps=0
  rt.reset(groups[0].iloc[0],history)
  for g in groups:
   row=g.iloc[0];cur=rt.adapter.reset(row);close(cur[OBS],row[OBS]);tar=core.target(row);a={c:float(row[c]) for c in ACT};n=rt.adapter.advance(prediction=dict(zip(TARGETS,tar[TARGETS].to_numpy(float))),action=a,exogenous=rt.adapter.exogenous(tar));close(n[OBS],tar[OBS]);maxerr=max(maxerr,float(np.max(np.abs(n[OBS].to_numpy(float)-tar[OBS].to_numpy(float)))));steps+=1
  longest=max(groups,key=len);rt.adapter.reset(longest.iloc[0])
  for _,row in longest.iloc[:301].iterrows():
   tar=core.target(row);a={c:float(row[c]) for c in ACT};n=rt.adapter.advance(prediction=dict(zip(TARGETS,tar[TARGETS].to_numpy(float))),action=a,exogenous=rt.adapter.exogenous(tar));close(n[OBS],tar[OBS]);steps+=1
  resets.append({'split':split,'all_reset_points':len(groups),'teacher_steps':steps,'longest_available':len(longest),'long_teacher_steps':min(301,len(longest)),'max_first_step_diff':maxerr})
 negatives=negative_tests(out,crop)
 ar=evaluate_action_response(rt.model,ds,minimum_support=100,continuous_change_threshold=.10,probe_count=256).to_artifact();viol=0;nonfinite=0;probeinfo=[]
 for a,v in ar['actions'].items():
  lo,hi=v['counterfactual_low'],v['counterfactual_high']
  if not np.isfinite(lo) or not hi>lo:probeinfo.append({'action':a,'status':'no_variable_range'});continue
  q=ds.x.iloc[np.linspace(0,len(ds.x)-1,min(256,len(ds.x)),dtype=int)].copy()
  for value in [lo,hi]:
   q[a]=value;y=rt.model.predict(q).next_observation.to_numpy(float);nf=int((~np.isfinite(y)).sum());bad=int(((y<[-60,-60,0,0])|(y>[80,80,100,100])).sum());nonfinite+=nf;viol+=bad;probeinfo.append({'action':a,'value':value,'nonfinite':nf,'physical_violations':bad})
 actionpass=ar['eligible'] and not ar['fully_action_insensitive'] and nonfinite==0 and viol==0
 write(out/'action_response.json',{'report':ar,'probes':probeinfo,'pass':actionpass});write(out/'reset_history_validation.json',resets)
 gate={'crop':crop,'status':'PASS' if actionpass else 'BLOCKED','binding':True,'negative_before_deserialize':negatives,'one_step_rows':len(pred),'one_step_max_abs_diff':float(np.max(np.abs(pred.to_numpy()-ref.to_numpy()))),'runtime_parity':True,'sequential_steps':len(trace),'full56_parity':True,'own_residual_baseline':True,'action_response':actionpass,'probe_nonfinite':nonfinite,'probe_physical_violations':viol,'all_resets':sum(v['all_reset_points'] for v in resets),'manifest_sha256':pin}
 write(out/'gate.json',gate);print('HANDOFF',crop,gate['status'],flush=True)

def registry(train,cfg):
 episodes=np.array(sorted(train.episode_id.astype(str).unique()));rng=np.random.default_rng(cfg.seed);rng.shuffle(episodes);n=min(len(episodes)-1,max(1,round(len(episodes)*cfg.calibration_episode_fraction)));cal=set(episodes[:n]);mask=train.episode_id.astype(str).isin(cal).to_numpy();frames=[]
 for role,m,cap in [('reference',~mask,cfg.max_reference_rows),('calibration',mask,cfg.max_calibration_rows)]:
  ix=np.flatnonzero(m)
  if len(ix)>cap:ix=np.sort(rng.choice(ix,cap,replace=False))
  f=train.iloc[ix][['row_id','episode_id','reg_date']].copy();f['position']=ix;f['role']=role;f['split']='train';frames.append(f)
 return pd.concat(frames,ignore_index=True),{'reference':sorted(set(episodes)-cal),'calibration':sorted(cal)}

def support(crop):
 check();assert read(H/crop/'gate.json')['status']=='PASS';out=S/crop;out.mkdir(parents=True,exist_ok=False);train=pd.read_parquet(P/crop/'train_scaled.parquet');val=pd.read_parquet(P/crop/'validation_scaled.parquet');cfg=SupportBuildConfig(**C['support']['config'])
 a,_=build_support_artifact(train,crop=crop,state_columns=OBS,action_columns=ACT,episode_column='episode_id',output_dir=out,config=cfg);m=read(a);reg,ep=registry(train,cfg);reg.to_parquet(out/'fit_row_registry.parquet',index=False);write(out/'episode_membership.json',ep);assert not set(ep['reference'])&set(ep['calibration']);assert not set(reg.row_id)&set(val.row_id)
 model=StateActionSupportModel(a);ref=train.iloc[reg.loc[reg.role.eq('reference'),'position']];cal=train.iloc[reg.loc[reg.role.eq('calibration'),'position']];np.testing.assert_array_equal(model.reference_state_raw,ref[OBS].to_numpy(float));np.testing.assert_array_equal(model.reference_action_raw,ref[ACT].to_numpy(float))
 cal_scores=model.scores(cal[OBS],cal[ACT]);thresholds={}
 for kind in ['state','joint']:
  for lev,p in [('warning',.95),('severe',.99)]:
   key=kind+'_'+lev;t=conformal_quantile(cal_scores[kind],p);assert t==m['thresholds'][key] and np.isfinite(t);assert np.mean(cal_scores[kind]<=t)>=p;thresholds[key]=t
  assert thresholds[kind+'_warning']<=thresholds[kind+'_severe']
 pd.DataFrame({'row_id':cal.row_id.to_numpy(),**cal_scores}).to_parquet(out/'calibration_scores.parquet',index=False)
 second=out/'independent_rebuild';a2,_=build_support_artifact(train.copy(),crop=crop,state_columns=OBS,action_columns=ACT,episode_column='episode_id',output_dir=second,config=cfg);reg2,ep2=registry(train.copy(),cfg);pd.testing.assert_frame_equal(reg,reg2);assert ep==ep2 and read(a2)==m;reg2.to_parquet(second/'fit_row_registry.parquet',index=False);write(second/'episode_membership.json',ep2)
 r2=StateActionSupportModel(a2);np.testing.assert_array_equal(model.reference_state_raw,r2.reference_state_raw);np.testing.assert_array_equal(model.reference_action_raw,r2.reference_action_raw)
 vs=model.scores(val[OBS],val[ACT]);rs=r2.scores(val[OBS],val[ACT]);local=model.local_action_support(val[OBS]);other=r2.local_action_support(val[OBS]);assert local==other
 for k in vs:close(vs[k],rs[k])
 independent=[]
 for i in np.unique(np.linspace(0,len(val)-1,8).astype(int)):
  state=val.iloc[i][OBS].to_numpy(float)*model.state_weights;act=val.iloc[i][ACT].to_numpy(float)*model.action_weights;joint=np.r_[state/np.sqrt(2),act/np.sqrt(2)]
  for kind,rr,q in [('state',model._state_reference,state),('joint',model._joint_reference,joint)]:
   kth=np.partition(np.sqrt(((rr-q)**2).sum(axis=1)),49)[49];close(kth,vs[kind][i]);independent.append({'position':int(i),'kind':kind,'difference':float(abs(kth-vs[kind][i]))})
 rows=val[[c for c in ['row_id','episode_id','reg_date','series_id','source_confidence'] if c in val]].copy();rows['source_confidence']=C['selected'][crop]['command_source'];summ=[]
 for kind in vs:
  rows[kind+'_distance']=vs[kind]
  for lev in ['warning','severe']:rows[kind+'_'+lev]=vs[kind]>thresholds[kind+'_'+lev]
  summ.append({'crop':crop,'kind':kind,'rows':len(val),'warning_threshold':thresholds[kind+'_warning'],'severe_threshold':thresholds[kind+'_severe'],'warning_rate':float(rows[kind+'_warning'].mean()),'severe_rate':float(rows[kind+'_severe'].mean()),**{f'p{q}':float(np.percentile(vs[kind],q)) for q in [0,50,90,95,99,100]}})
 rows.to_parquet(out/'validation_support_scores.parquet',index=False);pd.DataFrame(summ).to_csv(out/'support_summary.csv',index=False)
 groups=[]
 for keys in [['source_confidence'],['source_confidence','episode_id']]:
  for v,g in rows.groupby(keys,dropna=False):
   v=v if isinstance(v,tuple) else (v,);groups.append({**dict(zip(keys,v)),'rows':len(g),**{k:float(g[k].mean()) for k in ['state_warning','state_severe','joint_warning','joint_severe']}})
 pd.DataFrame(groups).to_csv(out/'source_episode_summary.csv',index=False)
 localrows=[]
 for i,v in enumerate(local):
  z={'row_id':str(val.iloc[i].row_id)}
  for a,(lo,hi) in v['continuous'].items():assert 0<=lo<=hi<=1;z[a+'_lower']=lo;z[a+'_upper']=hi
  for a,values in v['binary'].items():z[a+'_mask']=values;z[a+'_empty']=not values
  localrows.append(z)
 pd.DataFrame(localrows).to_parquet(out/'validation_local_support.parquet',index=False)
 ls={'continuous_adjustments':m['local_action_support']['continuous_conformal_adjustments'],'binary_minimum':m['local_action_support']['binary_minimum_neighbor_count'],'validation':{}}
 for a in ACT[:3]:ls['validation'][a]={'mean_width':float(np.mean([z[a+'_upper']-z[a+'_lower'] for z in localrows])),'logged_outside_rate':float(np.mean([(val.iloc[i][a]<z[a+'_lower'] or val.iloc[i][a]>z[a+'_upper']) for i,z in enumerate(localrows)]))}
 for a in ACT[3:]:ls['validation'][a]={'empty_masks':sum(z[a+'_empty'] for z in localrows),'logged_disallowed_rate':float(np.mean([val.iloc[i][a] not in z[a+'_mask'] for i,z in enumerate(localrows)]))}
 write(out/'local_action_support.json',ls);write(out/'thresholds.json',{'state_conformal95':thresholds['state_warning'],'state_conformal99':thresholds['state_severe'],'joint_conformal95':thresholds['joint_warning'],'joint_conformal99':thresholds['joint_severe'],'fit_split':'train_only'})
 for kind,dim in [('state',56),('joint',62)]:write(out/(kind+'_support.json'),{'kind':kind,'dimension':dim,'support':rec(a),'arrays':rec(out/'support_reference.npz'),'thresholds':{k:v for k,v in thresholds.items() if k.startswith(kind)},'contract_sha256':CONTRACT_HASH})
 comparison={'passed':True,'exact_membership_groups_weights_thresholds_adjustments_arrays':True,'full_validation_reload_scores_local_parity':True,'independent_k50':independent,'hashes':{n:{'first':sha(out/n),'second':sha(second/n)} for n in ['state_action_support.json','support_reference.npz','fit_row_registry.parquet','episode_membership.json']}}
 write(out/'deterministic_reload_validation.json',comparison)
 files={'handoff':rec(H/crop/'manifest.json'),'schema':C['selected'][crop]['schema'],'scaler':C['selected'][crop]['observation_scaler'],'train':rec(P/crop/'train_scaled.parquet'),'support':rec(a),'reference':rec(out/'support_reference.npz'),'fit_registry':rec(out/'fit_row_registry.parquet'),'thresholds':rec(out/'thresholds.json'),'local':rec(out/'local_action_support.json')}
 write(out/'binding.json',{'version':'geas35.step15.hlv15_b60_r6.4ch.v1','crop':crop,'contract_sha256':CONTRACT_HASH,'files':files,'code':[rec(p) for p in code()]})
 _,loaded=load_bound_support(out/'binding.json',manifest_sha256=sha(out/'binding.json'),crop=crop,root=R);sc=loaded.scores(val[OBS],val[ACT]);assert loaded.local_action_support(val[OBS])==local
 for k in vs:close(vs[k],sc[k])
 write(out/'gate.json',{'crop':crop,'status':'PASS','train_rows':len(train),'validation_rows':len(val),'reference_rows':len(ref),'calibration_rows':len(cal),'episode_disjoint':True,'validation_fit_rows':0,'exact56_state_62_joint':True,'calibration_coverage':m['empirical_calibration_coverage'],'thresholds':thresholds,'deterministic_reload':True,'local_support':True,'bound_loader':True,'provenance':C['selected'][crop]['command_source']});print('SUPPORT',crop,'PASS',flush=True)

def regression():
 check();p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(R/'tests'),'-p','test_transition_four_channel_phase0.py'],env={**os.environ,'PYTHONPATH':str(R/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True);(H/'history_regression.log').write_text(p.stdout+p.stderr);write(H/'history_regression.json',{'exit_code':p.returncode,'passed':p.returncode==0,'scope':'synthetic >288, midnight, source/identity/gap, invalid prediction, alias, carry, future leakage'});assert p.returncode==0

def report():
 check();reg=read(H/'history_regression.json')['passed'];rows=[];blocked=[]
 for c in C['selected']:
  hg=read(H/c/'gate.json') if (H/c/'gate.json').exists() else {'status':'BLOCKED'};sg=read(S/c/'gate.json') if (S/c/'gate.json').exists() else {'status':'BLOCKED'}
  ok=reg and hg['status']=='PASS' and sg['status']=='PASS'
  if not ok:blocked.append(c)
  rows.append({'crop':c,'handoff':hg,'support':sg,'passed':ok})
 status='TRANSITION 4CH HANDOFF/SUPPORT — PASS / STEP16 READY' if not blocked else ('TRANSITION 4CH HANDOFF/SUPPORT — BLOCKED' if len(blocked)==3 else 'TRANSITION 4CH HANDOFF/SUPPORT — PARTIAL / '+', '.join(blocked)+' BLOCKED')
 write(S/'final_gate.json',{'status':status,'crops':rows,'history_regression':reg,'contract_sha256':CONTRACT_HASH,'Step16_executed':False,'PPO_executed':False,'online_deployment':'BLOCKED'})
 lines=['# Transition 4CH Handoff / Step15 support', '',status,'','고정 모델 bytes와 canonical Phase0만 사용했다. 재학습/Step16/PPO/Test/P0/Future-F/GEAS2.5 작업은 실행하지 않았다.','', '| crop | handoff/parity/action | support/determinism/reload | state95 | state99 | joint95 | joint99 |','|---|---|---|---:|---:|---:|---:|']
 for r in rows:
  g=r['support'];t=g.get('thresholds',{});lines.append('| '+' | '.join(str(v) for v in [r['crop'],r['handoff']['status'],g['status'],*[t.get(k) for k in ['state_warning','state_severe','joint_warning','joint_severe']]])+' |')
 lines+=['','## Validation score-only','', '| crop | kind | warning (severe 포함) | severe |','|---|---|---:|---:|']
 for r in rows:
  p=S/r['crop']/'support_summary.csv'
  if p.exists():
   for _,v in pd.read_csv(p).iterrows():lines.append(f"| {r['crop']} | {v['kind']} | {v.warning_rate:.6f} | {v.severe_rate:.6f} |")
 lines+=['','Train episode-disjoint calibration seed42/k50/conformal95·99를 그대로 사용했다. Validation은 threshold/local support fitting에 사용하지 않았다. 높은 Validation OOD는 기술 gate와 별개로 다음 sequential rollout에서 확인할 결과이며 기준을 조정하지 않았다.','', '각 crop에 exact reference/calibration registry, 독립 rebuild, source/episode 분포, local continuous interval/binary mask와 빈 mask, action-response probe, 96-step full56 parity trace, negative binding tests를 저장했다. hard weather bounds와 empirical support는 구분한다.','', 'cucumber api_transmitted_command, melon/strawberry legacy_command_snapshot_assumption을 유지한다. 연구용 fallback은 실제 actuator 안전성 승인과 다르며 online deployment BLOCKED를 유지한다.','', '다음 단계는 새 4CH Step16 연구 환경 검증이다. 이번에는 실행하지 않았고 PPO/Future-F reward integration 승인을 부여하지 않는다.']
 (S/'final_handoff_support.md').write_text('\n'.join(lines)+'\n');print(status,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','handoff','support','regression','report']);ap.add_argument('--crop',choices=list(C['selected']));args=ap.parse_args()
 try:
  if args.stage=='freeze':freeze()
  elif args.stage=='handoff':handoff(args.crop)
  elif args.stage=='support':support(args.crop)
  elif args.stage=='regression':regression()
  else:report()
 except Exception as e:
  dest=H if args.stage in ['handoff','regression','freeze'] else S;dest.mkdir(exist_ok=True,parents=True);write(dest/f'{args.stage}_{args.crop}_failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'contract_sha256':CONTRACT_HASH});raise
