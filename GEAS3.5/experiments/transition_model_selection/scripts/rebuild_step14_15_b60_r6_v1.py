"""Fixed selected handoff/support rebuild only; no fit, Test or RL entry point."""
from pathlib import Path
import os,sys,json,shutil,tempfile,argparse
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from geas35.models.transition.handoff_b60_r6_v1 import BoundTransitionRuntime,VERSION,SELECTED,sha
from geas35.models.transition.action_response import evaluate_action_response
from geas35.models.transition.batch_sensor_rollout_b_v1 import BatchSensorUpdater
from geas35.rl.support_v1 import SupportBuildConfig,build_support_artifact,StateActionSupportModel,conformal_quantile
import geas35.models.transition.command_carry_b_v1 as core
P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1'
D=ROOT/'experiments/transition_model_selection/artifacts/command_carry_stale_b60_r6_v1'
H=ROOT/'experiments/transition_model_selection/artifacts/step14_b60_r6_v1'
S=ROOT/'experiments/rl_policy_training/artifacts/step15_b60_r6_v1'
core.P=P;core.D=D
OBS=core.OBS;ACT=core.ACTIONS

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n');tmp.replace(p)
def record(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def freeze():
 assert read(P/'gate_results.json')['status']=='PASS' and read(D/'final_gate.json')['status']=='PASS'
 for b in read(D/'selected_candidate_bindings.json'):assert SELECTED[b['crop']]==b['candidate'] and sha(ROOT/b['model_path'])==b['model_sha256']
 H.mkdir(exist_ok=False);S.mkdir(exist_ok=False)
 config=ROOT/'experiments/rl_policy_training/configs/step15_protocol_v1_4.json'
 files=[config,Path(__file__),ROOT/'src/geas35/models/transition/handoff_b60_r6_v1.py',ROOT/'src/geas35/rl/support_v1.py',ROOT/'src/geas35/models/transition/action_response.py',ROOT/'src/geas35/rl/sensor_history_env_v4.py',ROOT/'src/geas35/rl/causal_contract_env_v2.py',ROOT/'src/geas35/models/transition/batch_sensor_rollout_b_v1.py']
 inputs=[record(f) for f in sorted(P.rglob('*')) if f.is_file()]
 for b in read(D/'selected_candidate_bindings.json'):
  c=b['crop'];a=b['candidate']
  for f in [D/'selected_candidate_bindings.json',D/'final_gate.json',D/c/'selection.json',D/c/a/'fit/model.pkl',D/c/a/'fit/completed.json',D/c/a/'repeat/model.pkl',D/c/a/'repeat/completed.json',D/c/a/'rollout/predictions.npz',D/c/a/'rollout/completed.json',D/c/a/'repeat_rollout/completed.json']:
   inputs.append(record(f))
 pro={'version':VERSION,'selected':SELECTED,'dataset':str(P.relative_to(ROOT)),'support':read(config)['support'],'episode_column':'episode_id','distance':'existing Step15 group-weighted Euclidean kth neighbor; state continuous/binary and action continuous/binary groups; joint sqrt(2) normalization','action_response':{'minimum_support':100,'continuous_change_threshold':.10,'probe_count':256,'gate':'existing eligible: at least one sufficiently supported responsive action, not all six required'},'parity':{'rtol':1e-6,'atol':1e-8,'validation_windows_per_crop':8,'steps':12},'gates':['bound new model/scaler/schema','runtime/batch parity','fresh Train-only support and disjoint episode calibration 95/99 coverage','same action-response eligibility; finite and physical probes','quality metadata absent from model/support; no simulated raw stale detection'],'validation_ood':'descriptive; no post-hoc threshold adjustment or new Validation rate rejection rule','online_deployment':'BLOCKED actuator-safe fallback not validated','forbidden':['Test','model fit','PPO','HPO','Step16-19','Step19.5'],'inputs':inputs,'code':[record(f) for f in files]}
 write(H/'protocol.json',pro);write(H/'protocol_lock.json',{'sha256':sha(H/'protocol.json')});print('FROZEN',flush=True)
def check():
 pro=read(H/'protocol.json');assert sha(H/'protocol.json')==read(H/'protocol_lock.json')['sha256']
 for r in pro['inputs']+pro['code']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
 return pro

def handoff(crop,pro):
 name=SELECTED[crop];out=H/crop;out.mkdir(exist_ok=False)
 b=next(x for x in read(D/'selected_candidate_bindings.json') if x['crop']==crop)
 files={'model.pkl':ROOT/b['model_path'],'observation_scaler.json':P/crop/'observation_scaler.json','schema.json':P/'schema.json','runtime_contract.json':P/'runtime_contract.json'}
 for n,p in files.items():shutil.copyfile(p,out/n)
 selection=read(D/crop/'selection.json');fit=read(D/crop/name/'fit/completed.json')
 assert selection['status']=='PASS' and selection['selected']==name and fit['from_scratch'] and fit['reload_pass']
 assert selection['reproducibility']['model_hash_equal'] and sha(D/crop/name/'repeat/model.pkl')==b['model_sha256']
 for k,f in [('model_sha256','model.pkl'),('scaler_sha256','observation_scaler.json'),('schema_sha256','schema.json'),('runtime_contract_sha256','runtime_contract.json')]:assert b[k]==sha(out/f)
 bound=[record(P/crop/f'{split}_{suffix}.parquet') for split in ('train','validation') for suffix in ('physical','scaled','registry','windows','prepared_qc','provenance','quality_sidecar')]+[record(P/crop/'scaler_fit_registry.parquet'),record(D/'selected_candidate_bindings.json')]
 m={'version':VERSION,'crop':crop,'model_name':name,'target_formulation':'residual_physical_delta_v1','target_order':OBS[:3],'target_units':['degC','percent','ppm'],'input_order':OBS+ACT,'action_order':ACT,'action_units':['fraction [0,1]']*3+['binary 0/1']*3,'dataset_version':P.name,'bundle_hashes':{n:sha(out/n) for n in files},'input_bindings':bound,'selected_binding':b,'fresh_repeat_reload':selection['reproducibility'],'quality_contract':'B60 causal-forward/R6; admission metadata only, no generated-state stale detection','state_updater':'SensorHistoryStateUpdater v4: solar/alias/degree/condensation and sensor-history maintained','online_deployment':'BLOCKED','protocol_sha256':sha(H/'protocol.json')}
 write(out/'manifest.json',m)
 rt=BoundTransitionRuntime(out,crop=crop,manifest_sha256=sha(out/'manifest.json'),root=ROOT)
 ds,sc=core.load(crop,'validation');physical=pd.read_parquet(P/crop/'validation_physical.parquet');base=pd.read_parquet(P/crop/'validation_prepared_qc.parquet')
 reference=pd.read_parquet(D/crop/name/'fit/validation_predictions.parquet')
 runtime=rt.predict(physical[OBS],physical[ACT].to_numpy(float));np.testing.assert_allclose(runtime,reference,rtol=1e-6,atol=1e-8)
 arrays=np.load(D/crop/name/'rollout/predictions.npz');ids=arrays['start_row_id'];lookup=pd.Series(np.arange(len(physical)),index=physical.row_id.astype(str));candidates=np.flatnonzero(arrays['max_horizon']==12);chosen=candidates[np.unique(np.linspace(0,len(candidates)-1,8).astype(int))];traces=[]
 for k in chosen:
  start=int(lookup.loc[ids[k]]);i=pd.Index(base.reg_date).get_loc(physical.iloc[start].reg_date);initial=base.iloc[i].copy();initial[OBS]=physical.iloc[start][OBS]
  rt.reset(initial,base.iloc[max(0,i-12):i]);batch=BatchSensorUpdater(base,np.array([i]),physical.iloc[[start]][OBS].to_numpy(float))
  for step in range(12):
   row=physical.iloc[start+step];action=row[ACT].to_numpy(float);nxt=base.iloc[i+step+1].copy();nxt[OBS]=row[['next_'+c for c in OBS]].to_numpy(float)
   before=rt.updater.current_physical[OBS].to_numpy(float);np.testing.assert_allclose(before,batch.state[0],rtol=1e-6,atol=1e-8)
   pred,after=rt.advance(action,nxt);expected=arrays['predictions'][k,step];np.testing.assert_allclose(pred,expected,rtol=1e-6,atol=1e-8)
   np.testing.assert_allclose(after[['obs_prev_'+c for c in ACT]].to_numpy(float),action,rtol=0,atol=0)
   batch.advance(np.array([0]),pred.to_numpy(float)[None,:],nxt[OBS].to_numpy(float)[None,:],action[None,:],np.array([pd.Timestamp(row.target_timestamp).value]))
   np.testing.assert_allclose(after[OBS].to_numpy(float),batch.state[0],rtol=1e-6,atol=1e-8)
   traces.append({'start_row_id':ids[k],'step':step,'timestamp':row.reg_date,'current':before.tolist(),'action':action.tolist(),'prediction':pred.tolist(),'next':after[OBS].tolist(),'prediction_abs_diff':float(np.max(np.abs(pred-expected)))})
 pd.DataFrame(traces).to_parquet(out/'runtime_parity_trace.parquet',index=False)
 # Alterations must fail before deserializing/predicting.
 negatives=[]
 with tempfile.TemporaryDirectory() as tmp:
  copy=Path(tmp)/crop;shutil.copytree(out,copy);pinned=sha(copy/'manifest.json')
  for filename in ('model.pkl','observation_scaler.json','schema.json','runtime_contract.json'):
   p=copy/filename;raw=p.read_bytes();p.write_bytes(raw+b' ')
   try:BoundTransitionRuntime(copy,crop=crop,manifest_sha256=pinned,root=ROOT)
   except ValueError:negatives.append(filename)
   else:raise AssertionError('Tamper accepted')
   p.write_bytes(raw)
  try:BoundTransitionRuntime(copy,crop=next(c for c in SELECTED if c!=crop),manifest_sha256=pinned,root=ROOT)
  except ValueError:negatives.append('wrong_crop')
  else:raise AssertionError('Wrong crop accepted')
 action=evaluate_action_response(rt.model,ds).to_artifact();assert action['eligible'] and not action['fully_action_insensitive']
 violations=0;nonfinite=0
 for a,values in action['actions'].items():
  low=values['counterfactual_low'];high=values['counterfactual_high']
  if not np.isfinite(low) or not high>low:continue
  probe=ds.x.iloc[np.linspace(0,len(ds.x)-1,min(256,len(ds.x)),dtype=int)].copy()
  for v in (low,high):
   probe[a]=v;y=rt.model.predict(probe).next_observation.to_numpy(float);nonfinite+=int((~np.isfinite(y)).sum());violations+=int(((y<[-60,0,0])|(y>[80,100,10000])).sum())
 assert nonfinite==0 and violations==0
 write(out/'action_response.json',{'report':action,'probe_nonfinite':nonfinite,'probe_physical_violations':violations})
 result={'crop':crop,'status':'PASS','model':name,'model_sha256':sha(out/'model.pkl'),'manifest':record(out/'manifest.json'),'fresh_repeat_reload_verified':True,'one_step_rows':len(runtime),'one_step_max_abs_diff':float(np.max(np.abs(runtime.to_numpy()-reference.to_numpy()))),'rollout_steps':len(traces),'rollout_max_abs_diff':max(t['prediction_abs_diff'] for t in traces),'full_55_next_state_parity':True,'next_command_carry_parity':True,'negative_binding_cases':negatives,'action_response_eligible':action['eligible'],'simulation_raw_stale_detection':False}
 write(out/'gate.json',result);print('HANDOFF',crop,'PASS',flush=True)

def support(crop,pro):
 out=S/crop;out.mkdir(exist_ok=False);train=pd.read_parquet(P/crop/'train_scaled.parquet');cfg=SupportBuildConfig(**pro['support'])
 artifact,_=build_support_artifact(train,crop=crop,state_columns=OBS,action_columns=ACT,episode_column='episode_id',output_dir=out,config=cfg);m=read(artifact)
 assert m['state_columns']==OBS and len(OBS)==55 and m['action_columns']==ACT
 assert not any('stale' in c or 'recovery' in c for c in OBS+ACT)
 # Reconstruct the exact fixed builder row sampling and save audit identities.
 episodes=np.array(sorted(train.episode_id.astype(str).unique()));rng=np.random.default_rng(cfg.seed);rng.shuffle(episodes);count=min(len(episodes)-1,max(1,round(len(episodes)*cfg.calibration_episode_fraction)));cal=set(episodes[:count]);mask=train.episode_id.astype(str).isin(cal).to_numpy();positions=[]
 for role,msk,cap in [('reference',~mask,cfg.max_reference_rows),('calibration',mask,cfg.max_calibration_rows)]:
  ix=np.flatnonzero(msk)
  if len(ix)>cap:ix=np.sort(rng.choice(ix,cap,replace=False))
  f=train.iloc[ix][['row_id','episode_id','reg_date']].copy();f['position']=ix;f['role']=role;f['split']='train';positions.append(f)
 registry=pd.concat(positions,ignore_index=True);registry.to_parquet(out/'fit_row_registry.parquet',index=False)
 assert not set(positions[0].episode_id)&set(positions[1].episode_id)
 model=StateActionSupportModel(artifact);np.testing.assert_array_equal(model.reference_state_raw,train.iloc[positions[0].position][OBS].to_numpy(float))
 calframe=train.iloc[positions[1].position];scores=model.scores(calframe[OBS],calframe[ACT])
 for kind in ('state','joint'):
  for level,coverage in [('warning',.95),('severe',.99)]:
   assert np.isclose(conformal_quantile(scores[kind],coverage),m['thresholds'][kind+'_'+level],rtol=1e-12,atol=1e-12)
   assert m['empirical_calibration_coverage'][kind+'_'+level]>=coverage
 with tempfile.TemporaryDirectory() as tmp:
  other,_=build_support_artifact(train,crop=crop,state_columns=OBS,action_columns=ACT,episode_column='episode_id',output_dir=tmp,config=cfg)
  assert read(other)==m
  arr=np.load(Path(tmp)/'support_reference.npz');original=np.load(out/'support_reference.npz')
  for k in arr.files:np.testing.assert_array_equal(arr[k],original[k])
 summaries=[];groups=[]
 for split in ('train','validation'):
  frame=train if split=='train' else pd.read_parquet(P/crop/'validation_scaled.parquet');sc=model.scores(frame[OBS],frame[ACT])
  columns=[c for c in ('row_id','episode_id','reg_date','source_confidence','source_table','series_id') if c in frame];rows=frame[columns].copy()
  if 'source_confidence' not in rows:rows['source_confidence']='api_transmitted_command' if crop=='cucumber' else 'legacy_command_snapshot_assumption'
  for kind in ('state','joint'):
   rows[kind+'_distance']=sc[kind]
   for level in ('warning','severe'):rows[kind+'_'+level]=sc[kind]>m['thresholds'][kind+'_'+level]
   summaries.append({'crop':crop,'split':split,'kind':kind,'rows':len(frame),'warning_threshold':m['thresholds'][kind+'_warning'],'severe_threshold':m['thresholds'][kind+'_severe'],'warning_including_severe_rate':float(rows[kind+'_warning'].mean()),'severe_rate':float(rows[kind+'_severe'].mean()),**{f'p{q}':float(np.percentile(sc[kind],q)) for q in (0,50,90,95,99,100)}})
  rows.to_parquet(out/f'{split}_support_scores.parquet',index=False)
  for keys in (['source_confidence'],['source_confidence','episode_id']):
   for values,g in rows.groupby(keys,dropna=False):
    values=values if isinstance(values,tuple) else (values,)
    groups.append({'crop':crop,'split':split,'grouping':'+'.join(keys),**dict(zip(keys,values)),'rows':len(g),**{c:float(g[c].mean()) for c in ('state_warning','state_severe','joint_warning','joint_severe')}})
 pd.DataFrame(summaries).to_csv(out/'support_summary.csv',index=False);pd.DataFrame(groups).to_csv(out/'source_episode_summary.csv',index=False)
 write(out/'binding.json',{'version':'geas35.step15.b60_r6.v1','crop':crop,'handoff_manifest':record(H/crop/'manifest.json'),'input_scaler':record(P/crop/'observation_scaler.json'),'schema':record(P/'schema.json'),'train':record(P/crop/'train_scaled.parquet'),'support':record(artifact),'reference':record(out/'support_reference.npz'),'fit_registry':record(out/'fit_row_registry.parquet'),'protocol':record(H/'protocol.json')})
 write(out/'gate.json',{'crop':crop,'status':'PASS','train_rows':len(train),'reference_rows':m['reference_rows'],'calibration_rows':m['calibration_rows'],'fit_split':'train_only','validation_fit_rows':0,'episode_disjoint':True,'deterministic_rebuild':True,'exact_schema_55':True,'quality_metadata_excluded':True,'calibration_coverage_passed':True});print('SUPPORT',crop,'PASS',flush=True)

def report():
 pro=check();handoffs=[read(H/c/'gate.json') for c in SELECTED];supports=[read(S/c/'gate.json') for c in SELECTED]
 summaries=pd.concat([pd.read_csv(S/c/'support_summary.csv') for c in SELECTED],ignore_index=True);summaries.to_csv(S/'support_summary.csv',index=False)
 gate={'status':'PASS','Step14':'PASS','Step15':'PASS','Step16_entry':'PASS','Step16_executed':False,'online_deployment':'BLOCKED actuator-safe fallback not physically validated','handoffs':handoffs,'supports':supports,'inputs_unchanged':True,'Test_model_fit_PPO_HPO_Step16_19_Step19_5_executed':False}
 write(H/'step14_handoff_manifest.json',{'version':VERSION,'crops':{r['crop']:r['manifest'] for r in handoffs},'dataset':P.name,'status':'PASS','protocol':record(H/'protocol.json')});write(S/'final_gate.json',gate)
 lines=['# B60/R6 Step14 handoff / Step15 support — PASS','\nStep16 연구 검증 진입 가능. 이번 작업은 Step14/15까지만 실행했다. 실제 online deployment는 actuator-safe fallback 미검증으로 BLOCKED다. 기존 Step19.5 미통과 판정은 바뀌지 않는다.','\n| 작물 | 공식 모델 | runtime/batch parity | Train support rows | Validation state severe | Validation joint severe |','|---|---|---|---:|---:|---:|']
 for h,s in zip(handoffs,supports):
  c=h['crop'];ss=summaries.loc[summaries.crop.eq(c)&summaries.split.eq('validation')].set_index('kind');lines.append(f"| {c} | {h['model']} | PASS ({h['rollout_steps']} rollout steps + {h['one_step_rows']} one-step) | {s['train_rows']} | {ss.loc['state','severe_rate']:.4%} | {ss.loc['joint','severe_rate']:.4%} |")
 lines+=['\n## Binding과 검증','선정 fresh model pickle을 byte-identical 복사했다. Phase1/2 same-seed fresh-repeat hash와 reload 증거를 재확인했으며 이번에 fit하지 않았다. B60/R6 scaler/schema/runtime contract와 physical/scaled 데이터, 정확한 transition/window, QC sidecar, source provenance, scaler fit registry를 SHA256으로 결속했다. 전용 BoundTransitionRuntime은 caller-pinned manifest와 모든 입력 해시를 검사하고 model 내장 scaler/input/target order를 검증한다. 기존 기본 A 환경 factory를 공식 새 입력으로 사용하면 안 된다. 이 handoff 전용 로더가 새 runtime 진입점이다.','\n실제 Validation 전체 one-step 및 작물당 8개 균등 60분 window에서 current observation → current e_t → physical T/RH/CO₂ → next command carry → sensor/derived updater 순서를 검증했다. 저장 batch prediction과 55개 next state의 scalar/batch parity tolerance는 rtol=1e-6, atol=1e-8. 모델/scaler/schema/contract 변조 및 잘못된 crop 로딩 거부도 검증했다.','\nB60/R6는 raw supervisor/동결 admission metadata다. simulation-generated state에는 stale detector를 실행하지 않는다. 미래 기록 실내상태를 예측 상태로 사용하지 않으며 기존 v4 causal updater를 유지한다.','\n## Support 방법과 결과','기존 Step15 방법을 결과 전에 protocol.json에 고정했다. Train episode 20% calibration, seed42, 참조 최대10000행/보정 최대5000행, k50, 기존 continuous/binary group weights 및 joint sqrt(2) 정규화, conformal 95%/99%. Validation은 score만 계산한다. calibration 에피소드 분리, 원본 Train 행registry, 동일 재구축 array/config 및 coverage를 검증했다. 55 observation 순서와 새 scaler 그대로 사용하며 stale/recovery flag는 제외했다.','\nwarning 비율은 severe 포함이며 거리 > threshold 정의다. Validation 비율은 기술통계이며 결과 후 threshold를 조정하지 않았다. 높은 Validation OOD가 없다는 판정이나 긴 RL rollout 안전성 승인을 의미하지 않는다. 상세 threshold/p0/50/90/95/99/100 및 비율: [support_summary.csv](support_summary.csv). 작물 디렉터리에 전체 row score, source/episode 집계, fit registry, binding이 있다.','\n## 기존 대비 및 한계','v4 당시의 공식 B support 재구축은 없으므로 동일 방법 v4 support 변화 수치를 발명하지 않았다. 모집단은 v4 대비 melon Train -5, strawberry Train -5/Validation -194, cucumber 변화0이다. 새 scaler와 support를 모두 새 모집단에서 생성했으며 이전 support/threshold를 복사하지 않았다. v4 대비 거리/threshold 직접 비교는 미수행이며 승인 근거가 아니다.','\n기존 action-response minimum support100/continuous change0.10/probe256 기준을 그대로 재검사했다. 충분한 지지를 가진 action 중 반응이 존재한다는 기존 gate이며 여섯 action 모두 지원≥100을 뜻하지 않는다. action별 count/sensitivity는 Step14 각 작물 action_response.json에 기록했고 모든 probe nonfinite/physical violation0을 확인했다.','\nmelon/strawberry legacy_command_snapshot_assumption 및 cucumber api_transmitted_command 근거 수준 차이는 유지한다. 딸기 high-CO₂/high-vent Train coverage 부족은 미해결이며 stale QC 효과는 domain 분리다. 모든 target/horizon에서 Persistence보다 우월하다는 뜻이 아니다.','\n## 후속 판단','model/scaler/schema binding, runtime parity, 새 Train-only support, leakage/coverage/determinism, action-response, QC/simulation 분리 PASS로 Step16 연구 검증 진입 가능. Step16–19/PPO/Step19.5/HPO/Test는 실행하지 않았다. 실제 stale 시 actuator-safe handling 검증은 별도 배포 blocker로 유지한다.','\n[최종 gate](final_gate.json). Step14 manifest는 transition_model_selection/artifacts/step14_b60_r6_v1/step14_handoff_manifest.json이며 protocol/hash 및 작물별 parity trace/action-response가 함께 있다.']
 (S/'final_decision_ko.md').write_text('\n'.join(lines)+'\n');(H/'final_decision_ko.md').write_text('# Step14 B60/R6 — PASS\n\n'+ '\n'.join(lines[1:])+'\n')
 for folder in (H,S):write(folder/'artifact_integrity.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'})
 print('FINAL PASS',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','crop','report']);ap.add_argument('--crop',choices=list(SELECTED));args=ap.parse_args()
 if args.stage=='freeze':freeze()
 elif args.stage=='crop':
  p=check();handoff(args.crop,p);support(args.crop,p)
 else:report()
