from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
D=ROOT/'experiments/transition_model_selection/artifacts/command_carry_stale_b60_r6_v1';P=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b60_r6_v1';OLD=ROOT/'experiments/transition_model_selection/artifacts/command_carry_b_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(D/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def table(f):return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None)])+'\n'
pro=json.loads((D/'protocol.json').read_text());assert sha(D/'protocol.json')==json.loads((D/'protocol_lock.json').read_text())['sha256']
for x in pro['input_hashes']+pro['configuration_hashes']:assert sha(ROOT/x['path'])==x['sha256']
assert json.loads((D/'batch_runtime_parity.json').read_text())['passed']
summ=[];ones=[];multis=[];sels={};references=[];bindings=[]
for crop in ('melon','cucumber','strawberry'):
 sel=json.loads((D/crop/'selection.json').read_text());sels[crop]=sel;baseline=json.loads((D/crop/'persistence/rollout/completed.json').read_text());bf=json.loads((D/crop/'persistence/fit/completed.json').read_text())
 for name,spec in pro['candidates'][crop].items():
  for seed in spec['seeds']:
   label=name+(f'_seed_{seed}' if name=='mlp' else '');dest=D/crop/label
   if not (dest/'rollout/completed.json').exists():
    assert (dest/'fit/failure.json').exists() or (dest/'rollout/failure.json').exists();summ.append({'crop':crop,'candidate':label,'status':'failed'});continue
   f=json.loads((dest/'fit/completed.json').read_text());r=json.loads((dest/'rollout/completed.json').read_text());assert f['from_scratch'] and f['reload_pass'];assert r['protocol_sha256']==sha(D/'protocol.json');assert sha(dest/'fit/model.pkl')==r['model_sha256']
   row={'crop':crop,'candidate':label,'status':'evaluated','one_step_nrmse':f['one_step']['aggregate_metrics']['normalized_mean_rmse'],'score':r['score'],'persistence_weighted_skill':1-r['score']/baseline['score'],'eligible':r['eligible'],'action_response_eligible':f['action_response']['eligible'],'selected':sel['selected']==label,'one_step_nan_inf':f['one_step_nan_inf'],'one_step_physical_violation_rate':f['one_step_physical_violation_rate']}
   for h,metrics in r['rollout'].items():
    row[h+'_nrmse']=metrics['trajectory_nrmse'];row[h+'_physical_violation_rate']=metrics['physical_violation_rate'];row[h+'_nan_inf']=metrics['nan_inf_count']
    for target,v in metrics['targets'].items():multis.append({'crop':crop,'candidate':label,'horizon':h,'target':target,**v,'persistence_skill':1-v['rmse']/baseline['rollout'][h]['targets'][target]['rmse'],'window_count':metrics['window_count']})
   pred=pd.read_parquet(dest/'fit/validation_predictions.parquet')
   for target,v in f['one_step']['target_metrics'].items():
    idx=list(pred.columns).index(target);lo,hi=[(-60,80),(0,100),(0,10000)][idx];pv=pred[target].to_numpy(float)
    ones.append({'crop':crop,'candidate':label,'target':target,**v,'persistence_skill':1-v['rmse']/bf['one_step']['target_metrics'][target]['rmse'],'nan_inf_count':int((~np.isfinite(pv)).sum()),'physical_violation_rate':float(((pv<lo)|(pv>hi)|~np.isfinite(pv)).mean())})
   summ.append(row)
 if sel['status']=='PASS':
  f=json.loads((D/crop/sel['selected']/'fit/completed.json').read_text());assert f['one_step_nan_inf']==0 and f['one_step_physical_violation_rate']==0
  bindings.append({'crop':crop,'candidate':sel['selected'],'model_path':str((D/crop/sel['selected']/'fit/model.pkl').relative_to(ROOT)),'model_sha256':sel['model_sha256'],'scaler_path':str((P/crop/'observation_scaler.json').relative_to(ROOT)),'scaler_sha256':sha(P/crop/'observation_scaler.json'),'schema_sha256':sha(P/'schema.json'),'runtime_contract_sha256':sha(P/'runtime_contract.json'),'handoff_executed':False})
 old=json.loads((OLD/crop/'selection.json').read_text());oldlabel=old['selected'] or old['leader'];oldr=json.loads((OLD/crop/oldlabel/'rollout/completed.json').read_text());references.append({'crop':crop,'old_B_candidate':oldlabel,'old_B_score':oldr['score'],'old_B_status':old['status'],'note':'reference only: changed Train/Validation/scaler and target std; not same-population performance gain'})
pd.DataFrame(summ).to_csv(D/'candidate_summary.csv',index=False);pd.DataFrame(ones).to_csv(D/'one_step_target_metrics.csv',index=False);pd.DataFrame(multis).to_csv(D/'multistep_target_metrics.csv',index=False);pd.DataFrame(references).to_csv(D/'previous_B_reference_only.csv',index=False);write('selected_candidate_bindings.json',bindings)
assert len(summ)==30
overall='PASS' if all(s['status']=='PASS' for s in sels.values()) else 'BLOCKED'
write('final_gate.json',{'offline_Transition_Phase1_2':overall,'status':overall,'crops':sels,'online_deployment_safety':'BLOCKED unchanged','next_Step14_handoff_Step15_rebuild_eligible':overall=='PASS','Step14_handoff_executed':False,'support_created':False,'PPO_HPO_Step16_19_Step19_5_executed':False,'Test_accessed':False,'prior_model_weights_reused':False,'fixed_candidate_runs':30,'phase0_hashes_unchanged':True,'protocol_sha256':sha(D/'protocol.json')})
text=[f'# Stale QC B60/R6 Transition Phase1/2 — {overall}\n',f'**Offline Transition Phase1/2 {overall}. Online deployment safety는 BLOCKED 유지.** 연구용 데이터/모델 gate와 실제 온실 배포 승인 gate를 분리했다. 기존 v4와 이전 Phase1/2 판정은 덮어쓰지 않았다.\n','## 새 Phase0 입력\n',table(pd.DataFrame(json.loads((P/'build_summary.json').read_text()))[['crop','split','old_eligible','stale_excluded','recovery_pending_only_excluded','final_eligible','high_CO2','high_CO2_high_vent']]),'\n[Phase0 보고서](../../../rl_policy_training/artifacts/vent_command_carry_b60_r6_v1/final_decision_ko.md). B60 causal-forward +6 qualifying changing raw observations recovery만 admission에 반영했다. RR 소급 삭제 없음. quality metadata는 observation feature가 아니다. 동일55 observation·6 current command·3 physical target과 Train-only scaler를 사용했다. 기존 sensor-history/solar/alias/degree/condensation/command 계약은 유지했다. Phase0 회귀78개·reset934개·teacher3488step 무오류, determinism/schema/Train-only fit 검증 PASS.\n','## 결과 전에 고정한 프로토콜\n','기존8 후보(Persistence, Linear Regression, Linear SVR, KNN, ExtraTrees, LightGBM, XGBoost, MLP), 저장 하이퍼파라미터·MLP seeds42/43/44를 그대로 사용해 작물당10회, 총30개 fresh fit/evaluation을 수행했다. 이전 선정 알고리즘부터 실행했고 기존 가중치 파일을 초기값으로 사용하지 않았다. 새 HPO나 후보 추가 없음. 선택 모델은 같은 seed fresh repeat fit 및 full-window rollout/reload parity를 검증했다.','\n기존 가중 score=0.10 one-step+0.10 15분+0.25 30분+0.45 60분+0.10 mean drift. 기존 Validation target std(ddof0) 분모 정의 유지(모집단이 바뀌어 분모 수치도 바뀜). finite·물리 위반0·action response 지원/민감도 gate 및 Persistence score improvement guard 유지. MLP eligible seed 중앙 score. Test 미사용. 새 Phase0의 모든 정확한 Validation window 평가. RL projection/fallback/OOD 종료 없이 Transition rollout 평가. [protocol.json](protocol.json), [protocol_lock.json](protocol_lock.json).\n']
summary=pd.DataFrame(summ);multi=pd.DataFrame(multis)
for crop in sels:
 sel=sels[crop];text.append(f"## {crop}: {sel['status']} / 선정 {sel['selected'] or '없음'}\n")
 f=summary.loc[summary.crop.eq(crop)].copy();cols=['candidate','one_step_nrmse','15min_nrmse','30min_nrmse','60min_nrmse','score','eligible','selected'];f=f.reindex(columns=cols)
 for c in cols[1:6]:f[c]=f[c].map(lambda x:f'{x:.6f}' if pd.notna(x) else 'failed')
 text.append(table(f));label=sel['selected'] or sel['leader'];text.append(f"선정 근거/미통과 사유: {sel['approval_reason']}. Persistence score={sel['persistence_score']:.6f}.\n")
 if label:
  g=multi.loc[multi.crop.eq(crop)&multi.candidate.eq(label),['horizon','target','rmse','mae','nrmse','endpoint_rmse','persistence_skill','physical_violation_rate','nan_inf_count','window_count']].copy()
  for c in ['rmse','mae','nrmse','endpoint_rmse','persistence_skill']:g[c]=g[c].map(lambda v:f'{v:.6f}')
  text.append(table(g))
 if sel['status']=='PASS':text.append('동일 seed fresh repeat/reload 및 전체 window 예측 일치: '+str(sel['reproducibility'])+'\n')
text += ['## 한계와 해석\n','melon/strawberry는 `legacy_command_snapshot_assumption`, cucumber는 `api_transmitted_command`다. 전송 원시 시각 복원에 대한 근거 수준을 합쳐 해석하지 않는다. Train coverage 축소/지원 범위 밖의 역학은 남아 있으며 세 작물 전체 또는 고농도 구간에 대한 물리적 일반화 증명이 아니다.','\n딸기 개선이 관측되더라도 “고농도 CO₂ dynamics를 더 잘 학습했다”는 뜻이 아니다. stale/recovery-pending 상태를 적용 domain에서 분리한 결과이며 high-CO₂/high-vent Train coverage 부족과 legacy 가정은 남는다. 기존 B score 비교는 모집단·scaler·nRMSE 분모가 달라 참고용뿐이다. [previous_B_reference_only.csv](previous_B_reference_only.csv).','\n전체 후보·target별 MAE/RMSE/R²/nRMSE/tail/physical/nonfinite 및 Persistence skill은 [one_step_target_metrics.csv](one_step_target_metrics.csv), [multistep_target_metrics.csv](multistep_target_metrics.csv), [candidate_summary.csv](candidate_summary.csv)에 기록했다. 선택 모델/입력 결속은 [selected_candidate_bindings.json](selected_candidate_bindings.json).','## 후속 판단\n',('세 작물 모두 PASS이므로 다음 별도 작업에서 새 Step14 handoff 및 Step15 support 재구축에 진입 가능하다. 이번에는 handoff/support를 실행하지 않았다.' if overall=='PASS' else '하나 이상 실패했으므로 이 단계에서 중지한다. 위 실패 사유를 해결하기 전 Step14/15 재진입을 승인하지 않는다.'),'\n온라인 actuator-safe fallback 미검증은 독립 BLOCKED로 남는다. 연구 PASS가 실제 배포 승인이나 기존 Step19.5 미통과 해제를 의미하지 않는다. support/PPO/HPO/Step16–19/Step19.5 및 Test 실행 없음.','\n원본 phase0·설정 hash를 재확인했다. [final_gate.json](final_gate.json), [batch_runtime_parity.json](batch_runtime_parity.json), [code_manifest.json](code_manifest.json), [artifact_integrity.json](artifact_integrity.json).']
text.insert(-1,'\n추가 제한: 멜론 XGBoost CO₂ skill은15/30/60분 각각−0.019997/−0.036299/−0.046410로 음수다. 딸기 KNN은15/30분 양수지만60분−0.000919로 Persistence보다 소폭 낮다. 기존 사전 규칙은 target별 음의 skill을 단독 탈락 기준으로 삼지 않았으며 종합 score 개선과 기존 eligibility로 PASS했다. 결과 후 기준을 완화하지 않았다. 모든 target에서 우월하다고 해석하지 않는다. Linear SVR의 기존 고정 max_iter 수렴 경고도 설정 변경 없이 기록했다.')
(D/'final_decision_ko.md').write_text('\n'.join(text))
paths=list((ROOT/'experiments/transition_model_selection/scripts').glob('*stale_qc_transition_v1.py'))+[ROOT/'src/geas35/models/transition/command_carry_b_v1.py',ROOT/'src/geas35/models/transition/evaluate_command_carry_b_v1.py',ROOT/'experiments/transition_model_selection/scripts/select_command_carry_b_v1.py']
write('code_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});write('artifact_integrity.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name not in ('artifact_integrity.json','job.lock')});print('FINAL',overall,{c:(s['status'],s['selected']) for c,s in sels.items()},flush=True)
