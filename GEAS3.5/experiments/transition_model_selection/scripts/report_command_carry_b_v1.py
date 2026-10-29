"""Report all fixed B candidates, coverage and approval without touching Test."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from geas35.models.transition.command_carry_b_v1 import *
from geas35.models.transition.feature_variants import feature_variant_contract
pro=protocol()
for entry in pro['input_hashes']+pro['configuration_hashes']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
# Retain entire Phase 0 namespace unchanged, not only model input parquets.
for entry in json.loads((P/'artifact_integrity.json').read_text()):assert sha(P/entry['path'])==entry['sha256']
assert json.loads((D/'batch_runtime_parity.json').read_text())['passed']
testlog=Path('/tmp/transition_b1_tests.log').read_text();assert '5 passed' in testlog and ' failed' not in testlog
shutil.copyfile('/tmp/transition_b1_tests.log',D/'regression_tests.txt')
summary=[];one_rows=[];multi_rows=[];coverage=[];distribution=[];decisions={};candidate_details={};A=[];artifacts=[]
for crop in ('melon','cucumber','strawberry'):
 sel=json.loads((D/crop/'selection.json').read_text());decisions[crop]=sel
 baseline=json.loads((D/crop/'persistence/rollout/completed.json').read_text());candidate_details[crop]=[]
 for name,spec in pro['candidates'][crop].items():
  for seed in spec['seeds']:
   label=name+(f'_seed_{seed}' if name=='mlp' else '');dest=D/crop/label
   if not (dest/'rollout/completed.json').exists():
    assert (dest/'fit/failure.json').exists() or (dest/'rollout/failure.json').exists();continue
   f=json.loads((dest/'fit/completed.json').read_text());r=json.loads((dest/'rollout/completed.json').read_text());assert sha(dest/'fit/model.pkl')==r['model_sha256'];assert r['protocol_sha256']==sha(D/'protocol.json')
   fail=[]
   if name=='persistence':fail.append('reference_only')
   if not f['action_response']['eligible']:fail.append('action_response')
   for h,m in r['rollout'].items():
    if m['nan_inf_count']:fail.append(h+':nonfinite')
    if m['physical_violation_rate']:fail.append(h+':physical_violation')
    for target,v in m['targets'].items():
     br=baseline['rollout'][h]['targets'][target]['rmse'];skill=1-v['rmse']/br if br>0 else None
     multi_rows.append({'crop':crop,'source':json.loads((P/crop/'validation_summary.json').read_text())['source_confidence'],'candidate':label,'horizon':h,'target':target,**v,'persistence_skill':skill,'window_count':m['window_count']})
   p=pd.read_parquet(dest/'fit/validation_predictions.parquet').to_numpy(float)
   for j,(target,m) in enumerate(f['one_step']['target_metrics'].items()):
    j=TARGETS.index(target);v=p[:,j];lo,hi=[(-60,80),(0,100),(0,10000)][j];bad=~np.isfinite(v)|(v<lo)|(v>hi)
    basefit=json.loads((D/crop/'persistence/fit/completed.json').read_text());br=basefit['one_step']['target_metrics'][target]['rmse']
    one_rows.append({'crop':crop,'candidate':label,'target':target,**m,'nan_inf_count':int((~np.isfinite(v)).sum()),'physical_violation_rate':float(bad.mean()),'persistence_skill':1-m['rmse']/br if br>0 else None})
   row={'crop':crop,'candidate':label,'one_step_nrmse':f['one_step']['aggregate_metrics']['normalized_mean_rmse'],**{h+'_nrmse':r['rollout'][h]['trajectory_nrmse'] for h in ('15min','30min','60min')},'score':r['score'],'eligible':r['eligible'],'reasons':'|'.join(fail),'weighted_skill_vs_persistence':1-r['score']/baseline['score'],'60min_violation_rate':r['rollout']['60min']['physical_violation_rate'],'60min_nan_inf_count':r['rollout']['60min']['nan_inf_count'],'selected':sel['selected']==label}
   summary.append(row);candidate_details[crop].append(row)
 for split in ('train','validation'):
  stat=json.loads((P/crop/f'{split}_summary.json').read_text());a=pd.read_parquet(P/crop/f'{split}_provenance.parquet');qc=pd.read_parquet(P/crop/f'{split}_prepared_qc.parquet');eligible=a.eligible.to_numpy(bool)
  coverage.append({'crop':crop,'split':split,'source':stat['source_confidence'],'qc_rows':stat['qc_rows'],'B_eligible':stat['eligible'],'excluded':stat['excluded'],'excluded_fraction':stat['excluded']/stat['qc_rows'],'windows':stat['windows'],'exclusion_reasons':stat['exclusion_reasons']})
  for col in ['obs_indoor_temp_c','obs_indoor_humidity_pct','obs_indoor_co2_ppm','obs_outdoor_light','obs_outdoor_wind_speed']:
   for group,mask in [('eligible',eligible),('excluded',~eligible)]:
    v=qc.loc[mask,col].to_numpy(float);v=v[np.isfinite(v)]
    distribution.append({'crop':crop,'split':split,'group':group,'feature':col,'finite_rows':len(v),'mean':float(np.mean(v)) if len(v) else None,'std':float(np.std(v)) if len(v) else None,'q10':float(np.quantile(v,.1)) if len(v) else None,'q90':float(np.quantile(v,.9)) if len(v) else None})
 old=pro['old_A_reference'][crop];ap=ROOT/f"experiments/transition_model_selection/artifacts/step12_v1_4_root_cause/official_rerun/deduplicated_without_quality_flags/{crop}/{crop}/{old['candidate_name']}/training_summary.json";oldtrain=json.loads(ap.read_text())
 A.append({'crop':crop,'old_A_selected':old['candidate_name'],'A_train_rows':oldtrain['train_rows'],'A_validation_rows':oldtrain['validation_rows'],'reference_training_summary':str(ap.relative_to(ROOT)),'reference_training_summary_sha256':sha(ap),'A_stored_validation_score':old['validation_rollout_weighted_score'],'A_stored_60min_nrmse':old['validation_60min_nrmse'],'B_status':sel['status'],'B_selected':sel['selected'],'caveat':'reference only: feature contract and window populations differ; no A weights loaded; no Test results read'})
 if sel['status']=='PASS':
  label=sel['selected'];f=json.loads((D/crop/label/'fit/completed.json').read_text());assert f['one_step_nan_inf']==0 and f['one_step_physical_violation_rate']==0
  artifacts.append({'crop':crop,'candidate':label,'model':str((D/crop/label/'fit/model.pkl').relative_to(ROOT)),'model_sha256':sel['model_sha256'],'scaler':str((P/crop/'observation_scaler.json').relative_to(ROOT)),'scaler_sha256':sha(P/crop/'observation_scaler.json'),'dataset_schema_sha256':sha(P/'schema.json'),'runtime_contract_sha256':sha(P/'runtime_contract.json'),'input_contract':feature_variant_contract('deduplicated_without_quality_flags',action_columns=ACTIONS),'validation_only_selection':True,'formal_step14_handoff_executed':False})
pd.DataFrame(summary).to_csv(D/'candidate_summary.csv',index=False);pd.DataFrame(one_rows).to_csv(D/'one_step_target_metrics.csv',index=False);pd.DataFrame(multi_rows).to_csv(D/'multistep_target_metrics.csv',index=False);pd.DataFrame(distribution).to_csv(D/'coverage_feature_distribution.csv',index=False)
write(D/'coverage.json',coverage);write(D/'A_reference_only.json',A);write(D/'selected_candidate_bindings.json',artifacts)
overall='PASS' if all(x['status']=='PASS' for x in decisions.values()) else 'BLOCKED'
write(D/'final_gate.json',{'status':overall,'crops':decisions,'input_and_phase0_hashes_verified':True,'batch_scalar_parity':True,'regression_tests':5,'all_fixed_candidates_evaluated':len(summary)==30,'A_weights_reused':False,'Test_accessed':False,'support_created':False,'PPO_or_HPO_executed':False,'Step16_19_executed':False,'formal_Step14_15_handoff':False,'protocol_sha256':sha(D/'protocol.json')})
fmt=lambda v: f'{v:.6f}' if np.isfinite(v) else str(v)
text=['# B command-carry v4 Transition Phase 1/2 최종 판단',f'\n**전체 {overall}.** 작물별 승인과 세 작물 통합 Step 14/15 재진입을 구분한다. Phase 0 PASS와 기존 Step 19.5 미통과는 유지한다.\n',
'고정 입력: vent_command_carry_b_v4. 기존 8 알고리즘×3작물, MLP는 기존 42/43/44 restart까지 비교용 30개 새 fit과 PASS 후보 2개의 동일 seed 재fit을 수행했다. A 가중치는 읽지 않고 기존 저장 설정을 사용했다. 딸기 Linear Regression은 과거 최종 설정이 없어서 기존 YAML 기본 설정만 사용했다. HPO 및 결과 후 설정 변경은 없다. 55 observation 공식 schema에 맞춰 평가했고, 과거 57-feature QC ablation을 새 schema로 추가하지 않았다.\n',
'## 사전 선정 규칙\n',
'기존 weighted score는 one-step 0.10 + 15분 0.10 + 30분 0.25 + 60분 0.45 + drift 0.10이다. nRMSE 분모는 기존 정의대로 해당 작물 eligible Validation target 표준편차이며 scaler fit과 별개다. scaler는 Phase 0 Train-only 것을 그대로 사용했다. NaN/Inf/물리 위반 0과 기존 action-response support≥100 및 민감도 gate를 통과한 후보만 선정 대상이다. MLP는 eligible seed score 중앙값을 사용한다.\n',
'기존 최종 선정은 연구자 결정이었으며, 이번 사용자 위임에 따라 eligible 중 최저 점수를 선택했다. 추가로 이번 결과 전에 동결한 Persistence 가중 점수 개선 조건을 승인 guard로 적용했다. 이 guard는 기존 코드의 자동 필수 조건이었다고 주장하지 않는다. target별 음의 skill은 기존 프로토콜처럼 제한사항으로 보고한다. score 차이가 작다는 사실을 통계적 우월성으로 해석하지 않는다.\n',
'## 모집단 및 source\n','| 작물 | source | Train | Validation | Train 제외율(QC 대비) |\n|---|---|---:|---:|---:|']
for c in decisions:
 tr=next(x for x in coverage if x['crop']==c and x['split']=='train');va=next(x for x in coverage if x['crop']==c and x['split']=='validation');text.append(f"| {c} | {tr['source']} | {tr['B_eligible']:,} | {va['B_eligible']:,} | {tr['excluded_fraction']:.2%} |")
text.append('\n딸기는 원본 snapshot이 없는 QC 5분 삽입행/이력 부족 등이 많이 제외됐다. 오이는 자동 cycle 연결·연속 command 이력 조건으로 Train 모집단이 줄었다. 이는 성능 평가의 적용 범위를 제한하지만 제외가 성능 차이의 원인이라는 인과 결론은 아니다. eligible/excluded T/RH/CO₂/light/wind 분포는 coverage_feature_distribution.csv에 구분했다. legacy 두 작물은 snapshot 가정이며 실제 전송 timestamp·개도율 관측을 복원한 것이 아니다. API 오이와 동일 신뢰 수준으로 합치지 않는다.\n')
for c,sel in decisions.items():
 text.extend([f'## {c}: {sel["status"]}\n','| 후보 | one-step nRMSE | 15분 | 30분 | 60분 | weighted score | 기존 eligibility |\n|---|---:|---:|---:|---:|---:|---|'])
 for x in candidate_details[c]:text.append('| '+x['candidate']+' | '+' | '.join(fmt(x[k]) for k in ('one_step_nrmse','15min_nrmse','30min_nrmse','60min_nrmse','score'))+' | '+('PASS' if x['eligible'] else x['reasons'])+' |')
 text.append(f"\n최종 선정: **{sel['selected'] or '없음'}**. eligible 최상위: {sel['leader']}. 판단: {sel['approval_reason']}.\n")
 leader=sel['leader']
 if leader:
  text.append('eligible 최상위 후보의 horizon/target별 RMSE 및 Persistence skill(1−RMSE_model/RMSE_persistence):\n\n| horizon | target | RMSE | skill | endpoint RMSE |\n|---|---|---:|---:|---:|')
  for x in multi_rows:
   if x['crop']==c and x['candidate']==leader:text.append(f"| {x['horizon']} | {x['target']} | {fmt(x['rmse'])} | {fmt(x['persistence_skill']) if x['persistence_skill'] is not None else 'undefined'} | {fmt(x['endpoint_rmse'])} |")
 if sel['status']=='PASS':text.append('\n동일 seed 새 fit 및 reload, 전체 동일 window 예측 rtol=1e-6/atol=1e-8 반복 확인 완료. 재현성 결과: '+json.dumps(sel['reproducibility'],ensure_ascii=False)+'. 작물별 Step 14 모델 인계 준비는 가능하지만 세 작물 통합 인계는 전체 gate를 따른다.\n')
 else:text.append('\n최종 모델을 강제로 선택하지 않았다. 딸기는 안전한 최상위 LightGBM도 Persistence보다 가중 점수가 높고, 일부 선형/MLP 후보는 누적 rollout 물리 위반 또는 발산으로 제외됐다. 기준이나 후보를 바꾸지 않았다. 현 후보군으로는 딸기 Step 14 재구축 불가.\n')
text.extend(['## 미해결 제한 요약\n',
'멜론 XGBoost는 60분 CO₂ skill이 -0.032564로 Persistence보다 RMSE가 약 3.26% 높다. T/RH 개선과 전체 score 개선으로 동결 규칙을 통과했지만 CO₂ 장기 오차 한계는 남는다. 오이 ExtraTrees와 LightGBM score 차이는 약 0.000843으로 작으며 반복 재현성은 통계적 우월성 증명이 아니다.\n',
'딸기 LightGBM은 one-step 및 15분 평균 nRMSE는 Persistence보다 약간 낮지만 30/60분에는 역전된다. 특히 60분 CO₂ RMSE 101.201 ppm, skill -0.469386으로 baseline보다 약 46.94% 높다. 반면 T/RH skill은 양수다. 이 분해는 CO₂ 누적 예측을 우선 조사할 근거이며 legacy 제외 편향이 원인이라고 확정하지 않는다. 다음 비학습 진단은 CO₂ 초기값·시간대·episode·성장시기별 누적 오차 및 eligible/excluded 모집단 비교가 우선이다.\n',
'선정 모델의 안정성 증거는 동일 seed 반복 및 고정 Validation rollout이다. 모든 트리 모델을 여러 seed로 재선정한 결과가 아니며, seed 확장이나 새 설정 탐색은 이번 사전 동결 범위에 포함하지 않았다.\n',
'## 검증과 해석 범위\n',
'모든 3/6/12-step window registry를 그대로 평가했다. horizon별 전체 window 수는 coverage.json에 있다. 12-step 가능한 window만으로 3/6-step 모집단을 줄이지 않았다. Batch/scalar v4 parity는 96개 window×12 step=1,152개 비교(기록 상태 및 의도적 예측 변화), 이후 각 실제 모델의 3개 12-step 경로도 확인했다. selected 모델은 동일 seed fresh fit 뒤 전체 one-step과 전체 rollout을 다시 비교했다. NaN/Inf 발생 경로는 제거하거나 안전 fallback으로 대체하지 않고 실패율에 포함했다.\n',
'새 metric 집계는 기존 pooled RMSE/nRMSE 및 window 평균 drift와 회귀 테스트로 대조했다(5개 통과). RL safety projection/fallback/OOD 종료나 예측 상태 clipping을 사용하지 않았다. thermodynamic feature 자체의 기존 RH clip 정의는 유지했다. 선형 SVR의 기존 max_iter 수렴 경고와 MLP 발산도 원 로그에 남기고 설정을 사후 늘리지 않았다.\n',
'one_step_target_metrics.csv와 multistep_target_metrics.csv에 작물/후보/target별 MAE/RMSE/R²/nRMSE/q90/CVaR90, NaN/Inf, 물리 위반, endpoint drift 및 Persistence skill을 보존했다. candidate_summary.csv는 요약이며 선정 규칙은 protocol.json 및 protocol_lock.json으로 고정돼 있다. 입력 registry/schema/scaler/model/code 해시는 manifest에서 추적한다.\n',
'## 기존 A 결과: 참고 전용\n','| 작물 | A 선택 | A Train | B Train | A 저장 score | B 선택/상태 |\n|---|---|---:|---:|---:|---|'])
for x in A:
 tr=next(v for v in coverage if v['crop']==x['crop'] and v['split']=='train');text.append(f"| {x['crop']} | {x['old_A_selected']} | {x['A_train_rows']:,} | {tr['B_eligible']:,} | {x['A_stored_validation_score']:.6f} | {x['B_selected'] or x['B_status']} |")
text.extend(['\nA는 feature 시간 계약과 Train/Validation 모집단, rollout stride가 달랐다. 위 저장 Validation 숫자로 B의 개선을 입증하지 않는다. A 모델이나 Test metric을 읽어 새 선택에 사용하지 않았다.\n',
'## 다음 판단\n',
'전체 BLOCKED이면 세 작물 통합 Step 14/15로 넘어가지 않는다. PASS 작물의 모델/scaler/schema binding은 selected_candidate_bindings.json에 준비하되 공식 기존 환경이나 결과를 교체하지 않는다. 딸기는 기존 고정 후보의 target/horizon별 Persistence skill, episode·시간대·성장시기별 오차와 legacy 제외 모집단 차이를 추가 분해해 모델 한계와 학습 데이터 범위 문제를 구분하는 것이 다음 최소 비용 작업이다. 새 후보/HPO/기준 변경은 별도의 사전 정의 실험으로만 다룬다.\n',
'이번 작업에서 Test, support 생성, PPO, HPO 및 Step16~19를 실행하지 않았다. Phase 1/2 PASS는 미래 Step 14/15나 Step 19.5 성능 통과를 대신하지 않는다.\n'])
(D/'final_decision_ko.md').write_text('\n'.join(text))
files=[*[ROOT/'src/geas35/models/transition'/n for n in ('base.py','sklearn_models.py','boosting_models.py','hyperparameters.py','evaluator.py','action_response.py','rollout.py')],ROOT/'src/geas35/models/transition/command_carry_b_v1.py',ROOT/'src/geas35/models/transition/batch_sensor_rollout_b_v1.py',ROOT/'src/geas35/models/transition/evaluate_command_carry_b_v1.py',ROOT/'src/geas35/rl/sensor_history_env_v4.py',ROOT/'tests/test_transition_command_carry_b_v1.py',*sorted((ROOT/'experiments/transition_model_selection/scripts').glob('*command_carry_b_v1.py')),ROOT/'experiments/transition_model_selection/scripts/validate_batch_sensor_b_v1.py']
write(D/'code_manifest.json',[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files])
logs=D/'logs';logs.mkdir(exist_ok=True)
for pattern in ('transition_b1_fit_*.log','transition_b1_eval_*.log','transition_b1_select_*.log','transition_b1_batch_parity.log'):
 for log in Path('/tmp').glob(pattern):shutil.copyfile(log,logs/log.name)
write(D/'artifact_integrity.json',[{'path':str(p.relative_to(D)),'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'])
print(overall);print({c:(x['status'],x['selected']) for c,x in decisions.items()});print('candidates',len(summary))
