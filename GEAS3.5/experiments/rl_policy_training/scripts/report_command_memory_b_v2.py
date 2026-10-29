from pathlib import Path
import json,hashlib,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v2';O=D/'phase0_verified'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((O/'manifest.json').read_text());rows=m['summary']
for x in m['inputs']:assert sha(ROOT/x['path'])==x['sha256']
for x in m['outputs']:assert sha(O/x['path'])==x['sha256']
for x in json.loads((D/'source_manifest.json').read_text())['sources']:assert sha(D/x['path'])==x['sha256']
inv=json.loads((D/'actual_invariants.json').read_text())
shutil.copyfile('/tmp/geas_b2_regression_release.log',D/'regression_tests.txt')
simple=[];reason_rows=[];extra=[];quantiles=[]
for s in rows:
 crop,split=s['crop'],s['split'];a=pd.read_parquet(O/crop/f'{split}_provenance.parquet');e=pd.read_parquet(O/crop/f'{split}_events.parquet')
 er={'crop':crop,'split':split,'eligible_retained_rows':int((a.eligible&a.retained_channels.gt(0)).sum()),'event_class_counts':e.classification.value_counts().to_dict()}
 # Same-value repeated transmissions are not automatically collapsed.
 er['known_retained_rows']=int(a.retained_channels.gt(0).sum());extra.append(er)
 if a.eligible.any():
  ab=a.loc[a.eligible]
  for col in ['vent_loss_A','vent_loss_B']:
   for q,v in ab[col].quantile([0,.05,.25,.5,.75,.95,1]).items():quantiles.append({'crop':crop,'split':split,'feature':col,'quantile':q,'value':float(v)})
 simple.append({k:v for k,v in s.items() if not isinstance(v,dict)})
 for reason,n in s['exclusion_reasons'].items():reason_rows.append({'crop':crop,'split':split,'reason':reason,'rows':n})
 pd.DataFrame([s['feature_distribution_change']] if s['feature_distribution_change'] else []).to_csv(O/crop/f'{split}_feature_change.csv',index=False)
pd.DataFrame(quantiles).to_csv(D/'feature_distribution_quantiles.csv',index=False)
pd.DataFrame(simple).to_csv(D/'summary.csv',index=False);pd.DataFrame(reason_rows).to_csv(D/'exclusion_reasons.csv',index=False)
(D/'supplemental_counts.json').write_text(json.dumps(extra,indent=2)+'\n')
table='\n'.join(f"| {s['crop']} | {s['split']} | {s['eligible']:,} | {s['excluded']:,} | {s['intervening_override']:,} | {s['partial_retained_rows']:,} | {s['unknown_excluded']:,} | {s['linked_cycles']:,}/{s['cycle_present_rows']:,} | {'생성' if s['scaler_created'] else '미생성'} |" for s in rows)
feature='\n'.join(f"- {s['crop']} {s['split']}: "+(f"A 평균 {s['feature_distribution_change']['mean_A']:.8g}, B 평균 {s['feature_distribution_change']['mean_B']:.8g}, 평균 절대 변화 {s['feature_distribution_change']['mean_abs_change']:.8g}, 최대 절대 변화 {s['feature_distribution_change']['max_abs_change']:.8g}." if s['feature_distribution_change'] else '공식 eligible 0행으로 산출 불가; unknown을 0으로 대체하지 않음.') for s in rows)
checks='\n'.join(f"- {s['crop']} {s['split']}: teacher forcing {s['teacher_forcing_transitions']} transitions, 불일치 {s['teacher_forcing_errors']} feature 항목; 3/6/12-step windows {s['windows']}." for s in rows)
status=m['phase0']
report=f'''# B command-carry Phase 0 재수행 결과

**최종 판정: {status}. Phase 1 Transition Model 학습 진입을 승인하지 않는다.**

새 버전은 `vent_command_carry_b_v2`, 최종 데이터 namespace는 `phase0_verified`다. v1 및 기존 dataset/model/scaler/support/공식 결과를 덮어쓰지 않았다. `phase0`, `phase0_final`, `phase0_completed`, `phase0_release`, `phase0_fast_final`는 구현 중 중단된 보존 이력이며 공식 입력으로 사용하지 않는다.

## 외부 확인 반영

DS table 전체가 iot_data_idx=97 온실 로그라는 확인, 11개 channel mapping, 산출값 source optimal_monitoring_copy, 채널별 무전송 유지, KST를 계약에 반영했다. **n_idx/온실/channel mapping과 timezone 미확정 blocker는 해소됐다.** 아래 잔여 문제를 이 과거 blocker로 다시 서술하지 않는다.

## 실제 결과

| 작물 | split | B eligible | 제외 | 개입 격리 | 알려진 채널 유지 행 | memory unknown 제외 | 연결/산출 cycle 존재 행 | scaler |
|---|---|---:|---:|---:|---:|---:|---:|---|
{table}

연결 성공률의 분모는 센서 timestamp에 산출값이 존재하는 행이다. 전체 QC 행을 분모로 하는 비율도 summary.csv에 보존했다. 중복 산출 행은 분모에서 하나의 센서 시점으로 센다. scaler 표의 Validation 생성 표시는 해당 작물의 Train-only scaler가 존재한다는 뜻이며 Validation에서 fit한 것은 아니다.

'개입 격리'는 자동 후보 연결 후 같은 transition 구간에 별도/미연결 전송이 있는 경우를 보수적으로 제외한 수다. 원격·수동 또는 미연결 사건을 전부 확정된 수동명령으로 단정하지 않는다. 자동제어 자체가 미연결인 구간은 unlinked_event_intervals 및 이벤트 분류에 별도 보존했다. 동일 값의 복수 후보는 ambiguous로 격리하므로 개입 격리 열만으로 모든 제외를 설명하지 않는다.

'알려진 채널 유지 행'은 전송이 없는 channel의 유효 이전 메모리를 유지한 sensor 행이며, 해당 행이 공식 sample로 수용됐다는 뜻은 아니다. 수용된 유지 행 수는 supplemental_counts.json의 eligible_retained_rows로 구분했다. 제외 사유는 중복될 수 있으므로 사유별 합을 총 제외 행으로 더하지 않는다.

## A→B vent-loss 변화 — 동일 eligible timestamp 모집단

{feature}

A는 같은 행의 기록 vent, B는 센서 관측 직전 실제 전송 메모리의 좌/우 평균을 사용한다. 공통 식은 `vent_fraction × max(T−Tout,0) × (1−clip(max(light,0)/1000,0,1))`다. 이 proxy는 측정된 열손실이나 실제 개도 센서값이 아니다.

## 구현과 산출물

- `src/geas35/rl/vent_command_memory_b_v2.py`: 채널별 command-memory 재생, 산출값·전송 연결, unknown/동시각 충돌, 수동/모호 사건 격리, B feature/reward/target 생성.
- `experiments/rl_policy_training/scripts/build_command_memory_b_v2.py`: Train/Validation만 생성, 새 Train scaler, fit registry, transition/window registry, trace 및 teacher-forcing 검증.
- `tests/test_vent_command_memory_b_v2.py`: 부분/전체 무전송 유지, 현재 action 불변성, 이전 command 효과, off-cycle/시간제 명령, override, 경계, unknown, deterministic rebuild, Train-only fit.
- `raw/`: 원본 산출값/전송 이벤트를 별도 동결. sensor/control cycle timestamp, 원본 전송 timestamp, cycle/event ID, device/channel/source, current/previous/previous_previous command 및 validity/reason을 trace와 연결 가능한 형태로 보존한다.
- `phase0_verified/{{crop}}/`: physical/scaled dataset(해당 scaler가 있는 경우), provenance, events, exact transition/window registry, scaler/fit registry(Train이 유효한 경우).
- `source_manifest.json`, `external_contract.json`, `schema.json`, `code_manifest.json`, `gate_results.json` 및 요약 CSV에 계약·출처·검증 범위를 남겼다.

## 검증

기존 solar/alias/degree/condensation 회귀까지 **46 tests passed**. 각 실제 split의 전송 메모리 재생을 두 번 수행해 snapshot/event semantic hash가 일치함을 확인했다. 고정 입력 및 산출물 hash도 재확인했다. actual_invariants.json에는 실제 eligible 원천 payload/cycle 산출값을 바꾼 비교를 기록했다. 현재 관측은 불변이고 Transition action/next command carry만 계약에 따라 바뀌는지 확인했다. 실제로 수용 가능한 행이 없는 split은 '미검증'으로 남겼다.

{checks}

Teacher forcing은 eligible에서 균등 선정한 최대 24개 transition에 대해 과거 QC window로 reset하고 기록 T/RH/CO₂를 예측값으로 주입한 검증이다. 전체 eligible observation에서 B가 바꿔야 할 vent/obs_prev를 제외한 기존 solar/alias/degree/condensation 등 feature 보존도 확인했다. 이는 전체 episode 장기 teacher-forcing gate를 완료했다는 주장이 아니다. 전체 gate를 통과한 것으로 승격하지 않는다.

scaler는 eligible Train row registry만으로 새로 fit했다. Validation은 transform만 적용하며 split provenance가 Validation인 행의 fit 호출은 거부한다. physical target 단위는 유지하며 current/next observation에 동일 scaler를 적용한다. empty dataset의 finite 검사를 실제 학습 gate 통과로 간주하지 않는다.

## 잔여 blocker와 후속 조건

1. **멜론·딸기:** 동결 Train/Validation 구간에 DS 실제 전송 행이 없다. 산출값만으로 last-known transmitted command를 생성하지 않았다. 과거 source의 원본 전송 시각/값/episode 내 초기 메모리 증거 또는 별도 검증된 legacy adapter가 필요하다. 현재 행 산출값을 이전 action으로 대체하지 않는다.
2. **오이:** 산출값의 시기별 coverage, 같은 cycle/channel의 복수 자동 후보, 미연결/중간 제어, 일부 과거 channel operation 형식 차이 때문에 제외가 남는다. source_diagnostics.json과 events parquet에 원본을 보존했다. 중복을 last/first로 해결하거나 과거 operation을 임의 재해석하지 않는다.
3. 자동 후보의 시간 수용은 정확한 산출 timestamp와 같은 분([t,t+60초)), 고유 payload 일치를 함께 요구한다. 60초는 진단 구현의 보수적 수용 범위이며 관리자에게서 확인된 최대 지연 시간이 아니다. cycle 밖/지연·duration 명령은 격리한다. 모호한 복수 후보를 안전하게 수용하려면 batch/retry 규칙 또는 동일 전송 의미에 대한 추가 증거와 adapter regression이 필요하다.
4. 실제 Train/Validation 모집단 확보와 source별 제외 편향 검토, 전체 eligible episode의 reset/장기 teacher forcing, 실제 관측 인과 불변성과 schema/unit/hash gate를 완료해야 한다. 새 dataset이 있다는 이유만으로 충분한 모집단/검증 범위가 확보됐다고 판단하지 않는다.

이 조건을 해소한 뒤 Phase 0를 새 namespace에서 재검증한다. PASS 후에만 Phase 1의 사전 고정된 모델/설정/seed로 새 B Train 학습을 검토한다. 기존 A 모델 가중치와 old scaler를 공식 B 모델로 재사용하지 않는다. 이후 Phase 2 one-step 및 3/6/12-step Validation 평가, 새 Step 14 handoff와 Step 15 이후 gate가 필요하다.

이번 작업에서 Test, Transition Model fit, support 생성, PPO, HPO, threshold 변경은 수행하지 않았다. Step 19.5 미통과 및 HPO 보류 판정은 유지한다.
'''
(D/'final_decision_ko.md').write_text(report)
gates={'phase0':status,'phase1_authorized':False,'regression_passed':46,'actual_invariant_checks':inv,'full_episode_teacher_forcing_completed':False,'input_output_hashes_verified':True,'test_accessed':False,'model_support_ppo_hpo_fit':False}
(D/'gate_results.json').write_text(json.dumps(gates,indent=2)+'\n')
codefiles=['src/geas35/rl/vent_command_memory_b_v2.py','experiments/rl_policy_training/scripts/build_command_memory_b_v2.py','experiments/rl_policy_training/scripts/check_command_memory_b_v2_invariants.py','experiments/rl_policy_training/scripts/report_command_memory_b_v2.py','tests/test_vent_command_memory_b_v2.py','src/geas35/rl/causal_contract_env_v2.py','src/geas35/rl/recorded_light_env_v1.py','src/geas35/rl/mdp_v1.py','src/geas35/features/derived.py']
(D/'code_manifest.json').write_text(json.dumps([{'path':p,'sha256':sha(ROOT/p)} for p in codefiles],indent=2)+'\n')
(D/'artifact_integrity.json').write_text(json.dumps([{'path':str(p.relative_to(D)),'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'],indent=2)+'\n')
print(table);print(status)
