from pathlib import Path
import json,hashlib,datetime,collections
O=Path(__file__).resolve().parent;P=O.parent;R=P.parents[3];E=P.parent/'cucumber_profile_a_final_action_env_v1'
def load(n):return json.loads((O/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
c=json.loads((P/'structural_audit_contract.json').read_text());d=dict(c);h=d.pop('contract_payload_sha256');assert h=='070a6e7622b1ddb85630ec201263eaf10655e327dd35b4c728417d4db811401c';assert hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==h
assert all(sha(R/x['path'])==x['sha256'] for x in c['authoritative_inputs']);assert all(sha(P/n)==hh for n,hh in c['companion_sha256'].items());assert load('mode_gate.json')['PASS'] and load('analysis_gate.json')['PASS'];assert not (O/'technical_failure.json').exists()
s=load('state_failure_diagnostics.json');j=load('executed_joint_failure_diagnostics.json');f=load('candidate_support_feasibility.json');summary=load('decision_summary.json');con=load('support_contribution_summary.json');refs=load('complete_reference_summary.json');votes=load('exclusive_votes.json');provenance=load('failure_provenance_matrix.json');params=load('policy_parameter_integrity.json');reuse=load('reuse_parity_validation.json');th=c['frozen_thresholds']
assert all(x['identical'] for x in params)
def status(d):return 'SEVERE' if d>th['joint_severe'] else ('WARNING' if d>th['joint_warning'] else 'NONWARNING')
for x in j:
 for name,key in [('proposal','proposal_distance'),('preweather','preweather_distance'),('final','final_distance')]:x[name+'_status']=status(x[key])
write('executed_joint_failure_diagnostics.json',j)
def family(k):
 if k in c['target_order']:return 'four_indoor_channels'
 if k.startswith('obs_prev_'):return 'previous_commands'
 if 'degree' in k:return 'degree_history'
 if 'rh90' in k or 'low_vpd' in k:return 'humidity_vpd_rolling_history'
 if 'ach_' in k:return 'ACH'
 if 'vent_loss' in k or 'temp_diff' in k:return 'vent_loss_temperature_difference'
 if 'vpd' in k or 'dewpoint' in k or 'condensation' in k:return 'VPD_dew_condensation'
 return 'other_frozen_features'
for x in con:
 totals=collections.defaultdict(float)
 for q in x['features']:totals[family(q['feature'])]+=q['squared_contribution']
 x['feature_family_contribution']=dict(totals)
write('support_contribution_summary.json',con)
counts={g:dict(collections.Counter(x['vote'] for x in votes if x['group']==g)) for g in ['A','B','C']};labcounts={g:{k:sum(x['associations'][k] is True for x in s if x['group']==g) for k in s[0]['associations']} for g in ['A','B']};term=[x for x in j if x['primary_terminal_event']];top=collections.Counter(x['top10'][0] for x in con if x['mode']=='Fv1' and 'failure' in x['role'])
flags={'NEW DIAGNOSTIC ROLLOUT EXECUTED':'YES','SUPPORT QUERY EXECUTED':'YES','NEW AUTHORITATIVE Fv1 ROLLOUT EXECUTED':'NO','PPO OPTIMIZATION EXECUTED':'NO','HPO EXECUTED':'NO','TEST ACCESSED':'NO','ENVIRONMENT MODIFIED':'NO'}
status_final='FINAL-ACTION ENV STRUCTURAL AUDIT — COMPLETE / DECISION READY'
report=f'''# Final-Action Env v1 잔여 구조적 실패 비학습 감사 실행 결과

`{status_final}`

- RETRAINING = `{summary['retraining']}`
- HPO = `{summary['HPO']}`
- ENVIRONMENT VALIDITY = `{summary['environment_validity']}`

사전등록 payload `{h}`와 419개 동결 입력은 실행 전후 모두 일치한다. canonical Python 3.12.3 / 프로젝트 `.venv/bin/python` 및 동결 dependency gate가 통과했다. 정책은 seed42/update8, seed43/update8, seed44/update4만 사용했고 실제 tensor digest 전후가 동일하다. 사전등록·환경·support·reward·model·checkpoint를 변경하지 않았다.

## 핵심 결과

| 항목 | 결과 |
|---|---|
| A: legacy-persistent state 12건 | STRUCTURE 9 / MIXED 2 / UNRESOLVED 1 |
| B: migration 이후 드러난 state 5건 | STRUCTURE 3 / MIXED 1 / UNRESOLVED 1 |
| C: executed-final-joint 2건 | POLICY-exclusive 2 |
| COMPLETE 참조 | 기존 56 seed-run 그대로; 기술통계만 계산 |
| policy-exclusive | 2/19 |
| structure-exclusive | 12/19 |
| mixed / unresolved | 3/19 / 2/19 |
| recovery supported alternatives | terminal 6/6 event에서 발견 |
| migration이 새 실패를 만들었는가 | COMPLETE→FAILURE 0; 기존 censoring 이후 실패 노출이라는 해석 지지 |
| 재학습 | domain/rollout 구조 우세로 아직 정당화되지 않음 |
| HPO | optimizer 원인에 대한 직접 근거 없음 |
| 환경 타당성 | semantics valid; 구조적 policy-domain 문제 남음 |
| 다음 권고 gate | state-failure domain/recurrence 대응의 비학습 사전등록 |

STRUCTURE는 전체 10/19 이상 조건뿐 아니라 각 seed에서 POLICY보다 많아야 한다. 실제로 seed42=4 대 0, seed43=4 대 0, seed44=4 대 2이므로 동결 우세 조건을 만족했다. 중복 association을 exclusive 증거로 더하지 않았다. 같은 episode와 반복 seed가 포함되므로 통계적으로 독립인 19개 관측이라고 주장하지 않는다.

## 실행·재사용·정렬

A 12건은 이전 F와 migrated Fv1의 상태/행동/시각, legacy↔migrated history digest 및 model input이 정확히 일치했다. 다만 옛 TF 진단의 완전한 reset/rolling buffer 계보는 저장 자료만으로 입증할 수 없어 재사용을 승인하지 않았다. A/B의 필요한 R1/TF1/TF4 경로만 run 시작부터 Fv1 실패 시각까지 다시 계산했다. TF4는 자체 migrated 종료가 먼저 발생하면 즉시 멈춘다. 저장된 authoritative Fv1은 전혀 재실행하거나 teacher-force하지 않았다.

R0는 정확한 recorded node와 endpoint를 사용했다. TF1은 독립 reset으로 기록 current 56-state와 기존 tolerance 내 일치함을 검증한 후 t−5→t 결과를 비교했다. TF4는 기록 next 4CH만 한 번 주입하고 자체 policy/action/command/history를 유지했다. 모델 predict를 금지하여 이중 advancement나 예측 후 채널 덮어쓰기를 막았다. 실행 시각은 동일 crop/series/segment/episode/run의 exact timestamp이며 근접 join·종료 후 fill은 없다.

## State association — 중복 허용

| association TRUE 건수 | A (12) | B (5) |
|---|---:|---:|
'''
for k in ['RECORDED_DOMAIN','POLICY_ONE_STEP_ASSOCIATED','FREE_RECURRENCE_ASSOCIATED','FOUR_CHANNEL_RECURRENCE_ASSOCIATED','COMMAND_HISTORY_OR_DERIVED_ASSOCIATED']:report+=f"| {k} | {labcounts['A'][k]} | {labcounts['B'][k]} |\n"
report+='''
association의 중복과 exclusive 우세 판정은 다른 집계다. 특히 seed42·43의 `fe3e9f96d4…`는 TF4가 Fv1 실패보다 먼저 종료되어 해당 시각의 FOUR_CHANNEL_RECURRENCE comparator가 UNKNOWN이다. 이 2건을 정상 comparator로 채우지 않고 exclusive UNRESOLVED로 보존했다. Fv1 실패 이후의 TF4 severe를 앞선 실패의 원인으로 사용하지 않았다.

각 17건의 failure/first-warning에서 실제 k=50 반환 이웃을 사용했고 전체 56개 기여도 합계와 거리²가 rtol=1e-6, atol=1e-8 이내로 일치했다. reference hash/index, nearest/kth, 동률 정보, physical/scaled/weighted 차이 및 전 feature 기여도는 `support_contribution_summary.json`에 있다.

Fv1 state failure의 top-1 거리 기여 feature 횟수는 다음과 같다. 이는 종료 시점의 거리 설명이며 최초 인과 원인을 뜻하지 않는다.

'''
for k,n in top.most_common():report+=f'- `{k}`: {n}건\n'
report+='''
+5/+15/+30/+60분과 첫 warning/severe 직전의 기록/Fv1 support 및 4CH 편차는 `recorded_free_drift.json`에 보존했다. 최초 command/derived divergence와 전 feature 시각별 차이는 `command_history_divergence.json`에 있다. 선행한 timestamp만으로 command 또는 updater 버그를 확정하지 않는다.

## C: terminal 6개 decision

| run 접두사 | decision timestamp | state distance | pre-weather joint | final joint | 대안 |
|---|---|---:|---:|---:|---|
'''
for x in term:report+=f"| {x['run_id'][:10]} | {x['decision_timestamp']} | {x['state_distance']:.6f} | {x['preweather_distance']:.6f} | {x['final_distance']:.6f} | EXISTS |\n"
report+=f'''
주 terminal event는 각 run의 counter 1→2→3 연속 decision 3개씩이다. 종료 outcome은 마지막 decision의 +5분이다. 이전 final-severe event {len(j)-6}개도 별도 보존했으며 terminal 집계와 혼합하지 않았다.

6/6에서 LOCAL_GLOBAL_GEOMETRY_MISMATCH와 POLICY_INTERVENTION_PATH가 함께 확인됐다. 실제 local-admissible preweather action이 global joint에서는 severe였고, 날씨 제약 전후 점수가 동일하여 WEATHER_CAP_INDUCED는 0/6이었다. STATE_CONTEXT_DOMINATED도 0/6이다. 여섯 current state 모두 frozen finite candidate domain에 final-joint nonsevere 대안이 있었다.

후보는 local 하한/중점/상한×binary 조합에 해당 local domain 안의 실제 단계별 action만 더했다. 각 후보에 동일 current state의 frozen weather cap을 적용해 support만 조회했다. Transition, reward 최적화, 대안 action 실행은 없었다. 선택 가능한 전체 연속공간의 최적성이나 실제 physical counterfactual 성능을 주장하지 않는다.

RECOVERY는 local/fallback 경로를 시도하는 계약이며 global support 성공 보장이 아니다. 두 C run에서 같은 명시적 frozen invariant 위반은 발견되지 않았다. 대안이 있다는 사실은 회복 경로의 제한을 보여주지만 fallback 결함 또는 환경 semantics 위반을 자동 의미하지 않는다.

## COMPLETE 참조 분포

아래는 seed-run별 통계의 median이다. 원시 분모, 누락 수, min/p25/median/p75/p90/max, seed별 통계와 4CH 편차는 `complete_reference_summary.json`에 보존했다. projection은 실제 proposed→projected 값 변경, fallback은 fallback 선택 발생으로 집계했다. final action 부재를 정상 support로 세지 않았다.

| 집단 | 초기 state | 최대 state | proposal severe 비율 | final severe 비율 | projection 비율 | fallback 비율 |
|---|---:|---:|---:|---:|---:|---:|
'''
for g in ['A','B','C','REFERENCE_COMPLETE']:
 ss=refs['group_comparison'][g];v=[ss[k]['median'] for k in ['initial_state_distance','maximum_state_distance','proposal_severe_rate','final_severe_rate','projection_rate','fallback_rate']];report+='| '+g+' | '+' | '.join('null' if x is None else f'{x:.6f}' for x in v)+' |\n'
report+='''
COMPLETE 56건은 분포 비교에만 사용했다. 새로운 threshold·classifier·feature selector를 만들지 않았다.

## Migration·재학습·HPO 해석

75 seed-run 모두 legacy와 공통 구간의 physical state, next state, action 단계, model input, history digest가 일치했다. COMPLETE→FAILURE는 0이며, B/C의 7건은 기존 shadow case와 일치하고 모두 옛 proposal stop 이후에 종료했다. 따라서 현재 근거는 migration이 공통 경로를 손상시켰다는 설명보다, 옛 조기 종료에 가려졌던 downstream 실패가 드러났다는 설명을 지지한다. 단순 완료 건수만으로 판정하지 않았다.

C 2건의 정책/개입 경로 제한과 별도로, 전체 19건에서는 state-domain·recurrence 관련 exclusive 증거가 우세하다. 지금 PPO 재학습이나 HPO를 수행할 근거로 승격하지 않는다. 또한 이 감사에서는 optimizer/hyperparameter 문제가 원인이라는 직접 증거가 없다. 환경 semantics의 일관성은 정책 품질이나 실제 온실 배포 안전성을 보장하지 않는다.

다음 한 단계는 **17개 state-failure에 대한 domain/recurrence 대응 사전등록**이다. 이번의 기록-domain overlap, TF 비교 및 actual-distance 기여도를 입력으로, 변경 전에 검증할 대응 가설과 gate를 정해야 한다. 이번 실행에서는 어떤 대응도 적용하지 않았다.

## 재현성·실행 경계

초기 launcher는 프로젝트 src 탐색 경로 누락으로 진단 시작 전에 import가 실패했고 실행 전용 script의 경로만 보완했다. 첫 support 분석 시도는 과도한 BLAS 스레드로 중단했고, 완성된 진단 rollout을 그대로 재사용해 OMP/OPENBLAS/MKL=1로 분석만 다시 실행했다. 두 시도 로그를 보존했다. 데이터/수학적 gate 실패를 숨기거나 임계값을 조정한 것이 아니다. 최종 입력·정렬·reset·support closure·counter·prefix parity·parameter gate는 모두 통과했다.

'''
for k,v in flags.items():report+=f'- {k} = {v}\n'
(O/'final_structural_audit_execution.md').write_text(report)
write('execution_validation.json',{'PASS':True,'contract_payload':h,'frozen_inputs_unchanged':419,'preregistration_companions_unchanged':True,'state_cases':17,'primary_joint_events':6,'complete_reference':56,'exclusive_votes':summary['exclusive_counts'],'source_and_semantics_unchanged':True,'policy_digest_unchanged':True,'analysis_threads':{'OMP_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1,'MKL_NUM_THREADS':1},'attempt_notes':['initial import failed before diagnostic execution; src added to execution-only launcher','analysis interrupted before complete output due excessive threads; rerun same calculations with one thread, no diagnostic rollout rerun']})
manifest={'status':status_final,'completed_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'contract_payload':h,'contract_file_sha256':sha(P/'structural_audit_contract.json'),'input_count':419,'inputs':c['authoritative_inputs'],'frozen_environment':c['environment'],'selected_policies':[{'seed':x['seed'],'update':x['update'],'checkpoint':x['checkpoint']} for x in c['selected_policies']],'final_decisions':summary,'flags':flags,'artifact_sha256':{x.name:sha(x) for x in sorted(O.iterdir()) if x.is_file() and x.name!='execution_manifest.json'}}
write('execution_manifest.json',manifest)
print(status_final);print(summary['exclusive_counts']);print('artifact files',len(manifest['artifact_sha256']))
