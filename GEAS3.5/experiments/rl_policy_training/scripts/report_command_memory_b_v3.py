from pathlib import Path
import json,hashlib,re,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/rl_policy_training/artifacts/vent_command_carry_b_v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,default=str,ensure_ascii=False,allow_nan=False)+'\n')
rows=[];ver=[];hash_ok=True;fit=[];reason_rows=[];source_rows=[];collapse_rows=[];contradictions=[]
for crop in ('melon','cucumber','strawberry'):
 dest=D/crop;manifest=json.loads((dest/'build_manifest.json').read_text())
 for entry in manifest['inputs']:assert sha(ROOT/entry['path'])==entry['sha256']
 for entry in manifest['outputs']:assert sha(D/entry['path'])==entry['sha256']
 for split in ('train','validation'):
  s=json.loads((dest/f'{split}_summary.json').read_text());v=json.loads((dest/'verification'/f'{split}_summary.json').read_text());rows.append(s);ver.append(v)
  a=pd.read_parquet(dest/f'{split}_provenance.parquet');b=pd.read_parquet(dest/f'{split}_physical.parquet')
  for reason,n in s['exclusion_reasons'].items():reason_rows.append({'crop':crop,'split':split,'reason':reason,'count':n})
  source_rows.append({'crop':crop,'split':split,'source':s['source_confidence'],'eligible':len(b),'source_fraction_of_crop_split_eligible':1. if len(b) else None,'excluded_fraction':s['excluded']/s['qc_rows'],'legacy_disabled_eligible':0 if crop!='cucumber' else None,'legacy_boundary_excluded_fraction':s['boundary_legacy_rows']/s['qc_rows'] if s['boundary_legacy_rows'] is not None else None,'raw_snapshot_unknown_fraction':s['missing_invalid_legacy_snapshot_rows']/s['qc_rows'] if s['missing_invalid_legacy_snapshot_rows'] is not None else None})
  if crop=='cucumber':
   s['automatic_linked_rows']=int(a.linked.sum());s['automatic_link_rate']=float(a.linked.mean());s['partial_channel_hold_rows']=int(a.retained_channels.gt(0).sum());s['retained_channel_updates']=int(a.retained_channels.sum())
   off=pd.read_parquet(dest/f'{split}_collapse_off_provenance.parquet');collapse_rows.append({'split':split,'collapse_off_eligible':int(off.eligible.sum()),'collapse_on_eligible':int(a.eligible.sum()),'recovered':int((a.eligible&~off.eligible).sum()),'lost':int((~a.eligible&off.eligible).sum()),'ambiguous_off':int(off.ambiguous.sum()),'ambiguous_on':int(a.ambiguous.sum()),'conflict_off':int(off.conflicting_command.sum()),'conflict_on':int(a.conflicting_command.sum()),'override_off':int(off.intervening_override.sum()),'override_on':int(a.intervening_override.sum())})
   events=pd.read_parquet(dest/f'{split}_events.parquet');bad=events.loc[events.operation_or_value_conflict];bad.to_parquet(dest/f'{split}_operation_quarantine.parquet',index=False)
   contradictions.append({'split':split,'quarantined_event_count':len(bad),'examples':bad[['idx','reg_date','channel','operation','ratio']].head(8).astype(str).to_dict('records')})
 if manifest['scaler_created']:
  train=pd.read_parquet(dest/'train_physical.parquet');reg=pd.read_parquet(dest/'scaler_fit_registry.parquet');sc=json.loads((dest/'observation_scaler.json').read_text());assert reg.row_id.tolist()==train.row_id.tolist();assert reg['split'].eq('train').all()
  for col in sc['columns']:
   vals=train[col].to_numpy(float);assert np.isclose(vals.mean(),sc['means'][col],rtol=1e-12,atol=1e-12);std=vals.std(ddof=0);assert np.isclose(std if std>0 else 1.,sc['scales'][col],rtol=1e-12,atol=1e-12)
  fit.append({'crop':crop,'fit_rows':len(train),'train_only':True,'statistics_verified':True,'validation_fit_rows':0,'scaler_sha256':sha(dest/'observation_scaler.json')})
 else:fit.append({'crop':crop,'fit_rows':0,'train_only':False,'statistics_verified':False})
testlog=Path('/tmp/b3_regression_all.log').read_text();passed_tests=re.search(r'(\d+) passed',testlog);tests_ok=passed_tests is not None and ' failed' not in testlog and 'ERROR' not in testlog
shutil.copyfile('/tmp/b3_regression_all.log',D/'regression_tests.txt')
gates={'population_and_windows':all(x['eligible']>0 and all(n>0 for n in x['windows'].values()) for x in rows),'invariants':all(x['all_row_causal_invariant'] for x in rows) and all(x['actual_raw_invariants']>0 for x in ver),'teacher_forcing':all(x['pass'] and x['episode_resets']==x['eligible_episodes'] for x in ver),'scaler_train_only':all(x['statistics_verified'] for x in fit),'regression':tests_ok,'input_output_hashes':True,'deterministic_rebuild':all(x['deterministic_restore'] for x in rows)}
status='PASS' if all(gates.values()) else 'BLOCKED'
write(D/'gate_results.json',{'status':status,'gates':gates,'phase1_entry_eligible_under_project_contract':status=='PASS','training_executed':False,'test_accessed':False,'protocol_sha256':sha(D/'gate_protocol.json'),'tests_passed':int(passed_tests[1]) if passed_tests else 0,'verification_scope':ver,'scaler_audit':fit})
write(D/'operation_quarantine_summary.json',contradictions);write(D/'source_sensitivity.json',source_rows);pd.DataFrame(collapse_rows).to_csv(D/'collapse_comparison.csv',index=False);pd.DataFrame(reason_rows).to_csv(D/'exclusion_reasons.csv',index=False)
pd.DataFrame([{k:v for k,v in s.items() if not isinstance(v,dict)} for s in rows]).to_csv(D/'sample_summary.csv',index=False)
table='\n'.join(f"| {s['crop']} | {s['split']} | {s['eligible']:,} | {s['excluded']:,} | {s['unknown']:,} | {s['windows']['3']:,}/{s['windows']['6']:,}/{s['windows']['12']:,} |" for s in rows)
recovery='\n'.join(f"- {s['crop']} {s['split']}: v2 {s['v2_eligible']:,} → v3 {s['eligible']:,}, 신규 복구 {s['recovered_vs_v2']:,}, 제외로 전환 {s['lost_vs_v2']:,}." for s in rows)
collapse='\n'.join(f"- 오이 {s['split']}: 동일 operation 정규화 아래 collapse OFF {s['collapse_off_eligible']:,} → ON {s['collapse_on_eligible']:,}; 복구 {s['recovered']:,}, ambiguous {s['ambiguous_off']:,} → {s['ambiguous_on']:,}, conflict {s['conflict_on']:,}, override {s['override_on']:,}." for s in collapse_rows)
validation='\n'.join(f"- {v['crop']} {v['split']}: episode reset {v['episode_resets']}/{v['eligible_episodes']}, 가장 긴 연속 rollout {v['longest_run_steps']} steps, 중간 reset 이후 {v['midrun_reset_steps']} steps, 총 teacher-forcing {v['teacher_steps']} steps, feature/reward 불일치 {v['teacher_feature_reward_errors']}, 실제 원본 command 변경 검증 {v['actual_raw_invariants']}건." for v in ver)
source='\n'.join(f"- {s['crop']} {s['split']}: {s['source']} 단독, 제외율 {s['excluded_fraction']:.2%}"+(f", legacy 경계 이력부족 {s['legacy_boundary_excluded_fraction']:.2%}, 원본 snapshot 부재/invalid {s['raw_snapshot_unknown_fraction']:.2%}." if s['legacy_boundary_excluded_fraction'] is not None else '.') for s in source_rows)
nexttext=('프로젝트가 확정한 데이터 계약하에서 Phase 1에 진입 가능한 상태다. 다음 작업은 새 B Train으로 기존 선택 알고리즘/설정부터 재학습하고 기존 후보군을 사전 고정해 재선정한 뒤, one-step 및 15/30/60분 Validation 평가를 수행하는 것이다. 이번 작업에서는 실행하지 않았다.' if status=='PASS' else 'Phase 1 진입 불가. gate_results.json에서 실패한 항목을 먼저 해결하고 같은 tolerance/모집단 조건으로 재검증해야 한다. 이번 결과에 맞춰 PASS 조건을 낮추지 않았다.')
distribution='\n'.join(f"- {x['crop']} {x['split']}: 평균 {x['A_mean']:.6f} → {x['B_mean']:.6f}, 평균 절대 변화 {x['mean_abs_change']:.6f}, 최대 절대 변화 {x['max_abs_change']:.6f}." for x in rows)
report=f'''# B command-carry v3 Phase 0 최종 판단

**{status} — {nexttext}**

Step 19.5 미통과 진단 → 환경 계산/시간축 수정 → B 계약 → v2 BLOCKED → 사용자 프로젝트 결정 반영 → 이번 v3 Phase 0 재구축 순서다. 이번 판정은 dataset/scaler 수용 gate이며 전이모델/PPO 성능 또는 Step 14~19 통과 판정이 아니다. 기존 Step 19.5 미통과는 유지한다.

## 이전 BLOCKED 처리

- 동일 cycle/channel/수치 command의 충돌 없는 반복 전송을 collapse하고 모든 원본 event ID를 보존했다. 다른 값·off-cycle·순서 모호·시간제 명령은 제외했다. 임의 first/last로 상충하는 값을 선택하지 않았다.
- 멜론·딸기는 exact 원본 data_silla_enc snapshot을 legacy_command_snapshot_assumption으로 사용했다. observation은 직전 연속 snapshot, reward는 직전 두 snapshot, Transition action만 현재 snapshot을 사용한다. 원본에 없는 QC 보간/삽입 command와 경계는 unknown으로 제외했다.
- OPEN/ON 및 CLOSE/OFF 별칭 정규화로 channel7/8/11/12의 표현 차이 자체를 blocker에서 제거했다. numeric payload와 모순되는 OFF/CLOSE 양수 등은 별도 operation_quarantine parquet에 보존했다.
- solar/current alias/degree carry/condensation prehistory는 기존 검증된 정의를 유지했다. n_idx/온실97/channel mapping/KST는 재질문하지 않았다.

## 복구 표본과 평가 window

| 작물 | split | eligible | 제외 | command unknown | 3/6/12-step window |
|---|---|---:|---:|---:|---:|
{table}

{recovery}

{collapse}

collapse OFF/ON은 같은 새 operation 규칙을 사용하므로 collapse만의 효과다. v2/v3 차이는 operation과 source 가정도 달라 전체 계약 변경의 효과로 해석한다. legacy OFF 대조는 실제 전송 증거만 허용했던 v2의 0표본이며, 이번 복구 표본은 프로젝트 가정을 받아들였을 때만 얻는다. 상세 source별 unknown·boundary·conflict·override·ambiguous 및 주요 사유는 sample_summary.csv/exclusion_reasons.csv에 있다. 사유들은 중복될 수 있다.

## 현재 남은 blocker: 딸기 low-VPD 이력 경계

명령 provenance·collapse·legacy adapter의 기존 blocker와 별개로, 딸기 Train의 7개 reset 직후 전이에서 9개 feature 비교가 실패했다. 모두 action 시각 00:10 → target 00:15이다. low-VPD 7건, 이에 의존하는 ach_min_day/night 2건이며 reward 불일치는 없다.

`prepare_mdp_v1_frame` → `add_mdp_derived_features` → `_rolling_minutes`는 episode 경계를 구분하지 않고 최근 60분 VPD<0.5 kPa 행 수×5분을 계산한다. 반면 `CausalContractStateUpdater.reset`은 같은 crop/series/segment/episode의 연속 이력만 공급한다. 실제 7건에서 전체 과거 12행으로 재계산하면 기록값 35/60/10분과 정확히 일치하고, 같은 episode의 4행만 사용하면 runtime 20/0분과 정확히 일치했다. 2025-01-23은 low-VPD 10→0 차이로 ach_day 0.12→0.10, ach_night 0.055→0.05가 된다.

증거는 strawberry/verification/history_boundary_diagnosis.json 및 history_boundary_trace.csv.gz에 있으며, 재현 스크립트는 diagnose_history_command_memory_b_v3.py다. 이는 기록 상태를 teacher forcing한 비교이므로 Transition Model 누적 오차로 설명할 수 없다. 7건은 현재 사전 고정한 reset 검증 범위에서 발견된 수이며 전체 실패 전이 수라는 뜻은 아니다.

후속 수정은 rolling low-VPD/RH90 및 그에 의존하는 ACH의 episode 경계 계약을 dataset/runtime에서 일치시키는 것이다. 현재 경계 격리 정책을 유지한다면 B 전용 생성기에서 같은 crop/series/segment/episode의 연속 구간별 rolling을 적용하고, 새 namespace로 dataset/scaler를 다시 생성·검증해야 한다. 기존 기록의 연속 60분 정의를 유지하려면 command carry와 별개인 센서 prehistory의 episode 경계 허용 계약이 먼저 필요하다. 이번 결과를 통과시키려고 이력을 임의로 넘기거나 실패 표본을 사후 제외하지 않았다. 이 내부 계산 계약은 추가 DB 명세로 해결할 항목이 아니다.

## source 민감도와 해석 제한

{source}

legacy snapshot은 당시 유지된 command라는 프로젝트 가정이다. 실제 원본 전송시각, 센서 수집 후 적용 여부, 수동 개입의 정확한 순서, 실제 개도율/도달 지연을 복원한 것이 아니다. 가정이 어긋나면 action alignment·reward 변화량·Transition 동역학에 영향을 줄 수 있다. teacher forcing 통과는 선택한 feature 정의의 계산 일치 증거이며 가정의 물리적 진실성을 증명하지 않는다. API 기반 오이와 동일 근거 수준으로 합쳐 보고하지 않는다.

정확한 원본 snapshot이 없는 5분 삽입행과 crop/series/segment/episode/gap 경계는 제외했다. 한 작물 내 source는 고정되어 source 간 carry가 없다. 실제 0은 그대로 쓰고 unknown을 0 또는 현재 action으로 채우지 않았다.

## A→B feature 변화

{distribution}

동일 v3 eligible 행에서 기존 A와 새 B vent-loss를 비교했다. A→B 값은 무차원 개도율×양의 실내외 온도차×광량 보정 항으로 계산한 proxy이며 열량 측정값이 아니다.

## 비학습 검증

회귀 {int(passed_tests[1]) if passed_tests else 0}개 통과. gate_protocol.json은 실제 sample 결과 전에 저장했으며 rtol=1e-6, atol=1e-8을 유지했다.

{validation}

모든 행에서 현재 action만 변경해 observation이 불변인지 검사했고, 실제 eligible 원본 event/snapshot 및 cycle 산출값을 바꾸는 source별 검증도 수행했다. 이전 command 변경 시 vent-loss 식, next carry, exact target을 확인했다. 모든 eligible observation에서 vent/obs_prev 외의 기존 feature 보존을 검사했다.

실제 검증 범위는 각 eligible episode의 첫 reset, 각 crop/split의 최장 연속 구간 전체 rollout과 중간 reset이다. 모든 transition의 runtime teacher forcing을 수행했다는 뜻은 아니다. 288행 초과·자정·gap 누적 상태는 기존 회귀 테스트로 추가 검증했다. 실제 history 길이와 각 비교 오차는 verification의 압축 trace/오류 JSON에 남겼다. 하나라도 불일치하면 수정 없이 BLOCKED로 판단한다.

새 scaler는 각 작물의 eligible Train만 fit했다. fit row registry, 평균·표준편차를 독립 재계산하고 Validation fit 0행을 확인했다. Validation은 transform만 수행했다. exact transition/window registry, source provenance, input/scaler/schema/code/output hash를 보존했다. deterministic 명령 복원과 실제 제한 구간의 전체 feature/reward 재구축 hash를 비교했다.

## 산출물 및 다음 단계

새 namespace: vent_command_carry_b_v3. 작물 폴더의 *_physical.parquet, *_scaled.parquet, observation_scaler.json, scaler_fit_registry.parquet, *_registry.parquet, *_windows.parquet, *_provenance.parquet, *_events.parquet 및 verification/을 사용한다. 기존 v1/v2나 공식 결과를 덮어쓰지 않았다.

구현: src/geas35/rl/vent_command_memory_b_v3.py. 실행/검증/보고: experiments/rl_policy_training/scripts/*command_memory_b_v3.py. 테스트: tests/test_vent_command_memory_b_v3.py. 프로젝트 가정 상세는 project_decision_ko.md에 있다.

{nexttext}

Phase 1/2에서는 source별 성능과 제외 표본 편향을 분리하고 기존 A 가중치를 공식 B 모델로 재사용하지 않는다. Validation으로 one-step 및 3/6/12-step 누적 오차를 비교한 뒤 Step 14/15와 Step 16~19 재진입을 별도로 판단한다. 이번 작업에서 Test 읽기/생성, Transition Model fit, support 생성, PPO, HPO, threshold 완화는 수행하지 않았다.
'''
(D/'final_decision_ko.md').write_text(report)
files=[ROOT/'src/geas35/rl/mdp_v1.py',ROOT/'src/geas35/rl/vent_command_dataset_b_v1.py',ROOT/'src/geas35/rl/causal_contract_env_v2.py',ROOT/'src/geas35/rl/recorded_light_env_v1.py',ROOT/'src/geas35/features/derived.py',ROOT/'src/geas35/models/transition/causal_state.py',ROOT/'src/geas35/rl/vent_command_memory_b_v3.py',ROOT/'tests/test_vent_command_memory_b_v3.py',*sorted((ROOT/'experiments/rl_policy_training/scripts').glob('*command_memory_b_v3.py'))]
write(D/'code_manifest.json',[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files]);write(D/'artifact_integrity.json',[{'path':str(p.relative_to(D)),'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='artifact_integrity.json'])
print(status);print(table);print(gates)
