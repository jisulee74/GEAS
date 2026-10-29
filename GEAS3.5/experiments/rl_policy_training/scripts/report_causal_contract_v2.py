"""Assemble the completed candidate assessment from verified nontraining evidence."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/rl_policy_training/artifacts/step19_5_causal_contract_v2'
def read(name):return json.loads((D/name).read_text())
def main():
 result=read('verification_summary.json');teacher=read('teacher_summary.json');audit=read('consistency_audit.json');ends=read('residual_endpoints.json');assert result['status']=='complete'
 validation=[r for r in result['results'] if r['split']=='validation'];tr=teacher['episodes'];expected=teacher['all_expected_equal_pass'];vent=sum('obs_derived_vent_loss_proxy' in e['first_divergence'] for e in tr)
 text=['# Causal contract v2: 완료 결과와 진입 판단','',f"**상태: 확인된 alias·degree carry·condensation prehistory 수정 검증은 {'통과' if expected else '미통과'}, 전체 causal 계약 및 학습 진입은 보류.**",'',f"환경은 `geas35.rl.causal_contract.v2_candidate`다. 기존 solar contract v1을 기준으로 recorded-action 91개와 teacher-forcing {len(tr)}개 에피소드를 완료했다. 원본 입력 해시는 유지됐으며 Test·PPO 학습·HPO·threshold 변경은 없었다. 상세 구현과 vent 대안은 [계약 결정 문서](contract_decision_ko.md)에 있다.",'','## 검증 결과','',f"Teacher-forcing 관측 {sum(e['steps_observed'] for e in tr):,}행에서 vent-loss를 제외한 정의상 일치 항목의 `rtol=1e-6, atol=1e-8` 검사는 {'전부 통과' if expected else '불일치 있음'}했다. Vent-loss 최초 차이가 남은 에피소드는 {vent}/{len(tr)}개이며, 미해결을 명시적으로 제외 집계했다. 전체 계약 통과로 해석하지 않는다.",'','| feature | teacher 최대 절대오차 |','|---|---:|']
 for c in ['obs_current_vpd_kpa','obs_current_dewpoint_c','obs_current_condensation_margin_c','obs_derived_heating_degree_minutes','obs_derived_cooling_degree_minutes','obs_derived_condensation_margin_pred_10m','obs_derived_low_vpd_minutes_1h']:
  text.append(f"| {c} | {max(e['max_abs_errors'][c] for e in tr):.9g} |")
 text+=['',f"Reset prehistory는 최대 {max(e['prehistory_rows'] for e in tr)}행이며 reset 이후의 기록 내생 상태를 window에 추가하지 않는다. Solar 5개 feature의 recorded-action 최대 절대오차는 모든 작물/split에서 0이다.",'','회귀 테스트 **19 passed**: 중간 시작 heating/cooling carry, 자정, gap, 결측 fail-closed, 미래 누적값 무시, 과거 window의 episode/gap 필터, 결로 정의 일치, 288행을 넘는 1분/5분 간격 누적, Test 차단을 포함한다. 실제 5분 장기 테스트는 자정을 통과해도 누적 carry를 유지한다. 공식 Step 16~19 실행과는 별도인 변경 검증이다.','',f"재검산: base reward {sum(x['base_reward_checks'] for x in audit['summaries']):,}건, fallback {sum(x['fallback_checks'] for x in audit['summaries'])}건, 모든 실행 step의 state/joint 점수 및 alias 수식이 일치했다. 최대 base reward 오차 {max(x['max_base_reward_error'] for x in audit['summaries']):.3g}, 최대 alias 수식 오차 {max(x['max_alias_formula_error'] for x in audit['summaries']):.3g}. 계산 window 최적화 전후 완성 에피소드 {audit['window_optimization_parity_episodes']}개도 상태·보상 비교를 통과했다.",'','## 수정 전후 recorded-action','', '아래 이전 값은 solar contract v1이다. PPO 성능/Step 19.5 통과 결과가 아니다.','', '| Validation | melon 이전→이후 | cucumber 이전→이후 |','|---|---:|---:|']
 for label,key in [('실행 step','executed_steps'),('severe OOD / 실행 step','severe_ood_rate'),('fallback / 실행 step','fallback_rate')]:
  v=[]
  for r in validation:
   a,b=r['old'][key],r['new'][key];v.append(f'{a*100:.4f}%→{b*100:.4f}%' if 'rate' in key else f'{a:,}→{b:,}')
  text.append(f"| {label} | {' | '.join(v)} |")
 for label,key in [('state OOD 종료','severe_state_ood'),('반복 action OOD 종료','repeated_severe_action_ood'),('정상 종료','episode_boundary')]:
  text.append(f"| {label} | "+' | '.join(f"{r['old']['termination_reasons'].get(key,0)}→{r['new']['termination_reasons'].get(key,0)}" for r in validation)+' |')
 text+=['','| crop/split | state severe step | proposed joint severe step | 실제 실행 step |','|---|---:|---:|---:|']
 for a in audit['summaries']:text.append(f"| {a['crop']}/{a['split']} | {a['state_severe']} | {a['joint_severe_proposed']} | {a['executed']} |")
 text+=['','state/joint 사건은 중첩 가능하다. 종료 시점·길이·사유는 각 split의 `episode_changes.csv`, 기록/rollout solar 비교는 `feature_comparison.csv.gz`, teacher 최초 차이는 episode JSON의 `first_divergence`와 압축 trace에 보존했다.','', '## 남은 OOD와 내생 경로 차이','']
 for c in ['melon','cucumber']:
  e=[x for x in ends if x['crop']==c and x['split']=='validation'];a=next(x for x in audit['summaries'] if x['crop']==c and x['split']=='validation')
  text.append(f"- {c}: state OOD 종료 {len(e)}개 중 같은 시각 기록 상태도 severe인 경우 {sum(x['recorded_already_severe'] for x in e)}개. 전체 동결 유효 기록 행은 {a['recorded_valid_rows']:,}개이며 state severe {a['recorded_severe_counts']['state']}개, joint severe {a['recorded_severe_counts']['joint']}개다.")
 text+=['','| Validation 공통 prefix의 기록값 대비 MAE | 이전 | 이후 |','|---|---:|---:|']
 for a in audit['summaries']:
  if a['split']!='validation':continue
  for c,unit in [('obs_indoor_temp_c','°C'),('obs_indoor_humidity_pct','%p')]:
   e=a['trajectory_error'][c];text.append(f"| {a['crop']} {c} ({unit}) | {e['common_prefix_old_mae']:.5f} | {e['common_prefix_new_mae']:.5f} |")
 text+=['','공통 prefix는 동일 에피소드에서 양쪽에 존재하는 step만 비교해 실행 길이 차이의 영향을 줄였다. 행동 투영/fallback과 모델 입력 경로도 달라질 수 있으므로 위 값은 기록 경로와의 차이이며 모델 단독의 반사실적 예측 오차로 단정하지 않는다.','', '| 잔여 Validation 종료 | 최초 OOD step | 기록 상태도 severe | 거리 최대 성분 | T 차이 °C | RH 차이 %p |','|---|---:|---|---|---:|---:|']
 for e in ends:
  if e['split']!='validation':continue
  components=e['distance']['squared_distance_components'];top=max(components,key=components.get)
  text.append(f"| {e['crop']} ep{e['episode_index']} | {e['step']} | {e['recorded_already_severe']} | {top} | {e['indoor_difference']['obs_indoor_temp_c']:.4f} | {e['indoor_difference']['obs_indoor_humidity_pct']:.4f} |")
 text+=['','거리 기여도는 인과 중요도가 아니다. `residual_endpoints.json`에 timestamp·전체 성분·행동 변경 횟수·low-VPD 기록/rollout 값이 있다. 초기 상태부터 OOD인 경우는 모델 누적 오차로 분류하지 않는다.','', '오이 기록 분포 OOD는 환경 수정으로 없어지지 않는다. 전체 유효행 분모 4,858과 기존 episode rollout registry 분모 4,835는 다르며, state severe 864개는 동일하다. 이전 진단의 reference 선택 정확 재현 및 전체 non-calibration reference 비교(864→707개)는 그대로 유효한 별도 증거다. 이 단계에서는 support를 생성/변경하지 않았다.','', '## 최종 판단과 Step 영향','', '| 항목 | 판단 |','|---|---|','| 수정 오류 | current alias 동기화, degree carry 보존, 필터링 전 과거 QC window 공급을 별도 후보에 구현. 기존 환경은 보존. |','| Vent 시간 계약 | 외부 측정/명령/적용 timestamp 명세 부족. 기존 previous executed action 동작을 유지하고 미해결로 표시. 임의 복사/shift 없음. A/B 영향 비교는 계약 결정 문서 참조. |','| Transition 재학습 | 현재 필요 확정 불가. 계약 수정만으로 경로·종료가 바뀌었고 vent가 미해결이다. 정합한 입력 계약으로 Step 14 one-step/다중 horizon 및 기록 행동 대조를 한 뒤 판단. |','| Support 재설계 | 임의 재설계/완화는 금지. 오이 시간·생육·날씨 분포 커버리지와 정적 calibration의 적용범위 검토는 필요. 현재 결과만으로 새 threshold를 만들 근거가 없음. |','| Step 14 | 새 causal updater가 의미하는 다중 step 평가·handoff 의존성 재검증 필요. 가중치 재학습을 자동 요구하지 않음. 과거 source hash drift 한계는 유지. |','| Step 15 | 새 QC prehistory 입력·환경 버전의 manifest/무결성 재검증 필요. 현재 세 수정은 동결 dataset/scaler/support 재생성을 요구하지 않음. Vent 정의를 바꾸면 별도 영향 분석으로 재생성 범위가 커질 수 있음. |','| Step 16→17→18→19 진입 | **현재 승인하지 않음.** 전체 causal 계약의 vent 항목이 미해결이고 support 적용범위도 검토 필요. 후보에서의 비학습 부분 검증 통과를 전체 승인으로 대체하지 않음. |','| Step 19.5 재실행 | Vent 계약 확정, 필요한 Step 14/15 재검증, 정식 새 environment manifest, Step 16~19 순차 통과 및 calibration 계획 동결 후에만 가능. 기존 미통과 유지. |','', '다음 최소 작업은 외부 actuator timestamp 명세 확인 및 A/B 정의 결정이다. 명세가 확보되면 필요한 데이터·모델·support 영향 범위를 확정하고, 이번과 동일한 비학습 비교를 전체 계약 기준으로 통과시킨 뒤 정식 Step 16→17→18→19 실행 계획을 고정한다.','', '## 재현 파일','', '- 구현: `src/geas35/rl/causal_contract_env_v2.py`','- 회귀: `tests/test_causal_contract_env_v2.py`, `tests/test_recorded_light_env_v1.py`','- 실행: `verify_causal_contract_v2.py` → `teacher_force_causal_contract_v2.py` → `audit_causal_contract_v2.py` → `report_causal_contract_v2.py` (프로젝트 `experiments/rl_policy_training/scripts/`, `.venv/bin/python` 사용)','- 증거: `verification_manifest.json`, `verification_summary.json`, `teacher_summary.json`, `consistency_audit.json`, `residual_endpoints.json`, split별 원자적 저장 trace와 CSV. 원본 입력과 QC source 해시를 최종 재확인했다.','', 'Trace의 기본 `info.environment_version`은 기존 기반 환경 표기이며 이 후보의 식별은 별도 `diagnostic_environment_version`, 실행 manifest/result의 VERSION을 따른다. 정식 승인 시에는 전체 로깅·checkpoint 의존성을 새 공식 manifest로 일원화해야 한다.']
 (D/'final_decision_ko.md').write_text('\n'.join(text)+'\n')
 write={'candidate_version':'geas35.rl.causal_contract.v2_candidate','expected_equal_except_vent_pass':expected,'full_causal_contract_pass':False,'vent_status':'external_timestamp_specification_required','step16_19_admitted':False,'step19_5':'not_passed','new_training':False,'hpo':False,'test_accessed':False,'threshold_changed':False,'regression_tests':'19 passed'}
 (D/'decision.json').write_text(json.dumps(write,indent=2)+'\n')
 div=[]
 for e in tr:
  for c,v in e['first_divergence'].items():div.append({'crop':e['crop'],'split':e['split'],'episode_index':e['episode_index'],'feature':c,**v})
 pd.DataFrame(div).to_csv(D/'teacher_first_divergence.csv',index=False)
 files=[ROOT/'src/geas35/rl/causal_contract_env_v2.py',ROOT/'tests/test_causal_contract_env_v2.py',ROOT/'tests/test_recorded_light_env_v1.py',*list((ROOT/'experiments/rl_policy_training/scripts').glob('*causal_contract_v2.py')),*[p for p in D.glob('*') if p.is_file() and p.suffix in ('.json','.md','.csv') and p.name!='evidence_manifest.json']]
 (D/'evidence_manifest.json').write_text(json.dumps({'files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},indent=2)+'\n')
 print('Report complete:',D/'final_decision_ko.md')
if __name__=='__main__':main()
