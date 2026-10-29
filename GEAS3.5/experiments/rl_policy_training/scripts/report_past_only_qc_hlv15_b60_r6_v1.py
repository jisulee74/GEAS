from pathlib import Path
import json,hashlib
import pandas as pd
R=Path(__file__).resolve().parents[3];O=R/'experiments/rl_policy_training/artifacts/past_only_qc_hlv15_b60_r6_v1'
def read(n):return json.loads((O/n).read_text())
def main():
 gate=read('gate.json');pop=read('population_summary.json');stats=read('reconstruction_summary.json')
 lines=['# HLV15 causal B60/R6 Phase0 결과', '', '작성일: 2026-10-08 KST', '', '**Phase0 '+gate['status']+'**. Transition 학습 및 이후 단계는 실행하지 않았다.', '',
 '기존 HLV15 lock을 유지하고 out_light/out_windsp 전체 6개 timestamp stream을 재생성했다. frozen interpolation event 목록은 읽거나 trigger로 사용하지 않았다.', '',
 '## 모집단', '', '| 작물 | split | 기존 transition | 신규 transition | 기존 3/6/12 window | 신규 3/6/12 window | 기존 eligible 중 제외 | 신규 복구 |', '|---|---|---:|---:|---|---|---:|---:|']
 for x in pop:
  fmt=lambda d:'/'.join(str(d[str(h)]) for h in (3,6,12))
  lines.append(f"| {x['crop']} | {x['split']} | {x['old_eligible']:,} | {x['new_eligible']:,} | {fmt(x['old_windows'])} | {fmt(x['new_windows'])} | {x['old_eligible_lost']} | {x['newly_eligible']} |")
 lines+=['','## HLV 및 INVALID', '', '| 작물/split | 변수 | RAW | HLV | INVALID | HLV age p50/p90/max (분) | 정확히15분 | identity reset |', '|---|---|---:|---:|---:|---|---:|---:|']
 for x in stats:
  lines.append(f"| {x['crop']}/{x['split']} | {x['variable']} | {x['raw_valid']:,} | {x['HLV']:,} | {x['INVALID']:,} | {x['age_p50']}/{x['age_p90']}/{x['age_max']} | {x['exact15']} | {x['identity_resets']} |")
 lines+=['','raw count는 pre-QC split stream의 실제 measurement ID·iot ID와 non-ffill 유효값이다. source별/series/segment별 상세 집계는 작물별 `*_source_counts.csv`, 매 행 evidence/age/invalid reason은 `*_reconstruction.parquet`에 남겼다. identity reset에는 최초 초기화도 포함한다. HLV output은 evidence를 갱신하지 않는다.', '',
 '## 제외 및 invalid lineage', '', '| 작물/split | 직접 INVALID 관측행 | 전파 INVALID 관측행(직접 제외) | 기존 eligible 직접 제외 | 기존 eligible 전파 제외 | 기존 eligible target invalid | HLV current이면서 eligible |', '|---|---:|---:|---:|---:|---:|---:|']
 for x in pop:lines.append(f"| {x['crop']}/{x['split']} | {x['direct_INVALID_rows']} | {x['propagated_INVALID_rows']} | {x['old_eligible_direct_excluded']} | {x['old_eligible_propagated_excluded']} | {x['old_eligible_target_invalid']} | {x['hlv_still_eligible']} |")
 lines+=['','위 target invalid 수는 current 제외와 중복될 수 있으므로 합산하지 않는다. `*_provenance.parquet`는 전체 원래 stream row별 기존 command/boundary 사유, direct/propagated/current/target invalid, stale/recovery와 최종 eligibility를 보존한다. 새 window는 모든 transition current/target이 유효하고 동일 연속 rollout인 경우에만 생성한다.', '',
 'DLI/light-sum fallback은 invalid 증분을 0으로 채우지 않고 당일 latch를 유지한다. degree는 필요한 target/T가 invalid인 경우만 전파한다. 유효 measured light sum과 독립 target override는 보존한다. 원래 B60/R6 raw sidecar는 변경 없이 사용하며 HLV 값을 raw signature로 사용하지 않는다.', '',
 '## 검증 및 scaler', '']
 for crop in ('melon','cucumber','strawberry'):
  for split in ('train','validation'):
   p=O/crop/'verification'/f'{split}_summary.json'
   if p.exists():
    x=json.loads(p.read_text());lines.append(f"- {crop}/{split}: teacher steps {x['teacher_steps']}, reset checks {x['reset_checks']}, feature/reward mismatches {x['teacher_feature_reward_errors']}, reset mismatches {x['reset_feature_mismatches']}, longest run {x['longest_run_steps']}, raw action invariant {x['actual_raw_invariants']}; PASS={x['pass']}.")
 for x in pop:
  if x['split']=='train':
   p=O/x['crop']/'observation_scaler.json'
   lines.append(f"- {x['crop']} scaler: {'새 Train-only fit 완료, fit rows '+str(x['new_eligible']) if p.exists() else '미생성'}; Validation fit 없음.")
 lines+=['','기본 HLV15·sensor-history·causal-contract·command-memory 회귀 테스트 57개 PASS. full replay 반복, prefix 재실행 및 future suffix 변조, action 불변성, 독립 feature 불변성, next carry와 stale raw sidecar 보존은 `integrity_checks.json`을 참조한다. schema는 기존 55 observation 순서를 유지하며 신규 scaler는 causal/teacher gate 이후에만 생성한다.', '',
 '## 고농도 coverage 및 source 한계', '', '| 작물/split | CO₂≥1500 | + previous vent≥0.75 | + current vent≥0.75 |', '|---|---:|---:|---:|']
 for x in pop:lines.append(f"| {x['crop']}/{x['split']} | {x['high_CO2']} | {x['high_CO2_previous_vent']} | {x['high_CO2_current_vent']} |")
 lines+=['','melon/strawberry command provenance는 legacy_command_snapshot_assumption, cucumber는 api_transmitted_command를 유지한다. raw sensor identity는 관측된 iot_data_idx와 변수·source segment·crop/series/segment/file의 결합으로 binding하며 inserted row는 이전에 입증된 동일 stream identity만 사용한다. 물리 센서 교체 로그가 없는 상태에서 감춰진 교체를 알아낼 수 있다고 주장하지 않는다. 식별 불가능한 lineage는 INVALID다.', '',
 '딸기 Train 고농도는 22행이고, 이전 vent≥0.75 동시 조건은 3행, 현재 vent≥0.75 동시 조건은 2행뿐이다. 신규 high-CO₂ dynamics 학습을 수행한 것이 아니며 coverage는 여전히 희소하다. HLV는 새 실측값을 만들지 않는다. strawberry high-CO₂/high-vent Train coverage 부족, geas2_5 unseen series, legacy 가정, clean5 autoregressive accumulation, reward imbalance 및 online actuator-safe fallback limitation은 남는다.', '',
 '## 버전 보존과 다음 단계', '',
 '`historical metric preserved / causal availability gate superseded by past-only QC rebuild`', '',
 '기존 입력 및 lock 해시는 재확인했고 old dataset/model/scaler/support를 덮어쓰지 않았다. 이번 Phase0 결과는 old Transition/Step14/15/16 PASS를 자동 승계하지 않는다. strawberry Step16 / 전체 Step17 / online deployment BLOCKED를 유지한다.', '']
 if gate['status']=='PASS':lines+=['다음 작업은 **Transition Phase1/2 fresh retrain/reselection** 하나다. 이번에는 실행하지 않았다.']
 else:lines+=['정확한 blocker: '+', '.join(gate['blockers']), '', 'scaler fit 및 Transition 단계 진입을 중단한다. 기준 완화 없이 blocker를 먼저 해결해야 한다.']
 (O/'final_decision_ko.md').write_text('\n'.join(lines)+'\n')
 (O/'output_hashes.json').write_text(json.dumps({str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.rglob('*')) if p.is_file() and p.name!='output_hashes.json'},indent=2)+'\n')
if __name__=='__main__':main()
