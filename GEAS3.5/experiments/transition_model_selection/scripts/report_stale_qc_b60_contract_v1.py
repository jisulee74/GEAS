from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'experiments/transition_model_selection/artifacts/stale_qc_b60_contract_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(f):return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None)])+'\n'
base=pd.read_csv(D/'B60_final_impact.csv');ad=pd.read_csv(D/'admission_impact.csv');rec=pd.read_csv(D/'recovery_summary.csv');ev=pd.read_csv(D/'recovery_events.csv')
audit=[
{'path':'src/geas35/rl/model_driven_env.py:304','route':'RL rule-based safe fallback','inputs':'T,RH,current VPD,target temp min/max,is_daytime; local support full scaled observation','stale_dependency':'direct T/RH/light→daytime; T/RH→VPD; indirect full-state CO2/derived','evidence':'temp/humidity condition sets six actions; local projection then weather projection','safe_under_stale':'NOT VERIFIED; BLOCKER'},
{'path':'src/geas35/rl/model_driven_env.py:280','route':'local support/OOD warning projection','inputs':'all support state_columns + proposed action + frozen reference','stale_dependency':'state CO2/T/RH/light and derived features','evidence':'kNN of current state selects allowed action ranges; in-support does not establish freshness','safe_under_stale':'NOT VERIFIED'},
{'path':'src/geas35/rl/hybrid_ppo.py:106','route':'policy feasible masks/empty support fallback','inputs':'scaled observation/local support; same _safe_fallback_action; rain/wind','stale_dependency':'same full-state and rule dependencies','evidence':'binary no-local-evidence uses rule fallback; weather caps only constrain vent','safe_under_stale':'NOT VERIFIED'},
{'path':'src/geas35/rl/model_driven_env.py:348','route':'severe state OOD termination','inputs':'full state support distance, remaining scheduled steps','stale_dependency':'termination trigger uses stale-sensitive full state','evidence':'returns done/executed_action=None; no Transition call on this branch, but no physical command issued','safe_under_stale':'simulation stop is not actuator safe handling'},
{'path':'src/geas35/rl/model_driven_env.py:384','route':'severe joint OOD/repeated severe termination','inputs':'full state+action; rule fallback; severe streak','stale_dependency':'yes; model.predict still runs after fallback before repeated-severe termination','evidence':'fallback action and Transition use current state; termination later','safe_under_stale':'NOT VERIFIED; cannot satisfy stop policy/Transition contract'},
{'path':'src/geas35/rl/model_driven_env.py:222','route':'reset/natural end/time limit','inputs':'recorded episode initial observation, actions, exogenous provider; updater history','stale_dependency':'initial T/RH/CO2/light; aliases and rolling histories in candidate updater','evidence':'finite check only; reset does not prove sensor freshness or send a safe physical command','safe_under_stale':'NOT a safe path'},
{'path':'src/geas35/rl/mdp_v1.py:530','route':'weather hard constraints','inputs':'rain,wind,proposed action','stale_dependency':'no direct B signature but proposed action may already depend on stale values; weather independence unverified','evidence':'vent cap only, not a complete controller','safe_under_stale':'insufficient standalone'},
{'path':'src/geas35/realtime/engine.py:63','route':'online first blocked/joint-severe fallback: previous action','inputs':'previous_action and fallback_streak','stale_dependency':'no current sensor read in _fallback; stored action derived from earlier policy/state, could already be stale before60min','evidence':'hold once then fixed fallback; no evidence of safe hold duration under blind sensors','safe_under_stale':'NOT VERIFIED'},
{'path':'src/geas35/realtime/engine.py:66','route':'online configured fixed fallback','inputs':'safe_fallback mapping from caller/bundle','stale_dependency':'no current sensors in returned fixed vector','evidence':'_validated_action checks ranges/binary/heat-cool conflict only; factory requires six actions; no stale operating safety certificate','safe_under_stale':'sensor-independent computation but physical safety NOT VERIFIED; BLOCKER'},
{'path':'src/geas35/realtime/engine.py:68','route':'online _blocked / no-action path','inputs':'fallback state/mapping; sink','stale_dependency':'fallback may hold previous action; None emits no new command','evidence':'_blocked can emit fallback; no-command is not guaranteed neutralization; external contract says channels hold last transmitted commands','safe_under_stale':'cannot approve no-emission as safe'},
{'path':'src/geas35/realtime/engine.py:103','route':'timestamp stale / quality block / state OOD','inputs':'received_at−timestamp; quality; full-state support','stale_dependency':'timestamp age cannot catch fresh DB rows with repeated measurements; quality/OOD consumes sensors','evidence':'600s default timestamp age check differs from B60 repeated-value detection; blocks before policy for these branches','safe_under_stale':'no B60 hook currently; resulting fallback unverified'},
{'path':'src/geas35/realtime/policy.py:34','route':'online action safety/masks','inputs':'support full state,rain/wind,fallback; proposed action/feasible bounds','stale_dependency':'indirect full-state dependency; apply_action_safety alone only clips inputs','evidence':'heat/cool conflict removed; no freshness or safe environmental envelope proof','safe_under_stale':'NOT VERIFIED'}]
pd.DataFrame(audit).to_csv(D/'fallback_dependency_audit.csv',index=False)
text=['# B60 stale QC 계약 검증 — 재구축 gate BLOCKED\n',
'''**결론: 공식 Phase0 재구축 gate는 BLOCKED.** 프로젝트 결정인 signature B·60분·causal-forward는 후보 계약으로 고정했다. detector/boundary와 recovery 후보 구현 검증은 통과했지만, **센서 정체 시 사용할 기존 runtime 안전 처리 경로의 안전성이 검증되지 않았다.** 새 controller를 설계하거나 공식 dataset/model을 변경하지 않는다.

source audit C(물리 원인 undetermined), sensitivity A, 기존 v4 Phase0 PASS, strawberry/전체 Transition Phase1/2 BLOCKED를 유지한다. 외부 ground truth 문의를 하지 않았다. 60분은 사용자 프로젝트 결정에 따른 운영/QC 기준이며 센서 고장의 진실값·모델 성능 최적 threshold가 아니다. [contract_manifest.json](contract_manifest.json), [protocol_lock.json](protocol_lock.json).

## 고정한 causal 계약

raw CO₂1/CO₂2/T1/RH1/light가 동일 crop/series/segment/raw source에서5분 간격으로60분 exact repeat하면 **13번째 raw row부터 stale=True**다. 현재 및 과거 raw만 사용하고 이전 row를 소급 invalid 처리하지 않는다. timestamp gap·역순·중복, source/identity 변경, raw 부재, NaN/Inf/invalid에서 반복 history를 끊는다. 실제 raw stream이 연속되면 day/episode 경계는 끊지 않는다. 실제 raw에 없는 센서 health 정보까지 검증했다고 주장하지 않으며, 검사 가능한 invalid는 숫자 nonfinite/변환 실패 및 explicit-invalid 입력이다.

공식 후보 admission은 current 또는 target stale이면 transition 제외, trajectory t0..th에 stale 관측이 하나라도 있으면3/6/12-step window 제외다. RR whole-run은 감사 전용이며 admission에 쓰지 않는다. flag는 모델 외부 quality metadata다. sensor2/평균/ffill/상수로 센서값을 교체하지 않는다. raw supervisor는 모델이 예측한 상태가 아니라 실제 수집 raw에 적용한다.

## 기존 안전 경로 의존성 감사

다음은 코드의 입력 의존성과 검증 한계를 정리한 것이다. 함수명에 safe가 있다는 이유로 물리적 안전성을 승인하지 않는다.
''']
t=pd.DataFrame(audit)[['route','inputs','stale_dependency','safe_under_stale','path']];t.columns=['경로','입력','stale 의존','판정','코드'];text.append(table(t))
text.append('''직접 계보는 raw T/RH→대표 T/RH→current VPD/dewpoint/condensation margin, light→daytime/주야 target 선택이다. degree·lowVPD/RH90·ACH·condensation forecast·vent-loss 등 history/derived도 full-state local support/OOD를 통해 간접 영향을 준다. `_safe_fallback_action`의 직접 식은 온도가 상한을 넘거나 RH가 높으면 vent0.2, 주간 과열이면 shade1, 야간 저온이면 thermal1, 온도 비교로 heat/cool, RH 또는VPD로 fan을 정한 뒤 **full-state kNN local support projection**을 거친다. CO₂2를 직접 읽지 않아도 T/RH/light 및 대표 CO₂가 이미 영향을 준다.

온도만15→35°C로 바꾸고 projection을 identity stub으로 둔 비학습 함수 의존성 검사에서 fallback action이 달라짐을 확인했다. 이는 안전성 실험이 아니라 stale 입력 의존성 재현이다. 실제 환경/model을 실행하지 않았다.

실시간 `OnlineInferenceEngine`은 별도로 조사했다. 첫 fallback은 previous action 유지, 이후 caller/bundle의 고정 safe_fallback 또는None이다. 고정 벡터 자체는 현재 센서를 읽지 않지만, 검사는0..1 범위·binary·동시 heat/cool 금지에 그친다. 기존 realtime 테스트도 fake support/policy 및0벡터로 분기 동작을 확인하며 고온/저온/정체 상황의 물리 안전을 입증하지 않는다. `factory.py`는6개 action을 요구할 뿐 stale 안전 운영범위를 인증하지 않는다.

`_blocked`는 이름과 달리 fallback action이 있으면 sink로 전송할 수 있다. None이면 명령을 보내지 않지만 기존 외부 계약상 장치는 last command를 유지하므로 no-emission=안전 정지는 아니다. OOD termination은 simulation 종료일 뿐 실제 actuator 안전 처리도 아니다. timestamp stale(기본600초) 검사 역시 fresh timestamp로 동일값이 재저장되는 B60 정체를 식별하는 계약이 아니다.

따라서 **센서 독립적으로 계산 가능한 고정 fallback은 있지만, stale 상태에서도 사용할 수 있다고 검증된 safe path는 확인되지 않았다.** 현행 시스템이 어떤 경우에도 위험하다는 단정이 아니라 이번 gate를 통과시킬 근거가 없다는 판단이다. 새0벡터·정지·수동 전환 controller를 임의 설계하지 않았다. 현행 경로의 채택 가능한 stale 안전 처리/hold 지속시간/복귀 조건에 대한 별도 근거가 blocker다.

## Recovery 후보 사전 정의와 검증

B60 base stale와 runtime supervisor의 **recovery pending latch**를 분리했다. base exact-repeat가 끝났다고 즉시 supervisor를 정상으로 돌리지 않는다.

- B60 최초 발생 시 latch를 켠다. 이후 B signature가 직전 raw와 달라지고, frozen signature와도 다르며, finite/valid이고 동일identity/source에서5분 연속인 관측을 N개 요구한다.
- **각 qualifying row에서 직전 signature와 변화가 있어야 한다.** 첫 변화만 있고 새 상수값으로 다시 고정되면 count를0으로 돌린다. 이는 새 feature/threshold가 아니라 이번에 사전 정의한 보수적 recovery 후보다. 한 센서만 계속 변해도 tuple 변화 조건은 성립하므로 물리 복구 증명은 아니다.
- 첫 qualifying 변화가count1이며2/3/6개는 그 첫 변화 이후 최소5/10/25분 확인이다. “추가2/3/6개”로 해석하지 않는다. stale 시작부터 총 대기는 실제 event timestamp로 계산한다.
- gap/source/identity/raw 부재/invalid는 count를 reset하지만 latch를 해제하지 않는다. 새source 정상성을 이전source로 보증하지 않는다. gap 뒤 첫valid row는 baseline일 뿐 qualifying 변화로 세지 않는다.
- recovery latch는 새 raw run과 별도 안전 메타데이터다. episode/day 경계는 해제 사유가 아니다. split 이전에 미확인 stale latch가 있었는지는 추정하지 않으며 actual runtime은 supervisor state를 보존해야 한다.

후보 및 정의를 데이터 적용 전에 lock했고 모델 점수는 recovery 선택에 사용하지 않았다. synthetic boundary·미래 prefix 불변성·저장복원·전체 raw replay를 검사했다. 긴 확인일수록 blocked mask 포함관계2⊆3⊆6을 확인했다. 실제 결과:
''')
text.append(table(rec[['crop','split','steps','stale_entries','releases','latched_rows','additional_rows_beyond_B60','latched_B_eligible','stale_reentries_within60min_of_release','unresolved_at_split_end']].rename(columns={'additional_rows_beyond_B60':'B60 이후 추가행','stale_reentries_within60min_of_release':'복귀후60분내재진입'})))
text.append('''모든 후보에서 관측된 복귀 후60분 내 B60 재진입은0이었다. 그러나 B60 자체가60분 지속을 요구하므로 이 지표만으로 oscillation 안정성을 증명할 수 없다. 후보 간 정량적 오탐/복구 진실값은 여전히 없다. 딸기 Validation의 base stale run은8개지만 latch entry는7개다. 한 latch가 gap 등을 넘어 다음 run까지 유지된 경우가 있어 run과 latch episode를 동일하게 세지 않았다.

운영 안정성과 즉시 복귀 방지 관점의 **권장 recovery 후보는6-step**이다.2/3보다 추가 관측을 요구하므로 가장 보수적이지만 더 긴 운전 보류의 비용이 있으며, 현 데이터에서6이 통계적으로 우월하다는 뜻은 아니다. 아직 공식 recovery 정책을 승인하지 않고 안전 경로 blocker 해소 후 함께 확정해야 한다. [recovery_summary.csv](recovery_summary.csv), [recovery_events.csv](recovery_events.csv).

## B60 기준 최종 예상 모집단 — 삭제/재생성 없음

아래는 사용자 결정의 B60 causal flag 자체의 영향이다. recovery latch를 포함한 영향은 다음 표에 별도로 제시한다.
''')
text.append(table(base[['crop','split','stale_runs','masked_raw_rows','B_eligible_overlap','high_CO2_overlap','affected_transitions','affected_windows_3','affected_windows_6','affected_windows_12']].rename(columns={'masked_raw_rows':'causal stale row','B_eligible_overlap':'eligible stale row','high_CO2_overlap':'highCO2 stale row'})))
text.append(table(ad.loc[ad['mask'].eq('B_60_causal_forward'),['crop','split','original_rows','expected_transitions','expected_windows_3','expected_windows_6','expected_windows_12']]))
text.append('''현재B dataset의 행은 transition sample이라 expected_transitions가 예상 잔존행 수다. eligible stale 시작행162와 transition 제외164가 다른 것은 target stale 여부도 보기 때문이다. 기존 registry window만 필터링한 예상치며 새episode/window를 만들어 연결하지 않았다.

**6-step recovery까지 동일 admission에 포함할 경우의 조건부 예상치**(아직 미채택):
''')
text.append(table(ad.loc[ad['mask'].eq('latched_6'),['crop','split','excluded_transitions','expected_transitions','excluded_windows_3','expected_windows_3','excluded_windows_6','expected_windows_6','excluded_windows_12','expected_windows_12']]))
text.append('''2/3-step의 추가 영향도 [admission_impact.csv](admission_impact.csv)에 모두 보존했다. invalid 자체의 별도 신규 admission을 적용한 표가 아니라 stale+latch mask 효과다. 공식 dataset의 기존 QC invalid 처리 계약은 유지된다.

stale QC는 high-CO₂ Train coverage를 복구하지 않는다. B60 자체로 딸기 Train transition은0개 제외돼44,102행 그대로이며, 고농도 coverage가 새로 늘지도 않는다. recovery6 후보에서는44,097행이 예상될 뿐이다. 어떤 성능 변화가 생기더라도 이는 stale-invalid 상태를 모델 적용 domain에서 분리한 결과이며 “고농도 dynamics를 새로 학습했다”가 아니다. legacy snapshot 가정과 고농도/높은 환기 coverage 부족은 계속 남는다.

## Gate 및 다음 단계

|gate|판정|근거|
|---|---|---|
|B60 causal detector/boundary/recovery 후보 계약 검증|PASS(구현/비학습 검증 범위)|고정 signature·threshold, 미래 비의존성·경계·recovery 후보 재현; 공식 recovery6은 권고 상태|
|stale 발생 시 기존 runtime 안전 처리 경로 검증|BLOCKED|rule fallback은 stale 상태 의존; previous hold·고정 fallback·no-emission은 실제 stale 안전 사용 근거 없음|
|공식 Phase0 재구축 적용 승인|**BLOCKED**|두 gate 중 두 번째 미충족|

따라서 이번에는 `stale QC 반영 Phase0 재구축 → Transition 재학습/재선정`으로 넘어가지 않는다. fallback/controller를 새로 설계하지 않고, 적용 가능한 기존 안전 처리의 독립 입력/안전 사용 범위 및 recovery 조건 검증이 필요한 blocker로 기록한다. B60 후보 계약을 취소하는 판단은 아니며 데이터 gate와 runtime safety gate를 구분한다.

공식 데이터/모델 원본 hash 유지, Test 및 금지된 학습·단계 실행 없음. 산출물은 `stale_qc_b60_contract_v1`에 분리했다. [input_manifest.json](input_manifest.json), [contract_verification.json](contract_verification.json), [fallback_dependency_audit.csv](fallback_dependency_audit.csv), [gate.json](gate.json), [code_manifest.json](code_manifest.json), [artifact_integrity.json](artifact_integrity.json).
''')
assert (D/'contract_verification.json').exists()
(D/'final_decision_ko.md').write_text('\n'.join(text))
(D/'gate.json').write_text(json.dumps({'B60_project_candidate':'accepted operational/QC assumption not physical truth','detector_boundary_recovery_implementation':'PASS','recovery_recommendation_steps':6,'recovery_officially_adopted':False,'runtime_safe_handling':'BLOCKED insufficient stale safety evidence','Phase0_rebuild_authorized':False,'overall':'BLOCKED','existing_Phase0':'PASS unchanged','existing_Transition_Phase1_2':'BLOCKED unchanged'},indent=2)+'\n')
for p,h in json.loads((D/'input_manifest.json').read_text()).items():assert sha(ROOT/p)==h
paths=['src/geas35/rl/model_driven_env.py','src/geas35/rl/mdp_v1.py','src/geas35/rl/hybrid_ppo.py','src/geas35/rl/support_v1.py','src/geas35/rl/causal_contract_env_v2.py','src/geas35/rl/sensor_history_env_v4.py','src/geas35/features/derived.py','src/geas35/realtime/engine.py','src/geas35/realtime/policy.py','src/geas35/realtime/factory.py','tests/test_realtime_end_to_end.py']
paths += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/transition_model_selection/scripts').glob('*stale_qc_b60*') if p.is_file()]
(D/'code_manifest.json').write_text(json.dumps({p:sha(ROOT/p) for p in paths},indent=2)+'\n')
(D/'artifact_integrity.json').write_text(json.dumps({p.name:sha(p) for p in sorted(D.iterdir()) if p.is_file() and p.name!='artifact_integrity.json'},indent=2)+'\n')
print('Final BLOCKED gate/report saved; official artifact hashes unchanged.')
