# GEAS3.5 End-to-End Online Inference 구현 계획

## Summary

실시간 관측 수집부터 인과적 품질관리, PPO 정책 추론, 안전성 검사, 명령 기록 및 선택적 장치 전송까지 하나의 공통 추론 core로 연결한다.

기본 실행 모드는 `replay` 또는 `shadow`로 두며, `live`는 별도 설정과 안전 게이트를 모두 통과한 경우에만 활성화한다. 현재 진행 중인 GPU 실험과 독립적인 구조는 즉시 구현하고, 최종 PPO 선택·Test replay·artifact 고정은 calibration과 HPO가 끝난 뒤 수행한다.

논문에서는 최종적으로 다음 수준만 주장한다.

- End-to-end online inference pipeline 구현
- Offline Test replay 및 shadow-mode 검증
- 실제 장기 closed-loop 온실 검증은 향후 연구

## 지금 즉시 구현 가능한 단계

### Step 1 — Online inference 계약과 실행 모드 정의

- `replay`, `shadow`, `live` 세 실행 모드를 정의한다.
- 관측 입력, 품질관리 결과, 정책 입력, 정책 action, 실행 action, 안전 개입 및 출력 결과의 공통 자료형을 만든다.
- 품질모델, observation scaler, state-action support, transition model provenance 및 PPO policy를 하나의 deployment bundle로 검증하는 loader를 구현한다.
- 최종 PPO가 아직 없어도 fake policy와 기존 checkpoint로 전체 경로를 시험할 수 있도록 policy interface를 분리한다.
- Artifact hash나 schema가 다르면 시작 단계에서 fail-closed로 중단한다.
- Transition model은 PPO 학습 provenance 검증에 포함하되, 실제 정책 action 생성에는 사용하지 않는다.

### Step 2 — 인과적 실시간 전처리와 상태 관리 구현

- timestamp 순서를 강제하고 중복·역행·지연 관측을 탐지한다.
- 입력 스키마 통합, 수치 변환, 단위 통일 및 생육단계 계산을 공통 single-timestep API로 제공한다.
- 최근 288개 시점의 품질모델 문맥과 이전 실행 action을 메모리 상태로 관리한다.
- Offline에서 미래값을 사용하는 양방향 보간은 온라인 경로에서 금지한다.
- 동일 timestamp의 제어 이력만 결측 action 복원에 사용한다.
- 긴 관측 gap, 작기 변경 또는 segment 변경 시 history와 이전 action 상태를 초기화한다.
- 품질모델이 필수 policy observation을 확정하지 못하면 정책 출력을 장치로 전달하지 않는다.
- 기존 batch MDP 변환과 동일한 현재 시점 observation을 생성하는 causal observation builder를 추가한다.

### Step 3 — PPO policy adapter와 안전 계층 구현

- 기존 PPO checkpoint loader 위에 deterministic inference 전용 action provider를 구현한다.
- 정책이 출력하는 세 연속 action과 세 이산 action을 공식 순서와 범위로 검증한다.
- 학습 시 사용한 observation scaler와 state-action support를 그대로 적용하고 온라인 재학습이나 scaler 재추정은 금지한다.
- 학습 환경 내부에 결합된 feasible-action 계산을 순수 함수로 분리해 학습과 온라인 추론이 같은 제약을 사용하게 한다.
- 강우·강풍 제한, 연속 action 범위, 이산 action 허용 집합, heating/cooling 상호배제 및 severe OOD 처리를 적용한다.
- 정책 action과 안전 계층 이후의 executed action을 모두 보존한다.
- Fail-closed 시 최근 정상 action은 최대 한 제어 주기만 유지하고, 이후에는 사전 검증된 crop-specific fallback을 사용한다. Fallback도 확인할 수 없으면 명령을 보내지 않고 오류를 기록한다.

### Step 4 — 공통 단일 시점 추론 core 구현

- 하나의 `step(observation)` 호출이 전처리, QC, observation 구성, PPO 추론, 안전 검사 및 출력 결정을 순차 수행하게 한다.
- Replay와 live가 동일한 core를 사용하며 입력·출력 adapter만 교체하도록 한다.
- 매 step마다 다음 정보를 구조화해 반환한다.
  - 입력 timestamp와 데이터 신선도
  - 품질관리 상태와 unresolved 변수
  - policy action과 executed action
  - support projection, OOD 및 fallback 여부
  - inference latency와 실패 단계
  - 사용한 artifact hash
- 동일 timestamp 재처리는 action을 중복 생성하지 않도록 idempotent하게 처리한다.
- 한 단계의 실패가 history를 부분 갱신하지 않도록 state commit을 성공 시점 이후에 수행한다.

### Step 5 — Replay·Shadow·Live adapter 구현

- `ReplayObservationSource`는 Test 데이터를 timestamp 순으로 공급하고 결과를 파일에 저장한다.
- `DatabaseObservationSource`는 운영 데이터베이스에서 신규 관측만 polling하며 연결 재시도와 cursor 복구를 지원한다.
- `RecordOnlyControlSink`는 shadow action을 저장하되 장치 API를 호출하지 않는다.
- `HttpControlSink`는 기존 장치 API 계약을 별도 adapter로 캡슐화한다.
- API 주소, 인증정보, 온실 식별정보 및 device mapping은 환경변수나 외부 설정으로만 주입하고 코드에 저장하지 않는다.
- PPO의 단일 환기 action은 검증된 device mapping profile이 지정된 경우에만 좌·우 장치 명령으로 변환한다. 커튼, 난방, 냉방 및 팬도 명시적인 profile 없이는 live 시작을 거부한다.
- `live`는 기본 비활성화하고, 실행 모드 설정과 별도의 명시적 확인 옵션이 동시에 있어야 활성화한다.
- API 일부 전송 실패나 timeout 발생 시 자동 반복 전송하지 않고 실패를 기록한 뒤 fail-closed 상태로 전환한다.

### Step 6 — GPU 독립 테스트와 운영 가시성 구현

- Fake quality model, fake policy, 임시 scaler/support 및 mock API를 이용한 end-to-end 테스트를 작성한다.
- 정상 입력, 결측, 이상치, stale timestamp, 역행 timestamp, 작기 변경, 긴 gap, severe OOD 및 artifact hash 불일치를 검증한다.
- Replay와 shadow가 같은 입력에서 동일한 executed action을 생성하는지 확인한다.
- Live 모드 기본 차단, 설정 누락 차단, 중복 명령 차단 및 API 장애 시 fail-closed 동작을 검증한다.
- Policy, QC 및 API latency를 각각 기록하고 전체 step latency의 p50, p95 및 최대값을 집계한다.
- 실제 API 호출 없이 mock server까지만 사용하는 CI 테스트를 구성한다.
- 구조화된 실행 log와 run manifest에 코드 버전, artifact hash, 설정 hash 및 mode를 기록한다.

## GPU 실험 완료 후 수행할 단계

### Step 7 — v2.1 Calibration 최종화 및 Step 20 연결

- 세 작물의 v2.1 calibration이 모두 완료됐는지 확인하고 calibrated budget과 integrity manifest를 생성한다.
- 현재 Step 20이 기존 calibration 경로를 기대하는 부분을 v2.1 schema와 경로로 전환한다.
- Test 잠금 상태를 유지한 채 작물별 공식 HPO를 수행한다.
- HPO 완료 artifact의 trial 수, seed, rung, safety gate 및 Validation objective를 무결성 검사한다.
- 이 단계가 통과하기 전에는 Test policy evaluation이나 논문 결과 생성을 시작하지 않는다.

### Step 8 — 최종 Policy 선택 및 Deployment Artifact 생성

- Step 21 finalization을 새로 구현한다.
- 작물별로 safety gate를 통과한 설정 중 사전 정의된 Validation primary objective가 가장 높은 설정을 선택한다.
- 동률이면 severe OOD와 fallback 비율이 낮은 설정, 그래도 같으면 trial 번호가 작은 설정을 선택한다.
- 선택 설정의 세 seed 중 Validation reward가 중앙값인 seed의 checkpoint를 배포 policy로 사용한다.
- Test 결과는 선택에 사용하지 않는다.
- 학습 optimizer와 RNG state를 제외한 inference 전용 policy artifact를 생성한다.
- Policy config, observation schema, scaler, support, transition provenance 및 모든 hash를 deployment manifest에 고정한다.

### Step 9 — 공식 Test End-to-End Replay

- 고정된 품질모델과 최종 PPO deployment bundle을 사용해 작물별 Test를 최초 1회 replay한다.
- 각 episode와 segment 경계에서 history를 초기화하고 현재 및 과거 정보만 사용한다.
- Policy action, executed action, fallback, projection, OOD, unresolved QC 및 latency를 전 시점에 기록한다.
- 기존 RL Test metric뿐 아니라 end-to-end 유효 step 비율, QC 차단률, fallback률, API payload 생성 성공률 및 처리 지연을 산출한다.
- 동일 seed와 artifact에서 결과가 재현되는지 반복 replay로 검증한다.
- 이 결과를 논문의 RQ3와 online inference 구현 근거로 사용한다.

### Step 10 — Shadow 및 Opt-in Live 준비 검증

- 실제 운영 DB를 읽되 명령은 전송하지 않는 shadow mode를 일정 기간 실행한다.
- 관측 지연, 결측, 품질 차단, action 분포, 안전 개입 및 runtime latency를 검토한다.
- Shadow 결과가 사전 정의된 readiness gate를 통과하면 live 설정 파일을 생성할 수 있게 한다.
- 실제 live 실행은 validated device mapping, API 자격정보, 운영자 승인 및 emergency stop 확인 없이는 시작하지 않는다.
- 본 구현 작업에서는 live adapter와 안전 차단 기능까지 완성하되, 실제 장치에 명령을 보내는 field trial은 별도의 운영 승인과 향후 검증 범위로 남긴다.

## Public Interfaces

- `InferenceMode`: `replay | shadow | live`
- `ObservationSource`: 신규 시점의 관측과 선택적 제어 이력을 제공
- `PolicyActionProvider`: 현재 causal observation에서 deterministic PPO action 생성
- `ControlSink`: record-only 또는 HTTP 장치 출력
- `OnlineInferenceEngine.step()`: 하나의 timestamp를 원자적으로 처리
- `InferenceState`: history, 이전 executed action, 작기·segment 및 마지막 처리 timestamp 관리
- `InferenceResult`: QC, policy, safety, output 및 latency 결과 제공
- `DeploymentBundle`: policy와 모든 의존 artifact의 경로·schema·hash 고정

## Acceptance Criteria

- Replay, shadow 및 live가 동일한 inference core를 사용한다.
- Online 경로에서 미래 시점의 값이나 Test 이후 정보를 참조하지 않는다.
- 고정된 Test 입력에 대한 replay가 반복 실행에서 동일한 결과를 생성한다.
- 필수 observation 미확정, severe OOD, artifact mismatch 또는 stale input에서 정책 명령이 전송되지 않는다.
- `live`는 명시적인 opt-in과 검증된 device mapping 없이는 실행되지 않는다.
- Test는 policy 선택·HPO·threshold calibration에 사용되지 않는다.
- 논문에서 “구현 및 Test replay 검증”과 “실제 온실 closed-loop 검증”을 명확히 구분할 수 있는 manifest와 지표가 생성된다.

## Assumptions

- 현재 실행 중인 GPU 작업은 중단하거나 산출물을 수정하지 않는다.
- 작물별 전이모델 선택은 기존 연구자 handoff를 그대로 사용한다.
- 최종 PPO는 v2.1 calibration과 후속 HPO가 완료된 뒤에만 고정한다.
- 온라인 정책은 deterministic action을 사용한다.
- 실제 장치 API의 자격정보와 device mapping은 저장소에 커밋하지 않는다.
- Field deployment 및 작물 생산성 검증은 이번 코드 구현과 논문의 실험 범위에 포함하지 않는다.
