# RL HPO Budget Calibration 및 v2 Protocol 전환 계획

## 요약

- 실행 중인 Step 20 v1 HPO는 중단하되 checkpoint, SQLite DB, metrics를 수정·삭제하지 않고 `superseded v1` 증거로 보존한다.
- Step 15의 trial 수는 작물별 30개로 유지하되 `1M→2M` 고정 budget을 제거한다.
- Step 19와 Step 20 사이에 `Step 19.5: Learning-Curve-Based HPO Budget Calibration`을 추가한다.
- Step 19.5가 작물별 successive-halving rung budget을 확정하고, Step 20 v2는 해당 artifact의 hash와 budget을 변경 없이 사용한다.
- 24시간은 연구 결과를 바꾸는 hard limit가 아니라 병렬화와 자원 배치의 운영 목표로만 관리한다.

## 계획 문서 변경

### Step 15 수정

- 작물별 HPO trial 수 `30`과 sampler seed만 사전 고정한다.
- environment-step budget 필드를 제거하고 다음 계약으로 교체한다.
  - `budget_source_step: "19.5"`
  - `budget_status: "pending_calibration"`
  - `budget_artifact_schema: geas35.rl.step19_5.budget.v2`
- Train/Validation/Test 분리, normalized objective, safety gate, support threshold는 변경하지 않는다.
- Step 15 v1을 덮어쓰지 않고 `geas35.rl.step15.v2`와 별도 artifact 경로를 생성한다.

### Step 19.5 추가

- 목적: 실제 learning curve와 ranking stability를 근거로 작물별 HPO rung budget을 결정한다.
- 대표 설정은 Step 19 기본 PPO 설정 1개와 Step 20 search space에서 deterministic maximin 방식으로 뽑은 4개를 사용한다. 결과를 보고 설정을 교체하지 않는다.
- 각 설정은 seeds `42, 43, 44`로 Train에서 학습하고, 동일한 고정 Validation episode·exogenous trajectory에서 평가한다.
- 공통 평가 grid의 시작점은 대표 설정 중 가장 큰 rollout length이며 이후 environment steps를 두 배씩 증가시킨다. 숫자형 최종 budget은 사전 지정하지 않는다.
- 각 checkpoint에서 다음을 저장한다.
  - Validation `mean reward per scheduled valid step`
  - episodic/discounted return 보조 지표
  - policy/value loss, KL, entropy, explained variance, gradient norm
  - in-support, warning/severe OOD, projection, fallback, termination rate
  - seed별 값, median, bootstrap confidence interval
- ranking stability는 인접 checkpoint 간 Kendall rank correlation과 상위 1/3 configuration overlap으로 판단한다.
  - Kendall `τ ≥ 0.8`
  - 상위 1/3 집합 동일
  - 세 번 연속 checkpoint에서 만족
- learning stabilization은 최근 세 구간의 reward 변화가 현재 configuration 간 reward IQR의 5% 이하이고 bootstrap slope CI가 0을 포함하며, KL·entropy·value diagnostics가 유한하고 safety gate를 통과할 때 인정한다.
- 작물별 rung 결정 규칙:
  - 첫 rung: ranking stability가 처음 세 번 연속 확인된 checkpoint
  - 마지막 rung: ranking stability와 learning stabilization이 함께 처음 충족된 checkpoint
  - 마지막/첫 rung 비율이 4를 초과하면 그 사이의 관측 checkpoint 중 geometric midpoint에 가장 가까우면서 ranking criterion을 통과한 지점을 중간 rung으로 추가한다.
  - 기준이 충족되지 않으면 임의 budget을 선택하지 않고 `calibration_inconclusive`로 종료하여 Step 20 진입을 차단한다.
- Test는 접근하지 않는다.
- 산출물:
  - learning-curve table 및 plot
  - seed별 diagnostics
  - ranking-stability matrix
  - crop별 calibrated rung budget
  - 판단 근거 report
  - budget artifact 및 integrity manifest

### Step 20 수정

- 선행 조건을 “Step 19 trainability 통과 및 Step 19.5 budget calibration 통과”로 변경한다.
- 작물별 30 trials는 유지한다.
- successive-halving rung은 Step 19.5의 crop별 budget artifact에서만 읽는다.
- 공식 실행에서 CLI budget override를 금지하고 budget artifact hash가 다르면 fail-closed 처리한다.
- 기존 normalized objective, safety/OOD gate, paired Validation, multi-seed 재평가, Test 잠금은 변경하지 않는다.
- best configuration은 calibrated 최종 rung과 복수 seed 집계 결과로 결정한다.

## v2 구현 및 실행 구조

- 새 경로를 사용한다.
  - config: `step15_protocol_v2`, `step19_5_calibration_v2`, `step20_hpo_v2`
  - artifacts: `step15_v2`, `step19_5_v2`, `step20_v2`
  - study ID와 SQLite DB도 v1과 분리한다.
- 현재 v1 프로세스는 `Ctrl+C`로 중단한다. v1 checkpoint와 DB는 이동·수정하지 않고, v2 migration manifest에서 경로와 hash 및 중단 상태만 참조한다.
- 연구 protocol과 계산 최적화를 구분한다.
  - 연구 protocol: configs, seeds, episode pairing, rung 선택 규칙, objective, safety gate
  - 계산 최적화: worker 수, process affinity, ExtraTrees thread 수, vectorized execution, device placement
- crop-level parallelism:
  - Strawberry/Melon/Cucumber를 독립 process와 독립 crop DB로 실행하고 종료 후 통합한다.
- trial-level parallelism:
  - DB의 atomic lease로 미실행 trial을 claim한다.
  - lease heartbeat와 timeout을 기록하고 죽은 worker의 trial만 checkpoint에서 재개한다.
  - 동일 trial을 두 worker가 동시에 실행하지 못하게 한다.
- vectorized environment:
  - CPU process 기반 환경 여러 개를 동시에 실행하고 observation/action batch만 GPU PPO policy에 전달한다.
  - episode seed와 시작점은 `crop/config/seed/env_index/episode_counter`의 deterministic mapping으로 생성하여 worker 수나 완료 순서에 영향을 받지 않게 한다.
  - vector-env 수와 rollout batch 구성은 Train-only throughput benchmark로 선정한 뒤 Step 19.5와 Step 20에서 동일하게 고정·hash 기록한다.
  - ExtraTrees는 각 environment process에서 `n_jobs=1`로 유지하여 nested oversubscription을 방지한다.
- 24시간 목표는 throughput report와 예상 종료시간에만 사용한다. learning-curve 안정성 기준을 만족하지 않았다는 이유로 budget을 시간에 맞춰 강제 축소하지 않는다.

## 테스트 및 완료 기준

- Step 15 v2에 숫자형 HPO step budget이 없고 trial 수만 30으로 고정됐는지 검사한다.
- Step 19.5가 Train으로만 update하고 Validation은 평가에만 사용하며 Test 접근이 없는지 검사한다.
- synthetic learning curves로 다음을 검증한다.
  - 불안정 ranking에서는 budget을 확정하지 않음
  - 안정 ranking과 plateau에서 2-rung 결정
  - 간격이 큰 경우 관측 checkpoint로 3-rung 결정
  - seed noise가 큰 경우 조기 budget 확정을 거부
- Step 20 v2가 Step 19.5 budget과 hash가 일치할 때만 시작되는지 검사한다.
- CLI numeric budget override와 v1 artifact 입력을 거부하는지 검사한다.
- worker crash/resume, lease 만료, crop/trial 병렬 DB 무결성을 검사한다.
- scalar environment와 vectorized environment가 동일 seed/action sequence에서 reward, OOD, termination을 허용 오차 내 재현하는지 검사한다.
- v1 artifact가 변경되지 않았음을 전후 hash로 확인한다.
- 전체 회귀 테스트 통과 후에도 Step 20 v2 공식 30-trial 실행 전 상태는 `calibrated_ready`, 공식 실행 완료 후에만 `passed`로 변경한다.

## 확정 가정

- 작물별 dynamics와 learning speed가 다르므로 rung budget은 crop별로 결정한다.
- 대표 configuration 수는 5개, seed는 3개로 고정한다.
- 24시간은 운영 목표이며 연구 판단의 hard cutoff가 아니다.
- 기존 v1 결과는 공식 v2 calibration이나 HPO 선택에 사용하지 않는다.
- Step 21 이후 계획과 Train/Validation/Test 원칙은 변경하지 않는다.
