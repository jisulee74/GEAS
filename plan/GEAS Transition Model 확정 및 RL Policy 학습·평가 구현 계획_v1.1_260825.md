# GEAS Transition Model 확정 및 RL Policy 학습·평가 구현 계획

## 1. 현행 구현 판정

- 완료·재사용:
  - MDP v1의 State, 6개 Action, Reward, 5분 transition, episode boundary, Train-only observation scaler.
  - 세 작물 RL-ready Train/Validation/Test dataset과 provenance/hash chain.
  - ExtraTrees를 포함한 8개 Transition 후보의 HPO, one-step·rollout·resource·Test 평가.
  - Step 13의 24개 candidate 무결성, cold-load inference/reward, 재현성 검증.
  - explicit candidate loader, prediction/reward helper, Step 14 외부 연구자 결정 검증 및 handoff 생성기.
- 신규 구현:
  - 세 작물 ExtraTrees 연구자 결정 파일과 Step 14 공식 실행 산출물.
  - ExtraTrees 기반 model-driven RL simulator.
  - Train 데이터 기반 state/action support·OOD detector와 model-exploitation 방지 계층.
  - Gymnasium 호환 environment adapter, hybrid-action PPO, RL HPO·학습·평가 pipeline.
  - baseline, robustness/safety, 최종 Test 및 policy deployment handoff.
- 변경하지 않을 사항:
  - 기존 candidate artifact를 복제·재학습하거나 Transition 후보 비교를 다시 수행하지 않는다.
  - 기존 replay용 `MdpV1Env`와 Reward 계산은 회귀 검증용으로 유지한다.
  - crop별 독립 Policy를 학습하며 공통 crop-conditioned Policy는 이번 범위에서 제외한다.

## 2. Step-by-Step 구현

### Step 14. Transition Model 최종 확정 및 Handoff 완료

- 목적: 연구자의 ExtraTrees 결정을 기존 candidate artifact에 명시적으로 연결한다.
- 구현 내용:
  - strawberry, melon, cucumber 모두 `extra_trees`로 지정한 researcher decision JSON 작성.
  - 각 작물의 Validation rollout weighted score 1위와 주요 one-step/rollout 지표를 근거로 기록한다.
  - 상대적으로 느린 inference와 큰 artifact, 특히 cucumber 모델 크기를 resource trade-off 및 deployment constraint로 명시한다.
  - 기존 Step 14 runner로 hash, schema, evidence 값, cold-load prediction을 검증하고 artifact path만 handoff한다.
  - 실제 RL simulator에서 세 모델을 load하여 동일 feature 순서, 3-target 출력, reward handoff까지 smoke test한다.
- 선행 조건: Step 12·13 passed manifest와 연구자 결정.
- 완료 기준:
  - 세 작물 모두 명시적 ExtraTrees path로 load·inference·reward 계산이 성공한다.
  - artifact 복제 수가 0이고 자동 추천/선택 필드가 없다.
  - v1.4 Step 1~14의 통합 완료 조건과 hash chain이 모두 통과한다.
- 주요 산출물:
  - `researcher_decision.json`
  - `step14_handoff_manifest.json`
  - `explicit_candidate_deployment_handoff.json`
  - `step14_handoff_summary.md`
  - Transition v1.4 completion audit

### Step 15. RL Experiment 및 State-Action Support Contract 동결

- 목적: Test 접근 전에 논문 실험 프로토콜, simulator 계약, model-exploitation 방지 기준을 사전 고정한다.
- 구현 내용:
  - 작물별 ExtraTrees SHA-256, RL dataset/scaler, MDP·Reward config, feature/action 순서, library/container 정보를 단일 manifest에 연결한다.
  - Train은 PPO 학습, Validation은 HPO·checkpoint·policy 선택, Test는 최종 확정 후 1회 평가로 제한한다.
  - seed 목록, 30-trial HPO, trial당 1–2M environment steps, 최종 10 seeds를 pre-register한다.
  - 모든 metric 정의, aggregation, bootstrap CI, paired comparison, 실패·중단 처리 규칙을 고정한다.
  - Test loader는 final-evaluation flag와 protocol-freeze hash가 없으면 fail-closed하도록 설계한다.
  - 작물별 Train episode만 이용해 joint state-action support model을 구축한다.
    - ExtraTrees에 전달되는 실제 input schema를 기준으로 continuous feature는 기존 Train scaler를 사용한다.
    - binary·one-hot·quality feature는 의미를 보존한 거리로 처리한다.
    - episode 단위로 분리한 Train reference/calibration subset에서 k-nearest-neighbor nonconformity score를 계산한다.
    - warning/severe threshold는 각각 사전 고정한 95%/99% conformal coverage로 산출하며 Validation/Test로 재보정하지 않는다.
  - state-only score와 joint state-action score를 분리해 “이미 OOD인 상태”와 “상태는 정상이나 Policy Action이 support 밖인 경우”를 구분한다.
  - Train에서 유사 state 이웃의 관측 Action으로 local action support를 정의한다.
    - continuous Action은 conformal-calibrated local interval을 사용한다.
    - binary Action은 local Train support가 확인된 값만 feasible로 간주한다.
    - local evidence가 부족하면 임의 extrapolation 대신 기존 rule-based safe fallback을 사용한다.
- 선행 조건: Step 14.
- 완료 기준:
  - 동일 Train data/hash에서 support score와 threshold가 재현된다.
  - Train calibration coverage가 사전 정의 수준과 일치한다.
  - Validation/Test가 threshold·action envelope fit에 사용되지 않는다.
  - config 변경 시 새 protocol version을 요구하고 Test 접근 로그가 남는다.
- 주요 산출물:
  - RL protocol manifest
  - split-access policy
  - seed/budget registry
  - environment schema
  - crop별 state/action support artifact와 conformal calibration report

### Step 16. ExtraTrees 기반 RL Simulator 구현

- 목적: 기록된 다음 상태를 재생하는 기존 환경을 학습된 dynamics 환경으로 확장하고 OOD 이동을 매 step 감시한다.
- 구현 내용:
  - 내부 state는 unscaled canonical row로 유지하고 Policy observation만 기존 Train scaler로 변환한다.
  - 각 5분 step에서 현재 State와 제안·실행 Action의 state-only/joint OOD score를 먼저 계산한다.
  - 현재 State와 실행 Action을 ExtraTrees에 입력해 다음 온도·습도·CO₂를 예측한다.
  - 다음 timestamp의 외부환경은 provider에서 공급하고 시간·태양·생육 context와 deterministic derived feature는 기존 feature 함수를 재계산한다.
  - `obs_prev_*`는 현재 실행 Action으로 갱신하고 Reward는 기존 `compute_mdp_v1_reward`를 그대로 호출한다.
  - episode는 기존 rollout ID 및 날짜·series·segment·5분 단절 경계를 그대로 사용하며 마지막 valid transition에서 종료한다.
  - Train/Validation/Test별 recorded-weather provider를 분리하고 운영용 forecast-provider interface를 추가하되 예보 모델 자체는 범위에서 제외한다.
  - OOD 대응은 Train 기반 support 수준에 따라 고정한다.
    - in-support: Action을 그대로 실행한다.
    - warning joint OOD: local Train action envelope로 Action을 투영하고 conformal exceedance에 비례한 OOD penalty를 적용한다.
    - severe action OOD: 제안 Action을 거부하고 rule-based safe fallback Action을 실행하며 기존 Reward 범위의 최악값에 해당하는 penalty를 적용한다.
    - severe state OOD 또는 반복 severe OOD: rollout을 종료하고 남은 예정 step을 최악의 step reward로 채운 pessimistic terminal cost를 기록한다.
  - 조기 종료가 높은 reward를 만드는 exploitation을 막기 위해 HPO·평가 denominator는 실제 실행 step이 아니라 해당 episode의 예정 valid step 수를 사용한다.
  - `info`에 raw/executed/fallback Action, state/joint OOD score, threshold, intervention, terminal-cost 정보를 기록한다.
- 선행 조건: Step 14 handoff, Step 15 support artifact, 기존 MDP/scaler/feature utilities.
- 완료 기준:
  - observation/action shape와 dtype이 고정되고 NaN/Inf가 없다.
  - 같은 seed·초기점·action sequence에서 trajectory와 OOD 판정이 재현된다.
  - 1-step 결과가 explicit inference helper와 수치적으로 일치한다.
  - episode가 split·날짜·series 경계를 넘지 않는다.
  - Policy가 severe OOD termination으로 episode를 단축해 objective를 높일 수 없다.
- 주요 산출물: model-driven environment, exogenous-provider interface, OOD safety layer, environment manifest, trajectory trace.

### Step 17. Environment Dynamics·Reward·Constraint·OOD 검증

- 목적: Policy 학습 전 simulator 의미, support 판정, 안전 계약을 검증한다.
- 구현 내용:
  - zero/logged/extreme Action으로 action sensitivity와 반응 유무·범위·재현성을 검사한다.
  - logged Action rollout을 Step 12의 15/30/60분 지표와 교차 검증한다.
  - Reward 총합과 6개 penalty 분해가 기존 MDP 함수 결과와 일치하는지 검사한다.
  - 기존 강우·풍속 vent cap과 action range projector를 재사용하고 raw/executed Action, intervention reason을 기록한다.
  - termination과 truncation을 데이터 경계, severe OOD, training time-limit로 구분한다.
  - Train의 logged state-action이 기대 conformal coverage를 만족하는지 확인한다.
  - 의도적으로 support 밖 Action과 state perturbation을 입력하여 warning, projection, fallback, penalty, pessimistic termination이 순서대로 작동하는지 검증한다.
  - OOD detector가 crop·growth stage·day/night별 특정 정상 구간을 과도하게 배제하지 않는지 Train 내부 subgroup coverage를 보고한다.
- 선행 조건: Step 16.
- 완료 기준:
  - dynamics·reward·constraint·termination·OOD contract 테스트가 모두 통과한다.
  - Train/Validation/Test 환경이 서로 다른 episode registry만 사용한다.
  - logged-action trajectory가 severe OOD로 체계적으로 오판되지 않는다.
- 주요 산출물: environment validation report, constraint/OOD test matrix, rollout parity report, support coverage report.

### Step 18. Hybrid-Action PPO 구현

- 목적: continuous 3개와 binary 3개를 의미 손실 없이 공동 최적화하고 model exploitation을 학습 중 억제한다.
- 구현 내용:
  - PyTorch 기반 PPO actor-critic을 구현한다.
  - continuous head는 `[0,1]` bounded Beta distribution 3개, binary head는 Bernoulli distribution 3개를 출력한다.
  - joint log-probability와 entropy는 여섯 component의 합으로 계산하고 하나의 PPO clipped objective와 value loss를 사용한다.
  - dynamic hard bound와 Step 15의 local Train action support를 가능한 경우 distribution-to-feasible-range transform 또는 binary feasibility mask로 적용한다.
  - Policy가 제안한 raw Action과 환경이 실행한 Action을 모두 저장하고 support projection/fallback 비율을 학습 metric으로 기록한다.
  - warning OOD penalty, severe fallback, pessimistic terminal cost가 PPO return/GAE에 포함되도록 한다.
  - observation normalization은 기존 artifact scaler만 사용하고 학습 중 재추정하지 않는다.
  - GAE, clipped value loss, advantage normalization, gradient clipping, target-KL early stop, deterministic/stochastic inference mode를 지원한다.
  - checkpoint에는 network, optimizer, scheduler, RNG, normalization·environment·Transition·support artifact hash를 함께 저장한다.
- 선행 조건: Step 16~17.
- 완료 기준:
  - sampled continuous/binary Action이 각각 MDP와 local support 계약을 만족한다.
  - 저장 전후 동일 observation의 deterministic Action/value가 일치한다.
  - PPO ratio, entropy, GAE, mixed log-probability와 OOD penalty propagation 단위 테스트가 통과한다.
- 주요 산출물: hybrid PPO package, support-aware policy config/schema, checkpoint loader, training CLI.

### Step 19. PPO Trainability 및 구현 정상성 Pilot

- 목적: HPO 이전에 hybrid PPO와 support-aware training loop가 정상적으로 학습 가능한지만 확인한다.
- 구현 내용:
  - 세 작물의 축소 Train episode에서 NaN/Inf, loss 발산, exploding/vanishing gradient, invalid Action을 검사한다.
  - reward와 advantage가 상수가 아니고 Policy update에 사용할 유효한 learning signal을 제공하는지 확인한다.
  - policy/value loss, explained variance, KL, entropy, clip fraction, gradient norm이 유한하며 update에 반응하는지 확인한다.
  - 서로 다른 대표 State에 대해 Action distribution과 value output이 동일 상수로 붕괴하지 않고 state-dependent response를 보이는지 검사한다.
  - in-support trajectory 비율, warning/severe OOD 발생률, action projection/fallback 비율을 확인해 support layer가 학습을 전부 차단하거나 사실상 비활성화되지 않았는지 검증한다.
  - resume/checkpoint와 seed reproducibility를 짧은 run에서 검증한다.
  - 이 단계에서는 baseline 대비 성능 우위를 요구하지 않으며 reward가 낮다는 이유로 PPO를 탈락시키지 않는다.
  - 대안 알고리즘 검토는 구현 오류를 배제한 뒤에도 PPO objective 자체가 수치적으로 학습 불가능하거나 mixed distribution과 구조적으로 양립하지 않는 경우에만 시작한다.
- 선행 조건: Step 18.
- 완료 기준:
  - NaN/발산 없이 parameter update가 수행된다.
  - reward/advantage learning signal, KL/entropy, gradient, state-dependent Action 반응이 사전 정의 sanity 범위를 만족한다.
  - baseline 성능 우위 여부와 무관하게 HPO 실행 가능 판정을 내릴 수 있다.
- 주요 산출물: trainability decision record, pilot learning diagnostics, OOD/intervention curves, PPO continuation/fallback 판정.

### Step 20. Hyperparameter Search 설계 및 실행

- 목적: episode 길이와 Test에 영향을 받지 않고 작물별 PPO 설정을 선택한다.
- 구현 내용:
  - Optuna 등 재현 가능한 sampler를 사용해 작물별 30 trials를 수행한다.
  - 탐색 대상은 learning rate, rollout length, batch size, epochs, gamma, GAE lambda, clip range, entropy/value coefficients, target KL, network width/depth, Beta concentration floor이다.
  - 1차 successive-halving으로 불안정 trial을 중단하고 상위 설정은 복수 seed로 재평가한다.
  - primary objective는 Validation의 `mean reward per scheduled valid step`으로 고정한다.
    - 실제 누적 reward에 OOD penalty와 pessimistic terminal cost를 포함한다.
    - denominator는 원래 예정된 valid transition 수로 하여 episode 길이와 조기 종료 효과를 제거한다.
    - crop 내 episode를 동일 가중할지 step 가중할지는 protocol에서 고정하며 기본값은 전체 scheduled valid step 기준 micro-average다.
  - episodic return, discounted return, 실제 실행 step당 reward는 보조 지표로만 보고하고 HPO primary objective로 사용하지 않는다.
  - hard constraint violation은 기존과 같이 infeasible gate로 사용한다.
  - severe OOD rate와 safety fallback rate가 Train 기반 허용 기준을 넘는 trial도 infeasible 처리하며 threshold 자체는 HPO하지 않는다.
  - action instability와 warning OOD rate는 safety gate 통과 trial의 동률 해소 지표로만 사용한다.
  - 동일 Validation episode와 paired exogenous trajectories로 trial 간 비교한다.
- 선행 조건: Step 19 PPO trainability 통과.
- 완료 기준:
  - 작물별 best configuration이 length-normalized objective와 Test 미사용 원칙으로 선택된다.
  - episode 길이 또는 의도적 조기 종료만으로 trial 순위가 개선되지 않는다.
  - 모든 trial의 config, seed, budget, intermediate/final metrics, OOD 통계와 실패 사유가 보존된다.
- 주요 산출물: HPO database, trial table, best config, normalized-objective audit, HPO report.

### Step 21. 최종 RL Training Protocol

- 목적: 선택된 설정으로 통계적으로 평가 가능한 Policy ensemble을 생성한다.
- 구현 내용:
  - 작물별 best config를 10개 고정 seed로 Train environment에서 학습한다.
  - episode 시작점은 Train episode registry에서 seed 기반으로 sampling하며 Validation episode는 gradient update에 사용하지 않는다.
  - 각 rollout과 checkpoint에서 state/joint OOD score 분포, warning/severe 비율, projection, fallback, OOD termination을 기록한다.
  - OOD score 또는 fallback이 지속적으로 증가하는 run은 model exploitation 의심 상태로 표시하며 사전 정의 safety gate에 따라 중단한다.
  - 일정 step마다 atomic checkpoint와 RNG state를 저장하고 resume가 동일 trajectory를 재현하도록 한다.
  - checkpoint 선택은 Validation의 `mean reward per scheduled valid step`과 safety/OOD gate를 사용하며 공통 evaluation schedule과 patience를 적용한다.
  - Transition Model과 Policy RNG를 분리하고 모든 determinism flag와 하드웨어 정보를 기록한다.
- 선행 조건: Step 20.
- 완료 기준:
  - 30개 최종 run이 budget을 완료하거나 사전 정의 실패 사유를 가진다.
  - 독립 재실행에서 metric, OOD 통계와 deterministic rollout이 허용 오차 내 재현된다.
  - model exploitation safety gate를 통과한 checkpoint만 후속 평가 대상으로 전달된다.
- 주요 산출물: crop/seed별 checkpoints, training/OOD curves, run manifests, reproducibility bundle.

### Step 22. Baseline 정의 및 비교 실험

- 목적: RL 성능을 운영적으로 해석 가능한 기준과 비교한다.
- 구현 내용:
  - Logged-action replay: 실제 기록 Action을 동일 simulator에서 재생.
  - Persistence-action: episode 첫 Action 또는 직전 Action 유지.
  - Existing rule-based controller: 현재 MDP 목표온도·습도·VPD·weather constraint를 사용하는 결정론적 controller.
  - Safe random policy: Train local support 안의 무작위 정책으로 sanity lower bound만 제공.
  - PPO ablation: binary Action을 logged 값으로 고정한 continuous-only policy로 mixed control의 기여를 평가.
  - 모든 방법은 동일 초기 state, exogenous trajectory, episode, support detector와 seed로 paired 평가한다.
- 선행 조건: Step 17, Step 21.
- 완료 기준: 모든 baseline이 동일 I/O, constraint, OOD contract로 실행되고 PPO와 paired comparison이 가능하다.
- 주요 산출물: baseline registry, baseline configs, paired evaluation table.

### Step 23. Offline Policy Evaluation

- 목적: 실제 배포 전 환경 유지 성능, Action 품질, model-exploitation 여부를 다면적으로 평가한다.
- 구현 내용:
  - 성능: scheduled-valid-step normalized reward, episodic return, step reward, 6개 reward term, discounted/undiscounted return.
  - 환경 유지: 목표온도 band 내 시간, RH/VPD/결로 안전 범위 충족률, deviation magnitude·duration, CO₂ trajectory 기술통계.
  - Action behavior: 사용률, duty cycle, 평균·분산, saturation, switching count, ramp/reversal, heat–vent 동시 사용.
  - 안정성: state/action total variation, oscillation, recovery time, long-horizon drift, termination 이상률.
  - OOD: state/joint OOD score 분포, warning/severe 비율, support exceedance duration, projection/fallback/termination 비율.
  - 안전: raw 및 executed constraint violation, safety-layer intervention rate와 magnitude.
  - 평균, 표준편차, median, 95% bootstrap CI와 seed/episode paired effect size를 보고한다.
  - baseline 대비 정식 성능 우위 판단은 이 단계에서 수행한다.
  - 실제 behavior propensity가 없으므로 importance-sampling OPE를 주장하지 않고 “support-aware learned-simulator 기반 model-based offline evaluation”으로 명시한다.
- 선행 조건: Step 21~22.
- 완료 기준:
  - Validation에서 PPO와 모든 baseline의 동일 protocol 평가가 완료된다.
  - aggregate뿐 아니라 crop·facility/series·growth stage·day/night별 결과가 제공된다.
  - 성능 개선이 severe OOD나 과도한 safety fallback에 의존하지 않음을 확인한다.
- 주요 산출물: policy evaluation CSV/JSON, trajectory samples, OOD plots, statistical comparison report.

### Step 24. Robustness 및 Safety 평가

- 목적: 분포 변화와 Transition prediction error에 대한 Policy 취약성을 정량화한다.
- 구현 내용:
  - 외부환경 stress: 온도·습도·일사·풍속·강우의 Validation 기반 quantile scenario와 시간 블록 perturbation.
  - 초기 상태 stress: 목표 범위 밖 고온·저온·고습·저VPD·결로 근접 상태.
  - Transition error: Validation residual의 target 간 상관과 시간 블록을 보존한 bootstrap noise, bias/drift, error-scale sweep.
  - 모델 불확실성: ExtraTrees tree/sub-ensemble 또는 calibrated residual ensemble을 이용한 stochastic rollout.
  - sensor 문제: 허용된 quality flag, exogenous delay/missing fallback, bounded observation perturbation.
  - support-aware training이 없는 PPO ablation과 비교하여 OOD penalty·constraint·fallback이 model exploitation을 실제로 줄였는지 평가한다.
  - 각 scenario에서 normalized reward degradation, OOD duration, safety violation, intervention, recovery, catastrophic trajectory rate를 평가한다.
  - hard constraint 위반 또는 NaN/물리 범위 이탈은 평균 reward와 무관하게 fail 처리한다.
- 선행 조건: Step 23 및 Validation residual artifact.
- 완료 기준:
  - 사전 정의 stress matrix를 10개 policy seed에 동일하게 적용한다.
  - failure envelope와 안전 한계가 crop별로 문서화되고 심각 violation이 없는 checkpoint만 final 후보가 된다.
- 주요 산출물: robustness matrix, residual/noise calibration artifact, model-exploitation ablation, safety audit, failure-case trajectories.

### Step 25. Validation 기반 Policy 후보 확정

- 목적: Test 공개 전에 작물별 최종 checkpoint를 하나로 동결한다.
- 구현 내용:
  - mean reward per scheduled valid step, 환경 안전범위 충족, hard violation 0, severe OOD·intervention 한도, action stability를 계층적 기준으로 적용한다.
  - 우선 안전·OOD gate를 통과한 seed/checkpoint만 남기고 이후 normalized reward와 안정성으로 하나를 선택한다.
  - 우수 reward가 Train support 밖 state-action trajectory에 의존하는 checkpoint는 제외한다.
  - 선택 규칙과 동률 처리, checkpoint hash를 decision record에 고정한다.
  - 모든 config·code·dependency·Transition/scaler/support/exogenous hash를 freeze한다.
- 선행 조건: Step 23~24.
- 완료 기준: 작물별 단일 checkpoint가 Test를 보지 않고 확정되고 이후 수정이 잠긴다.
- 주요 산출물: final-policy candidate decision, frozen checkpoint index, Test authorization manifest.

### Step 26. Final Test 및 최종 Policy 확정

- 목적: 완전히 보류된 Test episode에서 최종 일반화 성능을 1회 측정하고 최종 특성을 기술한다.
- 구현 내용:
  - Step 25의 exact checkpoint와 baseline만 Test recorded-weather environment에서 평가한다.
  - Test 결과는 재학습, HPO, OOD threshold, safety constraint, checkpoint 교체, Policy 변경 또는 재선택에 사용하지 않는다.
  - Validation과 동일한 normalized reward, 환경·Action·OOD·안전 metric 및 paired statistics를 산출한다.
  - 사전 동결된 제한적 robustness scenario를 Test에서 실행하되 이는 `final characterization only`로 명시한다.
  - Test robustness는 알려지지 않은 일반화 특성과 failure mode를 기술하기 위한 것이며 어떤 parameter·threshold·constraint·checkpoint도 조정하지 않는다.
  - 일반 Test 또는 Test robustness에서 실패하여 수정이 필요한 경우 현재 protocol의 Policy는 배포 불가로 판정한다.
  - 수정은 기존 Test 결과를 tuning data로 전환하지 않고, 변경 이유와 Test 노출 사실을 기록한 새로운 protocol version으로 Step 15부터 재시작한다.
  - 새 protocol은 가능하면 새로운 holdout을 구성하며, 불가능하면 기존 Test가 더 이상 순수 final Test가 아님을 명시하고 별도 독립 검증이 확보될 때까지 배포를 보류한다.
  - Test 후 Train+Validation 재학습은 하지 않으며 배포 artifact는 실제 Test를 거친 checkpoint와 동일하게 유지한다.
- 선행 조건: Step 25 freeze 및 Test authorization.
- 완료 기준:
  - 세 작물 Test 평가와 접근 로그가 완전하다.
  - hard constraint/수치 안정성 gate를 통과하고 baseline 대비 효과와 CI가 보고된다.
  - Test robustness 결과가 final characterization으로만 표시되고 선택·조정 입력에서 제외되었음이 감사 가능하다.
  - 최종 확정 또는 배포 보류 판정이 명시된다.
- 주요 산출물: final Test report, final-characterization robustness report, test-access log, statistical tables, final policy decision record.

### Step 27. Deployment 및 연구 Handoff

- 목적: 평가된 Policy를 재현·감사·통합 가능한 단위로 전달한다.
- 구현 내용:
  - Policy checkpoint, architecture/config, observation/action schema, deterministic inference wrapper를 묶는다.
  - 연결된 ExtraTrees path/hash, scaler, MDP/Reward/constraint config, support/OOD artifact, exogenous-provider contract를 참조한다.
  - Python·PyTorch·NumPy·scikit-learn 버전, source commit/diff identifier, hardware와 전체 file checksum을 기록한다.
  - runtime에서 state/action OOD score를 계산하고 warning, projection, safe fallback, deployment stop 조건을 동일 계약으로 적용한다.
  - 입력 validation, out-of-distribution warning, safety-layer 적용, fallback rule-based controller 조건을 runtime contract로 명시한다.
  - model card에는 학습 데이터 범위, known limitation, model-based evaluation 한계, crop별 support·안전 envelope를 기록한다.
- 선행 조건: Step 26.
- 완료 기준:
  - 깨끗한 환경에서 bundle load와 고정 fixture inference가 재현된다.
  - artifact 변경·schema mismatch·잘못된 crop 연결은 fail-closed한다.
  - severe OOD에서 Policy Action 대신 검증된 fallback이 동작한다.
  - Transition artifact를 복제하지 않고 명시적 immutable reference를 유지한다.
- 주요 산출물:
  - `policy.pt`
  - `policy_manifest.json`
  - `policy_config.json`
  - `observation_action_schema.json`
  - `support_ood_manifest.json`
  - `evaluation_summary.json`
  - `model_card.md`
  - `deployment_handoff.json`
  - checksum/integrity report

## 3. 주요 인터페이스 변경

- 기존 `MdpV1Env`는 replay 검증용으로 유지하고 별도 model-driven environment를 추가한다.
- environment API는 Gymnasium 형식의 `reset(seed, options)`와 `step(action) -> observation, reward, terminated, truncated, info`를 제공한다.
- Action 외부 표현은 기존 순서의 길이 6 vector를 유지하되 Policy 내부에서는 Beta 3개와 Bernoulli 3개로 분리한다.
- environment의 `info`는 raw/executed Action, state/joint OOD score, support threshold, projection/fallback 및 OOD termination을 표준 필드로 제공한다.
- exogenous provider는 `reset(episode_context)` 및 다음 timestamp별 row 제공 계약을 사용하며 recorded/forecast/stress mode를 manifest에 기록한다.
- policy loader는 crop, checkpoint, Transition artifact, scaler, support artifact, schema hash가 모두 일치해야 load한다.
- Test 접근과 최종 checkpoint 선택은 별도 immutable manifests로 분리한다.

## 4. 테스트 및 논문 Acceptance Criteria

- 단위 테스트:
  - hybrid distribution sampling/log-probability/entropy, GAE, PPO clipping, checkpoint round-trip.
  - 5분 transition, episode boundary, derived feature 및 previous-action 갱신.
  - reward decomposition, dynamic constraint, deterministic seeding.
  - conformal support score·threshold 재현, local action envelope, OOD penalty/fallback/termination.
- 통합 테스트:
  - Step 14 handoff에서 support-aware environment까지 artifact cold-load.
  - 짧은 PPO train–resume–evaluate cycle.
  - Train/Validation/Test episode, scaler 및 OOD threshold leakage 방지.
  - baseline과 PPO의 동일 trajectory paired evaluation.
  - severe OOD 조기 종료가 normalized objective를 개선하지 않는지 확인.
- 통계·재현성:
  - 최종 10 seeds, episode bootstrap 95% CI, paired effect size와 원자료 보존.
  - HPO primary metric은 mean reward per scheduled valid step으로 고정한다.
  - 모든 표와 그림을 raw artifact에서 재생성 가능하게 한다.
  - 실패 run을 삭제하지 않고 원인과 포함/제외 규칙을 보고한다.
- 안전:
  - hard constraint violation 0을 최종 gate로 사용한다.
  - NaN/Inf, 물리 범위 이탈, artifact/schema mismatch는 즉시 실패한다.
  - safety-layer intervention과 원 Action을 모두 기록해 성능이 projector나 fallback에 과도하게 의존하는지 공개한다.
  - Test robustness는 final characterization에만 사용되며 어떠한 재선택·조정에도 사용되지 않는다.

## 5. 확정된 가정

- 작물별 독립 Policy 세 개를 학습·확정한다.
- 기본 알고리즘은 custom hybrid-action PPO다.
- 기본 예산은 작물별 HPO 30 trials, trial당 1–2M steps, 최종 10 seeds다.
- Transition·RL 선택과 HPO에는 Test를 사용하지 않는다.
- OOD support와 대응 threshold는 작물별 Train episode 및 conformal calibration으로만 정의한다.
- HPO primary objective는 OOD 비용을 포함한 mean reward per scheduled valid step이며 episodic return은 보조 지표다.
- 실제 behavior propensity가 없으므로 doubly robust/importance-sampling OPE는 수행하지 않는다.
- 운영용 기상예보 생성은 범위 밖이며 provider interface와 forecast replay 검증까지만 포함한다.
- CO₂는 State와 Transition target이지만 현 MDP v1 Reward에는 추가하지 않는다.
- Test 및 Test robustness는 final characterization 전용이며 실패 후 수정은 새로운 protocol version과 독립 검증 절차로만 수행한다.
