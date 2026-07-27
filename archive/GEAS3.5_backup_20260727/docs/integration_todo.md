# GEAS3.5 Integration TODO

## MDP v1 Follow-up Decisions

- MDP v2 고도화에서 transition model 위험/운영 구간별 평가를 추가한다.
  - MDP v1은 전체 구간 성능 평가를 우선 사용한다.
  - MDP v2에서는 야간, 고습, 결로위험, 난방 작동, 환기 작동, 난방-환기 동시 작동 구간별 R2/MAE/RMSE/Q90/CVaR90을 별도로 산출한다.
  - 단, 구간별 표본 수가 너무 적은 경우에는 참고 지표로만 사용하고 모델 선택 기준에서는 제외한다.
- MDP v2 고도화에서 VPD/결로위험 transition target 처리 방식을 비교한다.
  - MDP v1은 실내온도/실내습도 예측 결과로 VPD, 이슬점, 결로위험을 파생 계산한다.
  - 추후 실험에서는 VPD/결로위험을 직접 예측 target에 포함하는 방식과 파생 계산 방식의 성능 및 물리 일관성을 비교한다.
  - CO2는 MDP v1 state/action/transition/reward에서 제외한다. 추후 CO2 target 농도, CO2 제어 데이터, reward/safety 기준이 명확해지면 별도 확장으로 재검토한다.
- `GEAS3.5/docs/MDP v1 환경 설정 체크리스트 문서 작성 계획.md`에는 MDP v1 구현에 바로 반영할 확정 항목만 둔다.
- 기술문서 또는 데이터셋 근거가 명확하지 않은 제약조건은 MDP v1에 임의로 포함하지 않고 이 TODO에 남긴다.
- 장치별 최소 유지시간을 추후 검토한다.
  - GEAS Ver3.0 기술문서에는 박스 제약, 램프 제약, 안전 규칙, 투영 기반 보정은 명시되어 있다.
  - 다만 FCU 난방, FCU 냉방, 유동팬 등 MDP v1 직접 action 장치별 최소 유지시간은 구체적인 분 단위로 명시되어 있지 않다.
  - 추후 기존 운영 로그의 스위칭 패턴, 장치 보호 기준, 현장 운영 규칙을 근거로 결정한다.
- action 변경 가능 주기와 최소 유지시간을 장치별로 분리해 정의한다.
  - 천창/차광스크린/보온스크린: 5분마다 조정 가능하되 최대 변화량 제한과 방향 반전 penalty를 우선 적용한다.
  - FCU 난방/FCU 냉방/유동팬: 잦은 ON/OFF 방지를 위한 최소 유지시간 또는 스위칭 penalty 필요성을 검토한다.
  - 레일 3way 밸브, FCU 3way 밸브, 레일 순환펌프, FCU 순환펌프는 직접 action이 아니라 FCU/난방 제어 결과로 따라오는 하위 actuator 값으로 분리해 검토한다.
  - CO2 공급장치는 현재 로그상 정책 학습 근거와 적정 농도 기준이 부족하므로 MDP v1 직접 action에서는 제외하고, CO2 제어 데이터와 목표 기준 확보 후 별도 확장을 검토한다.
- MDP v1 이후 고도화 항목으로 action projection policy를 정교화한다.
  - 제약 위반 action을 차단할지, 가까운 안전 영역으로 보정할지, penalty만 줄지 항목별로 결정한다.
- transition model uncertainty 기반 action 제한 방식을 추후 검토한다.
  - transition model의 예측 불확실성이 큰 상태/action 영역에서는 PPO policy action을 제한하거나 safe controller로 전환하는 방식을 검토한다.
  - uncertainty 계산 후보: probabilistic ensemble 예측분산, 모델 간 예측분산, out-of-distribution score, validation residual 기반 위험구간 threshold.
  - action 처리 후보: 정책 후보에서 제외, 가까운 안전 action으로 projection, reward penalty 부여, rule-based safe controller fallback.
  - 특히 과거 데이터에 드문 조합인 야간 저온 상태의 강한 난방+강한 환기, 결로위험 상태의 급격한 환기, 고습 상태의 환기 부족 action을 우선 검토한다.
- 관수/양액 action을 MDP v2 또는 growth-aware MDP에서 검토한다.
  - MDP v1은 5분 단위 실내환경 제어를 우선 대상으로 하므로 관수/양액 action은 제외한다.
  - 관수/양액을 포함하려면 배지수분, 배지 EC, 배액, 생육 proxy 등 transition target과 수분/양액 관련 reward 및 safety constraint를 함께 설계한다.
- 물리 기반 에너지 사용량 penalty를 추후 검토한다.
  - MDP v1 초기 reward에서는 일반 에너지 penalty를 제외한다.
  - 현재 `Q_heat`, `Q_ventloss`는 난방-환기 동시사용 비효율 penalty 계산에만 사용한다.
  - 추후 에너지 비용 최적화를 명시 목표로 둘 때 `Q_heat`, `Q_ventloss`, 냉방 추정열량, 장치별 전력계수, 단가 정보를 분리된 reward term으로 설계한다.
  - 난방-환기 비효율 penalty와 에너지 비용 penalty가 같은 열손실을 중복으로 과도하게 벌점화하지 않도록 가중치와 normalization을 별도로 검증한다.
- GDPO 방식의 multi-reward normalization을 MDP v1 이후 고도화 항목으로 검토한다.
  - 현재 MDP v1은 train split 기반 reward normalization을 사용하지 않고, 도메인 기준 scaling/clipping으로 reward term을 0~1 범위에 맞춘다.
  - 추후 PPO rollout batch에서 온도, VPD, 과습, 결로, 에너지, action 안정성, 안전제약 reward term을 분리해 advantage를 구성하는 방식을 검토한다.
  - CO2 reward term은 CO2 target 농도와 제어 action이 MDP에 추가되는 경우에만 포함한다.
  - GDPO 논문처럼 항목별 normalization 이후 가중합하고 batch-wise advantage normalization을 적용할지 실험한다.
  - 단, GDPO는 GRPO/group rollout 기반 방법이므로 PPO에 적용할 때는 reward-term별 advantage 구성 방식으로 변형해 검증한다.

이 문서는 현재 placeholder로 남겨둔 연결 지점과 추후 구현해야 할 통합 작업을 정리한다.

## Implemented Transition Module Status

- learned transition module v1은 `src/geas35/models/transition/`에 구현되어 있다.
- 구현 완료 범위:
  - `BaseTransitionModel`, `TransitionDataset`, `TransitionPrediction` 공통 interface
  - dynamic next observation target resolver
  - RL dataset parquet/frame -> transition dataset builder
  - Linear Regression, Linear SVR, KNN baseline wrapper
  - LightGBM, CatBoost, XGBoost, MLP, TCN optional wrapper
  - one-step metric evaluation: R2, MAE, RMSE, Q90, CVaR90
  - `ActionProvider` 기반 rollout simulator
  - `LoggedActionProvider` 실제 구현
  - `PolicyActionProvider` interface-only placeholder
  - strategy/plugin 기반 model selector
  - trainer, artifact writer, `selected_transition_model.json`
  - transition experiment YAML config/runner/CLI
  - selected model load, `predict_next_observation`, predicted next state 기반 reward helper
- 기본 config 위치: `configs/experiments/transition/default.yaml`
- CLI entry point: `python -m geas35.experiments.transition.cli --config <config.yaml>`
- v1 learned transition reward 계산은 transition model이 reward를 직접 예측하지 않고, predicted next state를 기존 `compute_mdp_v1_reward()`에 전달하는 구조를 사용한다.
- 현재 구현된 rollout은 logged action sequence 기반이다. 실제 PPO policy와 연결되는 rollout은 `PolicyActionProvider` 구현 후 별도 통합한다.

## 1. Control Output Log Adapter

- 예정 경로: `src/geas35/data/realtime/control_log_reader.py`
- 실시간 DB의 inner-layer 제어 출력 로그를 읽어 `df_control_log`를 생성한다.
- `2_missing_values_handling.py`에서 action 결측 복원 시 같은 timestamp의 제어 출력 로그를 우선 사용한다.
- offline 실험용 adapter도 별도로 둔다.
  - 예정 경로: `src/geas35/data/offline/control_log_reader.py`
  - split dataset과 같은 timestamp 기준으로 merge 가능한 parquet/csv 형태를 지원한다.

## 2. Offline Pipeline Runner

- 예정 경로: `src/geas35/offline/run_split.py`
- 현재 stage runner들을 하나의 순차 파이프라인으로 연결한다.
  - `1_input_schema_prepared`
  - `2_missing_values_handled`
  - `3_unit_canonicalized`
  - `4_outliers_flagged`
- train/validation/test split별 실행 옵션을 제공한다.
- 각 stage manifest를 하나의 run manifest로 묶어 재현성을 남긴다.

## 3. Realtime Pipeline Runner

- 예정 경로: `src/geas35/realtime/run_once.py`
- 최신 DB row 로딩, 제어 출력 로그 로딩, 결측 처리, 단위 변환, 이상치 처리, feature engineering, RL policy inference, control output 생성을 연결한다.
- 이상치 또는 장기 결측으로 controller 입력 신뢰도가 낮을 때 Safe Controller로 전환한다.

## 4. TCN-Based Outlier Detection

- 예정 경로: `src/geas35/models/tcn/`
- TCN 기반 환경 상태 예측 모델을 구현한다.
- 최근 환경변수와 제어변수 시퀀스를 입력으로 사용한다.
- 다음 시점 환경변수 예측값인 `tcn_pred_value`를 생성한다.
- 예측 residual 기반 `tcn_outlier_score`를 계산한다.
- Warning/Outlier 구분 임계값 `T1`, `T2`를 산출한다.
  - `T1`: warning threshold
  - `T2`: outlier threshold
- 연속 TCN 대체 허용시간 `Lmax`를 산출한다.
- `Lmax`를 초과하는 장기 연속 대체가 발생하면 RL 제어를 중단하고 Safe Controller로 전환한다.
- TCN 연결 후 현재 placeholder 인터페이스와 통합한다.
  - 입력 column 예: `{feature}_tcn_outlier_score`
  - 입력 column 예: `{feature}_tcn_pred_value`
  - 현재 `4_outlier_handling.py`의 rule/TCN 결합 정책은 그대로 사용한다.

## 5. Outlier Handling Policy

- 현재 구현 위치: `src/geas35/preprocessing/4_outlier_handling.py`
- 원본 feature 값은 절대 수정하지 않는다.
- feature별로 다음 column을 생성한다.
  - `{feature}_raw_value`
  - `{feature}_controller_value`
  - `{feature}_rule_outlier_flag`
  - `{feature}_tcn_outlier_score`
  - `{feature}_tcn_pred_value`
  - `{feature}_ai_warning_flag`
  - `{feature}_ai_outlier_flag`
  - `{feature}_outlier_imputed_flag`
  - `{feature}_high_confidence_outlier_flag`
- 실제 TCN 모델이 연결되지 않은 경우에는 TCN 기반 대체를 수행하지 않고 raw value를 controller value로 유지한다.

## 6. Feature Engineering

- 예정 경로: `src/geas35/features/`
- GEAS3.0 baseline feature와 GEAS3.5 추가 feature를 분리해 정리한다.
- offline/realtime 양쪽에서 같은 feature builder를 사용하도록 한다.
- raw value, controller value, missing flag, outlier flag를 명시적으로 선택할 수 있게 한다.

## 7. RL Policy

- 예정 경로: `src/geas35/models/rl/`
- offline 학습 artifact 저장 경로를 정한다.
- realtime에서는 저장된 policy artifact만 load해서 action을 산출한다.
- 장기 결측, 장기 TCN 대체, high-confidence outlier 등 안전 조건을 policy inference 앞에서 확인한다.
- transition rollout과 실제 RL policy를 연결하려면 `PolicyActionProvider` 구현체를 추가한다.
- policy-backed rollout에서는 current predicted state를 policy observation으로 변환하고, policy action을 transition input action columns로 매핑하는 adapter가 필요하다.
- selected transition model inference API는 이미 `src/geas35/models/transition/inference.py`에 있으므로, realtime/policy 통합 단계에서는 이 API를 재사용한다.
