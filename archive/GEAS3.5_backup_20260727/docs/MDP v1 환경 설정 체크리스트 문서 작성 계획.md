# MDP v1 환경 설정 체크리스트

이 문서는 GEAS3.5에서 강화학습 환경(MDP v1)을 어떻게 설계했는지와, 앞으로 추가 논의가 필요한 항목을 한눈에 보기 위해 정리한다.  
세부 구현 로그보다는 state, action, transition, reward, 학습/검증 흐름의 핵심 결정사항을 중심으로 둔다.

## 설계 원칙

- [x] MDP v1은 생육 직접 최적화 전 단계의 **안전한 온실 환경 제어**를 목표로 한다.
- [x] 문서, 기존 GEAS 코드, 데이터에서 근거가 있는 항목만 MDP v1에 우선 반영한다.
- [x] 근거가 부족하거나 현장 장비 기준 확인이 필요한 항목은 `integration_todo.md`에 추후 검토 항목으로 남긴다.
- [x] MDP 환경과 transition model 입력은 같은 state 전처리 흐름을 공유한다.
- [x] 학습과 추론은 작물별로 독립된 모델 artifact를 사용한다.

## 1. 기본 설정

- [x] RL step 시간 간격: 5분
- [x] Episode 단위: 하루
- [x] Discount factor 초기값: `gamma = 0.99`
- [x] 기본 transition horizon: 5분 뒤 다음 상태
- [x] transition model 검증에는 15분, 30분, 60분 multi-step rollout 평가를 포함한다.

## 2. State 설정

- [x] 현재 실내 환경: 실내온도, 실내습도
- [x] 현재 실외 환경: 실외온도, 실외습도, 외부일사량, 외부풍속, 강우여부
- [x] 맥락 변수: 현재시간, 주간/전야/후야, 맑음/구름많음/흐림, 생육단계, DAT, 현재 적용 적정온도 범위
- [x] GEAS 파생변수: 생리 및 결로 안전, 광에너지 균형, 물리안전 및 기구한계 관련 변수
- [x] 데이터 품질 변수: 결측치 발생 여부, 이상치 발생 여부
- [x] 직전 action: 직전 천창, 차광스크린, 보온스크린, FCU 난방, FCU 냉방, 유동팬 상태
- [x] CO2는 MDP v1 state에서 제외한다.
- [x] 작물별로 독립 RL 모델을 학습하는 전제이므로 작물 one-hot은 제외한다.

### State 전처리 기준

- [x] MDP 환경과 transition model 입력은 `prepare_mdp_v1_frame` 기반 공통 observation을 사용한다.
- [x] 생육단계는 one-hot encoding을 사용한다.
- [x] 연속형 observation은 train split 기준 z-score 정규화를 적용한다.
- [x] binary, one-hot, quality flag는 0/1 값 그대로 사용한다.
- [x] 개도율 계열 값은 0~1 범위로 사용한다.
- [x] 실내온도와 실내습도는 현재 MDP v1 코드 기준 1번 센서 값을 사용한다.
- [x] 천창 개도율은 좌/우 천창 평균값을 사용한다.
- [x] 차광스크린과 보온스크린은 각각 단일 구동기 컬럼을 사용한다.

## 3. Action 설정

- [x] 연속 action: 천창 개도율, 차광스크린 개도율, 보온스크린 개도율
- [x] Binary action: FCU 난방 작동 여부, FCU 냉방 작동 여부, 유동팬 작동 여부
- [x] CO2 공급장치, 레일/FCU 3way 밸브, 순환펌프는 MDP v1 직접 action에서 제외한다.
- [x] 관수/양액 action은 MDP v1에서 제외하고, 이후 growth-aware MDP에서 검토한다.

## 4. Transition Model 설정

- [x] Transition model은 데이터 기반 environment model로 구성한다.
- [x] 입력 X: 현재 MDP v1 state + 현재 action
- [x] 예측 y: 5분 뒤 dynamic next observation subset
- [x] transition target에는 시간에 따라 변하는 dynamic observation만 포함하고, static metadata, one-hot metadata, quality/missing/outlier flag, 관리용 컬럼, `obs_prev_*` 직전 action observation은 제외한다.
- [x] `obs_prev_*` 직전 action observation은 rollout state update 과정에서 현재 action으로 deterministic하게 갱신한다.
- [x] Reward는 transition model이 직접 예측하지 않고, 예측된 next observation을 row-like state로 복원한 뒤 기존 `compute_mdp_v1_reward()`로 계산한다.
- [x] 후보 모델: Linear Regression, Linear SVR, KNN, LightGBM, CatBoost, XGBoost, MLP, TCN
- [ ] Probabilistic ensemble과 uncertainty 기반 안전 제어는 MDP v1 이후 고도화 항목으로 둔다.
- [x] 평가 지표: R2, MAE, RMSE, Q90, CVaR90
- [x] one-step prediction 평가와 15분/30분/60분 multi-step rollout 평가 경로를 구현한다.
- [x] model selector는 strategy/plugin 구조로 두고, mean RMSE, weighted RMSE, temperature priority, humidity priority, rollout weighted RMSE, custom score를 교체 가능하게 한다.
- [x] 학습 artifact는 crop/model 단위로 저장하고, selected model은 `selected_transition_model.json`으로 기록한다.
- [x] inference API는 selected artifact load, `predict_next_observation`, predicted next state 기반 reward helper를 제공한다.
- [x] v1 rollout action source는 `LoggedActionProvider`를 실제 구현으로 사용한다.
- [x] `PolicyActionProvider`는 이후 RL policy 연결을 위한 interface-only 상태로 둔다.
- [x] MDP v1에서는 전체 구간 성능을 우선 평가하고, 위험/운영 구간별 평가는 MDP v2 고도화 항목으로 넘긴다.

## 5. Reward 설정

- [x] 기본 구조: `reward = - 전체 penalty`
- [x] 온도 penalty: 생육단계별 적정온도 하한/상한 범위 이탈 정도
- [x] `P_rh`: 상대습도 90% 초과
- [x] `P_vpd`: `VPD < 0.3 kPa`
- [x] `P_cond`: 결로 여유 0.8도 미만
- [x] `P_hv`: 환기 열손실이 난방량의 0.25배를 초과하는 정도
- [x] `P_act`: 3-step 기준으로 연속적인 방향 반전이 발생하는 정도
- [x] 일반 에너지 사용량, CO2 사용량, 유동팬 사용 자체 penalty는 MDP v1 초기 reward에서 제외한다.
- [x] Reward term은 데이터 분포 기반 fit이 아니라 도메인 기준 scaling/clipping으로 0~1 범위에 맞춘다.
- [x] 최종 reward clipping은 초기에는 적용하지 않는다.

### Reward term scaling 기준

| Reward 항목 | 최대 penalty 기준 |
|---|---|
| `P_temp` | 목표온도 허용범위에서 3도 이상 이탈 |
| `P_rh` | RH 90% 초과 후 10%p 이상 초과 |
| `P_vpd` | VPD 0.3 kPa 미만, VPD 0 kPa에서 최대 |
| `P_cond` | 결로 여유 0.8도 미만, 결로 여유 0 이하에서 최대 |
| `P_hv` | `Q_heat > Q_max`일 때 초과 환기 열손실이 난방량의 0.5배 이상 |
| `P_act` | 개별 action 방향 반전 penalty를 0~1로 계산한 뒤 장비군별 가중 평균 |

### 초기 reward 가중치

| Reward 항목 | 초기 가중치 |
|---|---:|
| `w_temp` | 1/6 |
| `w_rh` | 1/6 |
| `w_vpd` | 1/6 |
| `w_cond` | 1/6 |
| `w_hv` | 1/6 |
| `w_act` | 1/6 |

초기 실험에서는 특정 penalty 항목에 임의 우선순위를 두지 않는 중립 baseline으로 시작한다.  
이후 reward 로그와 policy 평가 결과를 보고 필요한 경우 항목별 가중치를 조정한다.

현재 코드에는 별도의 `안전제약 위반 penalty` 가중치가 따로 있지는 않다.  
안전제약은 action 범위 제한, clipping, projection, 난방-환기 비효율 penalty 등을 통해 우선 반영한다.

## 6. Safety Constraint 설정

- [x] Action 값은 실제 장비 표현 범위 안으로 제한한다.
- [x] 연속 action은 0~1 범위로 제한한다.
- [x] Binary action은 0/1로 처리한다.
- [x] 강우 시 천창 개도율은 0으로 제한한다.
- [x] 강풍 시 천창 개도율은 최대 0.2로 제한한다.
- [x] 난방-환기 동시사용 비효율은 `P_hv` soft penalty에 반영한다.
- [x] MDP v1 hard constraint와 soft penalty의 역할을 분리한다.
- [x] MDP v1 제약 위반 action은 hard constraint는 projection, soft constraint는 penalty로 처리한다.

## 7. RL 학습 설정

- [x] PPO를 기본 후보 알고리즘으로 둔다.
- [x] 작물별 데이터셋으로 독립 모델을 학습하고, 추론 시 현재 작물에 맞는 모델을 로드한다.
- [ ] PPO observation/action space 최종 점검
- [x] Learned transition model 기반 rollout 방식 확정: v1은 `ActionProvider` abstraction을 사용하고 실제 구현은 `LoggedActionProvider` 기반 rollout으로 둔다.
- [ ] RL policy 연동 rollout 방식 확정: `PolicyActionProvider` 구현과 PPO policy 연결은 후속 단계에서 진행한다.
- [ ] Offline evaluation 방법 확정
- [ ] Baseline policy 설정: 기존 운영자 행동, rule-based GEAS, 무작위 정책 등

## 8. 검증 및 완료 기준

- [x] MDP v1 문서와 코드의 state/action/reward/transition module 구현 상태 일치 확인
- [ ] 샘플 데이터로 environment reset/step 동작 확인
- [ ] Reward term이 개별적으로 계산되고 로그에 남는지 확인
- [x] Transition target 생성 시점 정렬 검증 경로 구현: `prepare_mdp_v1_frame` -> RL dataset -> `TransitionDatasetBuilder`
- [x] Leakage feature 방지 경로 구현: dynamic target resolver denylist/allowlist, `next_*`, reward, flag, metadata 제외
- [x] Transition model 후보별 성능 비교와 selected artifact 생성 경로 구현
- [x] 5분 one-step 및 15/30/60분 rollout 성능 평가 코드 경로 구현
- [ ] 실제 crop별 RL dataset으로 transition experiment 실행 및 성능 결과 검토
- [ ] PPO 학습 전 environment sanity check 통과

## 추후 검토 항목

- [ ] 실내온도/습도 2번 센서 fallback 또는 대표값 집계 방식 추가 검토
- [ ] 장치별 최소 유지시간과 action 변경 가능 주기 검토
- [ ] 관수/양액 action 포함 여부 검토
- [ ] CO2 제어 목표와 reward 기준 검토
- [ ] 위험/운영 구간별 transition model 평가 추가
- [ ] GDPO 방식 reward normalization 고도화 검토
