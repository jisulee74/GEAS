# GEAS Transition Model 설계 계획 v1.4

## 1. Summary

GEAS Transition Model v1.4의 목적은 스마트팜 상태 전이를 근사하는 후보 모델을 동일한 조건에서 학습·평가하고, 연구자가 최종 모델을 판단할 수 있는 객관적인 비교 결과를 제공하는 것이다.

공식 learned transition은 현재 State와 6개 Action으로부터 5분 뒤 실내온도, 실내습도 및 실내 CO₂를 예측한다. 외부 기상과 context는 예측하지 않고 rollout 시 provider로 공급한다.

Framework는 후보별 HPO, Validation/Test 평가, rollout benchmark 및 resource benchmark를 수행하지만 최종 Transition Model을 자동으로 선정하지 않는다. 이 원칙은 GEAS QC Model Comparison Framework와 동일하다.

공식 후보는 다음 8개로 고정한다.

1. `persistence`
2. `linear_regression`
3. `linear_svr`
4. `knn`
5. `extra_trees`
6. `lightgbm`
7. `xgboost`
8. `mlp`

TCN, CatBoost, XGBRF, RandomForest 및 Probabilistic Ensemble은 v1.4 범위에서 제외한다. 기존 TCN·CatBoost 구현은 Transition artifact나 외부 consumer가 확인되지 않았으므로 wrapper, registry entry, package export 및 관련 테스트를 제거한다. 제거 직전 재검사에서 실제 외부 consumer나 load해야 할 pickle이 발견될 때만 deprecated compatibility shim을 유지한다.

## 2. Architecture and Benchmark Contract

### MDP temporal and episode contract

- 공식 RL step 간격은 정확히 5분이다.
- learned transition은 현재 State와 Action에서 5분 뒤의 내생 환경 상태를 예측한다.
- episode는 날짜 경계를 넘지 않는 최대 하루 단위다.
- crop, series, segment, episode identifier 변경, 5분 간격 단절 또는 필수 State/target invalid가 발생하면 같은 날짜 안에서도 episode를 분리한다.

### State contract

공식 State는 versioned MDP observation schema를 단일 기준으로 사용한다. 문서나 slide의 고정 변수 개수보다 artifact의 feature schema와 canonical column 목록이 우선한다.

- 실내 환경: 실내온도, 실내습도, 실내 CO₂ representative value
- 실내 CO₂ source: in_co2_representative
- CO₂ representative 정책: sensor1 유효값 → sensor2 fallback → 검증된 AI-imputed sensor1 → unavailable
- 실외 환경: 외부온도, 외부습도, 외부 순간·누적 일사량, 외부풍속, 강우
- 맥락: 시간 주기, 주간/야간, 태양주기, 맑음/흐림, 생육단계 one-hot, DAT, 생육단계별 목표온도
- 파생 변수: VPD, 이슬점, 결로 여유, 생리·결로 안전, 광에너지, 환기·열역학 및 제어한계 feature
- 감사 변수: 결측/이상치 quality flag
- 과거 행동: 직전 천창, 차광스크린, 보온스크린, 난방, 냉방, 순환팬 Action
- rollout에서는 현재 Action으로 다음 State의 과거 행동 feature를 갱신한다.

광·결로·열 관련 파생변수의 공식 계산 계약은 다음과 같다.

- 10분 후 예측 이슬점 여유의 canonical 변수명은 `dTcond_pred_10m`으로 고정한다. `obs_derived_condensation_margin_10m_c`는 사용하지 않는다.
- DLI는 PPFD 센서 입력이 아니라 외부일사 `I_out` [W·m⁻²]를 사용한 추정치다. `DLI_ext(t) = Σ I_out(k) Δt_k κ_solar→PPFD / 10^6` [mol·m⁻²·day⁻¹]이며 기본 변환계수 `κ_solar→PPFD=2.02 µmol·J⁻¹`는 config/artifact에 기록하고 현장 보정 가능해야 한다.
- 실측누적광 `S_meas(t)=Σ I_out(k)Δt_k/10^4` [J·cm⁻²]로 계산한다. 품질관리 완료 데이터의 `out_light_sum`이 유효하면 이를 우선 사용하고, 없을 때 동일 식으로 재계산한다.
- 청천누적광 `S_clear(t)` [J·cm⁻²]은 온실 위도·경도·시간대에 따른 태양 위치와 Haurwitz 청천 GHI를 계산한 뒤 일 단위로 적분한다. 검증된 upstream `cs_sum_jcm2`가 있으면 이를 우선 사용한다.
- `SumRatio(t)=S_meas(t)/S_clear(t)`이며 무차원 비율이다. 목표 누적광을 분모로 사용하지 않는다.
- `LightETA(t)=min{τ≥t | S_clear(τ)≥S_target}-t` [min]으로, 당일 청천 누적곡선을 역탐색해 계산한다. 임의의 기존 timestamp 열을 그대로 복사하지 않는다.
- `vent_loss_proxy(t)=u_vent(t)·max(T_in(t)-T_out(t),0)·[1-clip(I_out(t)/I_ref,0,1)]` [°C-equivalent proxy]로 고정한다. `u_vent`는 [0,1] 유효 천창 개도율, 기본 `I_ref=1000 W·m⁻²`이며 두 값과 식 버전을 artifact에 기록한다. 기술문서에는 환기열손실이 “일사와 온도차를 통합한 근사치”라고만 제시되어 정확한 식·계수는 없으므로, 이 식은 v1.4의 재현 가능한 운영 계약이다.
- Step 11에서는 좌표·시간대, DLI 변환계수, 목표누적광 및 `I_ref`를 crop/시설 config로 고정하고, 누락 또는 도달 불가능한 LightETA를 quality/provenance로 식별한다.

외부환경은 실제 운영에서 관측 또는 시점 정합된 기상예보로 공급한다.

### Action contract

공식 Action은 6개로 고정한다.

Continuous [0, 1]:

1. vent_pct: 좌·우 천창의 유효 개도율 평균
2. shade_curtain_pct: 차광스크린 개도율
3. thermal_curtain_pct: 보온스크린 개도율

Binary {0, 1}:

4. heat_run: FCU 난방 작동 여부
5. cool_run: FCU 냉방 작동 여부
6. fan_run: 순환팬 작동 여부

- Action NaN을 AI로 보간하지 않는다.
- 필수 source Action이 해결되지 않은 row는 RL transition 생성 시 제외하고 사유를 기록한다.
- CO₂ 공급 Action은 현재 공식 Action에 포함하지 않는다. 신뢰 가능한 공급 로그가 확보되면 별도 MDP version에서 추가한다.

### Preprocessing contract

- 연속형 State scaler는 Train split의 유효 transition만으로 z-score fit한다.
- Validation, Test, rollout 및 운영 inference에는 Train의 동일 mean/std를 적용한다.
- binary, one-hot, quality flag, time sine/cosine 및 이미 [0, 1]인 비율 변수는 z-score에서 제외한다.
- scaler column, mean, scale 및 source Train dataset hash를 crop별 dataset-level observation_scaler.json에 저장하고 모든 candidate manifest가 동일 scaler path/hash를 참조한다.
- 생육단계, 태양주기 및 범주형 context는 one-hot encoding한다.
- Validation/Test 통계로 scaler나 feature parameter를 다시 fit하지 않는다.

### Prediction contract

- 입력 X: current canonical State + current 6-dimensional Action
- 공식 출력 y:
  1. target_next_indoor_temp_c
  2. target_next_indoor_humidity_pct
  3. target_next_indoor_co2_ppm
- 세 target은 각각 5분 뒤 실내온도, 실내습도 및 실내 CO₂다.
- 외부 기상, 시간/context, one-hot metadata, quality flag, deterministic derived feature 및 과거 Action은 prediction target에서 제외한다.
- target resolver는 위 세 target만 공식 allowlist로 승인하고 누락·추가 target을 fail-fast한다.
- 각 candidate는 하나의 논리적 artifact이지만 내부적으로 target별 independent estimator를 기본 사용한다.
- 세 estimator는 동일 input schema를 사용하되 target별 hyperparameter를 가질 수 있다.
- joint shared-representation neural multi-output은 후속 ablation으로 분리한다.

### Multi-step rollout contract

- 15분은 3 steps, 30분은 6 steps, 60분은 12 steps다.
- 예측한 실내온도·습도·CO₂를 다음 State의 내생 변수로 재귀 입력한다.
- 외부온도·습도·일사·풍속·강우는 Transition Model이 예측하지 않는다.
- offline Validation/Test에서는 해당 timestamp의 기록된 외부환경을 exogenous provider로 사용한다.
- 실제 운영에서는 현재 관측과 시점 정합된 단기예보를 exogenous provider로 사용한다.
- 시간, 태양주기, 생육단계 및 deterministic context는 다음 timestamp에서 다시 계산한다.
- 외부 일사량은 State 입력이지만 target이 아니다. 실내 PAR 센서가 확보되면 별도 target ablation으로 검토한다.
- report에는 Action provider, exogenous provider 및 observed/forecast mode를 기록한다.

### Reward and downstream RL boundary

Reward는 다음 6개 soft penalty의 가중합에 음수를 취한다.

R = -(w_temp P_temp + w_cond P_cond + w_rh P_rh + w_vpd P_vpd + w_hv P_hv + w_act P_act)

1. 목표 범위를 벗어난 실내온도
2. 낮은 이슬점/결로 여유
3. 과도한 실내 상대습도
4. 낮은 VPD
5. 난방·환기 동시 사용
6. 제어 불안정성

- 각 penalty는 고정 domain threshold로 정규화하고 [0, 1]로 제한한다.
- 기본 가중치는 각각 1/6이며 config와 artifact에 기록한다.
- 실내 CO₂는 State/target에 포함하지만 v1.4 기본 Reward에는 즉시 추가하지 않는다.
- CO₂ penalty는 작물·생육단계별 목표와 CO₂ 공급 Action/log가 검증된 후 별도 ablation 및 MDP version으로 추가한다.
- discount factor는 Transition Model parameter가 아니다.
- downstream RL policy 기본값은 gamma=0.99이며 별도 RL algorithm config에서 관리한다.

### Candidate evaluation metrics

One-step Accuracy:

- R², MAE, RMSE, Q90 absolute error, CVaR90 absolute error, NRMSE
- 온도·습도·CO₂ target별, target group별 및 세 target aggregate metric

Multi-step Rollout:

- 15/30/60분 final-step RMSE/MAE
- trajectory mean RMSE/MAE
- target별 및 aggregate drift slope
- 물리적 범위 위반률
- NaN/Inf 발생 여부

Resource and Efficiency:

- inference latency median/p95
- peak CPU RSS 및 peak GPU allocated memory
- serialized model 및 artifact directory size
- training time 및 HPO total time
- benchmark device와 library version

### HPO objective

각 trainable candidate의 target별 hyperparameter는 Validation Rollout Weighted Score를 최소화한다.

S = 0.10 E_one-step + 0.10 E_15min + 0.25 E_30min + 0.45 E_60min + 0.10 D_drift

- 각 error 항은 온도·습도·CO₂의 target별 normalized error를 동일 가중 평균한다.
- 단위가 큰 CO₂가 objective를 지배하지 않도록 raw error를 직접 합산하지 않는다.
- 60분 rollout에 가장 높은 가중치를 둔다.
- Efficiency는 objective에 포함하지 않는다.
- Persistence는 HPO 없이 동일 Score를 계산한다.
- HPO는 candidate 내부 및 target별 configuration 결정이며 후보 간 최종 선택이 아니다.

### Comparative evaluation policy

- 모든 후보는 동일한 Validation split, schema, Train-only scaler, rollout horizon, Action/exogenous provider 및 metric으로 평가한다.
- Validation Score와 세부 지표 순위를 제공한다.
- Persistence 대비 improvement를 target별 및 aggregate로 제공하되 gate로 사용하지 않는다.
- Efficiency는 Validation Score와 분리한다.
- Framework는 winner, recommended 또는 selected model을 생성하지 않는다.
- 연구자가 성능, drift, resource 및 deployment constraint를 종합해 결정한다.

### Validation/Test policy

- Train: scaler fit, 모델 fit 및 HPO trial 학습
- Validation: HPO objective, candidate comparison 및 ranking
- Test: 8개 candidate best configuration의 별도 일반화 평가
- Test는 scaler fit, HPO, Validation Score, ranking 또는 자동 추천에 사용하지 않는다.
- Test 결과는 별도 파일과 report section에 저장한다.
- 최종 선택은 Validation 및 운영 조건을 기준으로 연구자가 수행한다.
- 모든 report에 test_used_for_selection: false를 기록한다.

## 3. Output Contract

```text
transition_results/
  config.json
  dataset_manifest.json
  observation_scaler.json
  exogenous_provider_config.json
  candidate_metrics.csv
  candidate_metrics.json
  candidate_rollout_metrics.csv
  candidate_resource_metrics.csv
  candidate_test_metrics.csv
  candidate_test_metrics.json
  transition_summary.md
  artifact_integrity.json

  artifacts/
    persistence/
    linear_regression/
    linear_svr/
    knn/
    extra_trees/
    lightgbm/
    xgboost/
    mlp/
```

각 candidate artifact directory에는 다음을 저장한다.

```text
model.pkl
manifest.json
feature_schema.json
hpo_results.json
best_config.json
training_summary.json
validation_one_step_metrics.json
validation_rollout_metrics.json
resource_metrics.json
test_metrics.json
```

Persistence에는 `model.pkl`, manifest 및 평가 결과를 저장하되 `hpo_results.json`과 `best_config.json`은 `not_applicable` 상태를 명시한다.

다음 output은 생성하지 않는다.

- `selection_report.json`
- `selected_transition_model.json`
- `selected_manifest`
- `selected_model/`
- selected model copy
- winner 또는 recommended model field
- 자동 inference handoff artifact

연구자가 최종 후보를 결정한 뒤에는 기존 candidate artifact를 그대로 사용한다. 모델을 별도 selected directory로 복사하지 않으며, downstream inference에는 `artifact root + crop + model_name` 또는 명시적 candidate artifact path를 전달한다.

## 4. Candidate Comparison Report

### CSV/JSON outputs

`candidate_metrics.csv/json`:

- candidate name
- model family
- 온도·습도·CO₂ target별 및 세 target aggregate Validation one-step MAE/RMSE/R²/Q90/CVaR90/NRMSE
- Validation Rollout Weighted Score
- target별 및 aggregate Persistence 대비 absolute/relative improvement
- candidate status
- 각 metric의 rank

`candidate_rollout_metrics.csv`:

- 온도·습도·CO₂ target별 및 aggregate 15/30/60분 final-step RMSE/MAE
- 온도·습도·CO₂ target별 및 aggregate 15/30/60분 trajectory RMSE/MAE
- target별 및 aggregate drift slope
- physical violation rate
- NaN/Inf status
- horizon별 rank와 drift rank

`candidate_resource_metrics.csv`:

- training time
- HPO total time
- inference median/p95
- peak CPU memory
- peak GPU memory
- serialized model size
- artifact directory size
- resource metric별 rank

`candidate_test_metrics.csv/json`:

- 8개 후보의 Test one-step 및 rollout metric
- Test 결과가 HPO와 Validation ranking에 사용되지 않았음을 나타내는 metadata
- Test metric에 대한 별도 descriptive rank
- Validation rank를 변경하거나 재계산하지 않음

### Markdown summary

`transition_summary.md`는 다음 내용을 제공한다.

- experiment config, 3-target schema, observation scaler 및 dataset provenance
- 후보별 Validation 성능 비교
- one-step Accuracy 비교
- 온도·습도·CO₂ target별 및 aggregate 15/30/60분 rollout 비교
- trajectory drift 비교
- resource usage 비교
- Validation Rollout Weighted Score 순위
- 각 개별 metric 순위
- Persistence 대비 improvement
- 별도 Test 결과 section
- invalid 또는 failed candidate 정보
- deployment trade-off를 해석하기 위한 사실 기반 설명

Summary는 winner, recommendation 또는 selected model을 작성하지 않는다. 문서 마지막에는 다음 문구를 그대로 포함한다.

> This framework intentionally does not perform automatic model selection.  
> Final transition-model selection is left to the researcher after considering rollout accuracy, resource usage, deployment constraints, and practical trade-offs.

## 5. Step-by-Step Roadmap

### Phase A — 현재 구현 완료

#### Step 1. Transition Core and Dynamic Target Schema — 완료

- 목적: learned transition 책임을 MDP, reward 및 replay environment와 분리한다.
- 구현 내용:
  - 공통 model, dataset, prediction interface
  - semantic target resolver foundation; 공식 3-target migration은 Step 11.4에서 수행
  - RL parquet dataset builder
  - deterministic, excluded 및 postprocessed column 분리
- 선행 조건: 기존 MDP v1 observation/action schema
- 완료 기준:
  - RL-ready frame에서 공통 input/target schema를 생성할 수 있다.
  - metadata, quality flag, deterministic 및 `obs_prev_*`가 target에서 제외된다.
  - missing dynamic target이 schema에 기록된다.

#### Step 2. Training, Evaluation and Rollout Foundation — 완료

- 목적: 후보 모델을 공통 interface로 학습·평가한다.
- 구현 내용:
  - Linear Regression, Linear SVR, KNN, MLP
  - LightGBM, XGBoost wrapper
  - independent multi-target foundation; target별 HPO와 CO₂ migration은 Step 11.4에서 수행
  - one-step evaluator
  - LoggedActionProvider와 3/6/12-step rollout
  - trajectory drift 계산
- 선행 조건: Step 1
- 완료 기준:
  - synthetic RL dataset에서 fit/predict/rollout이 동작한다.
  - prediction schema와 target schema가 일치한다.
  - metric payload가 JSON-safe 형식으로 생성된다.

#### Step 3. Legacy Experiment and Auto-Selection Skeleton — 교체 대상

- 목적: 현재 구현 상태를 명시하고 v1.4 migration 대상을 고정한다.
- 구현 내용:
  - 기존 YAML/CLI 및 candidate artifact 저장
  - 기존 automatic selection, selected manifest, selected Test evaluation
  - 기존 selected artifact 기반 inference helper
- 선행 조건: Step 1~2
- 완료 기준:
  - 현재 구현은 테스트로 동작함이 확인된다.
  - v1.4에서는 이 Step의 automatic selection 부분만 제거하고 candidate training/evaluation 기능은 재사용한다.

### Phase B — 현재 바로 구현 가능

#### Step 4. Official Candidate Registry 확정 및 제외 모델 제거

- 목적: 공식 후보를 8개로 고정하고 불필요한 구현을 정리한다.
- 구현 내용:
  - Persistence 및 ExtraTrees 추가
  - 공식 registry와 config를 8개 후보로 제한
  - TCN wrapper와 `deep.py` 제거
  - CatBoost wrapper를 boosting module에서 제거
  - TCN·CatBoost registry entry, package export, 테스트 및 문서 제거
  - XGBRF, RandomForest, Probabilistic Ensemble은 신규 구현하지 않음
- 선행 조건:
  - 저장소 내부 import/reference 재검사
  - Transition candidate artifact 및 pickle 재검사
- 완료 기준:
  - registry가 정확히 8개 후보만 제공한다.
  - 제외 모델 이름을 config에 지정하면 명확한 validation error가 발생한다.
  - TCN·CatBoost symbol과 dependency 참조가 남지 않는다.
  - load해야 할 legacy Transition artifact가 발견되면 해당 class path만 deprecated shim으로 격리한다.

#### Step 5. Shared Candidate Benchmark Objective

- 목적: 후보별 HPO와 Validation comparison에 동일한 rollout objective를 적용한다.
- 구현 내용:
  - Validation Rollout Weighted Score 단일 구현
  - persistence 포함 모든 후보에 동일 target normalization, Train-only scaler와 horizon 적용
  - trainable 7개 후보에 random-search HPO 적용
  - 기본 budget 20 trials/model, seed 42
  - 각 trial의 one-step/rollout/drift raw metric 저장
- 선행 조건: Step 4
- 완료 기준:
  - 후보별 best configuration이 공통 Validation objective로 결정된다.
  - HPO best configuration과 comparison score의 산식이 일치한다.
  - Persistence에는 HPO가 적용되지 않는다.
  - Test metric은 HPO objective에 전달되지 않는다.

#### Step 6. Accuracy, Rollout and Resource Benchmark 완성

- 목적: 연구자가 trade-off를 판단할 수 있는 전체 metric을 제공한다.
- 구현 내용:
  - 온도·습도·CO₂ target별 및 aggregate R², MAE, RMSE, Q90, CVaR90 및 NRMSE 보장
  - 15/30/60분 rollout과 drift 평가
  - training/HPO time 측정
  - 독립 subprocess 기반 inference time과 peak memory 측정
  - serialized model과 artifact directory size 측정
  - CPU/GPU 장치와 library version 기록
- 선행 조건: Step 4~5
- 완료 기준:
  - 8개 후보에 동일 metric schema가 적용된다.
  - unsupported metric은 누락하지 않고 status와 사유를 기록한다.
  - resource metric이 Validation Score에 포함되지 않는다.
  - benchmark protocol과 단위가 artifact에 저장된다.

#### Step 7. Automatic Selection 제거

- 목적: Framework의 책임을 candidate evaluation과 comparison으로 제한한다.
- 구현 내용:
  - automatic selector 호출 제거
  - selected model field와 selection result type 제거
  - `selection_report.json` 및 selected manifest writer/loader 제거
  - selected model copy/save 로직 제거
  - selected artifact 기반 inference helper 제거
  - explicit candidate artifact loader를 inference public API로 사용
  - 기존 selector의 score 계산 기능은 candidate benchmark objective와 ranking utility로 분리
- 선행 조건: Step 5~6
- 완료 기준:
  - 실행 결과에 selected, winner 또는 recommended model field가 없다.
  - Framework가 candidate 이름을 자동으로 inference handoff에 전달하지 않는다.
  - 모든 candidate artifact가 동등한 구조로 저장된다.
  - 특정 후보는 명시적 model name 또는 artifact path가 있어야 load할 수 있다.

#### Step 8. Candidate Comparison Outputs

- 목적: Validation과 resource 결과를 연구자가 검토할 수 있는 표준 산출물로 제공한다.
- 구현 내용:
  - candidate metric/rollout/resource CSV와 JSON 생성
  - Validation Rollout Weighted Score 종합 순위 생성
  - 개별 accuracy, rollout, drift 및 resource metric별 순위 생성
  - Persistence 대비 improvement 계산
  - `transition_summary.md` 생성
  - no-auto-selection integrity rule 추가
- 선행 조건: Step 6~7
- 완료 기준:
  - 보고서에 8개 후보가 모두 포함된다.
  - 순위는 제공하지만 selected/recommended/winner 필드는 없다.
  - Validation과 resource 결과가 분리된다.
  - Summary 마지막에 no-auto-selection 문구가 포함된다.
  - integrity checker가 selected artifact나 selected model field를 발견하면 실패한다.

#### Step 9. All-Candidate Test Evaluation

- 목적: 모든 후보의 최종 일반화 성능을 저장하되 candidate ranking과 분리한다.
- 구현 내용:
  - 후보별 best configuration을 Train으로 학습한 artifact에 Test 적용
  - 8개 후보의 Test one-step 및 rollout metric 저장
  - Test 결과를 별도 CSV/JSON 및 Markdown section에 기록
  - Validation rank와 Test descriptive rank를 별도 namespace로 유지
  - `test_used_for_hpo: false`, `test_used_for_validation_ranking: false`, `test_used_for_selection: false` 기록
- 선행 조건: Step 5~8
- 완료 기준:
  - 8개 후보 모두 Test artifact를 가진다.
  - Test 결과를 제거해도 Validation ranking 결과가 동일하다.
  - Test metric이 HPO 또는 Validation score 함수 입력으로 전달되지 않는다.
  - 문서에 최종 선택은 Validation 및 운영 조건을 기준으로 연구자가 수행한다고 명시한다.

#### Step 10. Tests and Documentation Migration

- 목적: 기존 auto-selection 계약을 v1.4 comparison 계약으로 교체한다.
- 구현 내용:
  - selected manifest/Test 테스트 제거
  - 8개 candidate artifact와 comparison output 테스트 추가
  - no selected/winner/recommended field 테스트
  - TCN·CatBoost import 및 registry 제거 테스트
  - HPO objective와 세 target normalized aggregation 일치 테스트
  - Test/scaler leakage 및 exogenous-provider boundary 테스트
  - explicit candidate inference/reward 테스트
  - README 용어를 Candidate Evaluation 또는 Model Comparison으로 변경
- 선행 조건: Step 4~9
- 완료 기준:
  - 전체 자동 테스트 실패 0건
  - synthetic end-to-end 실행이 8개 candidate artifact와 모든 comparison file을 생성한다.
  - selected artifact가 생성되지 않는다.
  - Test가 Validation rank를 변경하지 않는다.
  - 기존 MDP/replay/reward 기능에 regression이 없고 공식 3-target migration은 Step 11.4에서 검증한다.

### Phase C — QC 모델 완료 이후

#### Step 11. Existing QC Dataset Validation and Three-target RL Dataset 확정

- 목적: 기존 QC 완료 dataset을 재사용하고 CO₂를 포함한 공식 3-target Transition benchmark 입력을 확정한다.
- 선행 조건:
  - QC HPO·threshold calibration과 연구자의 작물별 결정 완료
  - offline_dataset_preparation/datasets/03_quality_controlled에 세 작물 Train/Validation과 manifest 존재
- 공식 QC 모델: strawberry=modern_tcn, melon=modern_tcn, cucumber=patch_tst
- 공통 정책:
  - 기존 Train/Validation은 검증 통과 시 재생성하거나 변경하지 않는다.
  - 오류는 자동 보정하지 않고 fail-fast하여 별도 QC 보정 작업으로 분리한다.
  - 누락된 Test만 같은 QC artifact와 전처리 계약으로 생성한다.
  - Test는 QC 선택, threshold calibration, scaler fit, HPO 및 Validation ranking에 사용하지 않는다.

##### Step 11.1. Existing QC Dataset and Selection Contract Freeze

- 기존 Train/Validation, QC manifest, 작물별 application JSON과 pickle을 입력으로 사용한다.
- manifest의 작물별 모델 mapping과 crop application을 상호 검증한다.
- model name, pickle, 8개 QC observation, calibrated threshold 및 실제 row count를 검증한다.
- 검증된 dataset/artifact 경로와 hash를 freeze manifest에 기록한다.
- 기존 Train/Validation은 수정하지 않는다.
- 불일치는 다음 단계 전에 fail-fast한다.

##### Step 11.2. Existing Train/Validation Schema and Provenance Validation

- 필수 source column, dtype, temporal ordering 및 episode boundary를 검사한다.
- in_co2_representative와 온도·습도 representative, 외부 일사량의 finite coverage를 기록한다.
- growth-stage, quality flag, imputation 및 Action restoration 계약을 검사한다.
- 실제 parquet와 manifest의 row/statistics를 대조한다.
- 직접 복원할 수 없는 기존 provenance는 legacy_unavailable로 기록하고 추정하지 않는다.
- 성공한 기존 Train/Validation을 immutable input으로 표시한다.

##### Step 11.3. Missing Test-only QC Completion

- 02_split의 Test에 작물별 확정 QC artifact를 적용한다.
- Test에 Train/Validation과 동일한 schema, unit, rule/AI QC 및 representative sensor 정책을 적용한다.
- 실내온도·습도·CO₂ representative value를 생성한다.
- crop-cycle manifest와 현재 row timestamp만으로 growth-stage를 생성한다.
- Test target, 미래 관측, metric 또는 prediction을 참조하지 않는 leakage 검사를 수행한다.
- 기존 Train/Validation은 읽기 전용으로 유지한다.
- 세 작물 Test가 모두 non-empty이고 공통 source schema를 만족해야 완료한다.

##### Step 11.4. Canonical MDP State, Action and Three-target Migration

- canonical State에 obs_indoor_co2_ppm을 추가하고 in_co2_representative를 source로 고정한다.
- continuous z-score 대상에 실내 CO₂를 추가한다.
- 공식 target을 next 실내온도, next 실내습도, next 실내 CO₂로 고정한다.
- 외부 일사와 기상/context는 State/provider 입력으로 유지하고 target에서 제외한다.
- 5분 step, 최대 하루 episode와 추가 boundary, 6개 Action의 이름·순서·범위를 검증한다.
- dataset builder, target resolver, evaluator, rollout, artifact schema 및 inference를 3-target으로 migration한다.
- target별 independent estimator와 target별 HPO configuration을 지원한다.
- recorded-weather offline provider와 forecast-based operating provider interface를 명시한다.
- 기존 2-target 테스트를 3-target으로 migration하고 필요한 2-target compatibility는 non-official mode로 격리한다.
- reward/replay 기능에 regression이 없어야 한다.

##### Step 11.5. RL-ready Dataset, Scaling and Row-exclusion Provenance

- 검증된 기존 Train/Validation과 신규 Test만 입력으로 사용한다.
- canonical State, 6개 Action, 세 next target, Reward 및 rollout identifier를 생성한다.
- Train 유효 transition만으로 scaler를 fit하고 Validation/Test에 동일하게 적용한다.
- scaler artifact에 column, mean, scale과 raw Train hash를 저장한다.
- Action NaN, 시간 단절, episode/series boundary 및 필수 State/target 결측의 제외 사유를 분리한다.
- crop/split별 입력·MDP·출력 row와 exclusion reason별 count를 기록한다.
- 세 작물의 Train/Validation/Test RL dataset이 non-empty이고 동일 3-target schema를 만족해야 완료한다.
- 산출물은 5_rl_dataset의 crop별 parquet, scaler artifact 및 rl_dataset_manifest.json이다.

##### Step 11.6. Hash Chain, Integrity Gate and Step 12 Handoff

- application JSON, pickle, threshold, State/Action/target schema, scaler 및 config SHA-256을 기록한다.
- 기존 Train/Validation은 reused_existing, 신규 Test는 generated_test_only로 구분한다.
- 02_split/Test, 03_quality_controlled 및 5_rl_dataset의 crop/split별 hash를 기록한다.
- freeze/QC manifest를 RL manifest parent provenance로 연결한다.
- code version, run identity, timestamp 및 전처리 설정을 기록한다.
- 실제 hash/schema/scaler/row count와 manifest를 대조한다.
- 세 작물 all-crop integrity gate를 통과해야 immutable RL manifest path/hash를 Step 12에 전달한다.

- Step 11 통합 완료 기준:
  - 기존 Train/Validation은 재생성 없이 사용되고 누락 Test만 생성된다.
  - State에 실내 CO₂와 외부 일사/광환경 입력이 포함된다.
  - output은 5분 뒤 실내온도·습도·CO₂ 세 target이다.
  - 5분 step, episode, 6개 Action, Train-only scaler 및 exogenous provider 계약을 만족한다.
  - 세 작물 RL Train/Validation/Test와 완전한 provenance가 존재한다.
  - Step 11.1~11.6이 순서대로 통과한다.

#### Step 12. Real-data Candidate Benchmark

- 목적: 실제 QC 완료 데이터에서 작물별 비교 결과를 생성한다.
- 구현 내용:
  - crop별 독립 config와 output directory 사용
  - 각 crop config에 Step 11의 immutable RL dataset manifest path/hash 기록
  - 작물별 8개 후보의 온도·습도·CO₂ target별 HPO
  - 세 target의 Validation accuracy, rollout, drift 및 resource benchmark
  - Validation 종합·개별 지표 순위 생성
  - 8개 후보 Test 평가와 별도 결과 저장
  - Persistence 대비 improvement 분석
- 선행 조건: Step 10~11
- 완료 기준:
  - 모든 작물에서 8개 후보 benchmark가 완료된다.
  - strawberry, melon, cucumber의 config, summary 및 비교 결과가 서로 다른 output root에 저장되고 상호 덮어쓰지 않는다.
  - 후보별 artifact와 온도·습도·CO₂ target별 및 aggregate Validation/Test/resource metric이 존재한다.
  - selected model 또는 automatic recommendation이 생성되지 않는다.
  - Test metric이 Validation ranking에 사용되지 않는다.
  - 연구자가 비교에 필요한 모든 raw metric과 rank를 확인할 수 있다.

### Phase D — 최종 완성 단계

#### Step 13. Reproducibility and Integrity Review

- 목적: 후보 비교 결과가 동일한 조건에서 재현되도록 보장한다.
- 구현 내용:
  - crop별 config, seed, dependency, device, 3-target schema, observation scaler, exogenous provider 및 Step 11 dataset manifest/hash 고정
  - candidate artifact와 참조 observation scaler/exogenous provider hash 검증
  - no-auto-selection integrity 검사
  - report와 raw artifact 값 대조
  - explicit candidate cold-load inference/reward smoke test
- 선행 조건: Step 12
- 완료 기준:
  - 깨끗한 환경에서 동일 benchmark를 재실행할 수 있다.
  - report와 artifact metric이 일치한다.
  - selected/winner/recommended field가 없다.
  - 모든 후보 artifact를 동일 scaler와 provider 계약으로 독립적으로 load할 수 있다.

#### Step 14. Researcher Decision and Deployment Handoff

- 목적: Framework 외부의 연구자 판단을 운영 모델 지정으로 연결한다.
- 구현 내용:
  - 연구자가 comparison report를 검토해 후보를 결정
  - 선택 이유, Validation metric, resource trade-off 및 deployment constraint를 별도 연구 기록으로 작성
  - downstream에는 candidate model name 또는 artifact path를 명시적으로 전달
  - Framework는 결정 파일이나 selected artifact를 자동 생성하지 않음
- 선행 조건: Step 12~13
- 완료 기준:
  - 연구자가 최종 후보와 근거를 명시적으로 기록한다.
  - downstream inference가 명시적 candidate artifact로 동작한다.
  - 선택 후보를 별도 directory로 복제하지 않는다.
  - 최종 결정이 Framework-generated recommendation으로 오해되지 않는다.

## 6. QC Dependency Boundary

QC 완료 전 즉시 구현 가능한 항목:

- 공식 8개 후보 registry
- Persistence 및 ExtraTrees
- TCN·CatBoost 제거
- 공통 HPO objective
- accuracy/rollout/resource benchmark
- automatic selection 제거
- candidate comparison report
- all-candidate Test evaluation
- integrity checker
- explicit candidate inference API
- synthetic 및 rule-only 기반 테스트

QC 완료 후에만 확정 가능한 항목:

- 연구자가 지정한 작물별 QC selection contract
- Train/Validation/Test 공통 QC 및 growth-stage schema
- 실내 CO₂가 포함된 canonical State와 3-target MDP schema
- 최종 QC 결과가 반영된 공식 RL dataset
- QC artifact부터 RL dataset까지 이어지는 hash/provenance chain
- 작물별 실제 HPO와 candidate benchmark
- 실제 GPU/CPU resource 결과
- 작물별 Validation/Test 비교표
- 연구자의 최종 모델 결정
- 운영 환경에 전달할 candidate artifact path

QC 모델은 Transition framework 구현의 선행 조건이 아니라 공식 실제 데이터 benchmark의 선행 조건이다.

## 7. Terminology Policy

다음 표현을 사용한다.

- Candidate Evaluation
- Candidate Benchmark
- Model Comparison
- Comparative Evaluation
- Validation Ranking
- Researcher Decision

다음 표현은 자동 결정을 의미할 수 있으므로 Framework output과 API에서 사용하지 않는다.

- Automatic Selection
- Selected Model
- Winner
- Recommended Model
- Best Transition Model
- Selection Report

`best_config`는 후보 내부 HPO 결과에 한해서만 사용하며 후보 간 최종 모델 선택을 의미하지 않는다.

## 8. Design Philosophy

- The framework provides objective benchmarking of all transition-model candidates.
- Final model selection is intentionally left to the researcher.
- This philosophy is identical to the GEAS QC model comparison framework.
- The framework is designed to support decision making rather than replace it.

## 9. Assumptions and Defaults

- 공식 timestep은 5분이다.
- episode는 최대 하루이며 crop/series/segment/episode 변경, 시간 단절 또는 invalid transition에서 추가 분리한다.
- canonical State와 6개 Action은 Section 2의 versioned schema를 따른다.
- 실내 CO₂ source는 in_co2_representative다.
- 공식 target은 5분 뒤 실내온도, 실내습도 및 실내 CO₂다.
- 외부 일사와 기상은 State/exogenous provider 입력이며 learned target이 아니다.
- candidate 내부 target별 independent estimator가 기본이다.
- continuous State scaler는 Train에서만 fit하고 Validation/Test/운영에서 재사용한다.
- 기존 Reward와 replay environment는 유지하되 State/target을 3-target으로 migration한다.
- 기본 Reward는 6개 bounded soft penalty와 동일 가중치 1/6을 사용한다.
- CO₂ Reward penalty는 별도 검증과 MDP version 없이 추가하지 않는다.
- downstream RL policy 기본 discount factor는 gamma=0.99이며 Transition HPO parameter가 아니다.
- Framework는 HPO와 ranking을 수행하지만 최종 후보를 자동 결정하지 않는다.
- Efficiency는 Validation Score와 분리한다.
- Test는 scaler fit, HPO, Validation ranking 또는 선택에 사용하지 않는다.
- Persistence는 baseline이며 gate가 아니다.
- 공식 benchmark는 8개 후보가 모두 실행되어야 유효하다.
- device와 library version을 후보별로 기록한다.
- TCN, CatBoost, XGBRF, RandomForest, Probabilistic Ensemble 및 automatic selection은 지원하지 않는다.
