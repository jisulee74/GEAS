# GEAS AI Quality Model 구현 계획 v1.4

## Summary

GEAS AI Quality Model은 **현재 시점 observation의 feature-level 품질을 판단하고, 조건을 만족하는 비정상 observation을 reconstruction으로 복원하는 quality layer**이다.

v1.4에서는 기존 v1.3 설계를 유지하고, **Hyperparameter Optimization Plan**만 추가한다.

구현 우선순위는 유지한다.

1. **Phase 1: ModernTCN**
2. **Phase 2: TimesNet**
3. **Phase 3: PatchTST**
4. **Final Model Selection**

핵심 정책도 유지한다.

- Observation만 AI Quality Model 입력/복원/이상치 탐지 대상
- Action은 AI/Median/Rule outlier/imputation 대상에서 제외
- 모든 AI 모델은 feature-level masked reconstruction task 수행
- Threshold는 hyperparameter가 아니라 calibration 단계에서 별도 결정
- Offline/Online은 동일한 절차와 `BaseQualityModel` interface를 공유
- Offline/Online이 반드시 동일한 model architecture를 써야 하는 것은 아님
- `forecast()`는 AI Quality Model 범위에서 사용하지 않음

## Reconstruction Task

모든 후보 AI Quality Model은 동일한 **feature-level masked reconstruction task**를 수행한다.

- 입력은 과거 lookback window + current timestep observation이다.
- sliding window는 temporal context 제공 목적이다.
- 복원 대상은 window 내부 마지막 시점, 즉 current timestep의 feature-level masked observation이다.
- timestamp 전체 masking은 사용하지 않는다.
- forecasting target은 사용하지 않는다.
- loss는 current timestep의 masked valid observation cell에만 적용한다.
- missing/rule-invalid/기존 invalid cell은 supervised target에서 제외한다.
- Action column은 입력, target, loss, imputation 어디에도 포함하지 않는다.

`reconstruct(df)`는 내부적으로 sliding-window current reconstruction을 수행하되, 외부 반환은 기존 구조와 맞춰 row-aligned DataFrame으로 제공한다.

## Confidence Definition

Confidence는 **AI Quality Model이 특정 observation cell을 비정상으로 판정하고 reconstruction으로 대체해도 된다고 보는 신뢰도**이다.

각 observation cell에 대해:

```text
score = abs(observed - reconstructed)
threshold = selected threshold
margin = score - threshold
confidence = sigmoid(margin / scale)
scale = max(validation_score_iqr, epsilon)
epsilon = 1e-6
```

- `validation_score_iqr`는 validation set 정상 후보 cell anomaly score IQR이다.
- column별 IQR을 우선 사용한다.
- column별 IQR이 0 또는 NaN이면 전체 observation score IQR을 fallback으로 사용한다.
- 그래도 사용할 수 없으면 `scale=1.0`을 사용한다.
- `score == threshold`이면 confidence는 약 `0.5`
- `score > threshold`이면 confidence가 커진다.
- `score < threshold`이면 confidence가 작아진다.

Pipeline output column:

```text
{observation}_ai_confidence
```

aggregate confidence는 만들지 않는다.

`predict_outlier()`는 score, threshold, confidence, outlier flag를 함께 계산한다.

`impute()`는 다음 정책을 따른다.

```text
impute_allowed = invalid_mask == True
                 and prediction is not NaN
                 and (
                     invalid reason is missing/rule_outlier
                     or ai_confidence >= imputation_confidence_threshold
                 )
```

기본값:

```text
imputation_confidence_threshold = 0.5
```

## Hyperparameter Optimization Plan

Hyperparameter Optimization은 **Threshold Calibration과 분리**한다.

- Hyperparameter는 모델 학습 구조와 training strategy를 결정한다.
- Threshold는 학습된 모델의 anomaly score를 outlier flag로 변환하기 위한 calibration 값이다.
- 따라서 threshold는 hyperparameter search 대상이 아니며, best hyperparameter configuration이 선택된 뒤 validation set에서 별도로 calibration한다.

### Common Hyperparameters

모든 AI Quality Model이 공통으로 갖는 search 대상:

- lookback window length
- masking ratio
- batch size
- optimizer type
- learning rate
- weight decay
- loss function variant
- training epoch upper bound
- early stopping patience
- random seed
- normalization/scaling strategy for observation input

### Model-specific Hyperparameters

- **ModernTCN**
  - convolution block depth
  - channel width
  - kernel/receptive-field configuration
  - dropout

- **TimesNet**
  - number of temporal blocks
  - top-k periods
  - period embedding/projection size
  - dropout

- **PatchTST**
  - patch length
  - patch stride
  - Transformer depth
  - attention head count
  - embedding dimension
  - dropout

구체적인 후보 값은 구현 단계의 config 파일에서 정의한다. v1.4 계획에서는 search 대상과 절차만 확정한다.

### Tuning Procedure

각 모델은 validation set으로 자신의 best hyperparameter configuration을 선택한다.

절차:

1. Train split으로 candidate configuration을 학습한다.
2. Validation split에서 reconstruction performance를 평가한다.
3. Validation split에서 synthetic anomaly detection score도 평가한다.
4. Early stopping은 train 중 validation reconstruction loss 기준으로 적용한다.
5. 각 모델별 best configuration을 선택한다.
6. 선택된 best configuration으로 해당 모델을 확정한다.
7. 확정된 model configuration에 대해 Threshold Calibration을 별도로 수행한다.
8. 모델별 best configuration + calibrated threshold끼리 최종 비교한다.

### Hyperparameter Objective

Hyperparameter selection은 threshold-dependent metric만으로 하지 않는다.

기본 objective:

- primary: validation synthetic masking RMSE
- secondary: validation synthetic masking MAE
- tertiary: validation anomaly score separability 또는 synthetic anomaly detection PR-AUC

F1, Precision, Recall은 threshold calibration 이후 산출되는 metric이므로 hyperparameter primary objective로 사용하지 않는다.

### Threshold Calibration

Threshold Calibration은 best hyperparameter configuration이 선택된 뒤 수행한다.

- 입력: fixed trained model + validation anomaly scores
- 출력: calibrated threshold
- 도구: 기존 `ThresholdOptimizer`
- objective: validation anomaly detection objective
- threshold 결과는 model artifact에 저장한다.
- threshold는 hyperparameter optimization 결과와 별도 field로 기록한다.

### Early Stopping

Early Stopping은 모든 AI 모델에 공통으로 적용하는 Training Strategy이다.

- 기준: validation reconstruction loss
- 적용 시점: 각 hyperparameter candidate training 중
- best checkpoint: validation reconstruction loss가 가장 낮은 checkpoint
- patience와 최소 개선폭은 common training config로 관리
- early stopping은 model selection metric이 아니라 학습 안정화 전략이다.

## Training-Inference Consistency

GEAS는 세 운영 단계에서 동일한 AI Quality Pipeline 절차를 사용한다.

- **Offline Dataset Preparation**
  - Raw dataset
  - Missing detection
  - Rule-based outlier detection
  - AI Quality Model outlier/confidence 계산
  - 조건을 만족하는 observation만 reconstruction
  - 정제 dataset으로 RL training/validation 수행

- **Online Inference**
  - Raw observation
  - 동일한 missing/rule/AI quality 절차 적용
  - 조건을 만족하는 observation만 reconstruction
  - 정제 observation을 RL policy에 입력

- **Offline Batch Refinement**
  - 새 데이터에 동일한 quality pipeline 절차 적용
  - AI Quality Model 재학습
  - RL policy 재학습

“동일한 pipeline”은 동일한 데이터 계약, flag semantics, quality decision 절차, `BaseQualityModel` interface를 의미한다. Offline과 Online의 실제 모델 구현체는 latency, memory, model size, 연산 비용에 따라 달라질 수 있다.

## Model Interface Plan

`BaseQualityModel` method signature는 유지한다.

- `fit(train_df, observation_columns, valid_mask=...)`
  - feature-level masked reconstruction task 기준으로 학습
  - best hyperparameter configuration이 주입된 model instance를 학습
  - Observation-only validation 적용
  - Action column 입력 시 실패

- `reconstruct(df, observation_columns)`
  - current timestep 기준 row-aligned reconstructed observation DataFrame 반환

- `anomaly_score(df, prediction_df)`
  - current timestep observation residual 기반 score 반환

- `predict_outlier(df, thresholds)`
  - score, threshold, confidence 계산
  - `QualityModelOutput` 반환
  - `confidence_scores` optional field 포함

- `impute(df, invalid_mask, prediction_df)`
  - invalid observation cell만 reconstruction 값으로 복원
  - AI-only outlier는 confidence gate 적용
  - Action은 절대 변경하지 않음

`forecast()`는 AI Quality Model 구현 대상에서 제외한다.

## Candidate Models

- **ModernTCN**
  - 1차 구현 대상
  - convolution 기반 reconstruction
  - online deployment 후보로 가장 유리

- **TimesNet**
  - 2차 구현 대상
  - 온실의 주기성 패턴 비교에 적합

- **PatchTST**
  - 3차 구현 대상
  - patch-based reconstruction 비교군

- **Future Work: iTransformer**
  - 현재 구현 대상 제외
  - 향후 다변량 관계 modeling이 중요해질 때 optional candidate로 검토

## Final Model Selection Criteria

최종 모델은 **모델별 best hyperparameter configuration + calibrated threshold**를 기준으로 비교한다.

단일 metric만으로 선택하지 않고 다음 항목을 종합 비교한다.

- **Reconstruction Performance**
  - MAE
  - RMSE
  - per-column error

- **Anomaly Detection Performance**
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC
  - confusion matrix

- **Online Inference Efficiency**
  - inference latency
  - memory usage
  - model size
  - online window 기준 처리 가능성

Validation은 hyperparameter selection과 threshold calibration에 사용한다. Test는 최종 선택된 model/config/threshold 평가에만 사용한다.

## Implementation Steps

### Step 1: Deep Quality Common Utilities

**Objective**
- 모든 AI 모델이 공유할 reconstruction dataset, mask, confidence 계산 기반을 만든다.

**Tasks**
- sliding-window dataset builder 설계
- feature-level current timestep masking 구현
- valid cell loss mask 생성
- confidence 계산 utility 구현
- torch lazy import helper 추가

**Deliverables**
- 공통 dataset/masking/confidence module
- unit tests
- 기존 `rule_only`, `median` regression 유지

**Tests**
- timestamp 전체 masking이 발생하지 않는지
- current timestep feature-level masking만 수행하는지
- Action column이 dataset에 포함되지 않는지
- confidence 수식이 score/threshold/scale에 맞게 계산되는지

### Step 2: QualityModelOutput Confidence Extension

**Objective**
- 기존 interface를 깨지 않고 confidence를 pipeline에 전달한다.

**Tasks**
- `QualityModelOutput.confidence_scores: pd.DataFrame | None = None` 추가
- 기존 `RuleOnlyQualityModel`, `MedianQualityModel` 반환값 backward-compatible하게 보강
- `{observation}_ai_confidence` column 생성 로직 추가

**Deliverables**
- confidence 포함 output container
- confidence column 생성
- legacy test 통과

**Tests**
- 기존 constructor 사용이 깨지지 않는지
- confidence가 없을 때도 기존 모델이 동작하는지
- confidence column shape/index/columns 검증

### Step 3: Confidence-Aware Imputation Policy

**Objective**
- AI-only outlier 복원 여부를 confidence로 gate한다.

**Tasks**
- invalid reason 구분 로직 정리
- missing/rule outlier는 기존처럼 reconstruction 가능 시 복원
- AI-only outlier는 `ai_confidence >= imputation_confidence_threshold`일 때만 복원
- artifact에 confidence threshold 저장

**Deliverables**
- confidence-aware imputation policy
- pipeline artifact field

**Tests**
- low-confidence AI outlier는 원본 유지
- high-confidence AI outlier는 reconstruction으로 복원
- missing/rule outlier 복원 정책 유지
- Action NaN 유지

### Step 4: Hyperparameter Optimization Framework

**Objective**
- 모델별 hyperparameter search와 threshold calibration을 분리한다.

**Tasks**
- common/model-specific hyperparameter config schema 설계
- candidate configuration runner 구현
- early stopping 공통 training strategy 연결
- validation reconstruction metric 기반 best config 선택
- best config artifact 저장

**Deliverables**
- hyperparameter optimization module
- best configuration artifact
- training history artifact

**Tests**
- threshold가 hyperparameter search에 포함되지 않는지 확인
- 모델별 best config가 validation metric으로 선택되는지 확인
- early stopping이 candidate training 중 적용되는지 확인
- selected config artifact schema 검증

### Step 5: ModernTCNQualityModel

**Objective**
- 첫 production candidate AI Quality Model 구현.

**Tasks**
- `ModernTCNQualityModel` class 구현
- feature-level masked reconstruction task 수행
- hyperparameter optimization framework 연결
- best config 선택 후 threshold calibration 수행
- `fit`, `reconstruct`, `anomaly_score`, `predict_outlier`, `impute` 연결
- `run_preprocessing_pipeline` candidate factory 등록

**Deliverables**
- ModernTCN model
- unit/integration tests
- synthetic evaluation report 생성 가능

**Tests**
- small synthetic series 학습/복원
- output shape 검증
- Action column reject
- confidence/outlier/imputation flow 검증
- HPO 후 threshold calibration 분리 검증
- existing full regression

### Step 6: TimesNetQualityModel

**Objective**
- 주기성 기반 reconstruction 후보 구현.

**Tasks**
- `TimesNetQualityModel` class 구현
- 동일 reconstruction task 적용
- hyperparameter optimization framework 연결
- 동일 confidence/imputation/evaluation interface 사용
- candidate factory 등록

**Deliverables**
- TimesNet model
- ModernTCN과 동일 test contract

**Tests**
- 동일 synthetic dataset으로 fit/reconstruct
- best config selection 검증
- confidence/outlier/imputation flow 검증
- full regression

### Step 7: PatchTSTQualityModel

**Objective**
- patch-based Transformer reconstruction 후보 구현.

**Tasks**
- `PatchTSTQualityModel` class 구현
- 동일 feature-level masked reconstruction task 적용
- hyperparameter optimization framework 연결
- 동일 evaluation/factory/artifact interface 사용

**Deliverables**
- PatchTST model
- comparative evaluation 가능 상태

**Tests**
- fit/reconstruct shape 검증
- patch reconstruction이 current timestep target을 반환하는지
- best config selection 검증
- full regression

### Step 8: Evaluation and Selection Report

**Objective**
- 모델별 best configuration과 calibrated threshold를 기준으로 최종 선택 report를 만든다.

**Tasks**
- reconstruction metrics 집계
- anomaly detection metrics 집계
- latency/memory/model size 측정
- hyperparameter selection 결과 기록
- threshold calibration 결과 기록
- validation selection과 test evaluation 분리 report 작성

**Deliverables**
- model comparison report schema
- artifact JSON
- selected model/config/threshold summary

**Tests**
- report에 best hyperparameter configuration 포함
- calibrated threshold 포함
- MAE/RMSE, Precision/Recall/F1/ROC-AUC/PR-AUC 포함
- latency/model size field 포함
- test 결과가 selection에 사용되지 않는지 검증

## Test Plan

- Unit tests
  - Action column reject
  - feature-level masking
  - confidence calculation
  - confidence-aware imputation
  - hyperparameter config validation
  - early stopping behavior
  - shape/index preservation
  - torch missing graceful failure

- Integration tests
  - 모델별 HPO → best config 선택 → threshold calibration
  - `run_preprocessing_pipeline` candidate selection
  - `3_missing_outliers_handled` output columns
  - Online-style single/current timestep reconstruction
  - `5_rl_dataset` Action NaN transition exclusion unchanged

- Regression tests
  - 기존 unittest 전체 통과
  - `rule_only`, `median` 결과 동일성 확인
  - BaseQualityModel method signature 변경 없음 확인

## Assumptions and Pre-Implementation Decisions

- PyTorch는 lazy import로 시작한다.
- Threshold는 hyperparameter가 아니라 calibration artifact로 관리한다.
- Hyperparameter 후보 값 자체는 implementation config에서 정의하되, common/model-specific 구분은 이 문서대로 고정한다.
- Early Stopping은 모든 AI 모델 공통 training strategy이다.
- Hyperparameter selection은 validation reconstruction 중심 metric으로 수행한다.
- Threshold calibration은 best hyperparameter configuration 선택 이후 validation anomaly objective로 수행한다.
- aggregate confidence column은 만들지 않는다.
- Online/Offline은 같은 interface와 절차를 공유하지만, 같은 architecture 사용을 강제하지 않는다.
- RL dataset stage와 MDP v1 로직은 AI Quality Model 구현 범위 밖이다.
