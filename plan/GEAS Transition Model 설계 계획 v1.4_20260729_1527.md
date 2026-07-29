# GEAS Transition Model 설계 계획 v1.4

## 1. Summary

GEAS Transition Model v1.4의 목적은 스마트팜 상태 전이를 근사하는 후보 모델을 동일한 조건에서 학습·평가하고, 연구자가 최종 모델을 판단할 수 있는 객관적인 비교 결과를 제공하는 것이다.

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

### Prediction contract

- 입력 \(X\): current observation + current action
- 출력 \(y\): semantic allowlist로 선택한 next dynamic observation
- deterministic time/context, one-hot metadata 및 quality flag는 target에서 제외한다.
- `obs_prev_*`는 현재 action으로 postprocess한다.
- reward는 예측 next observation을 row-like state로 복원한 뒤 기존 `compute_mdp_v1_reward()`로 계산한다.

### Candidate evaluation metrics

One-step Accuracy:

- R²
- MAE
- RMSE
- Q90 absolute error
- CVaR90 absolute error
- NRMSE
- target별, target group별 및 aggregate metric

Multi-step Rollout:

- 15분: 3 steps
- 30분: 6 steps
- 60분: 12 steps
- horizon별 final-step RMSE/MAE
- trajectory mean RMSE/MAE
- trajectory drift slope
- 물리적 범위 위반률
- NaN/Inf 발생 여부

Resource and Efficiency:

- inference latency median/p95
- peak CPU RSS
- peak GPU allocated memory
- serialized model size
- candidate artifact directory size
- training time
- HPO total time
- benchmark device와 library version

### HPO objective

각 trainable candidate의 hyperparameter는 Validation Rollout Weighted Score를 최소화하도록 자동 결정한다.

\[
S =
0.10E_{one-step}
+0.10E_{15min}
+0.25E_{30min}
+0.45E_{60min}
+0.10D_{drift}
\]

- \(E_{one-step}\): Validation one-step aggregate NRMSE
- \(E_{15min}\), \(E_{30min}\), \(E_{60min}\): horizon별 Validation rollout aggregate NRMSE
- \(D_{drift}\): 정규화된 Validation trajectory drift penalty
- 낮은 점수가 우수하다.
- 장기 rollout이 Transition Model의 핵심 목적이므로 60분 rollout에 가장 높은 가중치를 부여한다.
- Efficiency metric은 HPO objective에 포함하지 않는다.
- Persistence는 학습과 HPO를 수행하지 않지만 동일한 Validation Score를 계산한다.

HPO는 후보 내부의 best configuration을 정하는 절차일 뿐, 후보 간 최종 모델 선택을 의미하지 않는다.

### Comparative evaluation policy

- 모든 후보는 동일한 Validation split, schema, rollout horizon 및 metric으로 평가한다.
- Validation Rollout Weighted Score와 각 세부 지표의 순위를 제공한다.
- Persistence도 다른 후보와 동일하게 ranking에 포함한다.
- Persistence 대비 absolute/relative improvement를 보고하되 이를 탈락 gate로 사용하지 않는다.
- Efficiency는 별도 비교 지표로 제공하며 Validation Score에 포함하지 않는다.
- Framework는 efficiency tie-break, winner, recommended model 또는 selected model을 생성하지 않는다.
- 연구자가 Accuracy, rollout, drift, resource usage 및 deployment complexity를 종합하여 최종 모델을 결정한다.

### Validation/Test policy

- Train: 모델 fitting과 HPO trial 학습에 사용한다.
- Validation: HPO objective 계산, candidate comparison 및 모든 ranking에 사용한다.
- Test: 8개 후보의 best configuration에 대해 최종 일반화 성능을 별도로 산출한다.
- Test metric은 Validation Score, HPO, candidate ranking 또는 자동 추천에 사용하지 않는다.
- Test 결과는 별도 파일과 별도 report section에 저장한다.
- 연구자는 최종 모델 결정 시 Validation 비교 결과를 기준으로 하고 Test 결과는 선정 근거로 사용하지 않는 절차를 따른다.
- Framework는 `test_used_for_selection: false` 정책을 모든 report에 기록하지만, 최종 준수 책임은 연구자에게 있다.

## 3. Output Contract

```text
transition_results/
  config.json
  dataset_manifest.json
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
- Validation one-step MAE/RMSE/R²/Q90/CVaR90/NRMSE
- Validation Rollout Weighted Score
- Persistence 대비 absolute/relative improvement
- candidate status
- 각 metric의 rank

`candidate_rollout_metrics.csv`:

- 15/30/60분 final-step RMSE/MAE
- 15/30/60분 trajectory RMSE/MAE
- drift slope
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

- experiment config와 dataset provenance
- 후보별 Validation 성능 비교
- one-step Accuracy 비교
- 15/30/60분 rollout 비교
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
  - semantic allowlist/denylist 기반 target resolver
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
  - independent multi-target 학습
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
  - persistence 포함 모든 후보에 동일 normalization과 horizon 적용
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
  - R², MAE, RMSE, Q90, CVaR90 및 NRMSE 보장
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
  - HPO objective 일치 테스트
  - Test leakage 방지 테스트
  - explicit candidate inference/reward 테스트
  - README 용어를 Candidate Evaluation 또는 Model Comparison으로 변경
- 선행 조건: Step 4~9
- 완료 기준:
  - 전체 자동 테스트 실패 0건
  - synthetic end-to-end 실행이 8개 candidate artifact와 모든 comparison file을 생성한다.
  - selected artifact가 생성되지 않는다.
  - Test가 Validation rank를 변경하지 않는다.
  - 기존 MDP/replay/reward 기능에 regression이 없다.

### Phase C — QC 모델 완료 이후

#### Step 11. Final QC Artifact and RL Dataset 확정

- 목적: 공식 Transition candidate benchmark에 사용할 신뢰 가능한 dataset을 생성한다.
- 구현 내용:
  - 작물별 QC 모델과 threshold 확정
  - QC application artifact로 quality-controlled split 생성
  - 동일 결과에서 RL-ready Train/Validation/Test 생성
  - QC model hash, threshold, schema 및 row exclusion provenance 기록
- 선행 조건:
  - QC HPO와 threshold calibration 완료
  - 연구자의 QC 모델 결정 완료
- 완료 기준:
  - cucumber, strawberry, melon의 공식 RL dataset이 생성된다.
  - Transition schema와 action/next target 계약을 만족한다.
  - QC artifact부터 RL dataset까지 provenance를 추적할 수 있다.

#### Step 12. Real-data Candidate Benchmark

- 목적: 실제 QC 완료 데이터에서 작물별 비교 결과를 생성한다.
- 구현 내용:
  - 작물별 8개 후보 HPO
  - Validation accuracy, rollout, drift 및 resource benchmark
  - Validation 종합·개별 지표 순위 생성
  - 8개 후보 Test 평가와 별도 결과 저장
  - Persistence 대비 improvement 분석
- 선행 조건: Step 10~11
- 완료 기준:
  - 모든 작물에서 8개 후보 benchmark가 완료된다.
  - 후보별 artifact와 Validation/Test/resource metric이 존재한다.
  - selected model 또는 automatic recommendation이 생성되지 않는다.
  - Test metric이 Validation ranking에 사용되지 않는다.
  - 연구자가 비교에 필요한 모든 raw metric과 rank를 확인할 수 있다.

### Phase D — 최종 완성 단계

#### Step 13. Reproducibility and Integrity Review

- 목적: 후보 비교 결과가 동일한 조건에서 재현되도록 보장한다.
- 구현 내용:
  - config, seed, dependency, device 및 dataset hash 고정
  - candidate artifact hash 검증
  - no-auto-selection integrity 검사
  - report와 raw artifact 값 대조
  - explicit candidate cold-load inference/reward smoke test
- 선행 조건: Step 12
- 완료 기준:
  - 깨끗한 환경에서 동일 benchmark를 재실행할 수 있다.
  - report와 artifact metric이 일치한다.
  - selected/winner/recommended field가 없다.
  - 모든 후보 artifact를 독립적으로 load할 수 있다.

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

- 최종 QC 결과가 반영된 공식 RL dataset
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

- 기존 MDP v1, replay environment, reward function 및 legacy two-target 상수는 변경하지 않는다.
- Framework는 HPO와 candidate ranking을 수행하지만 최종 후보를 자동 결정하지 않는다.
- Validation Rollout Weighted Score는 비교용 종합 지표이며 selected model을 생성하지 않는다.
- Efficiency는 별도 지표로 제공하고 Validation Score에 포함하지 않는다.
- 모든 후보 Test 결과를 생성하지만 HPO와 Validation ranking에 사용하지 않는다.
- 연구자는 최종 결정 시 Test 결과를 선정 근거로 사용하지 않는 절차를 따른다.
- Persistence는 비교 baseline이며 gate가 아니다.
- 공식 benchmark는 8개 후보가 모두 실행되어야 유효하다.
- GPU 사용 여부와 device 정보를 후보별로 기록한다.
- v1.4에서는 TCN, CatBoost, XGBRF, RandomForest, Probabilistic Ensemble 및 automatic selection을 지원하지 않는다.
