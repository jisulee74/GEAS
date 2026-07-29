# GEAS Preprocessing Refactor Plan v1.2

## 1. Summary

이번 리팩토링의 직접 목표는 **AI 기반 결측치/이상치 처리용 Quality Model pipeline 구축**입니다. RL은 이번 리팩토링의 구현 대상이 아니라, Quality Model이 완료된 뒤 Clean Dataset을 입력으로 사용하는 다음 단계입니다.

전체 데이터 흐름:

```text
Raw Dataset
→ Resampling
→ Train / Validation / Test Split
→ Input Schema Preparation
→ Unit Canonicalization
→ Missing / Outlier Handling (AI Quality Model)
→ Clean Dataset 생성
→ RL Dataset 생성
→ RL Training
```

AI Quality Model 원칙:

- Train split으로 Quality Model을 학습합니다.
- Validation split으로 threshold tuning 및 model selection을 수행합니다.
- Test split은 학습/선택에 사용하지 않고, 선택 완료된 model과 threshold만 적용합니다.
- RL Training은 최종 선택된 AI Quality Model로 생성된 Clean Dataset 이후에만 수행합니다.

## 2. Dataset Stage Structure

최종 dataset stage는 다음처럼 분리합니다.

```text
datasets/iot97_historical/
  1_raw/
  2_resampled_5min/
  3_splits/
  4_preprocessed/
    1_input_schema_prepared/
    2_unit_canonicalized/
    3_missing_outliers_handled/
  5_rl_dataset/
```

`4_preprocessed/3_missing_outliers_handled`는 AI Quality Model의 최종 출력입니다.

상태:

- Observation missing/outlier 처리 완료
- Observation imputation 완료
- Action은 원본 또는 Action Log 복원값만 포함
- Action Log에도 없던 Action NaN은 그대로 유지
- 품질 flag 및 imputation trace 포함

`5_rl_dataset`은 RL 전용 dataset입니다.

상태:

- Observation 복원 완료
- Action NaN transition 제거 완료
- RL observation/action/target 형식 생성 완료
- RL valid transition mask와 제외 사유 manifest 포함

## 3. Column Policy

컬럼은 Observation, Action, Metadata를 서로 다른 정책으로 처리합니다.

| 구분 | Missing Flag | Rule Outlier | AI Outlier | Imputation |
|---|---|---|---|---|
| Observation / State | 수행 | 수행 | 수행 | AI 또는 baseline |
| Action / Control | 수행 | 제외 | 제외 | Action Log 동일 timestamp 복원만 허용 |
| Metadata | 필요 시 수행 | 필요 시 validation | 제외 | 추정하지 않음 |

공통 flag:

- `{column}_missing_flag`
- `missing_flag`

Observation 전용:

- `{obs}_raw_value`
- `{obs}_rule_outlier_flag`
- `{obs}_ai_anomaly_score`
- `{obs}_ai_outlier_flag`
- `{obs}_invalid_flag`
- `{obs}_imputed_flag`

Action 전용:

- `{action}_raw_value`
- `{action}_restored_flag`
- `{action}_restore_source`

`restore_source` 값:

```text
original
action_log
missing_unrestored
```

Aggregate flag:

- `rule_outlier_flag`
- `ai_outlier_flag`
- `invalid_flag`
- `imputed_flag`

Warning 단계는 만들지 않습니다. 품질 판정은 `normal / outlier` 2단계만 사용합니다.

## 4. Quality Model and Threshold Design

AI Quality Model interface는 Observation 전용입니다. Action은 절대 AI/Median으로 추정하지 않습니다.

공통 interface:

```python
class BaseQualityModel:
    def fit(train_df, observation_columns, *, time_column, group_columns, valid_mask=None): ...
    def reconstruct(df, observation_columns, *, time_column, group_columns): ...
    def forecast(df, observation_columns, *, horizon, time_column, group_columns): ...
    def anomaly_score(df, prediction_df, observation_columns): ...
    def predict_outlier(df, thresholds, observation_columns): ...
    def impute(df, invalid_mask, prediction_df, observation_columns): ...
```

Ablation 순서:

1. `Rule Only`
   - missing flag + rule outlier + invalid mask
   - Observation imputation 없음

2. `Rule + Median`
   - invalid Observation을 train median으로 복원
   - Action은 Action Log restore만 사용

3. `Rule + AI`
   - invalid Observation을 AI reconstruction/forecast 기반으로 복원
   - 후보: ModernTCN, TimesNet, iTransformer, PatchTST

Threshold 탐색은 별도 `ThresholdOptimizer` 클래스로 캡슐화합니다.

`ThresholdOptimizer` 책임:

- Validation Dataset 입력
- threshold 후보 탐색
- objective 계산
- 최적 threshold 반환
- 결과 summary 저장

초기 구현은 grid search optimizer로 시작하되, interface는 다음 방식으로 교체 가능하게 둡니다.

```text
GridSearchThresholdOptimizer
PercentileThresholdOptimizer
BayesianThresholdOptimizer
```

Synthetic Masking Evaluation:

- Validation 정상 후보 cell 일부를 masking합니다.
- Quality Model이 masking된 Observation을 복원합니다.
- MAE, RMSE를 계산합니다.
- 이 결과를 model selection과 threshold objective에 사용합니다.

Threshold는 고정 quantile이 아니라 validation objective로 자동 선택되는 hyperparameter입니다.

## 5. Files

수정 대상:

- `GEAS3.5/src/geas35/preprocessing/__init__.py`
- `GEAS3.5/src/geas35/preprocessing/2_missing_values_handling.py`
- `GEAS3.5/src/geas35/preprocessing/3_unit_canonicalization.py`
- `GEAS3.5/src/geas35/preprocessing/4_outlier_handling.py`
- `GEAS3.5/src/geas35/offline/prepare_unit_canonicalized_splits.py`
- `GEAS3.5/src/geas35/offline/prepare_missing_values_handled_splits.py`
- `GEAS3.5/src/geas35/offline/prepare_outliers_flagged_splits.py`
- `GEAS3.5/src/geas35/rl/mdp_v1.py`
- 관련 tests/docs

신규 생성:

- `GEAS3.5/src/geas35/preprocessing/2_unit_canonicalization.py`
- `GEAS3.5/src/geas35/preprocessing/3_missing_outliers_handling.py`
- `GEAS3.5/src/geas35/models/quality/base.py`
- `GEAS3.5/src/geas35/models/quality/rule_only.py`
- `GEAS3.5/src/geas35/models/quality/baseline.py`
- `GEAS3.5/src/geas35/models/quality/thresholds.py`
- `GEAS3.5/src/geas35/models/quality/evaluation.py`
- `GEAS3.5/src/geas35/models/quality/registry.py`
- `GEAS3.5/src/geas35/offline/prepare_missing_outliers_handled_splits.py`
- `GEAS3.5/src/geas35/offline/run_preprocessing_pipeline.py`
- `GEAS3.5/src/geas35/offline/prepare_rl_dataset.py`
- `GEAS3.5/scripts/run_preprocessing_pipeline.py`
- `GEAS3.5/scripts/prepare_rl_dataset.py`
- `GEAS3.5/configs/qc/preprocessing_quality_v1.json`
- `GEAS3.5/docs/preprocessing_pipeline.md`
- 신규 unit/integration tests

## 6. RL Dataset Boundary

AI Quality Model 출력인 `4_preprocessed/3_missing_outliers_handled`는 RL 학습용 최종 형식이 아닙니다. RL Dataset 생성은 별도 stage입니다.

`5_rl_dataset` 생성 정책:

- Clean Observation은 AI Quality Model이 복원한 값을 사용합니다.
- Action은 원본 또는 Action Log 복원값만 사용합니다.
- Action NaN이 남아 있는 transition은 `5_rl_dataset` 생성 시 제외합니다.
- 제외 이유는 manifest에 기록합니다.
- RL training은 `5_rl_dataset`만 입력으로 사용합니다.

Transition 제외 조건:

- RL action mapping에 필요한 Action 값이 NaN
- Observation target이 finite하지 않음
- timestep이 5분 간격이 아님
- crop/series/segment/episode boundary가 바뀜
- growth stage/context feature가 유효하지 않음

## 7. Risks

- LOCF 제거와 Action NaN transition 제외로 RL 학습 가능 transition 수가 줄 수 있습니다.
- Action Log coverage가 낮으면 복원 실패율이 높아질 수 있습니다.
- Synthetic masking은 실제 이상치 label이 아니므로 threshold 선택 proxy로만 해석해야 합니다.
- Deep time-series model 도입은 dependency/GPU/runtime 리스크가 있어 baseline pipeline 안정화 후 진행하는 것이 좋습니다.
- 기존 stage 이름과 flag 이름을 사용하는 테스트/문서가 많아 deprecation wrapper가 필요합니다.

## 8. Implementation Policy

이번 리팩토링은 절대 한 번에 전체 구현하지 않습니다. 반드시 step 단위로 진행합니다.

각 step 종료 조건:

- 구현 완료
- Unit Test 통과
- Integration Test 통과
- 기존 기능 Regression Test 통과
- 결과 확인 후 다음 step 진행

Step plan:

1. **Unit Canonicalization Stage Refactoring**
   - `2_unit_canonicalization.py` 생성
   - runner 경로를 `1_input_schema_prepared → 2_unit_canonicalized`로 변경
   - unit/integration/regression test 수행

2. **Missing Flag + Action Restore**
   - LOCF 제거
   - `{column}_missing_flag`, `missing_flag` 생성
   - Action Log restore 및 `{action}_restored_flag`, `{action}_restore_source` 생성
   - unit/integration/regression test 수행

3. **Observation-only Rule Outlier**
   - Action은 rule outlier에서 제외
   - Observation만 domain rule flag 적용
   - aggregate `rule_outlier_flag` 생성
   - unit/integration/regression test 수행

4. **Quality Model Interface + Rule-only Baseline**
   - `BaseQualityModel`
   - `RuleOnlyQualityModel`
   - Action이 interface에 들어가지 않는지 테스트
   - unit/integration/regression test 수행

5. **Median Baseline + Synthetic Masking Evaluation**
   - `MedianQualityModel`
   - validation synthetic masking MAE/RMSE 평가
   - unit/integration/regression test 수행

6. **ThresholdOptimizer**
   - `ThresholdOptimizer` interface
   - 초기 `GridSearchThresholdOptimizer`
   - validation objective 기반 threshold 선택
   - unit/integration/regression test 수행

7. **Combined Missing / Outlier Handling Stage**
   - `3_missing_outliers_handling.py`
   - `prepare_missing_outliers_handled_splits.py`
   - `4_preprocessed/3_missing_outliers_handled` manifest 생성
   - unit/integration/regression test 수행

8. **End-to-End Preprocessing Runner**
   - `run_preprocessing_pipeline.py`
   - train fit, validation selection, test transform 분리
   - artifacts 저장
   - unit/integration/regression test 수행

9. **RL Dataset Stage**
   - `5_rl_dataset` 생성 runner
   - Action NaN transition 제외
   - RL input format/manifest 생성
   - unit/integration/regression test 수행

10. **Docs and Compatibility Cleanup**
   - preprocessing docs 갱신
   - old runner/API deprecation wrapper 유지
   - old flag fallback 테스트
   - 전체 regression test 수행
