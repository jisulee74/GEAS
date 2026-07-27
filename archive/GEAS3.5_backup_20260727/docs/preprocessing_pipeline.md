# GEAS Preprocessing Pipeline

이 문서는 GEAS preprocessing refactor v1.2의 현재 구현 기준을 정리합니다.

## 전체 흐름

```text
datasets/iot97_historical/
  3_splits/
    {crop}/{train,validation,test}.parquet
  4_preprocessed/
    1_input_schema_prepared/
    2_unit_canonicalized/
    3_missing_outliers_handled/
      _artifacts/{crop}/selected_quality_model.json
      _artifacts/{crop}/validation_selection_report.json
      preprocessing_pipeline_manifest.json
  5_rl_dataset/
    {crop}/{train,validation,test}.parquet
    rl_dataset_manifest.json
```

직접 목표는 AI Quality Model 기반의 결측치/이상치 처리입니다. RL dataset은
Quality Model 출력이 완료된 뒤 별도 stage에서 생성합니다.

## Stage 책임

### 1_input_schema_prepared

- `reg_date` 파싱
- GEAS state/action feature column 보장
- numeric coercion
- 결측치 보간, action 복원, 이상치 탐지는 수행하지 않음

### 2_unit_canonicalized

- action percent 값을 0~1 scale로 정규화
- binary action을 0/1로 정규화
- humidity/range normalization
- missing/outlier/imputation은 수행하지 않음

### 3_missing_outliers_handled

- feature-level `{column}_missing_flag` 생성
- aggregate `missing_flag`는 생성하지 않음
- Action은 Action Log 동일 timestamp 값으로만 복원
- Action 복원 시 `{action}_restored_flag=1`
- Action Log에도 없으면 Action NaN 유지
- Action은 rule outlier, AI outlier, median/AI imputation 대상에서 제외
- Observation은 rule outlier, AI outlier, invalid mask, imputation 대상
- 최종 aggregate quality flags:
  - `rule_outlier_flag`
  - `ai_outlier_flag`
  - `invalid_flag`
  - `imputed_flag`

### 5_rl_dataset

- `4_preprocessed/3_missing_outliers_handled`를 입력으로 사용
- MDP v1 observation/action/transition target 생성
- Action NaN이 남아 있는 transition start는 여기서 제외
- `reward`, `next_obs_*`, `done`, `rl_valid_transition` 생성
- Action NaN 제외는 오류가 아니라 preprocessing data integrity 정책의 결과

## Quality Model 학습/선택 정책

`geas35.offline.run_preprocessing_pipeline`이 Step 8 runner입니다.

- Train split: Quality Model fit
- Validation split: threshold tuning 및 model selection
- Test split: 학습/선택에 사용하지 않고 선택된 model/threshold만 적용
- 현재 구현 후보:
  - `rule_only`
  - `median`
- 향후 ModernTCN, TimesNet, iTransformer, PatchTST는 같은
  `BaseQualityModel` interface에 추가합니다.

Artifacts:

- `selected_quality_model.json`
- `validation_selection_report.json`
- `preprocessing_pipeline_manifest.json`

## Compatibility

다음 legacy entry point는 기존 코드와 테스트를 위해 유지합니다.

- `geas35.preprocessing.3_unit_canonicalization`
  - 실제 구현은 `2_unit_canonicalization.py`로 이동
- `prepare_missing_values_handled_splits.py`
  - legacy intermediate stage runner
- `prepare_outliers_flagged_splits.py`
  - legacy outlier-flagged stage runner
- `apply_outlier_flags`
  - legacy rule/TCN-ready outlier policy

MDP v1 quality observation은 old flag와 new flag를 모두 지원합니다.

- `obs_quality_missing_imputed_flag`
  - `missing_imputed_flag`
  - fallback: `imputed_flag`, `outlier_imputed_flag`
- `obs_quality_outlier_flag`
  - `outlier_flag`
  - fallback: `invalid_flag`, `rule_outlier_flag`, `ai_outlier_flag`

## 실행 예시

```powershell
$env:PYTHONPATH='src'
python -m geas35.offline.prepare_input_schema_prepared_splits --splits train validation test
python -m geas35.offline.prepare_unit_canonicalized_splits --splits train validation test
python -m geas35.offline.run_preprocessing_pipeline --splits train validation test
python -m geas35.offline.prepare_rl_dataset_splits --splits train validation test
```

Step 10 기준 전체 regression:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
```
