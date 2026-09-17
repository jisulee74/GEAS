# Quality experiment artifact comparison

이 문서는 `quality_control_model_selection` 아래의 세 artifact 디렉터리가 어떤 실험 조건으로 생성되었는지 비교한다. 공통 조건은 반복하지 않고, artifact와 현재 코드에서 확인되는 **차이점만** 기록한다.

비교 대상은 다음 세 디렉터리다.

- `artifacts_backup_20260805`
- `artifacts_backup_20260806`
- `artifacts`

`quality_model_selection_all_vars_legacy/artifacts`는 전체 변수 대상 legacy 실험이므로 이 비교에서 제외한다.

## 핵심 차이

| 구분 | `artifacts_backup_20260805` | `artifacts_backup_20260806` | `artifacts` |
|---|---|---|---|
| 실행 완결성 | Strawberry만 3개 모델의 전체 평가가 존재한다. Melon과 Cucumber는 ModernTCN·TimesNet HPO 산출물만 있어 부분 실행이다. | 세 작물 × 3개 모델의 HPO, 임계값 보정, Validation/Test 평가 및 benchmark가 모두 존재한다. | 세 작물 × 3개 모델의 전체 산출물이 존재한다. |
| 이상치 크기 기준 | 평가 split 자체의 변수 표준편차를 사용했다. 따라서 Validation과 Test의 주입 크기가 각 split 분포에 따라 달라질 수 있다. | Train 정상값의 변수별 표준편차를 고정 기준으로 사용하고 Validation/Test에 재사용했다. | 8월 6일 방식과 동일하게 Train 정상값의 변수별 표준편차를 사용한다. |
| 이상치 방향 | 변수마다 한 방향으로만 주입했다. 변수 순서에 따라 `+8σ` 또는 `-8σ` 방향이 고정됐다. | 변수별 양·음 방향 주입 수가 가능한 한 균형을 이루도록 주입했다. | 8월 6일 방식과 동일하게 양·음 방향을 균형 있게 주입한다. |
| 물리 범위와의 관계 | 주입값에 도메인 정상범위 제한을 적용하지 않았다. 따라서 규칙 기반 검사만으로도 잡히는 범위 밖 값이 AI 평가에 포함될 수 있었다. | AI 평가 후보는 결측·규칙 이상·AI 이상·forward-fill 셀을 제외한 정상 셀로 제한하고, 주입값도 규칙 정상범위 안에서 경계까지 cap했다. | 8월 6일과 동일하게 규칙 정상범위 안에 주입한다. |
| 주입 이력 | `injection_stats`가 없어 변수별 실제 주입률, 방향, cap 횟수 및 실제 변화량을 artifact에서 감사할 수 없다. | `injection_stats`에 후보/주입 셀 수, 실제 주입률, 양·음 방향 수, cap 횟수, Train 표준편차 및 실제 변화량을 기록한다. | 8월 6일과 동일한 주입 통계를 기록한다. |
| 임계값 후보 범위 | 변수별 원본 Validation reconstruction error의 최소–최대 범위를 100등분했다. | 개선된 이상치 주입을 사용하지만, 임계값 후보는 여전히 변수별 원본 Validation reconstruction error 범위에서 생성했다. | 변수별 **이상치 주입 Validation score**의 최소–최대 범위를 100등분한다. 원본 Validation error 범위도 별도 필드로 함께 기록한다. |
| 후보 범위 추적성 | 후보 범위의 출처를 별도 필드로 기록하지 않았다. | 후보 범위의 출처를 별도 필드로 기록하지 않았다. | `candidate_range_source: injected_validation_scores`와 clean/injected 범위를 함께 기록한다. |

## 실행별 해석

### `artifacts_backup_20260805`

초기 실험 결과다. 합성 이상치는 선택된 셀에 변수별 `8 × 현재 split 표준편차`를 더하거나 빼는 방식이었고, 방향은 변수별로 하나만 사용했다. 도메인 정상범위로 제한하지 않았기 때문에 AI 모델의 문맥 기반 탐지 성능과 규칙 범위 위반 탐지 성능이 섞일 수 있다.

또한 Strawberry만 전체 파이프라인이 완료됐고 Melon/Cucumber는 HPO 도중의 부분 산출물이므로, 이 디렉터리를 세 작물 최종 성능 비교에 사용하면 안 된다.

### `artifacts_backup_20260806`

합성 이상치 주입 방식을 현실적인 AI 탐지 평가에 맞게 변경한 첫 완전 실행이다.

- 주입 대상은 규칙상 정상이고 결측/forward-fill flag가 없는 관측 셀이다.
- 변수별 크기는 Train 정상값 표준편차의 8배를 기준으로 한다.
- Validation/Test 모두 같은 Train 기준 스케일을 사용한다.
- 양·음 방향을 균형 있게 배정한다.
- 값이 규칙 정상범위를 벗어나지 않도록 가능한 경계까지 주입 크기를 줄인다.
- 실제 주입률과 cap 결과를 `injection_stats`에 기록한다.

다만 임계값 후보 범위는 원본 Validation reconstruction error의 최소–최대에 한정됐다. 합성 이상치의 score가 이 범위를 넘더라도 그 확장된 구간은 후보 임계값 탐색에 포함되지 않았다.

### `artifacts`

8월 6일의 이상치 주입 방식은 유지하고 임계값 후보 생성 범위를 수정한 실행이다. 변수별 합성 이상치가 주입된 Validation score의 최소–최대를 100등분하여 후보를 만들기 때문에, 주입 후 관측된 score 전체 범위에서 F1 최적 임계값을 찾는다.

각 변수의 다음 범위를 모두 저장한다.

- 원본 Validation reconstruction error 범위
- 이상치 주입 Validation score 범위
- 주입 score에 의해 확장된 상한
- 최종 선택 임계값과 후보별 지표

세 디렉터리 중 현재 실험 정의를 가장 충실하게 반영하는 결과는 `artifacts`다.

## 비교 시 주의사항

- 세 실행의 observation 변수 8개, 모델 후보 3개, HPO budget 50, lookback 288, 평가 이상치 비율 10%, 명목상 이상치 크기 8σ, random seed 42, 변수별 임계값 후보 수 100은 동일하다.
- 동일한 숫자 설정이라도 이상치 생성 코드와 임계값 후보 범위가 달라 성능 지표를 직접적인 재현 반복으로 간주하면 안 된다.
- HPO 결과와 선택된 best configuration의 차이는 실험 환경 차이뿐 아니라 실행 완료 시점 및 재학습 결과의 차이도 포함할 수 있다.
- artifact에는 입력 parquet의 content hash가 저장되지 않아, 세 실행이 바이트 단위로 완전히 동일한 dataset을 사용했는지는 artifact만으로 증명할 수 없다. 경로와 설정은 동일하다.
- 이 문서는 artifact에 기록된 조건과 현재 코드에서 복원 가능한 차이만 설명하며, 결과 수치의 우열을 평가하지 않는다.
