# GEAS Predictor Benchmark

`updated/GEAS3.0` 원본 코드는 건드리지 않고, inner-layer의 다음 시점 온도 예측기만 바꿔서 비교하는 오프라인 실험 워크스페이스입니다.

비교 대상은 다음 4가지입니다.

- `physics`: 현재 GEAS inner-layer의 물리식 기반 1-step 온도 예측
- `tiny_ttm`: edge 친화적인 경량 temporal-mixing 기반 예측기
- `physics_residual_tiny_ttm`: 물리식 예측값 + 경량 temporal residual 보정
- `random_forest`: 시계열 feature를 입력으로 쓰는 Random Forest baseline

## 실험 개요

1. DB에서 실제 외기/내기/제어 로그를 읽습니다.
2. 외기 로그 정보량과 결측률을 기준으로 가장 밀도 높은 공통 기간을 자동 선택합니다.
3. 선택된 기간의 앞부분으로 공통 물리 파라미터(theta)를 적합합니다.
4. 같은 theta, 같은 target profile, 같은 candidate search/constraint flow에서 predictor만 바꿔 replay simulation을 수행합니다.
5. `updated/evaluation/no_data_eval.py`의 9개 KPI 정의와 동일한 방식으로 결과를 계산합니다.
6. KPI 표, 예측 정확도, 런타임/모델 크기 비교 그래프를 저장합니다.

## 실행

권장 파이썬:

```bash
/home/ljs/miniconda3/envs/jisulee-geas-py310/bin/python
```

실행 예:

```bash
/home/ljs/miniconda3/envs/jisulee-geas-py310/bin/python /home/ljs/jslee/geas_predictor_benchmark/benchmark.py
```

옵션 예:

```bash
/home/ljs/miniconda3/envs/jisulee-geas-py310/bin/python /home/ljs/jslee/geas_predictor_benchmark/benchmark.py \
  --window-days 7 \
  --lookback 6 \
  --step-minutes 5 \
  --train-ratio 0.7
```

기본 고정 평가기간:

- `2025-03-17 00:00:00` ~ `2025-03-24 00:00:00`

## 출력

`results/run_YYYYMMDD_to_YYYYMMDD_YYYYMMDD_HHMMSS/` 아래에 다음이 저장됩니다.

- 루트:
  - `selected_period.json`
  - `theta_common.json`
  - `report.md`
- `csv/`
  - `kpi_summary.csv`
  - `prediction_summary.csv`
  - `efficiency_summary.csv`
  - `comparison_summary.csv`
  - `episode_<model>.csv`
- `png/`
  - `kpi_comparison.png`
  - `temperature_trajectory.png`
  - `efficiency_comparison.png`

## 주의

- `tiny_ttm`은 경량 edge 실험용 temporal-mixing surrogate 입니다.
- 현재 환경 제약상 대형 딥러닝 프레임워크 없이 `scikit-learn` 기반 경량 temporal feature mixer로 구현했습니다.
- 따라서 회의용 비교/PoC에는 적합하지만, 실제 배포용 Tiny Time Mixer와 1:1 동일 구현은 아닙니다.
