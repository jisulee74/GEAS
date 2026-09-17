# 실험 및 화면 해석 안내

## 확정된 실험 범위
오이(cucumber), 멜론(melon), 딸기(strawberry), ModernTCN·TimesNet·PatchTST 3개 후보. 최종 비교는 data/quality/validation_results.csv와 test_results.csv. Validation으로 비교·선정하고 Test는 최종 평가로 분리합니다. 실행 완료와 최종 모델 선택은 다른 상태입니다.

AI 대상 8개: in_medium_temp1, in_temp, in_temp2, in_hum, in_hum2, in_medium_hum1, in_co2, in_co2_2. 규칙 기반 대상 16개는 reference/offline/scripts/03_control_quality.py의 RULE_BASED_OUTLIER_COLUMNS 및 도메인 CSV 참조.

설정 YAML 기준: 모델별 HPO 50회, seed 42, lookback 288, 최대 100 epochs, patience 10. 평가 마스킹 비율 0.1, 이상치 주입 비율 0.1, 크기 8σ. 실제 실행 수와 최적 설정은 모델별 JSON을 읽으세요. 간소화된 프로젝트 구현을 원 논문의 완전 재현이라고 표현하지 마세요.

## 임계값 및 평가 해석
현재 artifacts는 injected_validation_scores 범위에서 변수별 100개 등간격 후보를 생성하는 버전입니다. 후보별 Validation 합성 이상치 F1 최대값으로 보정합니다. reference의 README/AI_TARGET_SCOPE에는 원본 Validation error 범위라는 이전 설명이 남아 있으므로 ARTIFACT_EXPERIMENT_COMPARISON.md와 실제 threshold_calibration.json 및 생성 코드를 우선합니다.

합성 이상치 크기는 Train 정상값 표준편차를 사용하고 방향을 균형화하며 규칙 정상범위 내로 제한합니다. 따라서 현장 라벨 기반 검증과 동일시하지 마세요. RMSE/MAE는 마스킹 복원 평가이고 자연 발생 결측의 숨은 정답을 검증한 결과가 아닙니다. 여러 변수 집계치에 임의로 °C 같은 단일 단위를 붙이지 마세요.

## 실제 보정 코드의 흐름
reference/offline/scripts/03_control_quality.py:_prepare_quality_controlled_frame 기준:
입력 스키마 정리 → 단위 통일 → 선택적 생육 정보 추가 → 결측·규칙 이상·AI 이상 탐지 및 모델 복원 → 비AI 변수 임시 보정 → 대표 센서값 생성.

reference/source/src/geas35/preprocessing/3_missing_outliers_handling.py에서 결측/규칙 이상은 복원 후보로 유지합니다. AI만 탐지한 이상치는 confidence가 제공되면 설정 임계값을 충족해야 복원하며, confidence가 없으면 기존 invalid mask를 사용합니다. 실제 수정 여부를 플래그로 기록합니다. 세부 결측·제어 로그 처리와 대표값 선택은 첨부 소스를 참조하세요.

비AI 임시 보정은 최대 3개 연속 행에 한정됩니다. 외부 온도·습도·풍속·강수량·기압은 양쪽 정상값을 이용한 시간 선형보간, 풍향은 원형 각도 보간, 일사량·누적 일사량은 보간 후 음수 제한, 강우 상태·PLC 상태는 직전 상태 유지입니다. 구간 경계·양끝 정상값 조건이 있으며 긴 공백이나 미해결 상태를 임의로 채우지 않습니다. 모든 결측 보정을 선형보간으로 설명하면 안 됩니다.

## 선택 상태
현재 적용 스크립트 DEFAULT_SELECTED_MODELS에는 딸기 ModernTCN, 멜론 ModernTCN, 오이 PatchTST가 설정되어 있습니다. 이는 '코드상 기본 적용 모델'로 표시할 수 있으나 이번 묶음에는 별도 심사자·선정 사유·실행 이력을 입증하는 선택 기록이 없습니다. 최종 적용 완료라고 단정하지 마세요.

## 자료가 없는 화면
QC 적용 후 원시/보정 시계열과 적용 건수는 포함하지 않았습니다. data/preprocessing은 리샘플링·분할 요약이며 최종 QC 보정 건수로 쓰면 안 됩니다. 보정 전후 시계열 화면은 자료 없음으로 표시하세요. 임의 그래프나 수치를 만들지 마세요.

## 화면별 자료
- 개요/복원/탐지/자원: 통합 Validation/Test CSV와 JSON.
- 변수별 상세: 작물의 reconstruction_metric_table.csv, anomaly_detection_metric_table.csv, 모델별 evaluation.json.
- HPO/학습: 모델별 hpo_results.json, best_config.json, training_history.csv.
- 임계값: 모델별 threshold_calibration.json.
- 벤치마크: 모델별 online_benchmark.json 및 작물별 online_benchmark_table.csv.
- 기존 그림: data/quality/figures 및 각 작물/figures. 정적 PNG의 split/범례를 확인하고 필터 결과인 것처럼 재표시하지 마세요.
모델별·작물별 중복 요약을 합산하지 말고 통합 CSV를 시작점으로 사용하세요.
