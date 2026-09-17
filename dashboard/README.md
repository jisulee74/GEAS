# GEAS 통합 실험 대시보드

데이터 품질 관리(QC), 전이모델 선정, 강화학습 및 제어 실험 준비 화면을 제공합니다.

- 웹사이트: https://jisulee74.github.io/GEAS/
- 실행 및 자동 배포: [DEPLOYMENT.md](DEPLOYMENT.md)

아래는 기존 전이모델 화면의 상세 설명입니다.

# GEAS Transition Model Selection Dashboard

스마트 온실 환경 자동제어 시스템 **GEAS** (Greenhouse Environment Automation System)의 모델 기반 강화학습(Model-Based RL) 파이프라인에서 생성된 **작물별 전이모델(Transition Model) 후보 선별 및 Validation 평가 결과 탐색용 반응형 웹 대시보드**입니다.

---

## 1. Transition Model 구조 및 실험 설정 (Architecture & Setup)

1. **Model Input (입력 변수)**:
   - **현재 온실 상태 ($s_t$)**: 실내 온도(°C), 실내 상대습도(%), 실내 CO₂ 농도(ppm), 외기 환경
   - **제어 Action ($a_t$)**: 환기창 개폐, 냉난방기, 유동팬, CO₂ 공급기, 보광등, 차광커튼 (6대 액션)
   - **Feature Variant**: 결측/이상치 품질 플래그 포함(`with_quality_flags`) vs 제외(`without_quality_flags`)
2. **Model Output (출력 및 전이 방식)**:
   - **잔차 전이 (Residual Transition)**: 5분 뒤 상태 변화량($\Delta s_{t+1}$)을 예측하여 현재 상태에 가산 ($s_{t+1} = s_t + \Delta s_{t+1}$)
   - **주요 예측 Target**: 실내 온도(°C), 실내 상대습도(%), 실내 CO₂ 농도(ppm)
3. **Multi-Horizon Recursive Rollout (다단계 재귀 롤아웃)**:
   - 5분 1-step 단기 예측 오차뿐 아니라 모델 예측값을 다음 시점 입력으로 반복 순환 투입하는 **15분(3-step), 30분(6-step), 60분(12-step) 롤아웃 안정성 및 오차 누적률** 평가
4. **공식 HPO 다단계 가중 목적함수 (HPO Multi-Horizon Objective)**:
   $$\text{Loss} = 0.10 \times \text{One-Step(5m)} + 0.10 \times \text{Rollout(15m)} + 0.25 \times \text{Rollout(30m)} + 0.45 \times \text{Rollout(60m)} + 0.10 \times \text{Drift Slope}$$

---

## 2. 대시보드 탭 구성

모든 탭 명칭은 직관적이고 일관된 용어로 구성되어 있으며, HPO 목적함수의 각 항에 대응하는 전용 분석 화면을 제공합니다:

1. **📊 개요 및 잠정 선도 후보 (Overview & Candidates)**:
   - 6대 핵심 KPI 카드 + 작물별 잠정 선도 후보(Provisional Leading Candidates) 카드 + 선택 근거 및 연구자 결정 워크스페이스 (JSON 내보내기)
2. **🏆 후보 모델 종합 순위 (Candidate Ranking - Table 1)**:
   - 15/30/60m 가중 롤아웃 NRMSE 기준 적격 우선 2단계 정렬 종합 순위표, 롤아웃 그룹 막대 차트, Persistence Skill 차트
3. **🎯 5분 단기 예측 성능 (One-Step Performance - Table 2)**:
   - HPO 목적함수 1항($0.10 \times \text{One-Step}$) 분석. 온도/습도/CO₂ Target 필터 및 NRMSE/R²/MAE/RMSE/Q90/CVaR90 비교
4. **🛡️ 다단계 롤아웃 및 안전성 (Rollout & Safety - Table 3)**:
   - 5m → 15m → 30m → 60m 오차 누적 궤적(Trajectory), 물리적 제약 위반, 반사실적 제어 감도(Counterfactual Sensitivity) 및 표본 부족(Data-Sparse) 상태 분석
5. **📉 롤아웃 드리프트 분석 (Rollout Drift Analysis - Table 6)**:
   - HPO 목적함수 5항($0.10 \times \text{Drift Slope}$) 분석. 15/30/60m Horizon별 및 Target별 편향 누적률(Mean/Median/Q90 Drift Slope NMAE) 비교
6. **⚡ 자원 효율성 및 속도 (Resource Efficiency - Table 4)**:
   - 추론 지연시간(Median/P95), 학습/HPO 시간, CPU/GPU 메모리, 모델 크기 벤치마크 및 Latency vs NRMSE / Size vs NRMSE 파레토 산점도
7. **🔬 품질플래그 효과 검증 (Quality-Flag Ablation - Table 5)**:
   - `with - without` 기준 Centered Diverging 막대 차트 + 95% Bootstrap 신뢰구간 에러바 + Standard Zoom / Full Scale 뷰 스위처

---

## 3. 실행 및 배포 방법

```bash
# Python 내장 웹 서버 실행 (포트 8080)
python -m http.server 8080
```

브라우저에서 `http://localhost:8080`에 접속하면 라이트 테마 기반의 고대비 반응형 대시보드가 구동됩니다.
