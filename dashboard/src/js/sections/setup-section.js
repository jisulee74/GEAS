/**
 * GEAS Dashboard - Section: Experimental Setup & Model Architecture (실험 설정 및 모델 구조)
 */

import { renderTooltipLi } from '../data/variable-tooltips.js';

export class SetupSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this._hasDocumentListener = false;
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="setup-page-layout">
        
        <!-- 1. Executive Summary & Mathematical Definition -->
        <div class="setup-hero-card">
          <div class="setup-hero-badge">
            <span>⚙️</span> Transition Model Architecture & Experimental Setup
          </div>
          <h2 class="setup-hero-title">온실 Transition Model 구조 및 실험 설계</h2>
          <p class="setup-hero-desc">
            현재 <strong>Transition Model</strong>은 시점 <strong>t의 온실 상태 관측값(s<sub>t</sub>)</strong>과 
            <strong>현재 제어 action(a<sub>t</sub>)</strong>을 입력받아, <strong>5분 뒤 실내 온도·습도·CO₂ 변화량</strong>을 예측하는 
            <strong>잔차 전이(Residual Transition) 모델</strong>입니다.
          </p>

          <div class="setup-math-grid">
            <div class="setup-math-box">
              <div class="setup-math-label">입력 벡터 구성 (Input Vector)</div>
              <div class="setup-math-eq">x<sub>t</sub> = [s<sub>t</sub>, a<sub>t</sub>]</div>
              <div class="setup-math-caption">상태 observation (s<sub>t</sub>) 55~57개 + 현재 제어 action (a<sub>t</sub>) 6개</div>
            </div>

            <div class="setup-math-box">
              <div class="setup-math-label">잔차 전이 출력 (Residual Transition)</div>
              <div class="setup-math-eq">ŷ<sub>t+1</sub> = y<sub>t</sub> + Δŷ<sub>t+1</sub></div>
              <div class="setup-math-caption">현재 측정값(y<sub>t</sub>)에 5분간의 예측 변화량(Δŷ<sub>t+1</sub>)을 가산하여 차기 상태 도출</div>
            </div>
          </div>
        </div>

        <!-- 2. Feature Variant Dimension Table -->
        <div class="setup-table-card">
          <div class="setup-section-header">
            <div class="setup-section-title">
              <span>📊</span> Feature Variant 차원 구성 비교 (Input Dimensions)
            </div>
            <div style="font-size: 0.82rem; color: #475569;">
              성능 동률 시 복잡도가 낮고 경량화된 <code style="background:#e0e7ff; color:#3730a3; padding:0.15rem 0.4rem; border-radius:4px; font-weight:600;">without_quality_flags (61차원)</code> 우선 채택
            </div>
          </div>

          <div class="setup-variant-table-container">
            <table class="setup-variant-table">
              <thead>
                <tr>
                  <th>Feature Variant (입력 구성)</th>
                  <th style="text-align: center;">상태 Observation (s<sub>t</sub>)</th>
                  <th style="text-align: center;">현재 Action (a<sub>t</sub>)</th>
                  <th style="text-align: center;">전체 입력 차원 (x<sub>t</sub>)</th>
                  <th>설명 및 채택 원칙</th>
                </tr>
              </thead>
              <tbody>
                <tr style="background: #f0fdf4;">
                  <td>
                    <div style="font-weight: 700; color: #166534; display: flex; align-items: center; gap: 0.4rem;">
                      <span>🌿</span> Without quality flags (기본/경량)
                    </div>
                  </td>
                  <td style="text-align: center; font-weight: 700; font-family: var(--font-family-mono); color: #0f172a;">55개</td>
                  <td style="text-align: center; font-weight: 700; font-family: var(--font-family-mono); color: #0f172a;">6개</td>
                  <td style="text-align: center; font-weight: 800; font-family: var(--font-family-mono); color: #059669; font-size: 1.05rem;">61개</td>
                  <td style="font-size: 0.85rem; color: #334155;">
                    품질 플래그 제외 기본형. <strong>동등 성능 시 모델 복잡도 및 센서 파이프라인 의존도를 최소화</strong>하기 위해 기본 채택.
                  </td>
                </tr>
                <tr>
                  <td>
                    <div style="font-weight: 700; color: #1e293b; display: flex; align-items: center; gap: 0.4rem;">
                      <span>🏷️</span> With quality flags (품질플래그 포함)
                    </div>
                  </td>
                  <td style="text-align: center; font-weight: 700; font-family: var(--font-family-mono); color: #0f172a;">57개 (+2)</td>
                  <td style="text-align: center; font-weight: 700; font-family: var(--font-family-mono); color: #0f172a;">6개</td>
                  <td style="text-align: center; font-weight: 800; font-family: var(--font-family-mono); color: #2563eb; font-size: 1.05rem;">63개</td>
                  <td style="font-size: 0.85rem; color: #334155;">
                    결측값 대체 여부 및 이상치 탐지 여부 플래그 2개가 추가된 확장 입력형.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- 3. Detailed Breakdown of 12 Input Categories (3 cols × 4 rows) -->
        <div class="setup-categories-card">
          <div class="setup-section-header">
            <div>
              <div class="setup-section-title">
                <span>📥</span> 상세 입력 구성
              </div>
              <div style="font-size: 0.82rem; color: #64748b; font-weight: 500; margin-top: 0.2rem;">
                전체 모델 입력 = <strong>현재 상태 Observation (1~9 환경·맥락·안전 파생 49개 + 10 직전 Action 6개)</strong> + <strong>11 현재 적용할 Action 6개</strong> (+ <strong>12 Quality Flag 2개</strong>)
              </div>
            </div>
            <div style="font-size: 0.8rem; background: #f8fafc; padding: 0.35rem 0.75rem; border-radius: 9999px; border: 1px solid #e2e8f0; color: #475569; font-weight: 600;">
              상태 이력(a<sub>t-1</sub>) vs 제어 조건(a<sub>t</sub>) 구분
            </div>
          </div>

          <div class="setup-cat-grid">
            
            <!-- [1행 - 1] 현재 실내환경 -->
            <div class="setup-cat-box">
              <div class="setup-cat-head">
                <span class="setup-cat-num">1</span>
                <span class="setup-cat-name">현재 실내환경</span>
                <span class="setup-cat-count">3개</span>
              </div>
              <ul class="setup-item-list">
                <li>실내 온도 (°C)</li>
                <li>실내 상대습도 (%)</li>
                <li>실내 CO₂ 농도 (ppm)</li>
              </ul>
            </div>

            <!-- [1행 - 2] 현재 실외환경 -->
            <div class="setup-cat-box">
              <div class="setup-cat-head">
                <span class="setup-cat-num">2</span>
                <span class="setup-cat-name">현재 실외환경</span>
                <span class="setup-cat-count">5개</span>
              </div>
              <ul class="setup-item-list">
                <li>외부 온도 (°C)</li>
                <li>외부 상대습도 (%)</li>
                <li>외부 일사량 (W/m²)</li>
                <li>외부 풍속 (m/s)</li>
                <li>강우 여부 (이진 0/1)</li>
              </ul>
            </div>

            <!-- [1행 - 3] 시간 조건 -->
            <div class="setup-cat-box">
              <div class="setup-cat-head">
                <span class="setup-cat-num">3</span>
                <span class="setup-cat-name">시간 조건</span>
                <span class="setup-cat-count">3개</span>
              </div>
              <ul class="setup-item-list">
                ${renderTooltipLi('time_sin_cos', '시간 주기 sin/cos 인코딩 (2개)')}
                ${renderTooltipLi('is_daytime', '주간 여부 (1개)')}
              </ul>
            </div>

            <!-- [2행 - 4] 일사·일조 조건 -->
            <div class="setup-cat-box">
              <div class="setup-cat-head">
                <span class="setup-cat-num">4</span>
                <span class="setup-cat-name">일사·일조 조건</span>
                <span class="setup-cat-count">7개</span>
              </div>
              <ul class="setup-item-list">
                ${renderTooltipLi('solar_cycle', '일사 주기: 주간, 초저녁, 심야 (3개)')}
                ${renderTooltipLi('sunshine_state', '일조 상태: 맑음, 부분 흐림, 흐림, 미확인 (4개)')}
              </ul>
            </div>

            <!-- [2행 - 5] 생육 정보 -->
            <div class="setup-cat-box">
              <div class="setup-cat-head">
                <span class="setup-cat-num">5</span>
                <span class="setup-cat-name">생육 정보</span>
                <span class="setup-cat-count">7개</span>
              </div>
              <ul class="setup-item-list">
                ${renderTooltipLi('growth_stage', '생육단계 1–6 one-hot encoding (6개)')}
                ${renderTooltipLi('dat', '정식 후 경과일수 (DAT, Days After Transplanting, 1개)')}
              </ul>
            </div>

            <!-- [2행 - 6] 목표 온도 정보 -->
            <div class="setup-cat-box">
              <div class="setup-cat-head">
                <span class="setup-cat-num">6</span>
                <span class="setup-cat-name">목표 온도 정보</span>
                <span class="setup-cat-count">5개</span>
              </div>
              <ul class="setup-item-list">
                ${renderTooltipLi('day_target_temp', '주간 목표온도 (°C)')}
                ${renderTooltipLi('night_target_temp', '야간 목표온도 (°C)')}
                ${renderTooltipLi('target_temp_low', '현재 목표온도 하한 (°C)')}
                ${renderTooltipLi('target_temp_high', '현재 목표온도 상한 (°C)')}
                ${renderTooltipLi('current_target_temp', '현재 대표 목표온도 (°C)')}
              </ul>
            </div>

            <!-- [3행 - 7] 생리 및 결로 안전 파생변수 -->
            <div class="setup-cat-box">
              <div class="setup-cat-head">
                <span class="setup-cat-num">7</span>
                <span class="setup-cat-name">생리 및 결로 안전 파생변수</span>
                <span class="setup-cat-count highlight">7개</span>
              </div>
              <ul class="setup-item-list">
                ${renderTooltipLi('current_vpd', '현재 VPD (Vapor Pressure Deficit, kPa)')}
                ${renderTooltipLi('current_dew_point', '현재 이슬점 (Dew Point Temperature, °C)')}
                ${renderTooltipLi('current_cond_margin', '현재 결로 여유 (Condensation Margin, °C)')}
                ${renderTooltipLi('high_humidity_duration', '최근 1시간 고습 지속시간')}
                ${renderTooltipLi('low_vpd_duration', '최근 1시간 저VPD 지속시간')}
                ${renderTooltipLi('pred_cond_margin_10m', '10분 후 예측 결로 여유')}
                ${renderTooltipLi('cond_risk_10m', '10분 후 결로 위험 여부')}
              </ul>
            </div>

            <!-- [3행 - 8] 광·에너지 균형 파생변수 -->
            <div class="setup-cat-box">
              <div class="setup-cat-head">
                <span class="setup-cat-num">8</span>
                <span class="setup-cat-name">광·에너지 균형 파생변수</span>
                <span class="setup-cat-count highlight">6개</span>
              </div>
              <ul class="setup-item-list">
                ${renderTooltipLi('dli', '누적 일사량 (DLI)')}
                ${renderTooltipLi('solar_acc', '측정 일사 누적값')}
                ${renderTooltipLi('clear_sky_solar_acc', '청천 일사 누적값')}
                ${renderTooltipLi('solar_ratio', '측정/청천 일사 비율')}
                ${renderTooltipLi('solar_target_eta', '목표 일사 도달 예상시간')}
                ${renderTooltipLi('vent_heat_loss_proxy', '환기 열손실 proxy')}
              </ul>
            </div>

            <!-- [3행 - 9] 물리 안전 및 기구 한계 파생변수 -->
            <div class="setup-cat-box">
              <div class="setup-cat-head">
                <span class="setup-cat-num">9</span>
                <span class="setup-cat-name">물리 안전 및 기구 한계 파생변수</span>
                <span class="setup-cat-count highlight">6개</span>
              </div>
              <ul class="setup-item-list">
                ${renderTooltipLi('temp_diff_in_out', '실내외 온도차 (T<sub>in</sub> − T<sub>out</sub>)')}
                ${renderTooltipLi('day_min_vent', '주간 최소 환기율')}
                ${renderTooltipLi('night_min_vent', '야간 최소 환기율')}
                ${renderTooltipLi('control_slew_rate_limit', '제어 변화 제한값')}
                ${renderTooltipLi('hdm', '난방도분 (Heating Degree Minutes)')}
                ${renderTooltipLi('cdm', '냉방도분 (Cooling Degree Minutes)')}
              </ul>
            </div>

            <!-- [4행 - 10] 직전 실행 action (상태 이력) -->
            <div class="setup-cat-box" style="border: 1.5px solid #c7d2fe; background: #f5f3ff;">
              <div class="setup-cat-head">
                <span class="setup-cat-num" style="background: #6366f1;">10</span>
                <span class="setup-cat-name" style="color: #4338ca;">직전 실행 Action (a<sub>t-1</sub>)</span>
                <span class="setup-cat-count" style="background: #e0e7ff; color: #3730a3;">6개</span>
              </div>
              <div style="font-size: 0.76rem; font-weight: 700; color: #4f46e5; margin-bottom: 0.35rem;">
                ⏱️ 상태 이력(State History)으로 관측(s<sub>t</sub>)에 포함
              </div>
              <ul class="setup-item-list">
                ${renderTooltipLi('prev_vent_opening', '직전 천창 개도율 (%)')}
                ${renderTooltipLi('prev_shade_screen', '직전 차광스크린 개도율 (%)')}
                ${renderTooltipLi('prev_thermal_screen', '직전 보온스크린 개도율 (%)')}
                ${renderTooltipLi('prev_heater', '직전 난방 작동 여부 (0/1)')}
                ${renderTooltipLi('prev_cooler', '직전 냉방 작동 여부 (0/1)')}
                ${renderTooltipLi('prev_circ_fan', '직전 순환팬 작동 여부 (0/1)')}
              </ul>
            </div>

            <!-- [4행 - 11] 현재 적용할 action (제어 입력) -->
            <div class="setup-cat-box setup-cat-box-accent">
              <div class="setup-cat-head">
                <span class="setup-cat-num" style="background: #059669;">11</span>
                <span class="setup-cat-name" style="color: #065f46;">현재 적용할 Action (a<sub>t</sub>)</span>
                <span class="setup-cat-count" style="background: #a7f3d0; color: #064e3b;">6개</span>
              </div>
              <div style="font-size: 0.76rem; font-weight: 700; color: #047857; margin-bottom: 0.35rem;">
                🎯 5분 뒤 환경변화(Δŷ<sub>t+1</sub>)를 유도하는 핵심 제어 조건
              </div>
              <ul class="setup-item-list">
                ${renderTooltipLi('curr_vent_opening', '<strong>천창 개도율</strong> (Continuous 0~100%)')}
                ${renderTooltipLi('curr_shade_screen', '<strong>차광스크린 개도율</strong> (Continuous 0~100%)')}
                ${renderTooltipLi('curr_thermal_screen', '<strong>보온스크린 개도율</strong> (Continuous 0~100%)')}
                ${renderTooltipLi('curr_heater', '<strong>난방 작동 여부</strong> (Binary 0/1)')}
                ${renderTooltipLi('curr_cooler', '<strong>냉방 작동 여부</strong> (Binary 0/1)')}
                ${renderTooltipLi('curr_circ_fan', '<strong>순환팬 작동 여부</strong> (Binary 0/1)')}
              </ul>
            </div>

            <!-- [4행 - 12] Quality Flags -->
            <div class="setup-cat-box" style="border: 1.5px dashed #93c5fd; background: #eff6ff;">
              <div class="setup-cat-head">
                <span class="setup-cat-num" style="background: #3b82f6;">12</span>
                <span class="setup-cat-name" style="color: #1e40af;">Quality Flag (선택적)</span>
                <span class="setup-cat-count" style="background: #bfdbfe; color: #1e3a8a;">2개</span>
              </div>
              <div style="font-size: 0.76rem; font-weight: 700; color: #2563eb; margin-bottom: 0.35rem;">
                🏷️ With-flags variant에만 상태 관측(s<sub>t</sub>)에 추가
              </div>
              <ul class="setup-item-list" style="color: #1e3a8a;">
                ${renderTooltipLi('imputation_flag', '결측값 대체 여부 (Imputation Flag, 0/1)')}
                ${renderTooltipLi('outlier_flag', '이상치 탐지 여부 (Outlier Flag, 0/1)')}
              </ul>
            </div>

          </div>
        </div>

        <!-- 4. Model Output & Multi-Horizon Rollout Setup -->
        <div class="setup-bottom-grid">
          
          <!-- Outputs Card -->
          <div class="setup-output-card">
            <div class="setup-section-title">
              <span>📤</span> 예측 대상 출력 변수 (3 Targets)
            </div>
            <p style="font-size: 0.85rem; color: #475569; margin: 0.5rem 0 1rem 0;">
              모델은 <strong>현재 적용할 제어 action(a<sub>t</sub>)</strong>이 인가되었을 때, 다음 세 변수의 <strong>5분간 변화량</strong>을 각각 독립적으로 예측합니다:
            </p>
            <div class="setup-target-items">
              <div class="setup-target-badge">
                <span class="target-icon">🌡️</span>
                <div class="target-info">
                  <div class="target-name">실내 온도 변화량 (ΔT)</div>
                  <div class="target-unit">단위: °C (5분 변화량)</div>
                </div>
              </div>
              <div class="setup-target-badge">
                <span class="target-icon">💧</span>
                <div class="target-info">
                  <div class="target-name">실내 상대습도 변화량 (ΔRH)</div>
                  <div class="target-unit">단위: % (5분 변화량)</div>
                </div>
              </div>
              <div class="setup-target-badge">
                <span class="target-icon">🫧</span>
                <div class="target-info">
                  <div class="target-name">실내 CO₂ 농도 변화량 (ΔCO₂)</div>
                  <div class="target-unit">단위: ppm (5분 변화량)</div>
                </div>
              </div>
            </div>

            <div style="font-size: 0.82rem; color: #64748b; margin-top: 1rem; line-height: 1.5; background: #f8fafc; padding: 0.75rem; border-radius: var(--radius-sm); border: 1px solid #e2e8f0;">
              💡 <strong>입력 특징 요약</strong>: 전체 입력 \(x_t = [s_t, a_t]\)는 단순 센서값만이 아니라 <strong>현재 환경 및 생육 맥락, 파생 안전지표, 직전 action(상태 이력 \(a_{t-1}\)), 현재 적용할 action(\(a_t\))</strong>을 함께 포함합니다.
            </div>
          </div>

          <!-- Multi-Horizon Rollout Evaluation Card -->
          <div class="setup-rollout-card">
            <div class="setup-section-title">
              <span>🔄</span> 다단계 재귀 롤아웃 및 HPO 목적함수
            </div>
            <p style="font-size: 0.85rem; color: #475569; margin: 0.5rem 0 0.75rem 0;">
              단기 1-step 예측에 과적합되지 않도록, 모델 출력을 차기 입력으로 순환 투입하는 
              <strong>15분(3-step), 30분(6-step), 60분(12-step) 재귀 롤아웃</strong>을 수행하여 누적 오차와 안정성을 평가합니다.
            </p>
            <div class="setup-loss-box">
              <div style="font-size: 0.78rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;">
                📐 Official HPO Multi-Horizon Loss Function:
              </div>
              <div style="font-family: var(--font-family-mono); font-size: 0.84rem; color: #047857; font-weight: 700;">
                Loss = 0.10×OneStep(5m) + 0.10×Rollout(15m) + 0.25×Rollout(30m) + 0.45×Rollout(60m) + 0.10×DriftSlope
              </div>
            </div>
          </div>

        </div>

        <!-- 5. Physical Feasibility Bounds (변수별 물리 허용 범위) -->
        <div class="setup-table-card">
          <div class="setup-section-header">
            <div>
              <div class="setup-section-title">
                <span>🛡️</span> 변수별 물리 허용 범위 (Physical Feasibility Bounds)
              </div>
              <div style="font-size: 0.82rem; color: #64748b; font-weight: 500; margin-top: 0.2rem;">
                다단계 재귀 롤아웃 시 예측값이 물리 한계를 벗어날 경우 <strong>Safety Breach</strong>로 판정되며 배포 부적격 처리됩니다.
              </div>
            </div>
            <div style="font-size: 0.78rem; background: #fee2e2; color: #991b1b; padding: 0.25rem 0.65rem; border-radius: 9999px; font-weight: 700; border: 1px solid #fecaca;">
              Safety Gate 허용 기준
            </div>
          </div>

          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-top: 0.5rem;">
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #ef4444; border-radius: var(--radius-md); padding: 1.1rem; box-shadow: var(--shadow-sm);">
              <div style="font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase;">실내 온도 (Indoor Temperature)</div>
              <div style="font-size: 1.45rem; font-weight: 800; font-family: var(--font-family-mono); color: #dc2626; margin: 0.35rem 0;">
                −60 ~ 80 °C
              </div>
              <div style="font-size: 0.78rem; color: #475569;">온실 내부 열역학적 물리 한계 범위</div>
            </div>

            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6; border-radius: var(--radius-md); padding: 1.1rem; box-shadow: var(--shadow-sm);">
              <div style="font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase;">실내 상대습도 (Indoor Humidity)</div>
              <div style="font-size: 1.45rem; font-weight: 800; font-family: var(--font-family-mono); color: #2563eb; margin: 0.35rem 0;">
                0 ~ 100 %
              </div>
              <div style="font-size: 0.78rem; color: #475569;">포화 수증기압 상한 및 건조 물리 제약</div>
            </div>

            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #10b981; border-radius: var(--radius-md); padding: 1.1rem; box-shadow: var(--shadow-sm);">
              <div style="font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase;">실내 CO₂ 농도 (Indoor CO₂)</div>
              <div style="font-size: 1.45rem; font-weight: 800; font-family: var(--font-family-mono); color: #059669; margin: 0.35rem 0;">
                0 ~ 10,000 ppm
              </div>
              <div style="font-size: 0.78rem; color: #475569;">CO₂ 시비 및 센서 측정 가능 한계 범위</div>
            </div>
          </div>
        </div>

      </div>
    `;

    this.bindEvents();
  }

  bindEvents() {
    if (!this.container) return;

    const items = this.container.querySelectorAll('.setup-item-with-tooltip');
    const clearAllOpen = () => {
      this.container?.querySelectorAll('.setup-item-with-tooltip.is-open').forEach(el => {
        el.classList.remove('is-open');
        el.setAttribute('aria-expanded', 'false');
      });
      this.container?.querySelectorAll('.setup-cat-box.has-open-tooltip').forEach(box => {
        box.classList.remove('has-open-tooltip');
      });
    };

    items.forEach(item => {
      item.addEventListener('click', (e) => {
        // If clicking inside popover, do nothing
        if (e.target.closest('.setup-tooltip-popover')) {
          return;
        }
        const wasOpen = item.classList.contains('is-open');
        const parentBox = item.closest('.setup-cat-box');
        clearAllOpen();
        if (!wasOpen) {
          item.classList.add('is-open');
          item.setAttribute('aria-expanded', 'true');
          parentBox?.classList.add('has-open-tooltip');
        }
      });
    });

    if (!this._hasDocumentListener) {
      this._hasDocumentListener = true;
      document.addEventListener('click', (e) => {
        if (!e.target.closest('.setup-item-with-tooltip')) {
          clearAllOpen();
        }
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          clearAllOpen();
        }
      });
    }
  }
}
