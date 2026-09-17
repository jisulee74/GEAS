/**
 * GEAS Dashboard - Quality Experiment Setup Section
 */

export class QualitySetupSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="section-container">
        <!-- 1. Header -->
        <div class="section-header-card">
          <div class="section-badge-row">
            <span class="badge badge-primary">Experiment Protocol</span>
            <span class="badge badge-warning">Dual Defense Architecture</span>
          </div>
          <h2 class="section-title">⚙️ 데이터 품질 관리 실험 설정 및 변수 분류 체계</h2>
          <p class="section-description">
            스마트 온실 환경 데이터의 물리적 무결성을 보장하기 위해 정의된 실험 프로토콜, 대상 센서 변수 24종(AI 대상 8종 + 규칙 대상 16종), 
            임시 보정 규칙 및 이중 센서 결합(Representative Sensor) 정책 명세입니다.
          </p>
        </div>

        <!-- 2. Experiment Setup Specification Cards -->
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">후보 모델군</div>
            <div class="metric-value text-accent">3 Architectures</div>
            <div class="metric-subtext">ModernTCN · TimesNet · PatchTST</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Lookback Window</div>
            <div class="metric-value text-info">288 Steps</div>
            <div class="metric-subtext">5분 주기 기준 과거 24시간 입력</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">HPO Budget & Patience</div>
            <div class="metric-value text-success">50 Trials · 10 Pat.</div>
            <div class="metric-subtext">Seed 42 · Max 100 Epochs</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">평가 노이즈 주입 설정</div>
            <div class="metric-value text-warning">10% Mask · 8σ</div>
            <div class="metric-subtext">방향 균형화 & 도메인 범위 내 Cap</div>
          </div>
        </div>

        <!-- 3. Target Scope: 8 AI Targets -->
        <div class="content-card">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <h3 class="card-title" style="margin: 0;">
              <span>🤖</span> 1. AI 딥러닝 기반 탐지 및 마스킹 복원 대상 (8 Targets)
            </h3>
            <span class="badge badge-primary">Core Microclimate & Substrate</span>
          </div>
          <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
            온실 내부 환경제어의 직접적 상태(State)를 구성하는 핵심 변수로, 복잡한 시계열 상호 상관성을 학습하여 이상치 탐지 및 대체 복원을 수행합니다.
          </p>
          <div class="table-responsive-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>변수명 (Column)</th>
                  <th>물리적 측정 항목</th>
                  <th>단위</th>
                  <th>물리 허용 범위</th>
                  <th>센서 위치 및 역할</th>
                  <th>AI 품질 처리 방식</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>in_temp</code></td>
                  <td>온실 실내온도 1 (주 센서)</td>
                  <td>°C</td>
                  <td>-5.0 ~ 55.0</td>
                  <td>구역 중앙 상단 (Primary)</td>
                  <td>이상치 탐지 & 결측 시 딥러닝 복원값 대체</td>
                </tr>
                <tr>
                  <td><code>in_temp2</code></td>
                  <td>온실 실내온도 2 (보조 센서)</td>
                  <td>°C</td>
                  <td>-5.0 ~ 55.0</td>
                  <td>구역 중앙 하단/대칭 (Secondary)</td>
                  <td>이상치 탐지 진단 (대표값 선출 시 2순위 활용)</td>
                </tr>
                <tr>
                  <td><code>in_hum</code></td>
                  <td>온실 실내상대습도 1 (주 센서)</td>
                  <td>%</td>
                  <td>0.0 ~ 100.0</td>
                  <td>구역 중앙 상단 (Primary)</td>
                  <td>이상치 탐지 & 결측 시 딥러닝 복원값 대체</td>
                </tr>
                <tr>
                  <td><code>in_hum2</code></td>
                  <td>온실 실내상대습도 2 (보조 센서)</td>
                  <td>%</td>
                  <td>0.0 ~ 100.0</td>
                  <td>구역 중앙 하단/대칭 (Secondary)</td>
                  <td>이상치 탐지 진단 (대표값 선출 시 2순위 활용)</td>
                </tr>
                <tr>
                  <td><code>in_co2</code></td>
                  <td>온실 실내 CO₂ 농도 1 (주 센서)</td>
                  <td>ppm</td>
                  <td>200.0 ~ 2500.0</td>
                  <td>작물 군락 높이 (Primary)</td>
                  <td>이상치 탐지 & 결측 시 딥러닝 복원값 대체</td>
                </tr>
                <tr>
                  <td><code>in_co2_2</code></td>
                  <td>온실 실내 CO₂ 농도 2 (보조 센서)</td>
                  <td>ppm</td>
                  <td>200.0 ~ 2500.0</td>
                  <td>작물 군락 대칭 높이 (Secondary)</td>
                  <td>이상치 탐지 진단 (대표값 선출 시 2순위 활용)</td>
                </tr>
                <tr>
                  <td><code>in_medium_temp1</code></td>
                  <td>배지/근권 온도 1</td>
                  <td>°C</td>
                  <td>0.0 ~ 45.0</td>
                  <td>슬래브 배지 내부 1구역</td>
                  <td>이상치 탐지 & 결측 시 딥러닝 복원값 대체</td>
                </tr>
                <tr>
                  <td><code>in_medium_hum1</code></td>
                  <td>배지/근권 수분함량 1</td>
                  <td>%</td>
                  <td>0.0 ~ 100.0</td>
                  <td>슬래브 배지 내부 1구역</td>
                  <td>이상치 탐지 & 결측 시 딥러닝 복원값 대체</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- 4. Target Scope: 16 Rule-based Targets -->
        <div class="content-card">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <h3 class="card-title" style="margin: 0;">
              <span>📏</span> 2. 규칙 기반(Rule-based) 이상치 판정 대상 (16 Targets)
            </h3>
            <span class="badge badge-neutral">reference/offline/scripts/03_control_quality.py RULE_BASED_OUTLIER_COLUMNS</span>
          </div>
          <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
            센서 하드웨어 결함, 단선, 통신 두절, 물리적 한계 초과 등을 탐지하기 위해 하드코딩된 규칙 검사 대상입니다.
          </p>
          <div class="table-responsive-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>분류</th>
                  <th>변수군 (Columns)</th>
                  <th>검사 규칙 및 임계값 조건</th>
                  <th>위반 시 처리</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>실내 환경 센서</strong></td>
                  <td><code>in_temp</code>, <code>in_temp2</code>, <code>in_hum</code>, <code>in_hum2</code>, <code>in_co2</code>, <code>in_co2_2</code></td>
                  <td>도메인 최소/최대 물리 임계값 초과 및 1스텝(5분) 급변 한계 초과</td>
                  <td><code>rule_outlier_flag = True</code> 부여 후 AI 복원 또는 센서2 대체</td>
                </tr>
                <tr>
                  <td><strong>배지/근권 환경 센서</strong></td>
                  <td><code>in_medium_temp1</code>, <code>in_medium_temp2</code>, <code>in_medium_hum1</code>, <code>in_medium_hum2</code></td>
                  <td>배지 생육 한계(-5~50°C, 0~100%) 초과 검사 (Sensor 2는 감사용 보존)</td>
                  <td>규칙 플래그 부여 및 Primary 센서 결측 복원 후보군 진입</td>
                </tr>
                <tr>
                  <td><strong>외부 기상 환경 센서</strong></td>
                  <td><code>out_temp</code>, <code>out_windsp</code>, <code>out_winddirec</code>, <code>out_rain</code>, <code>out_light</code>, <code>out_light_sum</code></td>
                  <td>풍속(0~60m/s), 일사량(>=0 W/m²), 누적일사(단조증가), 강우(0/1)</td>
                  <td>비AI 임시 보정 규칙(선형/각도/유지) 적용</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- 5. Non-AI Provisional Imputation Rules -->
        <div class="content-card">
          <h3 class="card-title">
            <span>⏱️</span> 3. 비AI 외부 기상변수 임시 단기 보정 규칙 (Provisional Short-Gap Imputation)
          </h3>
          <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
            외부 기상 및 PLC 상태 변수는 계산 복잡도를 낮추고 데이터 유실을 방지하기 위해 <strong>최대 3개 연속 행(15분 이하)</strong> 결측에 한하여 다음 규칙으로 보정됩니다.
            3개 행을 초과하는 결측은 인위적으로 보외(extrapolate)하지 않고 결측 상태를 보존합니다.
          </p>
          <div class="setup-grid">
            <div class="setup-box">
              <div class="setup-box-title">양방향 시간 선형보간 (Two-Sided Linear)</div>
              <div class="setup-box-content">
                <strong>대상:</strong> <code>out_temp</code>, <code>out_hum</code>, <code>out_windsp</code>, <code>out_rainfall</code>, <code>out_airpress</code><br>
                <strong>조건:</strong> 결측 구간 양 끝단에 정상 관측값이 존재할 때 시간 가중치 기반 선형 보간.
              </div>
            </div>

            <div class="setup-box">
              <div class="setup-box-title">원형 각도 보간 (Circular Interpolation)</div>
              <div class="setup-box-content">
                <strong>대상:</strong> <code>out_winddirec</code> (풍향, 0° ~ 360°)<br>
                <strong>조건:</strong> 360° 경계 불연속성(예: 359°와 1°)을 방지하기 위해 각도 삼각함수 벡터 공간(sin/cos)에서 가중 보간 후 역변환.
              </div>
            </div>

            <div class="setup-box">
              <div class="setup-box-title">비음수 제한 보간 (Bounded Interpolation)</div>
              <div class="setup-box-content">
                <strong>대상:</strong> <code>out_light</code> (순간일사량), <code>out_light_sum</code> (누적일사량)<br>
                <strong>조건:</strong> 일사량은 물리적으로 0 미만이 될 수 없으므로, 보간 후 <code>max(0.0, value)</code> 음수 절단(Cap) 적용.
              </div>
            </div>

            <div class="setup-box">
              <div class="setup-box-title">직전 상태 유지 (State Persistence)</div>
              <div class="setup-box-content">
                <strong>대상:</strong> <code>out_rain</code> (강우 감지 여부), <code>etc_plc_norm</code> (PLC 정상 통신 플래그)<br>
                <strong>조건:</strong> 이산적(Binary) 이벤트 상태이므로 보간을 금지하고 직전 유효 상태를 전방 채움(Forward-fill, 최대 3행).
              </div>
            </div>
          </div>
        </div>

        <!-- 6. Dual Sensor Failover Policy -->
        <div class="content-card">
          <h3 class="card-title">
            <span>🛡️</span> 4. 다중 센서 결합 및 대표 센서값(Representative Sensor) 합성 정책
          </h3>
          <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
            온실 내 동일 물리량에 대해 2개 센서가 설치된 경우, 강화학습 제어기(MDP State)에 인가할 단일 신뢰값을 선출하는 우선순위 메커니즘입니다.
          </p>
          <div class="table-responsive-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>대표 변수명</th>
                  <th>1순위 (Primary)</th>
                  <th>2순위 (Failover Secondary)</th>
                  <th>3순위 (AI Imputation Fallback)</th>
                  <th>최종 상태 (Unavailable)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>in_temp_representative</code></td>
                  <td><code>in_temp</code> (정상 시)</td>
                  <td><code>in_temp2</code> (센서1 이상 시)</td>
                  <td><code>in_temp</code>의 AI 복원값 (두 센서 모두 이상 시)</td>
                  <td><code>NaN</code> (미해결 상태 플래그)</td>
                </tr>
                <tr>
                  <td><code>in_hum_representative</code></td>
                  <td><code>in_hum</code> (정상 시)</td>
                  <td><code>in_hum2</code> (센서1 이상 시)</td>
                  <td><code>in_hum</code>의 AI 복원값 (두 센서 모두 이상 시)</td>
                  <td><code>NaN</code> (미해결 상태 플래그)</td>
                </tr>
                <tr>
                  <td><code>in_co2_representative</code></td>
                  <td><code>in_co2</code> (정상 시)</td>
                  <td><code>in_co2_2</code> (센서1 이상 시)</td>
                  <td><code>in_co2</code>의 AI 복원값 (두 센서 모두 이상 시)</td>
                  <td><code>NaN</code> (미해결 상태 플래그)</td>
                </tr>
                <tr>
                  <td><code>in_medium_temp_representative</code></td>
                  <td><code>in_medium_temp1</code> (정상 시)</td>
                  <td>- (배지2 센서 비활성 기간 장기화로 제외)</td>
                  <td><code>in_medium_temp1</code>의 AI 복원값</td>
                  <td><code>NaN</code> (미해결 상태 플래그)</td>
                </tr>
                <tr>
                  <td><code>in_medium_hum_representative</code></td>
                  <td><code>in_medium_hum1</code> (정상 시)</td>
                  <td>- (배지2 센서 비활성 기간 장기화로 제외)</td>
                  <td><code>in_medium_hum1</code>의 AI 복원값</td>
                  <td><code>NaN</code> (미해결 상태 플래그)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;
  }

  update() {
    this.render();
  }
}
