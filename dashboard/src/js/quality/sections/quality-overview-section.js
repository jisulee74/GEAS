/**
 * GEAS Dashboard - Quality Overview & Pipeline Workflow Section
 */

import { qualityDataLoader } from '../quality-data-loader.js';
import { qualityFilterStore } from '../quality-filter-store.js';
import { DataTable } from '../../components/data-table.js';
import { ChartRenderer } from '../../components/chart-renderer.js';
import { QualityCharts } from '../quality-charts.js';

export class QualityOverviewSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.table = null;
    this.chart = null;
  }

  async render() {
    if (!this.container) return;
    const filterState = qualityFilterStore.getState();
    const data = qualityDataLoader.getSummaryResults(filterState);
    const summaryData = await QualityCharts.getSummaryResults(filterState.split);

    this.container.innerHTML = `
      <div class="section-container">
        <!-- 1. Header & Executive Metrics -->
        <div class="section-header-card">
          <div class="section-badge-row">
            <span class="badge badge-primary">Data Quality Pipeline</span>
            <span class="badge badge-success">Split: ${filterState.split === 'test' ? 'Test (Holdout)' : 'Validation (Primary)'}</span>
            <span class="badge badge-neutral">3 Crops × 3 SOTA Architectures</span>
          </div>
          <h2 class="section-title">🌿 스마트 온실 데이터 품질 관리 파이프라인 개요 및 평가 종합</h2>
          <p class="section-description">
            GEAS 3.5 데이터 품질 관리 파이프라인은 복합 온실 환경 센서 데이터의 물리적 한계 및 비선형성을 보정하기 위해
            <strong>규칙 기반 검사(16개 물리 규칙)</strong>와 <strong>시계열 딥러닝 기반 이상치 탐지 및 마스킹 복원(8개 핵심 환경 변수)</strong>을 결합한
            하이브리드 전처리 아키텍처입니다. HPO 탐색과 임계값 보정을 거친 후보 모델군(ModernTCN, TimesNet, PatchTST)의 종합 평가 결과를 제공합니다.
          </p>
        </div>

        <!-- 2. Executive KPI Highlights -->
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">최고 이상치 탐지 F1-Score</div>
            <div class="metric-value text-accent" id="overview-kpi-best-f1">-</div>
            <div class="metric-subtext" id="overview-kpi-best-f1-desc">합성 이상치(8σ) 대상 탐지 최적 모델</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">최저 재구성 오차 (RMSE)</div>
            <div class="metric-value text-success" id="overview-kpi-best-rmse">-</div>
            <div class="metric-subtext" id="overview-kpi-best-rmse-desc">10% 무작위 마스킹 복원 성능</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">평균 1행 추론 지연시간</div>
            <div class="metric-value text-info" id="overview-kpi-avg-latency">-</div>
            <div class="metric-subtext">온실 제어 주기(5분=300,000ms) 대비 0.0001% 미만</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">최대 메모리 사용량 (Peak RAM)</div>
            <div class="metric-value text-warning" id="overview-kpi-peak-mem">-</div>
            <div class="metric-subtext">임베디드 엣지 게이트웨이 구동 안정권</div>
          </div>
        </div>

        <!-- 3. Pipeline Architecture Interactive Diagram -->
        <div class="content-card">
          <h3 class="card-title">
            <span>🔄</span> GEAS 3.5 하이브리드 품질 관리 6단계 파이프라인
          </h3>
          <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 1.25rem;">
            원천 센서 데이터 인입부터 MDP 강화학습 입력용 <code>대표 센서값(Representative Sensor)</code> 합성까지의 6단계 전처리 흐름입니다.
          </p>
          
          <div class="pipeline-workflow-grid">
            <div class="pipeline-step-card">
              <div class="step-num">Phase 1</div>
              <h4 class="step-title">입력 스키마 정리 및 단위 통일</h4>
              <ul class="step-list">
                <li>시간축 5분 격자 정렬 (Resampling)</li>
                <li>온도(°C), 습도(%), CO₂(ppm) 물리 단위 표준화</li>
                <li>컬럼명 및 도메인 데이터 스키마 표준 정의</li>
                <li>생육 단계(Growth Stage) 컨텍스트 병합</li>
              </ul>
            </div>

            <div class="pipeline-step-card">
              <div class="step-num">Phase 2</div>
              <h4 class="step-title">결측 탐지 및 이력 기록</h4>
              <ul class="step-list">
                <li>환경 상태 변수와 제어 변수의 결측 여부 검사</li>
                <li>변수별 <code>missing_flag</code> 기록</li>
                <li>복원 전 결측 이력과 복원 여부를 구분해 보존</li>
                <li>기반 함수: <code>add_missing_flags</code>, <code>prepare_missing_features</code></li>
              </ul>
            </div>

            <div class="pipeline-step-card highlight-step">
              <div class="step-num">Phase 3</div>
              <h4 class="step-title">제어로그 기반 제어값 복원</h4>
              <ul class="step-list">
                <li>결측인 제어값만 동일 시각의 제어로그로 복원</li>
                <li>기존 값이 있는 제어 항목은 유지</li>
                <li>동일 시각의 대응 로그가 없으면 결측 유지</li>
                <li>복원된 항목은 <code>restored_flag</code>로 기록</li>
                <li>환경 센서값의 AI 복원·보간과는 별개 과정임을 명시 (최근 로그 대입·시간 보간·AI 예측 배제)</li>
                <li>기반 함수: <code>restore_action_columns_from_control_log</code></li>
              </ul>
            </div>

            <div class="pipeline-step-card highlight-step">
              <div class="step-num">Phase 4</div>
              <h4 class="step-title">규칙·AI 기반 이상치 탐지</h4>
              <ul class="step-list">
                <li><strong>16개 규칙 기반 변수:</strong> 물리 허용범위 초과 및 급변(rate of change) 탐지</li>
                <li><strong>8개 AI 기반 변수:</strong> Lookback 288 스텝 시계열 모델(ModernTCN/TimesNet/PatchTST) 재구성 오차 기반 판정</li>
                <li>임계값: <code>injected_validation_scores</code> 기준 보정</li>
              </ul>
            </div>

            <div class="pipeline-step-card">
              <div class="step-num">Phase 5</div>
              <h4 class="step-title">환경 변수 결측·이상치 보정</h4>
              <ul class="step-list">
                <li><strong>8개 핵심 환경 변수:</strong> 딥러닝 모델의 복원값(Imputation)으로 대체</li>
                <li><strong>외부 기상 연속 변수:</strong> 최대 3행(15분) 양방향 시간 선형보간</li>
                <li><strong>풍향(각도):</strong> 360° 원형 각도 보간 / <strong>일사량:</strong> 비음수 바운디드 보간</li>
                <li><strong>강우/PLC:</strong> 직전 상태 유지(Persistence)</li>
                <li>기반 함수: <code>prepare_missing_outliers_handled_features</code></li>
              </ul>
            </div>

            <div class="pipeline-step-card accent-step">
              <div class="step-num">Phase 6</div>
              <h4 class="step-title">대표 센서값 생성</h4>
              <ul class="step-list">
                <li>동일 물리량 다중 센서 결합 (온도 1/2, 습도 1/2, CO₂ 1/2)</li>
                <li><strong>우선순위 1:</strong> 주 센서(Sensor 1) 정상 관측값</li>
                <li><strong>우선순위 2:</strong> 부 센서(Sensor 2) 정상 관측값</li>
                <li><strong>우선순위 3:</strong> Sensor 1 AI 복원값 (Fallback)</li>
                <li>강화학습 상태공간(State)으로 최종 전달</li>
              </ul>
            </div>
          </div>
        </div>

        <!-- 4. Interactive Summary Chart & Table -->
        <div class="two-column-grid">
          <div class="content-card">
            <h3 class="card-title">
              <span>📊</span> 모델별 이상치 탐지 F1-Score 비교
            </h3>
            <div style="height: 320px; position: relative;">
              <canvas id="overview-f1-chart"></canvas>
            </div>
          </div>

          <div class="content-card">
            <h3 class="card-title">
              <span>📉</span> 모델별 10% 마스킹 복원 RMSE 비교
            </h3>
            <div style="height: 320px; position: relative;">
              <canvas id="overview-rmse-chart"></canvas>
            </div>
          </div>
        </div>

        <!-- 5. Full Summary Data Table -->
        <div id="overview-summary-table-container"></div>

        <!-- 6. Multi-Crop Comparative Web Charts (Rendered from CSV) -->
        <div class="two-column-grid">
          <div class="content-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
              <h3 class="card-title" style="margin: 0; border: none;">
                <span>📊</span> 작물별 마스킹 복원 성능 (RMSE ↓)
              </h3>
              <span class="badge badge-primary">${filterState.split === 'test' ? 'Test 데이터' : 'Validation 데이터'}</span>
            </div>
            <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
              실제 <code>${filterState.split}_results.csv</code>의 작물별 10% 마스킹 복원 RMSE 수치입니다. (툴팁: MAE & 마스킹 셀 수)
            </p>
            <div style="height: 300px; position: relative;">
              <canvas id="overview-reconstruction-chart"></canvas>
            </div>
          </div>

          <div class="content-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
              <h3 class="card-title" style="margin: 0; border: none;">
                <span>🎯</span> 작물별 이상치 탐지 분류 성능 (F1 ↑)
              </h3>
              <span class="badge badge-success">${filterState.split === 'test' ? 'Test 데이터' : 'Validation 데이터'}</span>
            </div>
            <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
              실제 <code>${filterState.split}_results.csv</code>의 합성 이상치(8σ) 탐지 F1-Score 수치입니다. (툴팁: Precision/Recall/ROC-AUC)
            </p>
            <div style="height: 300px; position: relative;">
              <canvas id="overview-classification-chart"></canvas>
            </div>
          </div>
        </div>

        <!-- 7. Publication Artifact Reference (Clearly Labeled Static Reference) -->
        <div class="content-card" style="opacity: 0.9;">
          <details>
            <summary style="cursor: pointer; font-size: 0.88rem; font-weight: 700; color: #475569; padding: 0.25rem 0;">
              🖼️ 논문/보고서 제출용 원본 정적 아티팩트 보기 (Static Reference Figures - Non-Interactive)
            </summary>
            <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0.75rem 0;">
              ※ 아래 그림은 원본 고정 정적 이미지(figures/)이며, 실시간 필터 적용 대상이 아닙니다. 실시간 분석은 위의 인터랙티브 웹 차트를 이용하십시오.
            </p>
            <div class="reference-figures-grid">
              <div class="reference-figure-box">
                <div class="figure-label">Reconstruction Performance (Publication PNG)</div>
                <img src="quality_panel_bundle/data/quality/figures/reconstruction_performance.png" alt="Reconstruction Performance" class="ref-img" loading="lazy">
                <div class="figure-caption">작물 및 모델별 마스킹 복원 성능 (RMSE & MAE) 종합 비교</div>
              </div>
              <div class="reference-figure-box">
                <div class="figure-label">Anomaly Detection Ranking (Publication PNG)</div>
                <img src="quality_panel_bundle/data/quality/figures/anomaly_detection_ranking.png" alt="Anomaly Detection Ranking" class="ref-img" loading="lazy">
                <div class="figure-caption">합성 이상치(8σ) 탐지 F1-Score 순위 및 ROC-AUC / PR-AUC 비교</div>
              </div>
            </div>
          </details>
        </div>
      </div>
    `;

    this.updateKPIs(data);
    this.renderCharts(data);
    this.renderTable(data);
    QualityCharts.renderReconstructionPerformanceChart('overview-reconstruction-chart', summaryData, filterState.crop);
    QualityCharts.renderAnomalyClassificationChart('overview-classification-chart', summaryData, filterState.crop);
  }

  updateKPIs(data) {
    if (!data || data.length === 0) return;

    let bestF1 = -1, bestF1Model = '';
    let bestRmse = Infinity, bestRmseModel = '';
    let totalLatency = 0, totalPeakMem = 0;

    data.forEach(d => {
      if (d.f1_score > bestF1) {
        bestF1 = d.f1_score;
        bestF1Model = `${d.crop_label} · ${d.model}`;
      }
      if (d.rmse < bestRmse) {
        bestRmse = d.rmse;
        bestRmseModel = `${d.crop_label} · ${d.model}`;
      }
      totalLatency += d.inference_latency_ms_per_row;
      totalPeakMem = Math.max(totalPeakMem, d.peak_memory_mb);
    });

    const avgLatency = (totalLatency / data.length).toFixed(3);

    const f1El = document.getElementById('overview-kpi-best-f1');
    const f1DescEl = document.getElementById('overview-kpi-best-f1-desc');
    const rmseEl = document.getElementById('overview-kpi-best-rmse');
    const rmseDescEl = document.getElementById('overview-kpi-best-rmse-desc');
    const latEl = document.getElementById('overview-kpi-avg-latency');
    const memEl = document.getElementById('overview-kpi-peak-mem');

    if (f1El) f1El.textContent = bestF1.toFixed(4);
    if (f1DescEl) f1DescEl.textContent = `${bestF1Model}`;
    if (rmseEl) rmseEl.textContent = bestRmse.toFixed(3);
    if (rmseDescEl) rmseDescEl.textContent = `${bestRmseModel}`;
    if (latEl) latEl.textContent = `${avgLatency} ms/행`;
    if (memEl) memEl.textContent = `${totalPeakMem.toFixed(1)} MB`;
  }

  renderCharts(data) {
    if (!data || data.length === 0) return;

    const labels = data.map(d => `${d.crop.toUpperCase().slice(0, 3)} - ${d.model}`);
    const f1Scores = data.map(d => d.f1_score);
    const rocScores = data.map(d => d.roc_auc);
    const rmseScores = data.map(d => d.rmse);
    const maeScores = data.map(d => d.mae);

    // 1. F1 & ROC-AUC Chart
    ChartRenderer.getOrCreateChart('overview-f1-chart', {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'F1-Score',
            data: f1Scores,
            backgroundColor: '#4f46e5',
            borderRadius: 4
          },
          {
            label: 'ROC-AUC',
            data: rocScores,
            backgroundColor: '#06b6d4',
            borderRadius: 4
          }
        ]
      },
      options: {
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              afterBody: (context) => {
                const idx = context[0].dataIndex;
                const row = data[idx];
                return [
                  `Precision: ${row.precision.toFixed(4)}`,
                  `Recall: ${row.recall.toFixed(4)}`,
                  `MCC: ${row.mcc.toFixed(4)}`
                ];
              }
            }
          }
        },
        scales: {
          y: { min: 0.6, max: 1.0, title: { display: true, text: 'Score' } }
        }
      }
    });

    // 2. RMSE & MAE Chart
    ChartRenderer.getOrCreateChart('overview-rmse-chart', {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'RMSE',
            data: rmseScores,
            backgroundColor: '#f59e0b',
            borderRadius: 4
          },
          {
            label: 'MAE',
            data: maeScores,
            backgroundColor: '#10b981',
            borderRadius: 4
          }
        ]
      },
      options: {
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              afterBody: (context) => {
                const idx = context[0].dataIndex;
                const row = data[idx];
                return [`Masked Sample Cells: ${row.masked_cells.toLocaleString()}`];
              }
            }
          }
        },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: '오차값 (단위: 변수별 복합)' } }
        }
      }
    });
  }

  renderTable(data) {
    const columns = [
      { key: 'crop_label', label: '작물 (Crop)', sortable: true },
      { key: 'model', label: '모델 (Model)', sortable: true },
      { key: 'split', label: '평가 분할', sortable: true, format: v => v.toUpperCase() },
      { key: 'f1_score', label: 'F1-Score', sortable: true, format: v => v.toFixed(4) },
      { key: 'precision', label: 'Precision', sortable: true, format: v => v.toFixed(4) },
      { key: 'recall', label: 'Recall', sortable: true, format: v => v.toFixed(4) },
      { key: 'mcc', label: 'MCC', sortable: true, format: v => v.toFixed(4) },
      { key: 'fpr', label: 'FPR', sortable: true, format: v => v.toFixed(5) },
      { key: 'roc_auc', label: 'ROC-AUC', sortable: true, format: v => v.toFixed(4) },
      { key: 'pr_auc', label: 'PR-AUC', sortable: true, format: v => v.toFixed(4) },
      { key: 'rmse', label: 'RMSE', sortable: true, format: v => v.toFixed(3) },
      { key: 'mae', label: 'MAE', sortable: true, format: v => v.toFixed(3) },
      { key: 'inference_latency_ms_per_row', label: '지연시간 (ms/행)', sortable: true, format: v => v.toFixed(3) },
      { key: 'peak_memory_mb', label: 'Peak RAM (MB)', sortable: true, format: v => v.toFixed(1) },
      { key: 'model_size_mb', label: '모델 크기 (MB)', sortable: true, format: v => v.toFixed(3) },
      { key: 'injected_cells', label: '이상치 주입 수', sortable: true, format: v => v.toLocaleString() },
      { key: 'masked_cells', label: '마스킹 평가 수', sortable: true, format: v => v.toLocaleString() }
    ];

    this.table = new DataTable({
      containerId: 'overview-summary-table-container',
      title: '데이터 품질 관리 후보 모델 종합 평가표 (Integrated Results)',
      columns,
      data,
      pageSize: 10,
      initialSort: { key: 'f1_score', dir: 'desc' }
    });
  }

  update() {
    this.render();
  }
}
