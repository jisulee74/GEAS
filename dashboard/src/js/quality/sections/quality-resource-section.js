/**
 * GEAS Dashboard - Quality Resource Efficiency Section
 */

import { qualityDataLoader, QualityDataLoader } from '../quality-data-loader.js';
import { qualityFilterStore } from '../quality-filter-store.js';
import { DataTable } from '../../components/data-table.js';
import { QualityCharts } from '../quality-charts.js';

export class QualityResourceSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.table = null;
    this.activeMetric = 'inference_latency_ms_per_row';
  }

  async render() {
    if (!this.container) return;
    const filterState = qualityFilterStore.getState();
    const benchmarks = qualityDataLoader.getResourceMetrics(filterState);
    const summaryData = await QualityCharts.getSummaryResults(filterState.split);

    this.container.innerHTML = `
      <div class="section-container">
        <!-- 1. Header -->
        <div class="section-header-card">
          <div class="section-badge-row">
            <span class="badge badge-primary">Edge Benchmark</span>
            <span class="badge badge-success">Split: ${filterState.split === 'test' ? 'Test' : 'Validation'}</span>
            <span class="badge badge-info">Real-time Feasibility</span>
          </div>
          <h2 class="section-title">⚡ 모델 자원 효율성 및 실시간 연산 속도 (Computational Efficiency)</h2>
          <p class="section-description">
            스마트 온실 현장 엣지 제어기(Edge Gateway / Industrial PC)에서의 실시간 구동 적합성을 검증하기 위해 
            단일 관측치(1행) 추론 지연시간(Latency), 최대 메모리 점유량(Peak RAM), 모델 가중치 크기(Storage Size)를 정밀 계측한 결과입니다.
          </p>
        </div>

        <!-- 2. Real-time Feasibility Analysis Banner -->
        <div class="disclaimer-banner" style="background: #eff6ff; border-left: 4px solid #3b82f6; color: #1e40af; margin-bottom: 1.25rem;">
          <div>
            <strong>💡 현장 실시간 제어 적용성(Feasibility) 분석:</strong><br>
            스마트 온실 GEAS 제어 파이프라인의 1회 제어 루프 주기는 <strong>5분(300,000 ms)</strong>입니다. 
            후보 딥러닝 모델들의 1행 추론 지연시간은 평균 <strong>0.28ms ~ 0.37ms</strong>로, 5분 제어 예산의 <strong>약 0.0001%</strong>에 불과합니다.
            또한 최대 메모리 소모량 역시 <strong>50MB ~ 220MB</strong> 수준으로 임베디드 엣지 디바이스(Raspberry Pi, Jetson Nano 등)에서도 완벽하게 안정적으로 동작 가능합니다.
          </div>
        </div>

        <!-- 3. Interactive Web Chart (CSV/JSON rendered) -->
        <div class="content-card" style="margin-bottom: 1.25rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;">
            <div>
              <h3 class="card-title" style="margin: 0;">
                <span>📊</span> 엣지 자원 효율성 인터랙티브 비교 (Edge Resource Efficiency Chart)
              </h3>
              <p class="section-description" style="margin: 0.25rem 0 0 0; font-size: 0.85rem;">
                실제 벤치마크 계측 CSV(validation/test_results.csv) 수치 기반 동적 렌더링. 지표를 전환하여 모델별 자원 점유를 다각도로 비교하세요.
              </p>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">
              <span style="font-size: 0.82rem; font-weight: 600; color: #4b5563;">지표 선택:</span>
              <button id="btn-metric-latency" class="qc-btn ${this.activeMetric === 'inference_latency_ms_per_row' ? 'qc-btn-primary' : 'qc-btn-outline'}">⏱️ 추론 지연시간 (ms/row)</button>
              <button id="btn-metric-memory" class="qc-btn ${this.activeMetric === 'peak_memory_mb' ? 'qc-btn-primary' : 'qc-btn-outline'}">💾 Peak RAM (MB)</button>
              <button id="btn-metric-size" class="qc-btn ${this.activeMetric === 'model_size_mb' ? 'qc-btn-primary' : 'qc-btn-outline'}">📦 모델 크기 (MB)</button>
            </div>
          </div>
          <div style="height: 380px; position: relative;">
            <canvas id="resource-interactive-efficiency-chart"></canvas>
          </div>
        </div>

        <!-- 4. Detailed Data Table -->
        <div id="resource-benchmark-table-container" style="margin-bottom: 1.25rem;"></div>

        <!-- 5. Academic Reference Artifacts (Collapsible) -->
        <details class="content-card" style="margin-bottom: 1rem;">
          <summary style="font-weight: 600; cursor: pointer; color: #4b5563; outline: none; padding: 0.5rem 0;">
            <span>🖼️</span> 논문/보고서 참고용 정적 그림 보기 (Static Reference Artifact)
          </summary>
          <div class="reference-figures-grid" style="grid-template-columns: 1fr; margin-top: 1rem;">
            <div class="reference-figure-box" style="max-width: 800px; margin: 0 auto;">
              <div class="figure-label">Reference: computational_efficiency.png</div>
              <img src="quality_panel_bundle/data/quality/figures/computational_efficiency.png" alt="Computational Efficiency" class="ref-img" loading="lazy">
              <div class="figure-caption">과거 생성된 정적 벤치마크 플롯 (비교 및 검증용 참조 자료)</div>
            </div>
          </div>
        </details>
      </div>
    `;

    this.attachEvents(summaryData);
    this.renderChart(summaryData);
    this.renderTable(benchmarks);
  }

  attachEvents(summaryData) {
    const btnLatency = document.getElementById('btn-metric-latency');
    const btnMem = document.getElementById('btn-metric-memory');
    const btnSize = document.getElementById('btn-metric-size');

    const updateActiveButton = (metric) => {
      this.activeMetric = metric;
      [btnLatency, btnMem, btnSize].forEach(btn => {
        if (btn) {
          btn.classList.remove('qc-btn-primary');
          btn.classList.add('qc-btn-outline');
        }
      });
      if (metric === 'inference_latency_ms_per_row' && btnLatency) {
        btnLatency.classList.add('qc-btn-primary');
        btnLatency.classList.remove('qc-btn-outline');
      } else if (metric === 'peak_memory_mb' && btnMem) {
        btnMem.classList.add('qc-btn-primary');
        btnMem.classList.remove('qc-btn-outline');
      } else if (metric === 'model_size_mb' && btnSize) {
        btnSize.classList.add('qc-btn-primary');
        btnSize.classList.remove('qc-btn-outline');
      }
      this.renderChart(summaryData);
    };

    if (btnLatency) btnLatency.addEventListener('click', () => updateActiveButton('inference_latency_ms_per_row'));
    if (btnMem) btnMem.addEventListener('click', () => updateActiveButton('peak_memory_mb'));
    if (btnSize) btnSize.addEventListener('click', () => updateActiveButton('model_size_mb'));
  }

  renderChart(summaryData) {
    QualityCharts.renderComputationalEfficiencyChart('resource-interactive-efficiency-chart', summaryData, this.activeMetric);
  }

  renderTable(benchmarks) {
    const columns = [
      { key: 'crop', label: '작물', sortable: true, format: v => QualityDataLoader.formatCrop(v) },
      { key: 'model', label: '모델', sortable: true },
      { key: 'split', label: '평가 분할', sortable: true, format: v => v.toUpperCase() },
      { key: 'inference_latency_ms_per_row', label: '추론 지연시간 (ms/행)', sortable: true, format: v => v.toFixed(4) },
      { key: 'peak_memory_mb', label: 'Peak RAM (MB)', sortable: true, format: v => v.toFixed(2) },
      { key: 'model_size_mb', label: '모델 크기 (MB)', sortable: true, format: v => v.toFixed(4) },
      { key: 'measured_rows', label: '계측 행 수 (Measured Rows)', sortable: true, format: v => v.toLocaleString() },
      { key: 'repeats', label: '반복 계측 횟수', sortable: true }
    ];

    this.table = new DataTable({
      containerId: 'resource-benchmark-table-container',
      title: '온라인 추론 벤치마크 계측표 (Online Benchmark Table)',
      columns,
      data: benchmarks,
      pageSize: 10,
      initialSort: { key: 'inference_latency_ms_per_row', dir: 'asc' }
    });
  }

  update() {
    this.render();
  }
}

