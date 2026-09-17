/**
 * GEAS Dashboard - Quality Reconstruction & Masking Imputation Section
 */

import { qualityDataLoader, QualityDataLoader } from '../quality-data-loader.js';
import { qualityFilterStore } from '../quality-filter-store.js';
import { DataTable } from '../../components/data-table.js';
import { ChartRenderer } from '../../components/chart-renderer.js';
import { QualityCharts } from '../quality-charts.js';

export class QualityReconstructionSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.table = null;
    this.heatmapMetric = 'rmse';
  }

  async render() {
    if (!this.container) return;
    const filterState = qualityFilterStore.getState();
    const metrics = qualityDataLoader.getReconstructionMetrics(filterState);

    // Selected crop for figures
    const cropForFig = filterState.crop !== 'all' ? filterState.crop : 'cucumber';
    const summaryData = await QualityCharts.getSummaryResults(filterState.split);
    const reconRows = await QualityCharts.getReconstructionTable(cropForFig);

    this.container.innerHTML = `
      <div class="section-container">
        <!-- 1. Header -->
        <div class="section-header-card">
          <div class="section-badge-row">
            <span class="badge badge-primary">Imputation Benchmark</span>
            <span class="badge badge-success">Split: ${filterState.split === 'test' ? 'Test' : 'Validation'}</span>
            <span class="badge badge-info">10% Random Masking</span>
          </div>
          <h2 class="section-title">🩹 결측 및 마스킹 데이터 복원(Reconstruction) 성능</h2>
          <p class="section-description">
            정상 관측치 중 무작위로 10%를 마스킹 처리한 후 각 딥러닝 모델(ModernTCN, TimesNet, PatchTST)이 복원한 추정값과 
            원래 관측치 사이의 오차(RMSE, MAE)를 평가한 결과입니다.
          </p>
        </div>

        <!-- 2. Methodological Caution Banner -->
        <div class="qc-academic-callout" style="margin-bottom: 1.25rem;">
          <strong>💡 데이터 분석 지침:</strong> 센서 복원 오차는 물리 단위별 절대 편차가 상이합니다.
          실시간 차트와 아래의 변수별 정규화 히트맵을 통해 모델별 상대적 정밀도를 종합적으로 분석하십시오.
        </div>

        <!-- 3. Dynamic Charts (Per-variable & Global) -->
        <div class="two-column-grid">
          <div class="content-card">
            <h3 class="card-title">
              <span>📊</span> 변수별 복원 오차 비교 (RMSE ↓)
            </h3>
            <div style="height: 320px; position: relative;">
              <canvas id="recon-rmse-chart"></canvas>
            </div>
          </div>

          <div class="content-card">
            <h3 class="card-title">
              <span>📉</span> 모델별 종합 마스킹 복원 오차 (MAE ↓)
            </h3>
            <div style="height: 320px; position: relative;">
              <canvas id="recon-mae-chart"></canvas>
            </div>
          </div>
        </div>

        <!-- 4. Interactive Model x Variable Heatmap Matrix (Requirement 4) -->
        <div class="content-card">
          <div id="reconstruction-interactive-heatmap"></div>
        </div>

        <!-- 5. Global Reconstruction Comparative Chart (Requirement 1) -->
        <div class="content-card">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h3 class="card-title" style="margin: 0; border: none;">
              <span>📈</span> 전 작물 대상 마스킹 복원 종합 비교 (Global Performance)
            </h3>
            <span class="badge badge-primary">${filterState.split.toUpperCase()} 분할</span>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
            <code>${filterState.split}_results.csv</code>의 실제 복원 RMSE 수치를 바탕으로 오이, 멜론, 딸기의 모델별 성능을 비교합니다.
          </p>
          <div style="height: 300px; position: relative;">
            <canvas id="recon-global-chart"></canvas>
          </div>
        </div>

        <!-- 6. Detailed Data Table with Export -->
        <div id="reconstruction-table-container"></div>
      </div>
    `;

    this.renderCharts(metrics);
    this.renderTable(metrics);

    // 1. Render Interactive Model x Variable Heatmap Matrix
    QualityCharts.renderReconstructionHeatmap('reconstruction-interactive-heatmap', reconRows, {
      metric: this.heatmapMetric,
      split: filterState.split
    });
    this.bindHeatmapButtons(reconRows, filterState.split);

    // 2. Render Global Multi-crop Reconstruction Performance Chart
    QualityCharts.renderReconstructionPerformanceChart('recon-global-chart', summaryData, filterState.crop);
  }

  bindHeatmapButtons(reconRows, split) {
    const btnRmse = document.getElementById('btn-heatmap-rmse');
    const btnMae = document.getElementById('btn-heatmap-mae');
    if (btnRmse) {
      btnRmse.onclick = () => {
        this.heatmapMetric = 'rmse';
        QualityCharts.renderReconstructionHeatmap('reconstruction-interactive-heatmap', reconRows, {
          metric: 'rmse',
          split
        });
        this.bindHeatmapButtons(reconRows, split);
      };
    }
    if (btnMae) {
      btnMae.onclick = () => {
        this.heatmapMetric = 'mae';
        QualityCharts.renderReconstructionHeatmap('reconstruction-interactive-heatmap', reconRows, {
          metric: 'mae',
          split
        });
        this.bindHeatmapButtons(reconRows, split);
      };
    }
  }

  renderCharts(metrics) {
    if (!metrics || metrics.length === 0) return;

    // Filter non-all variables for per-variable chart
    const varRows = metrics.filter(m => m.column !== '__all__');
    const uniqueVars = [...new Set(varRows.map(m => m.column))];
    const uniqueModels = [...new Set(varRows.map(m => m.model))];

    // Build dataset per model for grouped bar chart
    const datasets = uniqueModels.map((model, idx) => {
      const colors = ['#4f46e5', '#06b6d4', '#10b981', '#f59e0b'];
      const data = uniqueVars.map(v => {
        const item = varRows.find(m => m.model === model && m.column === v);
        return item ? item.rmse : 0;
      });
      return {
        label: model,
        data,
        backgroundColor: colors[idx % colors.length],
        borderRadius: 4
      };
    });

    ChartRenderer.getOrCreateChart('recon-rmse-chart', {
      type: 'bar',
      data: {
        labels: uniqueVars.map(v => QualityDataLoader.formatColumnName(v).split(' ')[0]),
        datasets
      },
      options: {
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              title: (ctx) => {
                const varKey = uniqueVars[ctx[0].dataIndex];
                return QualityDataLoader.formatColumnName(varKey);
              }
            }
          }
        },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'RMSE' } }
        }
      }
    });

    // MAE by Model on __all__
    const allRows = metrics.filter(m => m.column === '__all__');
    ChartRenderer.getOrCreateChart('recon-mae-chart', {
      type: 'bar',
      data: {
        labels: allRows.map(r => `${QualityDataLoader.formatCrop(r.crop).split(' ')[0]} · ${r.model}`),
        datasets: [{
          label: '종합 MAE (__all__)',
          data: allRows.map(r => r.mae),
          backgroundColor: '#059669',
          borderRadius: 4
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'MAE' } }
        }
      }
    });
  }

  renderTable(metrics) {
    const columns = [
      { key: 'crop', label: '작물', sortable: true, format: v => QualityDataLoader.formatCrop(v) },
      { key: 'model', label: '모델', sortable: true },
      { key: 'split', label: '평가 분할', sortable: true, format: v => v.toUpperCase() },
      { key: 'column', label: '평가 대상 변수', sortable: true, format: v => QualityDataLoader.formatColumnName(v) },
      { key: 'rmse', label: 'RMSE (평균제곱근오차)', sortable: true, format: v => v.toFixed(4) },
      { key: 'mae', label: 'MAE (평균절대오차)', sortable: true, format: v => v.toFixed(4) },
      { key: 'masked_cells', label: '마스킹 평가 표본 수 (Masked Cells)', sortable: true, format: v => v.toLocaleString() }
    ];

    this.table = new DataTable({
      containerId: 'reconstruction-table-container',
      title: '변수별 마스킹 복원 세부 평가지표 (Reconstruction Metric Table)',
      columns,
      data: metrics,
      pageSize: 15,
      initialSort: { key: 'rmse', dir: 'asc' }
    });
  }

  update() {
    this.render();
  }
}
