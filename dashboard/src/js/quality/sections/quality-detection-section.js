/**
 * GEAS Dashboard - Quality Anomaly Detection Section
 */

import { qualityDataLoader, QualityDataLoader } from '../quality-data-loader.js';
import { qualityFilterStore } from '../quality-filter-store.js';
import { DataTable } from '../../components/data-table.js';
import { ChartRenderer } from '../../components/chart-renderer.js';
import { QualityCharts } from '../quality-charts.js';

export class QualityDetectionSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.table = null;
    this.prVariable = 'in_temp';
  }

  async render() {
    if (!this.container) return;
    const filterState = qualityFilterStore.getState();
    const metrics = qualityDataLoader.getAnomalyDetectionMetrics(filterState);

    // Selected crop for figures
    const cropForFig = filterState.crop !== 'all' ? filterState.crop : 'cucumber';
    const summaryData = await QualityCharts.getSummaryResults(filterState.split);
    const thresholdJson = await QualityCharts.getThresholdCalibration(cropForFig);

    const isTest = filterState.split === 'test';

    this.container.innerHTML = `
      <div class="section-container">
        <!-- 1. Header -->
        <div class="section-header-card">
          <div class="section-badge-row">
            <span class="badge badge-primary">Anomaly Detection Benchmark</span>
            <span class="badge badge-success">Split: ${isTest ? 'Test' : 'Validation'}</span>
            <span class="badge badge-warning">Synthetic 8σ Injection</span>
          </div>
          <h2 class="section-title">🔍 합성 이상치(Synthetic Anomaly) 탐지 성능</h2>
          <p class="section-description">
            센서 데이터셋에 8σ 규모의 가상 이상치를 주입하고, 각 모델의 재구성 오차 기반 판정 결과(F1, Precision, Recall, ROC/PR AUC)를 정량화한 지표입니다.
          </p>
        </div>

        <!-- 2. Test Split Notice if active -->
        ${isTest ? `
          <div class="qc-academic-callout" style="background: #fffbeb; border-left-color: #f59e0b; color: #92400e; margin-bottom: 1.25rem;">
            <strong>ℹ️ 평가 분할 안내:</strong> 임계값 보정(Threshold Calibration) 탐색 자료는 <strong>Validation 전용</strong>으로 산출되었습니다.
            Test 평가에서는 Validation에서 기확정된 최적 임계값을 적용하므로 아래 PR 곡선 및 ROC 운영점은 Validation 탐색 지표를 기준으로 표시됩니다.
          </div>
        ` : ''}

        <!-- 3. Dynamic Charts (Per-variable F1 & Global AUC) -->
        <div class="two-column-grid">
          <div class="content-card">
            <h3 class="card-title">
              <span>📊</span> 변수별 이상치 탐지 성능 (F1-Score ↑)
            </h3>
            <div style="height: 320px; position: relative;">
              <canvas id="detect-f1-chart"></canvas>
            </div>
          </div>

          <div class="content-card">
            <h3 class="card-title">
              <span>📈</span> 모델별 ROC-AUC 및 PR-AUC 종합 비교
            </h3>
            <div style="height: 320px; position: relative;">
              <canvas id="detect-auc-chart"></canvas>
            </div>
          </div>
        </div>

        <!-- 4. Interactive PR Curve & ROC Operating Point (Requirement 5) -->
        <div class="two-column-grid">
          <div class="content-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
              <h3 class="card-title" style="margin: 0; border: none;">
                <span>🎯</span> Validation 임계값 후보 기반 PR 곡선
              </h3>
              <div style="display: flex; align-items: center; gap: 0.4rem;">
                <label for="select-pr-variable" style="font-size: 0.78rem; font-weight: 600; color: #64748b;">변수:</label>
                <select id="select-pr-variable" class="qc-select">
                  ${Object.entries(QualityCharts.VARIABLE_INFO).map(([k, v]) => `
                    <option value="${k}" ${this.prVariable === k ? 'selected' : ''}>${v.label} [${v.unit}]</option>
                  `).join('')}
                </select>
              </div>
            </div>
            <p style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.75rem; line-height: 1.45;">
              ※ 100개 임계값 후보의 (Recall, Precision) 쌍 기반 곡선입니다. 원본 전체 연속 점수의 완벽 곡선이 아닙니다. 별(★) 마커는 선택된 최적 임계값 운영점입니다.
            </p>
            <div style="height: 300px; position: relative;">
              <canvas id="qc-pr-curve-chart"></canvas>
            </div>
          </div>

          <div class="content-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
              <h3 class="card-title" style="margin: 0; border: none;">
                <span>📉</span> 선택 임계값 조합의 ROC 운영점
              </h3>
              <span class="badge badge-info">단일 운영점 (FPR, TPR)</span>
            </div>
            <p style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.75rem; line-height: 1.45;">
              ※ 임의의 끝점 연결선 없이 최종 선택된 임계값 조합의 단일 운영점(FPR, TPR) 및 Random Guess 기준선(y=x)만 정직하게 표시합니다.
            </p>
            <div style="height: 300px; position: relative;">
              <canvas id="qc-roc-operating-point-chart"></canvas>
            </div>
          </div>
        </div>

        <!-- 5. Global Multi-crop Classification Comparison (Requirement 1) -->
        <div class="content-card">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h3 class="card-title" style="margin: 0; border: none;">
              <span>📊</span> 전 작물 대상 이상치 분류 성능 종합 비교 (Global Classification)
            </h3>
            <span class="badge badge-primary">${filterState.split.toUpperCase()} 분할</span>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
            <code>${filterState.split}_results.csv</code>의 실제 F1-Score 수치를 바탕으로 오이, 멜론, 딸기의 모델별 성능을 비교합니다.
          </p>
          <div style="height: 300px; position: relative;">
            <canvas id="qc-global-detection-chart"></canvas>
          </div>
        </div>

        <!-- 6. Detailed Data Table with Confusion Matrix & Export -->
        <div id="detection-table-container"></div>

        <!-- 7. Academic Reference Artifacts (Static Non-Interactive) -->
        <div class="content-card" style="opacity: 0.9;">
          <details>
            <summary style="cursor: pointer; font-size: 0.88rem; font-weight: 700; color: #475569; padding: 0.25rem 0;">
              🖼️ 논문/보고서 제출용 원본 정적 아티팩트 보기 (Static Reference Figures - Non-Interactive)
            </summary>
            <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0.75rem 0;">
              ※ 아래 이미지는 실험 산출물 디렉터리에 보관된 정적 이미지이며 실시간 필터 적용 대상이 아닙니다.
            </p>
            <div class="reference-figures-grid">
              <div class="reference-figure-box">
                <div class="figure-label">Anomaly Detection Classification Metrics</div>
                <img src="quality_panel_bundle/data/quality/figures/anomaly_detection_classification.png" alt="Classification Metrics" class="ref-img" loading="lazy">
                <div class="figure-caption">F1-Score, Precision, Recall, MCC, FPR 종합 분류 지표 비교</div>
              </div>
              <div class="reference-figure-box">
                <div class="figure-label">${QualityDataLoader.formatCrop(cropForFig)} ROC & PR Curves</div>
                <img src="quality_panel_bundle/data/quality/${cropForFig}/figures/roc_pr_curves.png" alt="ROC and PR Curves" class="ref-img" loading="lazy">
                <div class="figure-caption">${QualityDataLoader.formatCrop(cropForFig)} 변수별 ROC Curve 및 Precision-Recall Curve</div>
              </div>
            </div>
          </details>
        </div>
      </div>
    `;

    this.renderCharts(metrics);
    this.renderTable(metrics);

    // 1. Render PR Curve and ROC Operating Point Chart
    QualityCharts.renderPrCurveChart('qc-pr-curve-chart', thresholdJson, {
      variable: this.prVariable,
      activeModel: filterState.model !== 'all' ? filterState.model : 'all'
    });

    QualityCharts.renderRocOperatingPointChart('qc-roc-operating-point-chart', thresholdJson, 
      filterState.model !== 'all' ? filterState.model : 'all'
    );

    // 2. Render Global Multi-crop Classification Chart
    QualityCharts.renderAnomalyClassificationChart('qc-global-detection-chart', summaryData, filterState.crop);

    // 3. Bind Variable Selector for PR Curve
    const prVarSelect = document.getElementById('select-pr-variable');
    if (prVarSelect) {
      prVarSelect.onchange = (e) => {
        this.prVariable = e.target.value;
        QualityCharts.renderPrCurveChart('qc-pr-curve-chart', thresholdJson, {
          variable: this.prVariable,
          activeModel: filterState.model !== 'all' ? filterState.model : 'all'
        });
      };
    }
  }

  renderCharts(metrics) {
    if (!metrics || metrics.length === 0) return;

    // Filter non-all variables for per-variable chart
    const varRows = metrics.filter(m => m.column !== '__all__');
    const uniqueVars = [...new Set(varRows.map(m => m.column))];
    const uniqueModels = [...new Set(varRows.map(m => m.model))];

    const datasets = uniqueModels.map((model, idx) => {
      const colors = ['#4f46e5', '#06b6d4', '#10b981', '#f59e0b'];
      const data = uniqueVars.map(v => {
        const item = varRows.find(m => m.model === model && m.column === v);
        return item ? item.f1_score : 0;
      });
      return {
        label: model,
        data,
        backgroundColor: colors[idx % colors.length],
        borderRadius: 4
      };
    });

    ChartRenderer.getOrCreateChart('detect-f1-chart', {
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
          y: { min: 0.5, max: 1.0, title: { display: true, text: 'F1-Score' } }
        }
      }
    });

    // AUC comparison for __all__
    const allRows = metrics.filter(m => m.column === '__all__');
    ChartRenderer.getOrCreateChart('detect-auc-chart', {
      type: 'bar',
      data: {
        labels: allRows.map(r => `${QualityDataLoader.formatCrop(r.crop).split(' ')[0]} · ${r.model}`),
        datasets: [
          {
            label: 'ROC-AUC',
            data: allRows.map(r => r.roc_auc),
            backgroundColor: '#0284c7',
            borderRadius: 4
          },
          {
            label: 'PR-AUC',
            data: allRows.map(r => r.pr_auc),
            backgroundColor: '#7c3aed',
            borderRadius: 4
          }
        ]
      },
      options: {
        plugins: { legend: { position: 'top' } },
        scales: {
          y: { min: 0.2, max: 1.0, title: { display: true, text: 'AUC Score' } }
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
      { key: 'f1_score', label: 'F1-Score', sortable: true, format: v => v.toFixed(4) },
      { key: 'precision', label: 'Precision (정밀도)', sortable: true, format: v => v.toFixed(4) },
      { key: 'recall', label: 'Recall (재현율)', sortable: true, format: v => v.toFixed(4) },
      { key: 'roc_auc', label: 'ROC-AUC', sortable: true, format: v => v.toFixed(4) },
      { key: 'pr_auc', label: 'PR-AUC', sortable: true, format: v => v.toFixed(4) },
      { key: 'true_positive', label: 'TP (진양성)', sortable: true, format: v => v.toLocaleString() },
      { key: 'false_positive', label: 'FP (위양성)', sortable: true, format: v => v.toLocaleString() },
      { key: 'false_negative', label: 'FN (위음성)', sortable: true, format: v => v.toLocaleString() },
      { key: 'true_negative', label: 'TN (진음성)', sortable: true, format: v => v.toLocaleString() },
      { key: 'injected_cells', label: '주입 표본수 (Injected)', sortable: true, format: v => v.toLocaleString() }
    ];

    this.table = new DataTable({
      containerId: 'detection-table-container',
      title: '변수별 이상치 탐지 세부 지표 및 혼동행렬 (Anomaly Detection Metric Table)',
      columns,
      data: metrics,
      pageSize: 15,
      initialSort: { key: 'f1_score', dir: 'desc' }
    });
  }

  update() {
    this.render();
  }
}
