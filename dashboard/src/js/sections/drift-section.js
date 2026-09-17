/**
 * GEAS Dashboard - Section: Rollout Drift Analysis (Table 6)
 */

import { dataService } from '../core/data-loader.js';
import { filterStore } from '../core/filter-store.js';
import { DataTable } from '../components/data-table.js';
import { ChartRenderer } from '../components/chart-renderer.js';
import {
  formatNumber,
  formatCrop,
  formatModel,
  formatTarget,
  renderEligibilityBadge,
  renderEligibilityGuide
} from '../core/formatters.js';

export class DriftSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.selectedHorizon = '60min'; // 'all', '15min', '30min', '60min'
    this.selectedTarget = 'all_targets'; // 'all_targets', 'indoor_temperature', 'indoor_humidity', 'indoor_co2'
    this.tableInstance = null;
    this.chartInstance = null;
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <!-- Academic Explanation Callout for Drift Slope -->
        <div class="academic-callout">
          <div>
            <strong>📐 Rollout Drift Slope Term in HPO Loss (+ 0.10 × Drift Slope NMAE):</strong><br>
            Recursive rollout 중 시계열 오차가 특정 방향으로 지속적으로 누적·편향(Drift)되는 현상을 정량화한 지표입니다. 
            Drift Slope NMAE가 낮을수록 장기 롤아웃 시 실내 기후가 비현실적으로 발산하지 않고 안정적인 동적 평형을 유지합니다.
          </div>
        </div>

        <!-- Controls Bar -->
        <div class="onestep-controls-bar">
          <div class="onestep-target-selector">
            <span style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase;">Horizon:</span>
            <button type="button" class="target-pill-btn ${this.selectedHorizon === '60min' ? 'active' : ''}" data-horizon="60min">60-Min Rollout</button>
            <button type="button" class="target-pill-btn ${this.selectedHorizon === '30min' ? 'active' : ''}" data-horizon="30min">30-Min Rollout</button>
            <button type="button" class="target-pill-btn ${this.selectedHorizon === '15min' ? 'active' : ''}" data-horizon="15min">15-Min Rollout</button>
            <button type="button" class="target-pill-btn ${this.selectedHorizon === 'all' ? 'active' : ''}" data-horizon="all">All Horizons</button>
          </div>

          <div class="onestep-target-selector">
            <span style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase;">Target State:</span>
            <button type="button" class="target-pill-btn ${this.selectedTarget === 'all_targets' ? 'active' : ''}" data-drift-target="all_targets">All Targets</button>
            <button type="button" class="target-pill-btn ${this.selectedTarget === 'indoor_temperature' ? 'active' : ''}" data-drift-target="indoor_temperature">🌡️ Temp</button>
            <button type="button" class="target-pill-btn ${this.selectedTarget === 'indoor_humidity' ? 'active' : ''}" data-drift-target="indoor_humidity">💧 Humidity</button>
            <button type="button" class="target-pill-btn ${this.selectedTarget === 'indoor_co2' ? 'active' : ''}" data-drift-target="indoor_co2">🌿 CO₂</button>
          </div>
        </div>

        <!-- Drift Chart Card -->
        <div class="chart-card">
          <div class="chart-header">
            <div class="chart-title-group">
              <h3 class="chart-title" id="drift-chart-title">📊 Rollout Drift Slope NMAE Distribution</h3>
              <span class="chart-subtitle" id="drift-chart-subtitle">Comparing Mean, Median, and Q90 Tail Drift across candidate models (Lower is better)</span>
            </div>
          </div>
          <div class="chart-canvas-wrapper">
            <canvas id="drift-slope-chart"></canvas>
          </div>
        </div>

        <!-- Table 6 Data Table -->
        <div id="table-6-container"></div>

        <!-- Table 6 Key Metric Explanation Card: Overall Physical Violation Rate -->
        <div class="drift-metrics-guide-card" style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: var(--radius-lg); padding: 1.25rem 1.5rem; display: flex; flex-direction: column; gap: 0.85rem; box-shadow: var(--shadow-sm);">
          <div style="font-size: 0.98rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <span>🛡️</span> Table 6 주요 지표 상세 설명: Overall Physical Violation Rate (종합 물리 위반율)
            </div>
            <span style="font-size: 0.78rem; color: #991b1b; background: #fee2e2; padding: 0.2rem 0.6rem; border-radius: 9999px; font-weight: 700;">
              0.000 = 물리 제약 완벽 준수
            </span>
          </div>

          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #ef4444; border-radius: var(--radius-md); padding: 1rem;">
            <div style="font-weight: 700; color: #991b1b; font-size: 0.9rem; margin-bottom: 0.4rem;">
              💡 종합 물리 허용 범위 위반율 (Overall Physical Violation Rate) 안내
            </div>
            <p style="font-size: 0.84rem; color: #334155; margin: 0 0 0.65rem 0; line-height: 1.6;">
              이 열의 수치는 표시된 개별 Target(예: 실내 CO₂ 단독)만의 위반율이 아니라, <strong>해당 작물·모델·Variant·Horizon 조건의 전체 예측 환경 변수(온도·습도·CO₂)를 아우르는 종합 물리 위반 비율</strong>입니다. 따라서 동일 조건 내에서 온도·습도·CO₂ 행에 같은 값이 표시됩니다.
            </p>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem; margin-bottom: 0.65rem;">
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 4px; padding: 0.45rem 0.7rem; font-size: 0.82rem;">
                <span style="color: #475569;">실내 온도:</span> <code style="color: #15803d; font-weight: 700; margin-left: 0.25rem;">−60 ~ 80°C</code>
              </div>
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 4px; padding: 0.45rem 0.7rem; font-size: 0.82rem;">
                <span style="color: #475569;">실내 상대습도:</span> <code style="color: #15803d; font-weight: 700; margin-left: 0.25rem;">0 ~ 100%</code>
              </div>
              <div style="background: #ffffff; border: 1px solid #dcfce7; border-radius: 4px; padding: 0.45rem 0.7rem; font-size: 0.82rem;">
                <span style="color: #475569;">실내 CO₂:</span> <code style="color: #15803d; font-weight: 700; margin-left: 0.25rem;">0 ~ 10,000 ppm</code>
              </div>
            </div>

            <p style="font-size: 0.8rem; color: #475569; margin: 0; line-height: 1.55;">
              • <strong>0 (0.000)</strong>: 해당 horizon에서 세 환경 변수 모두 물리 허용 범위 위반이 전혀 발생하지 않음.<br>
              • <strong>0보다 큼 (&gt;0)</strong>: 하나 이상의 환경 변수에서 비현실적인 예측이 발생하여 강화학습 배포 부적격(Ineligible)으로 판정됨.
            </p>
          </div>
        </div>

        <!-- High-Visibility Eligibility Gating Criteria Guide -->
        ${renderEligibilityGuide()}
      </div>
    `;

    this.bindEvents();
    this.renderTable();
    this.renderChart();
  }

  bindEvents() {
    this.container.querySelectorAll('[data-horizon]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.container.querySelectorAll('[data-horizon]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.selectedHorizon = btn.getAttribute('data-horizon');
        this.update();
      });
    });

    this.container.querySelectorAll('[data-drift-target]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.container.querySelectorAll('[data-drift-target]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.selectedTarget = btn.getAttribute('data-drift-target');
        this.update();
      });
    });
  }

  update() {
    const rows = this.getFilteredRows();
    if (this.tableInstance) {
      this.tableInstance.setData(rows);
    }
    this.renderChart();
  }

  getFilteredRows() {
    let rows = filterStore.filterRows(dataService.getTable6());
    if (this.selectedHorizon !== 'all') {
      rows = rows.filter(r => r.horizon === this.selectedHorizon);
    }
    if (this.selectedTarget !== 'all_targets') {
      rows = rows.filter(r => r.target === this.selectedTarget);
    } else {
      rows = rows.filter(r => r.target === 'all_targets');
    }
    return rows;
  }

  renderTable() {
    const rows = this.getFilteredRows();

    const columns = [
      {
        key: 'crop',
        label: 'Crop',
        format: (val) => formatCrop(val)
      },
      {
        key: 'model',
        label: 'Model',
        format: (val) => `<strong>${formatModel(val)}</strong>`
      },
      {
        key: 'feature_variant',
        label: 'Variant',
        format: (val) => `<span class="badge badge-variant">${val === 'with_quality_flags' ? 'With Flags' : 'Without Flags'}</span>`
      },
      {
        key: 'horizon',
        label: 'Horizon',
        format: (val) => `<span style="font-family: var(--font-family-mono); font-weight: 600;">${val}</span>`
      },
      {
        key: 'target',
        label: 'Target',
        format: (val) => val === 'all_targets' ? 'All Targets' : (TARGET_LABELS[val] || val)
      },
      {
        key: 'deployment_eligible',
        label: 'Eligible',
        format: (val, row) => renderEligibilityBadge(val, row.baseline_only)
      },
      {
        key: 'drift_slope_nmae',
        label: 'Mean Drift NMAE',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono); font-weight: 700; color: #0284c7;">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'drift_slope_nmae_median',
        label: 'Median Drift',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'drift_slope_nmae_q90',
        label: 'Q90 Tail Drift',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'drift_slope_nmae_cvar90',
        label: 'CVaR90 Risk',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'physical_violation_rate',
        label: 'Overall Physical Violation Rate',
        tooltip: 'Fraction of all predicted temperature, humidity, and CO₂ values outside predefined physical bounds at the selected rollout horizon. This value is not target-specific.',
        align: 'right',
        format: (val) => {
          const num = Number(val || 0);
          return num > 0 ? `<span class="badge badge-physical-violation">${num.toExponential(2)}</span>` : `<span style="color: #059669;">0</span>`;
        }
      },
      {
        key: 'validation_window_count',
        label: 'Windows',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${val}</span>`
      }
    ];

    this.tableInstance = new DataTable({
      containerId: 'table-6-container',
      title: 'Table 6: Rollout Drift Slope Details & Tail Error Risk (Validation Only)',
      columns,
      data: rows,
      initialSort: { key: 'drift_slope_nmae', dir: 'asc' },
      prioritizeEligible: true,
      pageSize: 15
    });
  }

  renderChart() {
    const rows = this.getFilteredRows();
    if (rows.length === 0) return;

    const titleEl = document.getElementById('drift-chart-title');
    if (titleEl) {
      titleEl.textContent = `📊 Drift Slope NMAE (${this.selectedHorizon.toUpperCase()} - ${this.selectedTarget === 'all_targets' ? 'All Targets' : TARGET_LABELS[this.selectedTarget]})`;
    }

    const displayRows = [...rows]
      .sort((a, b) => (a.drift_slope_nmae || 0) - (b.drift_slope_nmae || 0))
      .slice(0, 24);

    const labels = displayRows.map(r => `${formatModel(r.model)} (${r.crop.slice(0, 3)} - ${r.feature_variant === 'with_quality_flags' ? 'w/flags' : 'wo/flags'})`);

    this.chartInstance = ChartRenderer.getOrCreateChart('drift-slope-chart', {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Mean Drift Slope NMAE',
            data: displayRows.map(r => r.drift_slope_nmae),
            backgroundColor: '#0284c7'
          },
          {
            label: 'Median Drift Slope NMAE',
            data: displayRows.map(r => r.drift_slope_nmae_median),
            backgroundColor: '#0d9488'
          },
          {
            label: 'Q90 Tail Error',
            data: displayRows.map(r => r.drift_slope_nmae_q90),
            backgroundColor: '#d97706'
          }
        ]
      },
      options: {
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.raw?.toFixed(5)}`
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: 'Drift Slope NMAE (Lower is better)' },
            beginAtZero: true
          },
          x: {
            ticks: { maxRotation: 45, minRotation: 20 }
          }
        }
      }
    });
  }
}
