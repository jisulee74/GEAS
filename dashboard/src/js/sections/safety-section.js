/**
 * GEAS Dashboard - Section 4: Rollout Progression & Safety / Action-Response Gates (Table 3)
 */

import { dataService } from '../core/data-loader.js';
import { filterStore } from '../core/filter-store.js';
import { DataTable } from '../components/data-table.js';
import { ChartRenderer } from '../components/chart-renderer.js';
import {
  formatNumber,
  formatCrop,
  formatModel,
  renderEligibilityBadge,
  renderActionResponseBadge,
  renderEligibilityGuide
} from '../core/formatters.js';

export class SafetySection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.tableInstance = null;
    this.lineChartInstance = null;
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <!-- Status & Criteria Legend Guide -->
        <div class="action-support-guide">
          <div class="support-guide-item">
            <span class="badge badge-safe">✓ Action-Sensitive</span>
            <span>Controls induce meaningful state changes</span>
          </div>
          <div class="support-guide-item">
            <span class="badge badge-insufficient-support">ℹ️ Data-Sparse</span>
            <span>Neutral: Lack of observational action variation</span>
          </div>
          <div class="support-guide-item">
            <span class="badge badge-action-insensitive">⚠️ Action-Insensitive</span>
            <span>Unfit for RL Simulator (zero control responsiveness)</span>
          </div>
          <div class="support-guide-item">
            <span class="badge badge-physical-violation">⚠️ Safety Breach</span>
            <span>Rollout states violated thermodynamic bounds</span>
          </div>
        </div>

        <div class="safety-grid-row">
          <!-- Rollout Trajectory Error Growth Line Chart -->
          <div class="chart-card">
            <div class="chart-header">
              <div class="chart-title-group">
                <h3 class="chart-title">📈 Error Accumulation Trajectory Across Rollout Horizons</h3>
                <span class="chart-subtitle">NRMSE Growth across 5m (One-Step), 15m, 30m, and 60m recursive rollouts</span>
              </div>
            </div>
            <div class="chart-canvas-wrapper">
              <canvas id="safety-trajectory-chart"></canvas>
            </div>
          </div>

          <!-- Counterfactual Sensitivity Chart -->
          <div class="chart-card">
            <div class="chart-header">
              <div class="chart-title-group">
                <h3 class="chart-title">🎛️ Counterfactual Action Sensitivity</h3>
                <span class="chart-subtitle">Mean & Minimum counterfactual response per model (Higher responsiveness is required)</span>
              </div>
            </div>
            <div class="chart-canvas-wrapper">
              <canvas id="safety-sensitivity-chart"></canvas>
            </div>
          </div>
        </div>

        <!-- Table 3 Data Table -->
        <div id="table-3-container"></div>

        <!-- Table 3 Key Metric Explanation Card: Drift Slope -->
        <div class="safety-metrics-guide-card" style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: var(--radius-lg); padding: 1.25rem 1.5rem; display: flex; flex-direction: column; gap: 0.85rem; box-shadow: var(--shadow-sm);">
          <div style="font-size: 0.98rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <span>📉</span> Table 3 주요 지표 상세 설명: Drift Slope (오차 누적 드리프트 기울기)
            </div>
            <span style="font-size: 0.78rem; color: #0369a1; background: #e0f2fe; padding: 0.2rem 0.6rem; border-radius: 9999px; font-weight: 700;">
              HPO 목적함수 10% 반영 지표
            </span>
          </div>
          
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #0284c7; border-radius: var(--radius-md); padding: 1rem;">
            <div style="font-weight: 700; color: #0369a1; font-size: 0.9rem; margin-bottom: 0.4rem;">
              💡 Drift Slope의 정의와 물리적 의미
            </div>
            <p style="font-size: 0.84rem; color: #334155; margin: 0 0 0.65rem 0; line-height: 1.6;">
              <strong>Drift Slope</strong>는 <strong>Recursive rollout이 진행될수록 정규화된 예측 오차가 얼마나 빠르게 증가(누적)하는지</strong>를 나타내는 지표입니다.
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0.65rem; margin-bottom: 0.75rem;">
              <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 0.55rem 0.75rem;">
                <div style="font-weight: 700; color: #0f172a; font-size: 0.82rem;">• 작을수록 유리</div>
                <div style="font-size: 0.78rem; color: #64748b;">시간 경과에 따른 오차 발산이 억제되어 장기 rollout이 구조적으로 안정적임을 의미합니다.</div>
              </div>
              <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 0.55rem 0.75rem;">
                <div style="font-weight: 700; color: #b91c1c; font-size: 0.82rem;">• 양수이고 클수록 위험</div>
                <div style="font-size: 0.78rem; color: #64748b;">미래 시점(15m → 30m → 60m)으로 갈수록 오차가 빠르게 누적됨을 나타냅니다.</div>
              </div>
              <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 0.55rem 0.75rem;">
                <div style="font-weight: 700; color: #059669; font-size: 0.82rem;">• 0에 가까울 때</div>
                <div style="font-size: 0.78rem; color: #64748b;">다단계 롤아웃이 진행되는 동안 추가적인 오차 증가율이 극히 미미하고 평탄하게 유지됩니다.</div>
              </div>
              <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 0.55rem 0.75rem;">
                <div style="font-weight: 700; color: #4338ca; font-size: 0.82rem;">• 음수일 때</div>
                <div style="font-size: 0.78rem; color: #64748b;">해당 평가 구간에서 롤아웃 후반부의 오차가 오히려 감소하거나 안정화되는 경향을 보입니다.</div>
              </div>
            </div>

            <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 4px; padding: 0.65rem 0.85rem; font-size: 0.8rem; color: #92400e; line-height: 1.55;">
              ⚠️ <strong>연구 해석 주의사항 (Academic Caveat):</strong><br>
              Drift가 작다고 해서 반드시 정확한 모델은 아닙니다. <strong>처음부터 큰 오차를 일정하게 유지하는 모델도 drift는 작게 계산</strong>될 수 있으므로, 반드시 <strong>15·30·60분 NRMSE 및 Persistence Skill Score와 함께 종합적으로 비교·검토</strong>해야 합니다.
            </div>
          </div>
        </div>

        <!-- High-Visibility Eligibility Gating Criteria Guide -->
        ${renderEligibilityGuide()}
      </div>
    `;

    this.renderTable();
    this.renderCharts();
  }

  update() {
    const rows = filterStore.filterRows(dataService.getTable3());
    if (this.tableInstance) {
      this.tableInstance.setData(rows);
    }
    this.renderCharts();
  }

  renderTable() {
    const rows = filterStore.filterRows(dataService.getTable3());

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
        key: 'deployment_eligible',
        label: 'Overall Gate',
        format: (val, row) => renderEligibilityBadge(val, row.model === 'persistence')
      },
      {
        key: 'action_response_eligible',
        label: 'Action Gate',
        format: (val, row) => renderActionResponseBadge(val, row.fully_action_insensitive, row.actions_insufficient_support)
      },
      {
        key: 'actions_status_ok',
        label: 'Action Support',
        align: 'center',
        tooltip: 'Actions with sufficient observational support vs total action space',
        format: (val, row) => `${row.actions_with_sufficient_support} / ${row.actions_total}`
      },
      {
        key: 'mean_counterfactual_sensitivity',
        label: 'Mean Sensitivity',
        align: 'right',
        format: (val) => val === null ? `<span style="color: var(--text-muted); font-size: 0.78rem;">N/A (Data-Sparse)</span>` : `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'minimum_counterfactual_sensitivity',
        label: 'Min Sensitivity',
        align: 'right',
        format: (val) => val === null ? `<span style="color: var(--text-muted); font-size: 0.78rem;">N/A (Data-Sparse)</span>` : `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'physical_violation_rate_60min',
        label: '60m Violation',
        align: 'right',
        format: (val) => {
          const num = Number(val || 0);
          if (num > 0) return `<span class="badge badge-physical-violation">${num.toExponential(2)}</span>`;
          return `<span style="color: #34d399;">0</span>`;
        }
      },
      {
        key: 'nan_inf_count_total',
        label: 'NaN/Inf',
        align: 'center',
        format: (val) => (val > 0 ? `<span style="color: #ef4444; font-weight: 700;">⚠️ ${val}</span>` : `<span style="color: var(--text-muted);">0</span>`)
      },
      {
        key: 'drift_slope_nmae',
        label: 'Drift Slope',
        align: 'right',
        tooltip: 'Recursive rollout 진행에 따른 정규화 예측 오차 증가율. 작을수록 장기 롤아웃 안정성 우수.',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      }
    ];

    this.tableInstance = new DataTable({
      containerId: 'table-3-container',
      title: 'Table 3: Safety, Stability & Action-Response Verification (Validation Only)',
      columns,
      data: rows,
      initialSort: { key: 'action_response_eligible', dir: 'desc' },
      pageSize: 15
    });
  }

  renderCharts() {
    const t1Rows = filterStore.filterRows(dataService.getTable1());
    const t3Rows = filterStore.filterRows(dataService.getTable3());
    if (t1Rows.length === 0) return;

    // Filter top models to display in trajectory chart without cluttering
    const sampleModels = [...t1Rows]
      .filter(r => (r.validation_rollout_weighted_nrmse || 0) < 1.0)
      .slice(0, 7);

    const horizons = ['5-min (1-Step)', '15-min', '30-min', '60-min'];

    const datasets = sampleModels.map((m, idx) => ({
      label: `${formatModel(m.model)} (${m.crop.slice(0, 3)})`,
      data: [m.one_step_nrmse, m.rollout_15min_nrmse, m.rollout_30min_nrmse, m.rollout_60min_nrmse],
      borderColor: ChartRenderer.modelColors[m.model] || ChartRenderer.defaultColors[idx % 8],
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      tension: 0.2,
      pointRadius: 4
    }));

    this.lineChartInstance = ChartRenderer.getOrCreateChart('safety-trajectory-chart', {
      type: 'line',
      data: {
        labels: horizons,
        datasets
      },
      options: {
        scales: {
          y: {
            title: { display: true, text: 'Rollout NRMSE' },
            beginAtZero: true
          }
        }
      }
    });

    // Sensitivity Bar Chart
    const sensitivitySample = [...t3Rows]
      .filter(r => r.mean_counterfactual_sensitivity !== null)
      .slice(0, 12);

    const sensLabels = sensitivitySample.map(r => `${formatModel(r.model)} (${r.crop.slice(0, 3)})`);

    ChartRenderer.getOrCreateChart('safety-sensitivity-chart', {
      type: 'bar',
      data: {
        labels: sensLabels,
        datasets: [
          {
            label: 'Mean Sensitivity',
            data: sensitivitySample.map(r => r.mean_counterfactual_sensitivity),
            backgroundColor: '#06b6d4'
          },
          {
            label: 'Min Sensitivity',
            data: sensitivitySample.map(r => r.minimum_counterfactual_sensitivity),
            backgroundColor: '#6366f1'
          }
        ]
      },
      options: {
        scales: {
          y: {
            title: { display: true, text: 'Counterfactual Sensitivity Score' },
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
