/**
 * GEAS Dashboard - Section 2: Candidate Ranking (Table 1)
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
  renderEligibilityGuide,
  compareCandidates
} from '../core/formatters.js';

export class RankingSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.tableInstance = null;
    this.rolloutChart = null;
    this.skillChart = null;
    this.viewMode = 'focus'; // 'focus' (NRMSE <= 2.0) or 'full' (all models including divergent)
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <!-- Official HPO Objective Formula Notice -->
        <div class="academic-callout">
          <div>
            <strong>📐 Official HPO Objective Function (Weighted Rollout Metric):</strong><br>
            <code>Loss = 0.10 × One-Step(5m) + 0.10 × Rollout(15m) + 0.25 × Rollout(30m) + 0.45 × Rollout(60m) + 0.10 × Drift Slope</code>
          </div>
          <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.25rem;">
            Primary model selection criterion. Prioritizes long-term rollout fidelity (60m: 45%) and drift resistance while balancing 1-step accuracy.
          </div>
        </div>

        <!-- Visual Comparison Charts with Outlier View Switcher -->
        <div class="ranking-charts-row">
          <!-- Rollout NRMSE Grouped Bar Chart -->
          <div class="chart-card">
            <div class="chart-header">
              <div class="chart-title-group">
                <h3 class="chart-title">📊 Recursive Rollout NRMSE by Horizon</h3>
                <span class="chart-subtitle">15m, 30m, and 60m recursive rollouts (Lower is better)</span>
              </div>
              <div class="chart-controls">
                <div class="toggle-pill-group">
                  <button type="button" class="toggle-pill-btn ${this.viewMode === 'focus' ? 'active' : ''}" data-ranking-view="focus">Focus (≤ 2.0)</button>
                  <button type="button" class="toggle-pill-btn ${this.viewMode === 'full' ? 'active' : ''}" data-ranking-view="full">Full Scale (All)</button>
                </div>
              </div>
            </div>

            <div id="ranking-outlier-notice" style="font-size: 0.75rem; color: #fbbf24; background: var(--accent-warning-bg); border: 1px solid var(--accent-warning-border); padding: 0.35rem 0.65rem; border-radius: 4px; display: none;"></div>

            <div class="chart-canvas-wrapper">
              <canvas id="ranking-rollout-chart"></canvas>
            </div>
          </div>

          <!-- Persistence Skill Chart -->
          <div class="chart-card">
            <div class="chart-header">
              <div class="chart-title-group">
                <h3 class="chart-title">📈 Persistence Skill Score by Horizon</h3>
                <span class="chart-subtitle">Skill = 1 - (RMSE_model / RMSE_persistence) | > 0 denotes beating Persistence baseline</span>
              </div>
            </div>
            <div class="chart-canvas-wrapper">
              <canvas id="ranking-skill-chart"></canvas>
            </div>
          </div>
        </div>

        <!-- Comprehensive Candidate Ranking Table -->
        <div id="table-1-container"></div>

        <!-- Table 1 Key Metrics Explanation Card: Safety Breach & Skill (60m) -->
        <div class="ranking-metrics-guide-card" style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: var(--radius-lg); padding: 1.25rem 1.5rem; display: flex; flex-direction: column; gap: 0.85rem; box-shadow: var(--shadow-sm);">
          <div style="font-size: 0.98rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 0.5rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem;">
            <span>📖</span> Table 1 주요 지표 상세 설명 (Safety Breach & Skill 60M)
          </div>
          
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem;">
            <!-- Safety Breach Explanation -->
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #ef4444; border-radius: var(--radius-md); padding: 1rem;">
              <div style="font-weight: 700; color: #991b1b; font-size: 0.9rem; display: flex; align-items: center; justify-content: space-between;">
                <span>🛡️ Safety Breach (물리 허용 범위 위반율)</span>
                <span style="font-size: 0.75rem; background: #fee2e2; color: #991b1b; padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 700;">0.000 = 완벽 준수</span>
              </div>
              <p style="font-size: 0.83rem; color: #334155; margin: 0.5rem 0 0 0; line-height: 1.55;">
                <strong>15·30·60분 rollout 중 예측값이 물리 허용 범위를 벗어난 비율의 최댓값</strong>입니다.<br>
                • <strong>0 (0.000)</strong>: 모든 horizon(15/30/60분)에서 물리 범위 위반이 전혀 없었음을 의미 (안전성 통과).<br>
                • <strong>0보다 큼 (&gt;0)</strong>: 세 horizon 중 하나 이상에서 비현실적인(unphysical) 예측이 발생했음을 의미 (배포 부적격).
              </p>
            </div>

            <!-- Skill (60m) Explanation & Formula -->
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #f59e0b; border-radius: var(--radius-md); padding: 1rem;">
              <div style="font-weight: 700; color: #92400e; font-size: 0.9rem; display: flex; align-items: center; justify-content: space-between;">
                <span>📈 Skill (60m) (Persistence 대비 예측 개선도)</span>
                <span style="font-size: 0.75rem; background: #fef3c7; color: #92400e; padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 700;">양수(&gt;0) = Baseline 능가</span>
              </div>
              <div style="font-family: var(--font-family-mono); font-size: 0.82rem; background: #ffffff; border: 1px solid #cbd5e1; padding: 0.35rem 0.6rem; border-radius: 4px; margin: 0.4rem 0; color: #0f172a; font-weight: 700;">
                Skill<sub>60m</sub> = 1 - (RMSE<sub>model, 60m</sub> / RMSE<sub>persistence, 60m</sub>)
              </div>
              <p style="font-size: 0.83rem; color: #334155; margin: 0.25rem 0 0 0; line-height: 1.55;">
                60분 rollout에서 해당 모델이 <strong>Persistence baseline보다 얼마나 나은지</strong>를 나타냅니다.<br>
                • <strong>양수 (&gt;0)</strong>: Persistence보다 우수 (예: <code>+0.20</code>은 Persistence 대비 RMSE 약 20% 감소).<br>
                • <strong>0</strong>: Persistence와 동일 | <strong>음수 (&lt;0)</strong>: Persistence보다 오차가 커 롤아웃 시 성능 열화.
              </p>
            </div>
          </div>
        </div>

        <!-- High-Visibility Eligibility Gating Criteria Guide -->
        ${renderEligibilityGuide()}
      </div>
    `;

    this.bindEvents();
    this.renderTable();
    this.renderCharts();
  }

  bindEvents() {
    this.container.querySelectorAll('[data-ranking-view]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.container.querySelectorAll('[data-ranking-view]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.viewMode = btn.getAttribute('data-ranking-view');
        this.renderCharts();
      });
    });
  }

  update() {
    const filteredRows = filterStore.filterRows(dataService.getTable1());
    if (this.tableInstance) {
      this.tableInstance.setData(filteredRows);
    }
    this.renderCharts();
  }

  renderTable() {
    const rawRows = filterStore.filterRows(dataService.getTable1());

    // Table 1 Columns with explicit HPO Formula Tooltip & Two-Tier Ranking
    const columns = [
      {
        key: 'crop',
        label: 'Crop',
        format: (val) => formatCrop(val)
      },
      {
        key: 'model',
        label: 'Candidate Model',
        format: (val, row) => `<strong>${formatModel(val)}</strong>`
      },
      {
        key: 'feature_variant',
        label: 'Feature Variant',
        format: (val) => `<span class="badge badge-variant">${val === 'with_quality_flags' ? 'With Flags' : 'Without Flags'}</span>`
      },
      {
        key: 'deployment_eligible',
        label: 'Eligibility Gate',
        format: (val, row) => renderEligibilityBadge(val, row.baseline_only)
      },
      {
        key: 'validation_rollout_weighted_nrmse',
        label: 'Weighted NRMSE',
        align: 'right',
        tooltip: 'HPO Objective: 0.10*1-step + 0.10*15m + 0.25*30m + 0.45*60m + 0.10*drift',
        format: (val, row) => {
          const isBest = row.deployment_eligible && !row.baseline_only && val < 0.25;
          return `<span style="font-family: var(--font-family-mono); font-weight: 700; color: ${isBest ? '#34d399' : 'inherit'};">${formatNumber(val, 4)}</span>`;
        }
      },
      {
        key: 'one_step_nrmse',
        label: '1-Step (5m)',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'rollout_15min_nrmse',
        label: '15m Rollout',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'rollout_30min_nrmse',
        label: '30m Rollout',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'rollout_60min_nrmse',
        label: '60m Rollout',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'skill_vs_persistence_60min',
        label: 'Skill (60m)',
        align: 'right',
        format: (val, row) => {
          if (row.baseline_only) return `<span style="color: var(--text-muted);">0.0000 (Base)</span>`;
          const num = Number(val);
          const color = num > 0 ? '#34d399' : '#f87171';
          return `<span style="font-family: var(--font-family-mono); font-weight: 600; color: ${color};">${num > 0 ? '+' : ''}${formatNumber(num, 4)}</span>`;
        }
      },
      {
        key: 'physical_violation_rate_max',
        label: 'Safety Breach',
        align: 'right',
        format: (val) => {
          const rate = Number(val || 0);
          if (rate > 0) {
            return `<span class="badge badge-physical-violation" title="Exceeded valid greenhouse physical bounds">⚠️ ${rate.toExponential(2)}</span>`;
          }
          return `<span style="color: #34d399; font-size: 0.8rem;">0 (Safe)</span>`;
        }
      },
      {
        key: 'nan_inf_count_total',
        label: 'NaN/Inf',
        align: 'center',
        format: (val) => {
          const count = Number(val || 0);
          return count > 0 ? `<span style="color: #ef4444; font-weight: 700;">⚠️ ${count}</span>` : `<span style="color: var(--text-muted);">0</span>`;
        }
      }
    ];

    this.tableInstance = new DataTable({
      containerId: 'table-1-container',
      title: 'Table 1: Candidate Screening & Multi-Horizon Rollout Ranking (Validation Only)',
      columns,
      data: rawRows,
      initialSort: { key: 'validation_rollout_weighted_nrmse', dir: 'asc' },
      prioritizeEligible: true, // Two-tier sorting: Deployable candidates first, then NRMSE
      pageSize: 15
    });
  }

  renderCharts() {
    const filteredRows = filterStore.filterRows(dataService.getTable1());
    if (filteredRows.length === 0) return;

    // Separate focus list and divergent outliers
    const outlierNotice = document.getElementById('ranking-outlier-notice');
    let displayList = [...filteredRows];

    const divergentModels = displayList.filter(r => (r.validation_rollout_weighted_nrmse || 0) > 2.0);

    if (this.viewMode === 'focus') {
      displayList = displayList.filter(r => (r.validation_rollout_weighted_nrmse || 0) <= 2.0);
      if (outlierNotice) {
        if (divergentModels.length > 0) {
          outlierNotice.style.display = 'block';
          outlierNotice.innerHTML = `⚠️ <strong>${divergentModels.length} divergent candidate(s)</strong> (e.g. ${divergentModels.map(m => `${m.model} [${m.crop}]`).join(', ')}) with NRMSE > 2.0 are hidden in Focus View. Switch to <strong>Full Scale</strong> to view all values.`;
        } else {
          outlierNotice.style.display = 'none';
        }
      }
    } else {
      if (outlierNotice) {
        outlierNotice.style.display = 'block';
        outlierNotice.innerHTML = `ℹ️ <strong>Full Scale View Active:</strong> Showing all candidates including high-error and divergent models without any clipping.`;
      }
    }

    displayList.sort((a, b) => (a.validation_rollout_weighted_nrmse || 0) - (b.validation_rollout_weighted_nrmse || 0));
    const finalSlice = displayList.slice(0, 18);

    const labels = finalSlice.map(r => `${formatModel(r.model)} (${r.crop.slice(0, 3)} - ${r.feature_variant === 'with_quality_flags' ? 'w/flags' : 'wo/flags'})`);

    // 1. Rollout Grouped Bar Chart
    this.rolloutChart = ChartRenderer.getOrCreateChart('ranking-rollout-chart', {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: '15-Min Rollout NRMSE',
            data: finalSlice.map(r => r.rollout_15min_nrmse),
            backgroundColor: '#06b6d4'
          },
          {
            label: '30-Min Rollout NRMSE',
            data: finalSlice.map(r => r.rollout_30min_nrmse),
            backgroundColor: '#6366f1'
          },
          {
            label: '60-Min Rollout NRMSE',
            data: finalSlice.map(r => r.rollout_60min_nrmse),
            backgroundColor: '#a855f7'
          }
        ]
      },
      options: {
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.raw?.toFixed(4)}`
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: 'Rollout NRMSE (Lower is better)' },
            beginAtZero: true
          },
          x: {
            ticks: { maxRotation: 45, minRotation: 20 }
          }
        }
      }
    });

    // 2. Persistence Skill Chart (Unclipped)
    this.skillChart = ChartRenderer.getOrCreateChart('ranking-skill-chart', {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: '15-min Skill',
            data: finalSlice.map(r => r.skill_vs_persistence_15min),
            backgroundColor: '#10b981'
          },
          {
            label: '30-min Skill',
            data: finalSlice.map(r => r.skill_vs_persistence_30min),
            backgroundColor: '#3b82f6'
          },
          {
            label: '60-min Skill',
            data: finalSlice.map(r => r.skill_vs_persistence_60min),
            backgroundColor: '#f59e0b'
          }
        ]
      },
      options: {
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const val = ctx.raw;
                const isOutlier = val < -2;
                return `${ctx.dataset.label}: ${val > 0 ? '+' : ''}${val?.toFixed(4)}${isOutlier ? ' (Divergent Model)' : ''}`;
              }
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: 'Persistence Skill (>0 beats baseline)' }
          },
          x: {
            ticks: { maxRotation: 45, minRotation: 20 }
          }
        }
      }
    });
  }
}
