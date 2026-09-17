/**
 * GEAS Dashboard - Section 6: Quality-Flag Feature Ablation (Table 5)
 */

import { dataService } from '../core/data-loader.js';
import { filterStore } from '../core/filter-store.js';
import { DataTable } from '../components/data-table.js';
import { ChartRenderer } from '../components/chart-renderer.js';
import {
  formatNumber,
  formatCrop,
  formatModel,
  formatVariantEffect,
  renderQualityEffectBadge,
  renderEligibilityBadge,
  renderEligibilityGuide
} from '../core/formatters.js';

export class AblationSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.tableInstance = null;
    this.zoomMode = 'standard'; // 'standard' or 'full'
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <!-- Academic Ablation Principle Callout -->
        <div class="ablation-header-callout">
          <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 0.25rem;">
            🔬 Paired Quality-Flag Ablation & Selection Rule (Validation Rollout Windows)
          </div>
          <div>
            <strong>Difference Definition:</strong> <code>with_quality_flags - without_quality_flags</code>.<br>
            • <span style="color: #34d399; font-weight: 600;">Negative Value (&lt; 0):</span> Quality flags significantly improved rollout NRMSE.<br>
            • <span style="color: #f87171; font-weight: 600;">Positive Value (&gt; 0):</span> Omitting quality flags achieved lower rollout error.<br>
            • <span style="color: var(--text-secondary); font-weight: 600;">95% CI spanning 0:</span> No statistically significant difference. Under parsimony principles, the leaner <code>without_quality_flags</code> variant is preferred.
          </div>
        </div>

        <!-- Diverging Bar Chart Card with Outlier View Switcher -->
        <div class="chart-card">
          <div class="chart-header">
            <div class="chart-title-group">
              <h3 class="chart-title">📊 Paired Rollout NRMSE Difference & 95% Bootstrap CI</h3>
              <span class="chart-subtitle">Centered diverging effect size with 95% confidence interval bars</span>
            </div>
            <div class="chart-controls">
              <div class="toggle-pill-group">
                <button type="button" class="toggle-pill-btn ${this.zoomMode === 'standard' ? 'active' : ''}" data-zoom="standard">Standard Zoom (±0.08)</button>
                <button type="button" class="toggle-pill-btn ${this.zoomMode === 'full' ? 'active' : ''}" data-zoom="full">Full Scale (All Outliers)</button>
              </div>
            </div>
          </div>

          <div id="ablation-diverging-chart-container" class="ablation-diverging-container"></div>
        </div>

        <!-- Table 5 Data Table -->
        <div id="table-5-container"></div>

        <!-- High-Visibility Eligibility Gating Criteria Guide -->
        ${renderEligibilityGuide()}
      </div>
    `;

    this.bindEvents();
    this.renderTable();
    this.renderChart();
  }

  bindEvents() {
    const zoomBtns = this.container.querySelectorAll('[data-zoom]');
    zoomBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        zoomBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.zoomMode = btn.getAttribute('data-zoom');
        this.renderChart();
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
    const { crop, model } = filterStore.getState();
    let rows = dataService.getTable5();

    if (crop !== 'all') {
      rows = rows.filter(r => r.crop.toLowerCase() === crop.toLowerCase());
    }
    if (model !== 'all') {
      rows = rows.filter(r => r.model.toLowerCase() === model.toLowerCase());
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
        key: 'quality_flag_effect',
        label: 'Ablation Effect',
        format: (val) => renderQualityEffectBadge(val)
      },
      {
        key: 'with_flags_deployment_eligible',
        label: 'With-Flags Gate',
        format: (val, row) => renderEligibilityBadge(val, row.baseline_only)
      },
      {
        key: 'without_flags_deployment_eligible',
        label: 'Without-Flags Gate',
        format: (val, row) => renderEligibilityBadge(val, row.baseline_only)
      },
      {
        key: 'with_flags_paired_rollout_nrmse',
        label: 'With-Flags NRMSE',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'without_flags_paired_rollout_nrmse',
        label: 'Without-Flags NRMSE',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'difference_with_minus_without',
        label: 'Difference (With - Without)',
        align: 'right',
        format: (val) => {
          const num = Number(val);
          const color = num < 0 ? '#34d399' : num > 0 ? '#f87171' : 'var(--text-secondary)';
          return `<span style="font-family: var(--font-family-mono); font-weight: 700; color: ${color};">${num >= 0 ? '+' : ''}${formatNumber(num, 5)}</span>`;
        }
      },
      {
        key: 'bootstrap_ci95_lower',
        label: '95% CI Lower',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 5)}</span>`
      },
      {
        key: 'bootstrap_ci95_upper',
        label: '95% CI Upper',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 5)}</span>`
      },
      {
        key: 'paired_validation_window_count',
        label: 'Paired Windows',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${val}</span>`
      }
    ];

    this.tableInstance = new DataTable({
      containerId: 'table-5-container',
      title: 'Table 5: Quality-Flag Feature Ablation (Paired Validation Rollout Windows)',
      columns,
      data: rows,
      initialSort: { key: 'difference_with_minus_without', dir: 'asc' },
      pageSize: 15
    });
  }

  renderChart() {
    const rows = this.getFilteredRows();
    ChartRenderer.renderDivergingAblationChart({
      containerId: 'ablation-diverging-chart-container',
      data: rows,
      zoomMode: this.zoomMode
    });
  }
}
