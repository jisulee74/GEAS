/**
 * GEAS Dashboard - Section 3: One-Step Target Performance (Table 2)
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
  TARGET_LABELS,
  renderEligibilityBadge,
  renderEligibilityGuide
} from '../core/formatters.js';

export class OneStepSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.selectedTarget = 'indoor_temperature'; // 'indoor_temperature', 'indoor_humidity', 'indoor_co2', 'all'
    this.selectedMetric = 'nrmse'; // 'nrmse', 'mae', 'rmse', 'r2', 'absolute_error_q90', 'absolute_error_cvar90'
    this.tableInstance = null;
    this.chartInstance = null;
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <!-- Control Bar for Target & Metric Selection -->
        <div class="onestep-controls-bar">
          <div class="onestep-target-selector">
            <span style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase;">Target State:</span>
            <button type="button" class="target-pill-btn ${this.selectedTarget === 'indoor_temperature' ? 'active' : ''}" data-target="indoor_temperature">🌡️ Temp (°C)</button>
            <button type="button" class="target-pill-btn ${this.selectedTarget === 'indoor_humidity' ? 'active' : ''}" data-target="indoor_humidity">💧 Humidity (%)</button>
            <button type="button" class="target-pill-btn ${this.selectedTarget === 'indoor_co2' ? 'active' : ''}" data-target="indoor_co2">🌿 CO₂ (ppm)</button>
            <button type="button" class="target-pill-btn ${this.selectedTarget === 'all' ? 'active' : ''}" data-target="all">All Targets</button>
          </div>

          <div style="display: flex; align-items: center; gap: 0.75rem;">
            <label style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase;">Metric:</label>
            <select class="filter-select" id="onestep-metric-select" style="width: auto;">
              <option value="nrmse" selected>NRMSE (Normalized - Default)</option>
              <option value="r2">R² (Variance Explained - Higher is better)</option>
              <option value="mae">MAE (Mean Absolute Error)</option>
              <option value="rmse">RMSE (Root Mean Square Error)</option>
              <option value="absolute_error_q90">Q90 (90th percentile Tail Error)</option>
              <option value="absolute_error_cvar90">CVaR90 (Conditional Tail Risk)</option>
            </select>
          </div>
        </div>

        <!-- Academic Direction Notice -->
        <div class="academic-callout">
          <div>
            <strong>📌 Target Evaluation Direction:</strong> 
            <span id="onestep-metric-guide">NRMSE is dimensionless and normalized by target variance (Lower is better).</span>
          </div>
        </div>

        <!-- Metric Chart -->
        <div class="chart-card">
          <div class="chart-header">
            <div class="chart-title-group">
              <h3 class="chart-title" id="onestep-chart-title">📊 5-Min One-Step Model Comparison</h3>
              <span class="chart-subtitle" id="onestep-chart-subtitle">Comparing candidate models on selected target state</span>
            </div>
          </div>
          <div class="chart-canvas-wrapper">
            <canvas id="onestep-chart"></canvas>
          </div>
        </div>

        <!-- Table 2 Data Table -->
        <div id="table-2-container"></div>

        <!-- High-Visibility Eligibility Gating Criteria Guide -->
        ${renderEligibilityGuide()}
      </div>
    `;

    this.bindEvents();
    this.renderTable();
    this.renderChart();
  }

  bindEvents() {
    const targetBtns = this.container.querySelectorAll('[data-target]');
    targetBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        targetBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.selectedTarget = btn.getAttribute('data-target');
        this.update();
      });
    });

    const metricSelect = document.getElementById('onestep-metric-select');
    if (metricSelect) {
      metricSelect.addEventListener('change', (e) => {
        this.selectedMetric = e.target.value;
        this.updateMetricGuide();
        this.update();
      });
    }
  }

  updateMetricGuide() {
    const guideEl = document.getElementById('onestep-metric-guide');
    if (!guideEl) return;
    if (this.selectedMetric === 'r2') {
      guideEl.innerHTML = `<strong>R² (Coefficient of Determination)</strong>: Higher is better (1.0 = perfect prediction). Reflects explained variance.`;
    } else if (this.selectedMetric === 'nrmse') {
      guideEl.innerHTML = `<strong>NRMSE (Normalized RMSE)</strong>: Lower is better. Enables fair comparison across Temperature, Humidity, and CO₂.`;
    } else {
      guideEl.innerHTML = `<strong>${this.selectedMetric.toUpperCase()}</strong>: Native physical unit error (Lower is better).`;
    }
  }

  update() {
    const rows = this.getFilteredData();
    if (this.tableInstance) {
      this.tableInstance.setData(rows);
    }
    this.renderChart();
  }

  getFilteredData() {
    let rows = filterStore.filterRows(dataService.getTable2());
    if (this.selectedTarget !== 'all') {
      rows = rows.filter(r => r.target === this.selectedTarget);
    }
    return rows;
  }

  renderTable() {
    const rows = this.getFilteredData();

    const columns = [
      {
        key: 'crop',
        label: 'Crop',
        format: (val) => formatCrop(val)
      },
      {
        key: 'target',
        label: 'Target State',
        format: (val) => `<strong>${TARGET_LABELS[val] || val}</strong>`
      },
      {
        key: 'model',
        label: 'Model',
        format: (val) => formatModel(val)
      },
      {
        key: 'feature_variant',
        label: 'Variant',
        format: (val) => `<span class="badge badge-variant">${val === 'with_quality_flags' ? 'With Flags' : 'Without Flags'}</span>`
      },
      {
        key: 'deployment_eligible',
        label: 'Eligible',
        format: (val) => renderEligibilityBadge(val)
      },
      {
        key: 'nrmse',
        label: 'NRMSE (↓)',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono); font-weight: 700; color: #38bdf8;">${formatNumber(val, 4)}</span>`
      },
      {
        key: 'r2',
        label: 'R² (↑)',
        align: 'right',
        format: (val) => {
          const num = Number(val);
          const color = num >= 0.95 ? '#34d399' : num >= 0.8 ? '#fbbf24' : '#f87171';
          return `<span style="font-family: var(--font-family-mono); color: ${color};">${formatNumber(num, 4)}</span>`;
        }
      },
      {
        key: 'mae',
        label: 'MAE (↓)',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 3)}</span>`
      },
      {
        key: 'rmse',
        label: 'RMSE (↓)',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 3)}</span>`
      },
      {
        key: 'absolute_error_q90',
        label: 'Q90 (↓)',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 3)}</span>`
      },
      {
        key: 'absolute_error_cvar90',
        label: 'CVaR90 (↓)',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 3)}</span>`
      }
    ];

    this.tableInstance = new DataTable({
      containerId: 'table-2-container',
      title: 'Table 2: 5-Minute One-Step Target Detailed Metrics (Validation Only)',
      columns,
      data: rows,
      initialSort: { key: 'nrmse', dir: 'asc' },
      pageSize: 15
    });
  }

  renderChart() {
    const rows = this.getFilteredData();
    if (rows.length === 0) return;

    const titleEl = document.getElementById('onestep-chart-title');
    const subTitleEl = document.getElementById('onestep-chart-subtitle');

    if (titleEl) {
      titleEl.textContent = `📊 5-Min One-Step ${this.selectedMetric.toUpperCase()} (${this.selectedTarget === 'all' ? 'All Targets' : TARGET_LABELS[this.selectedTarget]})`;
    }

    // Prepare chart grouped by model
    const displayRows = rows.slice(0, 24);
    const labels = displayRows.map(r => `${formatModel(r.model)} (${r.crop.slice(0, 3)} - ${r.feature_variant === 'with_quality_flags' ? 'w/flags' : 'wo/flags'})`);
    const values = displayRows.map(r => r[this.selectedMetric]);

    const isR2 = this.selectedMetric === 'r2';

    this.chartInstance = ChartRenderer.getOrCreateChart('onestep-chart', {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: `${this.selectedMetric.toUpperCase()}`,
            data: values,
            backgroundColor: displayRows.map(r => ChartRenderer.modelColors[r.model] || '#6366f1'),
            borderRadius: 4
          }
        ]
      },
      options: {
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            title: { display: true, text: `${this.selectedMetric.toUpperCase()} (${isR2 ? 'Higher is better' : 'Lower is better'})` },
            beginAtZero: !isR2
          },
          x: {
            ticks: { maxRotation: 45, minRotation: 20 }
          }
        }
      }
    });
  }
}
