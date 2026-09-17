/**
 * GEAS Dashboard - Section 5: Resource Efficiency & Trade-offs (Table 4)
 */

import { dataService } from '../core/data-loader.js';
import { filterStore } from '../core/filter-store.js';
import { DataTable } from '../components/data-table.js';
import { ChartRenderer } from '../components/chart-renderer.js';
import {
  formatNumber,
  formatCrop,
  formatModel,
  formatLatency,
  formatMemory,
  renderEligibilityBadge,
  renderEligibilityGuide
} from '../core/formatters.js';

export class ResourceSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.tableInstance = null;
    this.latencyScatter = null;
    this.sizeScatter = null;
    this.scatterViewMode = 'focus'; // 'focus' (NRMSE <= 1.0) or 'full' (all points)
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <!-- Control Bar for Scatter Scale -->
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
          <div class="academic-callout" style="flex: 1; min-width: 280px; margin: 0;">
            <div>
              <strong>⚡ Operational Trade-off & Pareto Frontier:</strong><br>
              Inference latency directly bounds real-time control cycle throughput, while model size impacts edge device memory footprint. Optimal transition models lie at the bottom-left frontier (low error, low resource usage).
            </div>
          </div>
          <div class="toggle-pill-group">
            <button type="button" class="toggle-pill-btn ${this.scatterViewMode === 'focus' ? 'active' : ''}" data-scatter-view="focus">Focus Pareto View (≤ 1.0 NRMSE)</button>
            <button type="button" class="toggle-pill-btn ${this.scatterViewMode === 'full' ? 'active' : ''}" data-scatter-view="full">Full Scale (All Points)</button>
          </div>
        </div>

        <div id="resource-outlier-notice" style="font-size: 0.75rem; color: #fbbf24; background: var(--accent-warning-bg); border: 1px solid var(--accent-warning-border); padding: 0.4rem 0.75rem; border-radius: 4px; display: none;"></div>

        <!-- Scatter Plots for Pareto Efficiency Frontiers -->
        <div class="resource-charts-grid">
          <!-- Scatter 1: Latency vs Error -->
          <div class="chart-card">
            <div class="chart-header">
              <div class="chart-title-group">
                <h3 class="chart-title">⚡ Latency vs. Rollout NRMSE</h3>
                <span class="chart-subtitle">X: Median Latency (ms) | Y: Validation Rollout Weighted NRMSE</span>
              </div>
            </div>
            <div class="chart-canvas-wrapper">
              <canvas id="resource-latency-scatter"></canvas>
            </div>
          </div>

          <!-- Scatter 2: Model Size vs Error -->
          <div class="chart-card">
            <div class="chart-header">
              <div class="chart-title-group">
                <h3 class="chart-title">💾 Model Artifact Size vs. Rollout NRMSE</h3>
                <span class="chart-subtitle">X: Serialized Model Size (MB) | Y: Validation Rollout Weighted NRMSE</span>
              </div>
            </div>
            <div class="chart-canvas-wrapper">
              <canvas id="resource-size-scatter"></canvas>
            </div>
          </div>
        </div>

        <!-- Table 4 Data Table -->
        <div id="table-4-container"></div>

        <!-- High-Visibility Eligibility Gating Criteria -->
        ${renderEligibilityGuide()}
      </div>
    `;

    this.bindEvents();
    this.renderTable();
    this.renderCharts();
  }

  bindEvents() {
    this.container.querySelectorAll('[data-scatter-view]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.container.querySelectorAll('[data-scatter-view]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.scatterViewMode = btn.getAttribute('data-scatter-view');
        this.renderCharts();
      });
    });
  }

  update() {
    const rows = filterStore.filterRows(dataService.getTable4());
    if (this.tableInstance) {
      this.tableInstance.setData(rows);
    }
    this.renderCharts();
  }

  renderTable() {
    const rows = filterStore.filterRows(dataService.getTable4());

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
        label: 'Eligible',
        format: (val) => renderEligibilityBadge(val)
      },
      {
        key: 'inference_latency_median_ms',
        label: 'Median Latency',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatLatency(val)}</span>`
      },
      {
        key: 'inference_latency_p95_ms',
        label: 'P95 Latency',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatLatency(val)}</span>`
      },
      {
        key: 'training_time_seconds',
        label: 'Train Time (s)',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 2)} s</span>`
      },
      {
        key: 'hpo_total_time_seconds',
        label: 'HPO Time (s)',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatNumber(val, 1)} s</span>`
      },
      {
        key: 'peak_cpu_memory_mb',
        label: 'CPU Memory',
        align: 'right',
        format: (val) => formatMemory(val)
      },
      {
        key: 'serialized_model_size_mb',
        label: 'Model Size',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatMemory(val)}</span>`
      },
      {
        key: 'artifact_directory_size_mb',
        label: 'Artifact Dir',
        align: 'right',
        format: (val) => `<span style="font-family: var(--font-family-mono);">${formatMemory(val)}</span>`
      }
    ];

    this.tableInstance = new DataTable({
      containerId: 'table-4-container',
      title: 'Table 4: Computational Resource & Operational Latency Benchmarks',
      columns,
      data: rows,
      initialSort: { key: 'inference_latency_median_ms', dir: 'asc' },
      pageSize: 15
    });
  }

  renderCharts() {
    const t4Rows = filterStore.filterRows(dataService.getTable4());
    const t1Rows = dataService.getTable1();
    if (t4Rows.length === 0) return;

    // Join T4 with T1 to get validation_rollout_weighted_nrmse for each point
    const allPoints = [];
    t4Rows.forEach(r4 => {
      const match = t1Rows.find(r1 =>
        r1.crop === r4.crop &&
        r1.model === r4.model &&
        r1.feature_variant === r4.feature_variant
      );
      if (match && match.validation_rollout_weighted_nrmse !== null) {
        allPoints.push({
          crop: r4.crop,
          model: r4.model,
          variant: r4.feature_variant,
          latency: r4.inference_latency_median_ms,
          size: r4.serialized_model_size_mb,
          nrmse: match.validation_rollout_weighted_nrmse,
          eligible: r4.deployment_eligible
        });
      }
    });

    const outlierNotice = document.getElementById('resource-outlier-notice');
    const divergentPoints = allPoints.filter(p => p.nrmse > 1.0);

    let displayPoints = allPoints;
    if (this.scatterViewMode === 'focus') {
      displayPoints = allPoints.filter(p => p.nrmse <= 1.0);
      if (outlierNotice) {
        if (divergentPoints.length > 0) {
          outlierNotice.style.display = 'block';
          outlierNotice.innerHTML = `⚠️ <strong>${divergentPoints.length} divergent model point(s)</strong> with NRMSE > 1.0 are outside the focus window. Switch to <strong>Full Scale</strong> to view all points.`;
        } else {
          outlierNotice.style.display = 'none';
        }
      }
    } else {
      if (outlierNotice) {
        outlierNotice.style.display = 'block';
        outlierNotice.innerHTML = `ℹ️ <strong>Full Scale Active:</strong> Showing all ${allPoints.length} model benchmark points without clipping.`;
      }
    }

    // Scatter 1: Latency vs NRMSE
    const datasetsLatency = displayPoints.map(p => ({
      label: `${formatModel(p.model)} (${p.crop.slice(0, 3)} - ${p.variant === 'with_quality_flags' ? 'w/flags' : 'wo/flags'})`,
      data: [{ x: p.latency, y: p.nrmse }],
      backgroundColor: ChartRenderer.modelColors[p.model] || '#6366f1',
      borderColor: p.variant === 'with_quality_flags' ? '#ffffff' : 'transparent',
      borderWidth: p.variant === 'with_quality_flags' ? 2 : 0,
      pointRadius: p.eligible ? 8 : 5,
      pointStyle: p.variant === 'with_quality_flags' ? 'circle' : 'triangle'
    }));

    this.latencyScatter = ChartRenderer.getOrCreateChart('resource-latency-scatter', {
      type: 'scatter',
      data: { datasets: datasetsLatency },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: Latency = ${ctx.raw.x.toFixed(2)} ms, NRMSE = ${ctx.raw.y.toFixed(4)}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Median Inference Latency (ms) (Lower is faster)' },
            beginAtZero: true
          },
          y: {
            title: { display: true, text: 'Weighted Rollout NRMSE (Lower is better)' },
            beginAtZero: true
          }
        }
      }
    });

    // Scatter 2: Model Size vs NRMSE
    const datasetsSize = displayPoints.map(p => ({
      label: `${formatModel(p.model)} (${p.crop.slice(0, 3)})`,
      data: [{ x: p.size, y: p.nrmse }],
      backgroundColor: ChartRenderer.modelColors[p.model] || '#6366f1',
      borderColor: p.variant === 'with_quality_flags' ? '#ffffff' : 'transparent',
      borderWidth: p.variant === 'with_quality_flags' ? 2 : 0,
      pointRadius: p.eligible ? 8 : 5,
      pointStyle: p.variant === 'with_quality_flags' ? 'circle' : 'triangle'
    }));

    this.sizeScatter = ChartRenderer.getOrCreateChart('resource-size-scatter', {
      type: 'scatter',
      data: { datasets: datasetsSize },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: Size = ${ctx.raw.x.toFixed(2)} MB, NRMSE = ${ctx.raw.y.toFixed(4)}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Serialized Model Size (MB) (Lower is leaner)' },
            beginAtZero: true
          },
          y: {
            title: { display: true, text: 'Weighted Rollout NRMSE (Lower is better)' },
            beginAtZero: true
          }
        }
      }
    });
  }
}
