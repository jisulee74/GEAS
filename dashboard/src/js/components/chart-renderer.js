/**
 * GEAS Dashboard - Chart Renderer Component (Chart.js & Custom SVG Visualizers)
 */

export class ChartRenderer {
  static defaultColors = [
    '#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#3b82f6', '#14b8a6'
  ];

  static modelColors = {
    persistence: '#94a3b8',
    linear_svr: '#38bdf8',
    knn: '#34d399',
    extra_trees: '#a78bfa',
    lightgbm: '#fbbf24',
    xgboost: '#f87171',
    mlp: '#f43f5e',
    linear_regression: '#818cf8'
  };

  /**
   * Helper to create or destroy existing Chart.js instances on a canvas
   */
  static getOrCreateChart(canvasId, config) {
    const canvas = typeof canvasId === 'string' ? document.getElementById(canvasId) : canvasId;
    if (!canvas) return null;

    if (canvas._chartInstance) {
      canvas._chartInstance.destroy();
    }

    // Configure theme colors for Chart.js (Crisp Light Theme)
    const textColor = '#334155';
    const gridColor = '#f1f5f9';

    const mergedOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#0f172a', font: { family: 'Inter', size: 12, weight: 600 } }
        },
        tooltip: {
          backgroundColor: '#ffffff',
          titleColor: '#0f172a',
          bodyColor: '#334155',
          borderColor: '#cbd5e1',
          borderWidth: 1,
          padding: 10,
          boxPadding: 4,
          cornerRadius: 8,
          shadowOffsetX: 0,
          shadowOffsetY: 4,
          shadowBlur: 12,
          shadowColor: 'rgba(0, 0, 0, 0.08)'
        },
        ...(config.options?.plugins || {})
      },
      scales: {
        x: {
          ticks: { color: textColor, font: { family: 'Inter', size: 11, weight: 500 } },
          grid: { color: gridColor },
          ...(config.options?.scales?.x || {})
        },
        y: {
          ticks: { color: textColor, font: { family: 'Inter', size: 11, weight: 500 } },
          grid: { color: gridColor },
          ...(config.options?.scales?.y || {})
        }
      },
      ...(config.options || {})
    };

    const newChart = new Chart(canvas, {
      ...config,
      options: mergedOptions
    });

    canvas._chartInstance = newChart;
    return newChart;
  }

  /**
   * Render Diverging Bar Chart for Table 5 Quality Flag Ablation with 95% Bootstrap CI
   */
  static renderDivergingAblationChart({ containerId, data, zoomMode = 'standard' }) {
    const container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!container) return;

    if (!data || data.length === 0) {
      container.innerHTML = `<div style="text-align: center; padding: 3rem; color: var(--text-muted);">No ablation data available under current filters.</div>`;
      return;
    }

    // Sort: improved first, then neutral, then worsened
    const sorted = [...data].sort((a, b) => a.difference_with_minus_without - b.difference_with_minus_without);

    // Zoom limits: Standard view restricts range to [-0.08, 0.05] or dynamic max, flagging extreme outliers
    const outlierThreshold = -0.15;
    const maxVal = zoomMode === 'standard' ? 0.05 : Math.max(...sorted.map(d => Math.abs(d.difference_with_minus_without || 0)), 0.05);
    const minVal = zoomMode === 'standard' ? -0.08 : Math.min(...sorted.map(d => d.difference_with_minus_without || 0), -0.08);

    const range = Math.max(Math.abs(minVal), Math.abs(maxVal));

    let rowsHtml = sorted.map(item => {
      const diff = item.difference_with_minus_without;
      const ciLower = item.bootstrap_ci95_lower;
      const ciUpper = item.bootstrap_ci95_upper;
      const isOutlier = zoomMode === 'standard' && diff < minVal;

      // Calculate bar percentage from center (50%)
      let barWidthPct = 0;
      let barLeftPct = 50;
      let barClass = 'no-diff';

      if (diff < 0) {
        // Improved (With quality flags better) -> goes to the left of center
        barClass = 'improved';
        const displayVal = Math.max(diff, minVal);
        barWidthPct = (Math.abs(displayVal) / range) * 50;
        barLeftPct = 50 - barWidthPct;
      } else if (diff > 0) {
        // Worsened (Without quality flags better) -> goes to the right of center
        barClass = 'worsened';
        const displayVal = Math.min(diff, maxVal);
        barWidthPct = (displayVal / range) * 50;
        barLeftPct = 50;
      }

      // CI calculation (clamped to container)
      let ciLeftPct = 50;
      let ciWidthPct = 0;
      if (ciLower !== null && ciUpper !== null) {
        const clampedLower = Math.max(Math.min(ciLower, maxVal), minVal);
        const clampedUpper = Math.max(Math.min(ciUpper, maxVal), minVal);
        const posLower = 50 + (clampedLower / range) * 50;
        const posUpper = 50 + (clampedUpper / range) * 50;
        ciLeftPct = Math.min(posLower, posUpper);
        ciWidthPct = Math.abs(posUpper - posLower);
      }

      const outlierNotice = isOutlier
        ? `<span style="color: #f87171; font-weight: 700; font-size: 0.72rem; margin-left: 0.35rem;" title="Extreme outlier (Diff: ${diff.toFixed(4)}). Switch to Full Scale to expand axis.">⚠️ Outlier [${diff.toFixed(2)}]</span>`
        : '';

      return `
        <div class="diverging-row">
          <div class="diverging-label">
            <span>${item.model.replace(/_/g, ' ').toUpperCase()}${outlierNotice}</span>
            <span class="diverging-label-sub">${item.crop} (${item.paired_validation_window_count} windows)</span>
          </div>
          <div class="diverging-bar-area" title="Diff: ${diff >= 0 ? '+' : ''}${diff.toFixed(6)} | 95% CI: [${ciLower?.toFixed(6)}, ${ciUpper?.toFixed(6)}]">
            <div class="diverging-center-line"></div>
            <div class="diverging-bar ${barClass}" style="left: ${barLeftPct}%; width: ${Math.max(barWidthPct, 2)}%;"></div>
            ${ciWidthPct > 0 ? `<div class="diverging-ci-line" style="left: ${ciLeftPct}%; width: ${Math.max(ciWidthPct, 2)}%;"></div>` : ''}
          </div>
          <div class="diverging-value-text">
            <strong>${diff >= 0 ? '+' : ''}${diff.toFixed(5)}</strong>
            <div style="font-size: 0.68rem; color: var(--text-muted);">CI: [${ciLower?.toFixed(4)}, ${ciUpper?.toFixed(4)}]</div>
          </div>
        </div>
      `;
    }).join('');

    container.innerHTML = `
      <div class="diverging-chart-wrapper">
        <div style="display: grid; grid-template-columns: 180px 1fr 140px; gap: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--border-color); font-size: 0.75rem; font-weight: 700; color: var(--text-secondary);">
          <div>CROP / MODEL</div>
          <div style="display: flex; justify-content: space-between; padding: 0 0.5rem;">
            <span style="color: #34d399;">◀ With Flags Better (Negative Diff)</span>
            <span style="color: var(--text-muted);">| 0 (No Diff) |</span>
            <span style="color: #f87171;">Without Flags Better (Positive Diff) ▶</span>
          </div>
          <div style="text-align: right;">DIFF & 95% CI</div>
        </div>
        ${rowsHtml}
      </div>
    `;
  }
}
