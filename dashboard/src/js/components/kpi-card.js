/**
 * GEAS Dashboard - KPI Card Component
 */

export function createKpiCard({ title, value, unit = '', icon = '📊', subtitle = '', accent = '#6366f1', trend = null }) {
  let trendHtml = '';
  if (trend) {
    const trendClass = trend.type || 'neutral';
    trendHtml = `<span class="kpi-trend ${trendClass}">${trend.text}</span>`;
  }

  return `
    <div class="kpi-card" style="--kpi-accent: ${accent};">
      <div class="kpi-header">
        <span class="kpi-title">${title}</span>
        <div class="kpi-icon">${icon}</div>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${value}</span>
        ${unit ? `<span class="kpi-unit">${unit}</span>` : ''}
      </div>
      <div class="kpi-footer">
        ${trendHtml}
        <span>${subtitle}</span>
      </div>
    </div>
  `;
}
