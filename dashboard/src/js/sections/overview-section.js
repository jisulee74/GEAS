/**
 * GEAS Dashboard - Section 1: Overview & Provisional Leading Candidates
 */

import { dataService } from '../core/data-loader.js';
import { filterStore } from '../core/filter-store.js';
import { createKpiCard } from '../components/kpi-card.js';
import {
  formatNumber,
  formatPercent,
  formatCrop,
  formatModel,
  formatVariant,
  renderEligibilityBadge,
  compareCandidates
} from '../core/formatters.js';

export class OverviewSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.researcherDecisions = {
      strawberry: { model: 'lightgbm', variant: 'without_quality_flags', note: 'Lowest weighted rollout NRMSE (0.2373) with zero safety breaches and sub-16ms latency.' },
      melon: { model: 'xgboost', variant: 'without_quality_flags', note: 'Superior rollout stability (0.1205) and high counterfactual action sensitivity.' },
      cucumber: { model: 'lightgbm', variant: 'without_quality_flags', note: 'Tied NRMSE (0.1117) with with_flags; selected without_quality_flags under parsimony principle.' }
    };
  }

  render() {
    if (!this.container) return;

    const filterState = filterStore.getState();
    const t1Filtered = filterStore.filterRows(dataService.getTable1());
    const t3Filtered = filterStore.filterRows(dataService.getTable3());
    const t4Filtered = filterStore.filterRows(dataService.getTable4());
    
    // Filter Table 5 according to active crop and model
    let t5Filtered = dataService.getTable5();
    if (filterState.crop !== 'all') {
      t5Filtered = t5Filtered.filter(r => r.crop.toLowerCase() === filterState.crop.toLowerCase());
    }
    if (filterState.model !== 'all') {
      t5Filtered = t5Filtered.filter(r => r.model.toLowerCase() === filterState.model.toLowerCase());
    }

    // Dynamic KPI counts based on active filter
    const totalCrops = new Set(t1Filtered.map(r => r.crop)).size;
    const totalCandidates = t1Filtered.length;
    const deployableCandidates = t1Filtered.filter(r => r.deployment_eligible && !r.baseline_only).length;
    const safeCandidates = t1Filtered.filter(r => (r.physical_violation_rate_max === 0 || !r.physical_violation_rate_max) && (r.nan_inf_count_total === 0 || !r.nan_inf_count_total)).length;
    const superiorToPersistence = t1Filtered.filter(r => !r.baseline_only && r.skill_vs_persistence_60min > 0).length;

    const improvedCount = t5Filtered.filter(r => r.quality_flag_effect === 'with_flags_improved').length;
    const noDiffCount = t5Filtered.filter(r => r.quality_flag_effect === 'no_significant_difference').length;
    const worsenedCount = t5Filtered.filter(r => r.quality_flag_effect === 'with_flags_worsened').length;

    // Find Provisional Leading Candidates per crop within filtered dataset
    const availableCrops = filterState.crop === 'all' ? ['strawberry', 'melon', 'cucumber'] : [filterState.crop.toLowerCase()];
    const cropRankings = {};

    for (const crop of availableCrops) {
      const candidates = t1Filtered.filter(r =>
        r.crop.toLowerCase() === crop &&
        r.deployment_eligible &&
        !r.baseline_only &&
        r.model !== 'persistence'
      );

      if (candidates.length > 0) {
        // Sort using deterministic parsimony comparator (cucumber tie-break prefers without_quality_flags)
        candidates.sort(compareCandidates);
        cropRankings[crop] = {
          first: candidates[0],
          second: candidates.length > 1 ? candidates[1] : null,
          totalEligible: candidates.length
        };
      } else {
        cropRankings[crop] = null;
      }
    }

    this.container.innerHTML = `
      <div class="overview-grid">
        <!-- KPI Summary Cards -->
        <div class="kpi-grid">
          ${createKpiCard({
            title: 'Active Filter Crops',
            value: totalCrops,
            unit: 'Crops',
            icon: '🌱',
            subtitle: availableCrops.map(c => formatCrop(c).split(' ')[0]).join(', '),
            accent: '#10b981'
          })}
          ${createKpiCard({
            title: 'Screened Candidates',
            value: totalCandidates,
            unit: 'Models',
            icon: '🧪',
            subtitle: 'Matching current filter',
            accent: '#6366f1'
          })}
          ${createKpiCard({
            title: 'Deployment-Eligible',
            value: deployableCandidates,
            unit: 'Candidates',
            icon: '✅',
            subtitle: `${((deployableCandidates / Math.max(totalCandidates, 1)) * 100).toFixed(0)}% pass rate`,
            accent: '#06b6d4',
            trend: { type: 'positive', text: 'Gate Qualified' }
          })}
          ${createKpiCard({
            title: 'Zero Safety Violations',
            value: safeCandidates,
            unit: 'Models',
            icon: '🛡️',
            subtitle: 'Zero thermodynamic limit breach',
            accent: '#3b82f6'
          })}
          ${createKpiCard({
            title: 'Outperformed Persistence',
            value: superiorToPersistence,
            unit: 'Models',
            icon: '📈',
            subtitle: 'Skill > 0 @ 60-min rollout',
            accent: '#f59e0b'
          })}
          ${createKpiCard({
            title: 'Filtered Flag Effects',
            value: `${improvedCount}/${noDiffCount}/${worsenedCount}`,
            unit: 'Imp / No-diff / Worse',
            icon: '🔬',
            subtitle: `${noDiffCount} pairs had no significant gain`,
            accent: '#8b5cf6'
          })}
        </div>

        <!-- Provisional Leading Candidates per Crop -->
        <div>
          <div style="margin-bottom: 0.85rem; display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">
            <div>
              <h3 style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 0.5rem;">
                <span>🎯 Validation-based Provisional Leading Candidates</span>
                <span style="font-size: 0.75rem; font-weight: 600; padding: 0.15rem 0.5rem; background: var(--accent-info-bg); color: #38bdf8; border: 1px solid var(--accent-info-border); border-radius: 4px;">Validation Scope</span>
              </h3>
              <span style="font-size: 0.8rem; color: var(--text-secondary);">
                Lowest Validation Rollout Weighted NRMSE among Gate-Qualified learned models. Final model selection is determined by researcher evaluation.
              </span>
            </div>
            <div style="font-size: 0.78rem; color: #475569; font-weight: 500;">
              Parsimony Tie-Break: <code style="background: #f1f5f9; color: #4338ca; padding: 0.1rem 0.35rem; border-radius: 4px;">without_quality_flags</code> prioritized when ΔNRMSE ≈ 0
            </div>
          </div>

          <div class="champion-grid">
            ${availableCrops.map(crop => {
              const rankData = cropRankings[crop];
              if (!rankData || !rankData.first) {
                return `
                  <div class="champion-card ${crop}">
                    <div class="champion-crop-name">${formatCrop(crop)}</div>
                    <div style="color: var(--accent-danger); font-size: 0.85rem; font-weight: 600; margin-top: 0.5rem;">
                      No deployment-eligible candidates matching current filters.
                    </div>
                  </div>
                `;
              }

              const best = rankData.first;
              const runnerUp = rankData.second;
              let marginHtml = '';

              if (runnerUp) {
                const diff = (runnerUp.validation_rollout_weighted_nrmse || 0) - (best.validation_rollout_weighted_nrmse || 0);
                const relPct = (diff / (best.validation_rollout_weighted_nrmse || 1)) * 100;
                const isTie = Math.abs(diff) < 1e-5;

                marginHtml = `
                  <div class="champion-runner-up" style="background: transparent !important; border: none !important; padding: 0.4rem 0 0 0 !important; margin-top: 0.25rem !important;">
                    <div style="color: #475569; display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem;">
                      <span>Runner-up: <strong style="color: #0f172a;">${formatModel(runnerUp.model)} (${runnerUp.feature_variant === 'with_quality_flags' ? 'w/flags' : 'wo/flags'})</strong></span>
                      <span style="font-weight: 700; color: ${isTie ? '#0284c7' : '#059669'};">${isTie ? 'Statistical Tie (0.00%)' : `+${diff.toFixed(4)} (+${relPct.toFixed(1)}%)`}</span>
                    </div>
                    ${isTie ? `<div style="font-size: 0.72rem; color: #4338ca; margin-top: 0.25rem; font-weight: 500;">ℹ️ Selected without_quality_flags due to identical performance and lower pipeline complexity.</div>` : ''}
                  </div>
                `;
              }

              return `
                <div class="champion-card ${crop}">
                  <div class="champion-header">
                    <div>
                      <div class="champion-crop-name">
                        <span>🌱</span> ${formatCrop(crop)}
                      </div>
                      <div class="champion-model-name">${formatModel(best.model)}</div>
                    </div>
                    <div>
                      ${renderEligibilityBadge(best.deployment_eligible, best.baseline_only)}
                    </div>
                  </div>

                  <div style="font-size: 0.82rem; color: #475569; display: flex; align-items: center; justify-content: space-between;">
                    <span>Variant: <strong style="color: #4338ca;">${formatVariant(best.feature_variant)}</strong></span>
                    <span style="font-size: 0.74rem; color: #64748b; font-weight: 600;">Qualified: ${rankData.totalEligible} models</span>
                  </div>

                  <div class="champion-stats-grid">
                    <div class="champ-stat-item">
                      <span class="champ-stat-label">Weighted Rollout NRMSE</span>
                      <span class="champ-stat-val" style="color: #059669;">${formatNumber(best.validation_rollout_weighted_nrmse, 4)}</span>
                    </div>
                    <div class="champ-stat-item">
                      <span class="champ-stat-label">1-Step 5m NRMSE</span>
                      <span class="champ-stat-val">${formatNumber(best.one_step_nrmse, 4)}</span>
                    </div>
                    <div class="champ-stat-item">
                      <span class="champ-stat-label">60m Rollout NRMSE</span>
                      <span class="champ-stat-val">${formatNumber(best.rollout_60min_nrmse, 4)}</span>
                    </div>
                    <div class="champ-stat-item">
                      <span class="champ-stat-label">60m Persistence Skill</span>
                      <span class="champ-stat-val" style="color: ${best.skill_vs_persistence_60min > 0 ? '#059669' : '#dc2626'};">
                        ${best.skill_vs_persistence_60min > 0 ? '+' : ''}${formatNumber(best.skill_vs_persistence_60min, 4)}
                      </span>
                    </div>
                  </div>

                  ${marginHtml}
                </div>
              `;
            }).join('')}
          </div>
        </div>

        <!-- Decision Rationale & Trade-Offs Deep Dive -->
        <div class="overview-summary-banner">
          <div class="overview-summary-title">
            <span>🔬 Multi-Criteria Decision Rationale & Trade-Offs (선택 근거 분석)</span>
          </div>
          <div class="overview-summary-text">
            본 전이모델 평가는 단순 단기 오차(5m One-step NRMSE)뿐 아니라 <strong>다단계 롤아웃 안정성, 물리적 제약 준수(Safety), 반사실적 제어 감도(Action Sensitivity), 운영 지연시간 및 파이프라인 복잡도</strong>를 종합하여 평가합니다.
          </div>

          <div class="overview-key-findings-grid">
            <div class="finding-card">
              <div class="finding-icon">🍓</div>
              <div class="finding-content">
                <div class="finding-title">Strawberry: LightGBM (without_quality_flags)</div>
                <div class="finding-desc">
                  가중 롤아웃 NRMSE <strong>0.2373</strong>으로 전체 1위. 60분 롤아웃 시 Persistence 대비 +18.7% 우수하며, 추론 지연시간 15.2ms, 물리 위반 0건으로 완벽한 안정성을 기록함.
                </div>
              </div>
            </div>

            <div class="finding-card">
              <div class="finding-icon">🍈</div>
              <div class="finding-content">
                <div class="finding-title">Melon: XGBoost (without_quality_flags)</div>
                <div class="finding-desc">
                  가중 롤아웃 NRMSE <strong>0.1205</strong>로 선도. 60분 롤아웃 NRMSE 0.1741을 달성하며 제어 action 감도 및 롤아웃 드리프트 억제력이 가장 우수함.
                </div>
              </div>
            </div>

            <div class="finding-card">
              <div class="finding-icon">🥒</div>
              <div class="finding-content">
                <div class="finding-title">Cucumber: LightGBM (without_quality_flags - Tie-Break)</div>
                <div class="finding-desc">
                  LightGBM의 with_flags와 without_flags가 가중 NRMSE <strong>0.1117</strong>로 동률을 기록함. Table 5 부트스트랩 검증 상 차이가 0.0000(유의차 없음)이므로 파이프라인 복잡도가 낮은 <code>without_quality_flags</code>를 잠정 선도 모델로 채택함.
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Interactive Researcher Decision Workspace -->
        <div class="chart-card">
          <div class="chart-header">
            <div class="chart-title-group">
              <h3 class="chart-title">📝 Researcher Decision & Handoff Workspace</h3>
              <span class="chart-subtitle">시스템 자동 권장과 연구자의 최종 채택 모델을 분리 기록하고 공식 Handoff 설정(JSON/CSV)으로 내보냅니다.</span>
            </div>
            <div class="chart-controls">
              <button type="button" class="table-btn" id="export-decision-json-btn">
                <span>💾</span> Export Decision (JSON)
              </button>
            </div>
          </div>

          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 0.5rem;">
            ${availableCrops.map(crop => {
              const current = this.researcherDecisions[crop] || { model: 'lightgbm', variant: 'without_quality_flags', note: '' };
              return `
                <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 1rem; display: flex; flex-direction: column; gap: 0.75rem;">
                  <div style="font-weight: 700; font-size: 0.95rem; color: var(--text-primary); display: flex; justify-content: space-between;">
                    <span>${formatCrop(crop)}</span>
                    <span style="font-size: 0.75rem; color: #34d399;">System: ${formatModel(cropRankings[crop]?.first?.model || 'lightgbm')}</span>
                  </div>

                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div>
                      <label style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Final Model:</label>
                      <select class="filter-select" data-decision-crop="${crop}" data-decision-field="model" style="padding: 0.35rem 0.5rem; font-size: 0.8rem;">
                        <option value="lightgbm" ${current.model === 'lightgbm' ? 'selected' : ''}>LightGBM</option>
                        <option value="xgboost" ${current.model === 'xgboost' ? 'selected' : ''}>XGBoost</option>
                        <option value="extra_trees" ${current.model === 'extra_trees' ? 'selected' : ''}>ExtraTrees</option>
                        <option value="knn" ${current.model === 'knn' ? 'selected' : ''}>KNN</option>
                        <option value="linear_svr" ${current.model === 'linear_svr' ? 'selected' : ''}>Linear SVR</option>
                      </select>
                    </div>

                    <div>
                      <label style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Feature Variant:</label>
                      <select class="filter-select" data-decision-crop="${crop}" data-decision-field="variant" style="padding: 0.35rem 0.5rem; font-size: 0.8rem;">
                        <option value="without_quality_flags" ${current.variant === 'without_quality_flags' ? 'selected' : ''}>Without Flags</option>
                        <option value="with_quality_flags" ${current.variant === 'with_quality_flags' ? 'selected' : ''}>With Flags</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Selection Note / Rationale:</label>
                    <textarea class="filter-input" data-decision-crop="${crop}" data-decision-field="note" style="width: 100%; min-height: 50px; font-size: 0.78rem; resize: vertical;" placeholder="Enter scientific rationale for this crop...">${current.note}</textarea>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      </div>
    `;

    this.bindDecisionEvents();
  }

  bindDecisionEvents() {
    this.container.querySelectorAll('[data-decision-crop]').forEach(el => {
      el.addEventListener('input', (e) => {
        const crop = e.target.getAttribute('data-decision-crop');
        const field = e.target.getAttribute('data-decision-field');
        if (!this.researcherDecisions[crop]) this.researcherDecisions[crop] = {};
        this.researcherDecisions[crop][field] = e.target.value;
      });
    });

    const exportBtn = document.getElementById('export-decision-json-btn');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => {
        const payload = {
          export_timestamp: new Date().toISOString(),
          evaluation_scope: 'validation_only',
          source_manifest: dataService.manifest,
          researcher_decisions: this.researcherDecisions
        };
        const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2));
        const dlAnchor = document.createElement('a');
        dlAnchor.setAttribute('href', dataStr);
        dlAnchor.setAttribute('download', 'geas_transition_model_decision.json');
        document.body.appendChild(dlAnchor);
        dlAnchor.click();
        document.body.removeChild(dlAnchor);
      });
    }
  }

  update() {
    this.render();
  }
}
