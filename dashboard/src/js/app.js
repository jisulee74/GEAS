/**
 * GEAS Dashboard - Application Entry Point & Controller
 * Integrated Dashboard for Data Quality Management, Transition Model Selection, and RL Control
 */

import { dataService } from './core/data-loader.js';
import { filterStore } from './core/filter-store.js';
import { formatCrop, formatModel } from './core/formatters.js';

// Transition Model Sections (Canonical clean imports for shared singletons)
import { OverviewSection } from './sections/overview-section.js';
import { SetupSection } from './sections/setup-section.js';
import { RankingSection } from './sections/ranking-section.js';
import { OneStepSection } from './sections/onestep-section.js';
import { SafetySection } from './sections/safety-section.js';
import { DriftSection } from './sections/drift-section.js';
import { ResourceSection } from './sections/resource-section.js';
import { AblationSection } from './sections/ablation-section.js';

// Quality Management Services & Sections
import { qualityDataLoader, QualityDataLoader } from './quality/quality-data-loader.js';
import { qualityFilterStore } from './quality/quality-filter-store.js';
import { QualityOverviewSection } from './quality/sections/quality-overview-section.js';
import { QualitySetupSection } from './quality/sections/quality-setup-section.js';
import { QualityReconstructionSection } from './quality/sections/quality-reconstruction-section.js';
import { QualityDetectionSection } from './quality/sections/quality-detection-section.js';
import { QualityHpoThresholdSection } from './quality/sections/quality-hpo-threshold-section.js';
import { QualityResourceSection } from './quality/sections/quality-resource-section.js';
import { QualityDeploymentSection } from './quality/sections/quality-deployment-section.js';

// RL Section Placeholder
import { RlControlSection } from './sections/rl-control-section.js';

class App {
  constructor() {
    this.currentExperiment = 'quality'; // 'quality' | 'transition' | 'rl_control'
    this.currentQualityTab = 'quality-overview';
    this.currentTransitionTab = 'overview';

    this.transitionSections = {};
    this.qualitySections = {};
    this.rlSection = null;
  }

  async init() {
    console.log('Initializing GEAS Integrated Research Dashboard...');

    // 1. Initialize Section Controllers immediately
    this.initTransitionSections();
    this.initQualitySections();
    this.rlSection = new RlControlSection('tab-panel-rl-control');

    // 2. Bind ALL Navigation and Filter Events immediately (event delegation)
    this.bindMasterExperimentNavigation();
    this.bindQualityTabNavigation();
    this.bindTransitionTabNavigation();
    this.bindQualityFilterEvents();
    this.bindTransitionFilterEvents();
    this.bindProvenanceModal();

    // 3. Expose global helpers on window for direct/inline calls
    window.switchExperiment = (exp) => this.switchExperiment(exp);
    window.switchQualityTab = (tab) => this.switchQualityTab(tab);
    window.switchTransitionTab = (tab) => this.switchTransitionTab(tab);
    window.dataService = dataService;
    window.filterStore = filterStore;
    window.qualityDataLoader = qualityDataLoader;
    window.qualityFilterStore = qualityFilterStore;

    // 4. Ingest datasets FIRST so that when sections render, data is 100% available!
    try {
      await Promise.allSettled([
        dataService.initialize(),
        qualityDataLoader.initialize()
      ]);
      console.log('All datasets loaded successfully.', {
        t1: dataService.getTable1().length,
        t2: dataService.getTable2().length,
        t3: dataService.getTable3().length,
        t4: dataService.getTable4().length,
        t5: dataService.getTable5().length,
        t6: dataService.getTable6().length,
        qualityVal: qualityDataLoader.validationResults.length
      });
    } catch (err) {
      console.error('Error loading datasets:', err);
    }

    // 5. Populate dropdown options with loaded datasets
    this.populateTransitionFilterOptions();

    // 6. Subscribe to filter store updates
    filterStore.subscribe((state) => {
      this.updateTransitionFilterSummary(state);
      this.updateCurrentTransitionSection();
    });

    qualityFilterStore.subscribe((state) => {
      this.updateQualityFilterSummary(state);
      this.updateCurrentQualitySection();
    });

    // 7. Initial render with fully ingested datasets
    this.switchExperiment(this.currentExperiment);
    this.updateQualityFilterSummary(qualityFilterStore.getState());
    this.updateTransitionFilterSummary(filterStore.getState());
  }

  // =========================================================================
  // Master Experiment Navigation (Quality vs Transition vs RL)
  // =========================================================================

  bindMasterExperimentNavigation() {
    document.addEventListener('click', (e) => {
      const btn = e.target.closest('.master-nav-btn[data-experiment]');
      if (btn) {
        const expKey = btn.getAttribute('data-experiment');
        if (expKey) {
          this.switchExperiment(expKey);
        }
      }
    });
  }

  switchExperiment(expKey) {
    console.log('Switching Experiment Pipeline to:', expKey);
    this.currentExperiment = expKey;

    // 1. Update master nav button states
    document.querySelectorAll('.master-nav-btn[data-experiment]').forEach(btn => {
      const match = btn.getAttribute('data-experiment') === expKey;
      btn.classList.toggle('active', match);
    });

    // 2. Toggle experiment containers
    const qualityContainer = document.getElementById('container-exp-quality');
    const transitionContainer = document.getElementById('container-exp-transition');
    const rlContainer = document.getElementById('container-exp-rl');

    if (qualityContainer) qualityContainer.style.display = expKey === 'quality' ? 'block' : 'none';
    if (transitionContainer) transitionContainer.style.display = expKey === 'transition' ? 'block' : 'none';
    if (rlContainer) rlContainer.style.display = expKey === 'rl_control' ? 'block' : 'none';

    // 3. Render active tab of current experiment safely
    try {
      if (expKey === 'quality') {
        this.switchQualityTab(this.currentQualityTab);
      } else if (expKey === 'transition') {
        this.switchTransitionTab(this.currentTransitionTab);
      } else if (expKey === 'rl_control') {
        if (this.rlSection) this.rlSection.render();
      }
    } catch (err) {
      console.error('Error switching experiment view:', err);
    }
  }

  // =========================================================================
  // Quality Management Sub-Navigation & Filters
  // =========================================================================

  initQualitySections() {
    this.qualitySections = {
      'quality-overview': new QualityOverviewSection('tab-panel-quality-overview'),
      'quality-setup': new QualitySetupSection('tab-panel-quality-setup'),
      'quality-reconstruction': new QualityReconstructionSection('tab-panel-quality-reconstruction'),
      'quality-detection': new QualityDetectionSection('tab-panel-quality-detection'),
      'quality-hpo': new QualityHpoThresholdSection('tab-panel-quality-hpo'),
      'quality-resource': new QualityResourceSection('tab-panel-quality-resource'),
      'quality-deployment': new QualityDeploymentSection('tab-panel-quality-deployment')
    };
  }

  bindQualityTabNavigation() {
    document.addEventListener('click', (e) => {
      const btn = e.target.closest('#quality-tabs-nav .tab-btn[data-tab]');
      if (btn) {
        const tabKey = btn.getAttribute('data-tab');
        if (tabKey) {
          this.switchQualityTab(tabKey);
        }
      }
    });
  }

  switchQualityTab(tabKey) {
    console.log('Switching Quality Sub-Tab to:', tabKey);
    this.currentQualityTab = tabKey;

    // Update active tab buttons
    document.querySelectorAll('#quality-tabs-nav .tab-btn[data-tab]').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabKey);
    });

    // Update active tab panels
    document.querySelectorAll('#container-exp-quality .tab-panel').forEach(panel => {
      panel.classList.toggle('active', panel.id === `tab-panel-${tabKey}`);
    });

    // Conditionally toggle variable filter visibility
    const varFilterGroup = document.getElementById('quality-variable-filter-group');
    const perVarTabs = ['quality-reconstruction', 'quality-detection', 'quality-hpo'];
    if (varFilterGroup) {
      varFilterGroup.style.display = perVarTabs.includes(tabKey) ? 'flex' : 'none';
    }

    // Render section safely
    const section = this.qualitySections[tabKey];
    if (section) {
      try {
        section.render();
      } catch (err) {
        console.error(`Error rendering quality section ${tabKey}:`, err);
      }
    }
  }

  updateCurrentQualitySection() {
    const section = this.qualitySections[this.currentQualityTab];
    if (section) {
      try {
        if (typeof section.update === 'function') {
          section.update();
        } else {
          section.render();
        }
      } catch (err) {
        console.error('Error updating quality section:', err);
      }
    }
  }

  bindQualityFilterEvents() {
    const cropSelect = document.getElementById('quality-filter-crop');
    const modelSelect = document.getElementById('quality-filter-model');
    const splitSelect = document.getElementById('quality-filter-split');
    const varSelect = document.getElementById('quality-filter-variable');

    cropSelect?.addEventListener('change', (e) => qualityFilterStore.setFilter('crop', e.target.value));
    modelSelect?.addEventListener('change', (e) => qualityFilterStore.setFilter('model', e.target.value));
    splitSelect?.addEventListener('change', (e) => qualityFilterStore.setFilter('split', e.target.value));
    varSelect?.addEventListener('change', (e) => qualityFilterStore.setFilter('variable', e.target.value));

    document.addEventListener('click', (e) => {
      if (e.target.closest('#quality-filter-reset-btn')) {
        qualityFilterStore.resetFilters();
        if (cropSelect) cropSelect.value = 'all';
        if (modelSelect) modelSelect.value = 'all';
        if (splitSelect) splitSelect.value = 'validation';
        if (varSelect) varSelect.value = 'all';
      }
    });
  }

  updateQualityFilterSummary(state) {
    const activeFiltersEl = document.getElementById('quality-active-filters-summary');
    if (!activeFiltersEl) return;

    const tags = [];
    if (state.crop !== 'all') tags.push(`작물: ${QualityDataLoader.formatCrop(state.crop)}`);
    else tags.push('전체 작물');

    if (state.model !== 'all') tags.push(`모델: ${state.model}`);
    else tags.push('전체 모델');

    tags.push(`분할: ${state.split === 'test' ? 'Test (Holdout)' : 'Validation (기본)'}`);

    if (state.variable !== 'all') {
      tags.push(`변수: ${QualityDataLoader.formatColumnName(state.variable).split(' ')[0]}`);
    }

    activeFiltersEl.innerHTML = tags.map(t => `<span class="active-filter-pill">${t}</span>`).join('');
  }

  // =========================================================================
  // Transition Model Sub-Navigation & Filters (Preserved Exactly)
  // =========================================================================

  initTransitionSections() {
    this.transitionSections = {
      overview: new OverviewSection('tab-panel-overview'),
      setup: new SetupSection('tab-panel-setup'),
      ranking: new RankingSection('tab-panel-ranking'),
      onestep: new OneStepSection('tab-panel-onestep'),
      safety: new SafetySection('tab-panel-safety'),
      drift: new DriftSection('tab-panel-drift'),
      resource: new ResourceSection('tab-panel-resource'),
      ablation: new AblationSection('tab-panel-ablation')
    };
  }

  bindTransitionTabNavigation() {
    document.addEventListener('click', (e) => {
      const btn = e.target.closest('#transition-tabs-nav .tab-btn[data-tab]');
      if (btn) {
        const tabKey = btn.getAttribute('data-tab');
        if (tabKey) {
          this.switchTransitionTab(tabKey);
        }
      }
    });
  }

  switchTransitionTab(tabKey) {
    console.log('Switching Transition Sub-Tab to:', tabKey);
    this.currentTransitionTab = tabKey;

    // Update active tab buttons
    document.querySelectorAll('#transition-tabs-nav .tab-btn[data-tab]').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabKey);
    });

    // Update active tab panels
    document.querySelectorAll('#container-exp-transition .tab-panel').forEach(panel => {
      panel.classList.toggle('active', panel.id === `tab-panel-${tabKey}`);
    });

    // Render section safely
    const section = this.transitionSections[tabKey];
    if (section) {
      try {
        section.render();
      } catch (err) {
        console.error(`Error rendering transition section ${tabKey}:`, err);
      }
    }
  }

  updateCurrentTransitionSection() {
    const section = this.transitionSections[this.currentTransitionTab];
    if (section) {
      try {
        if (typeof section.update === 'function') {
          section.update();
        } else {
          section.render();
        }
      } catch (err) {
        console.error('Error updating transition section:', err);
      }
    }
  }

  populateTransitionFilterOptions() {
    try {
      const crops = dataService.getUniqueCrops();
      const models = dataService.getUniqueModels();

      const cropSelect = document.getElementById('filter-crop');
      const modelSelect = document.getElementById('filter-model');

      if (cropSelect && crops.length > 0) {
        cropSelect.innerHTML = '<option value="all">All Crops (전체 작물)</option>' +
          crops.map(c => `<option value="${c}">${formatCrop(c)}</option>`).join('');
      }

      if (modelSelect && models.length > 0) {
        modelSelect.innerHTML = '<option value="all">All Models (전체 모델)</option>' +
          models.map(m => `<option value="${m}">${formatModel(m)}</option>`).join('');
      }
    } catch (e) {
      console.warn('Could not populate transition filters:', e);
    }
  }

  bindTransitionFilterEvents() {
    const cropSelect = document.getElementById('filter-crop');
    const variantSelect = document.getElementById('filter-variant');
    const modelSelect = document.getElementById('filter-model');
    const eligibilitySelect = document.getElementById('filter-eligibility');
    const baselineCheckbox = document.getElementById('filter-baseline');

    cropSelect?.addEventListener('change', (e) => filterStore.setFilter('crop', e.target.value));
    variantSelect?.addEventListener('change', (e) => filterStore.setFilter('featureVariant', e.target.value));
    modelSelect?.addEventListener('change', (e) => filterStore.setFilter('model', e.target.value));
    eligibilitySelect?.addEventListener('change', (e) => filterStore.setFilter('eligibility', e.target.value));
    baselineCheckbox?.addEventListener('change', (e) => filterStore.setFilter('showBaseline', e.target.checked));

    document.addEventListener('click', (e) => {
      if (e.target.closest('#filter-reset-btn')) {
        filterStore.reset();
        if (cropSelect) cropSelect.value = 'all';
        if (variantSelect) variantSelect.value = 'all';
        if (modelSelect) modelSelect.value = 'all';
        if (eligibilitySelect) eligibilitySelect.value = 'all';
        if (baselineCheckbox) baselineCheckbox.checked = true;
      }
    });
  }

  updateTransitionFilterSummary(state) {
    const activeFiltersEl = document.getElementById('active-filters-summary');
    if (!activeFiltersEl) return;

    const tags = [];
    if (state.crop !== 'all') tags.push(`Crop: ${formatCrop(state.crop)}`);
    if (state.featureVariant !== 'all') tags.push(`Variant: ${state.featureVariant === 'with_quality_flags' ? 'With Flags' : 'Without Flags'}`);
    if (state.model !== 'all') tags.push(`Model: ${formatModel(state.model)}`);
    if (state.eligibility !== 'all') tags.push(`Eligibility: ${state.eligibility === 'eligible_only' ? 'Eligible' : 'Ineligible'}`);
    if (!state.showBaseline) tags.push('Hide Baselines');

    const t1Count = dataService.getTable1 ? dataService.getTable1().length : 0;
    if (tags.length === 0) {
      activeFiltersEl.innerHTML = `<span class="active-filter-pill">All Records (${t1Count} Candidates)</span>`;
    } else {
      activeFiltersEl.innerHTML = tags.map(t => `<span class="active-filter-pill">${t}</span>`).join('');
    }
  }

  bindProvenanceModal() {
    const modal = document.getElementById('provenance-modal');
    document.addEventListener('click', (e) => {
      if (e.target.closest('#provenance-modal-btn')) {
        modal?.showModal();
      }
      if (e.target.closest('#close-provenance-modal-btn')) {
        modal?.close();
      }
      if (e.target === modal) {
        modal?.close();
      }
    });
  }
}

// Bootstrap application reliably
const startApp = () => {
  if (!window.app) {
    window.app = new App();
    window.app.init();
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', startApp);
} else {
  startApp();
}
