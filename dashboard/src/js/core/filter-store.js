/**
 * GEAS Dashboard - Reactive Global Filter Store
 */

class FilterStore {
  constructor() {
    this.state = {
      crop: 'all',
      featureVariant: 'all',
      model: 'all',
      eligibility: 'all', // 'all', 'eligible_only', 'ineligible_only'
      showBaseline: true,
      searchQuery: ''
    };
    this.listeners = [];
  }

  getState() {
    return { ...this.state };
  }

  setFilter(key, value) {
    if (this.state[key] === value) return;
    this.state[key] = value;
    this.notify();
  }

  setFilters(partialState) {
    let changed = false;
    for (const [k, v] of Object.entries(partialState)) {
      if (this.state[k] !== v) {
        this.state[k] = v;
        changed = true;
      }
    }
    if (changed) this.notify();
  }

  reset() {
    this.state = {
      crop: 'all',
      featureVariant: 'all',
      model: 'all',
      eligibility: 'all',
      showBaseline: true,
      searchQuery: ''
    };
    this.notify();
  }

  subscribe(listener) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  notify() {
    for (const listener of this.listeners) {
      listener(this.getState());
    }
  }

  /**
   * Applies the current filter state to any given dataset row array
   */
  filterRows(rows, options = {}) {
    if (!Array.isArray(rows)) return [];
    const { crop, featureVariant, model, eligibility, showBaseline } = this.state;

    return rows.filter(row => {
      // 1. Crop Filter
      if (crop !== 'all' && row.crop && row.crop.toLowerCase() !== crop.toLowerCase()) {
        return false;
      }

      // 2. Feature Variant Filter
      if (featureVariant !== 'all' && row.feature_variant && row.feature_variant !== featureVariant) {
        return false;
      }

      // 3. Model Filter
      if (model !== 'all' && row.model && row.model.toLowerCase() !== model.toLowerCase()) {
        return false;
      }

      // 4. Baseline Filter
      const isBaseline = row.baseline_only === true || row.baseline_only === 'True' || row.baseline_only === 'true' || row.model === 'persistence';
      if (!showBaseline && isBaseline) {
        return false;
      }

      // 5. Eligibility Filter
      if (eligibility !== 'all') {
        const isEligible = row.deployment_eligible === true || row.deployment_eligible === 'True' || row.deployment_eligible === 'true';
        if (eligibility === 'eligible_only' && !isEligible) return false;
        if (eligibility === 'ineligible_only' && isEligible) return false;
      }

      return true;
    });
  }
}

export const filterStore = new FilterStore();
