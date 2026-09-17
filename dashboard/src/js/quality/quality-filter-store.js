/**
 * GEAS Dashboard - Quality Panel Filter State Store
 * Independent filter store for Data Quality Management experiments
 */

class QualityFilterStore {
  constructor() {
    this.initialState = {
      crop: 'all',          // 'all' | 'cucumber' | 'melon' | 'strawberry'
      model: 'all',         // 'all' | 'ModernTCN' | 'TimesNet' | 'PatchTST'
      split: 'validation',  // 'validation' (default) | 'test'
      variable: 'all'       // 'all' | column name
    };
    this.state = { ...this.initialState };
    this.listeners = new Set();
  }

  getState() {
    return { ...this.state };
  }

  setFilter(key, value) {
    if (this.state[key] !== value) {
      this.state[key] = value;
      this.notify();
    }
  }

  setFilters(partialState) {
    let changed = false;
    for (const [key, val] of Object.entries(partialState)) {
      if (this.state[key] !== val) {
        this.state[key] = val;
        changed = true;
      }
    }
    if (changed) {
      this.notify();
    }
  }

  resetFilters() {
    this.state = { ...this.initialState };
    this.notify();
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify() {
    for (const listener of this.listeners) {
      try {
        listener(this.getState());
      } catch (err) {
        console.error('Error in QualityFilterStore subscriber:', err);
      }
    }
  }
}

export const qualityFilterStore = new QualityFilterStore();
