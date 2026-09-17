/**
 * GEAS Dashboard - Data Ingestion, Normalization & Store
 */

import { EMBEDDED_DATA } from '../data/embedded-data.js';

class DataService {
  constructor() {
    this.manifest = null;
    this.tables = {
      table_1: [],
      table_2: [],
      table_3: [],
      table_4: [],
      table_5: [],
      table_6: []
    };
    this.isLoaded = false;
  }

  async initialize() {
    try {
      // 1. Try to load embedded data first
      if (EMBEDDED_DATA && EMBEDDED_DATA.tables) {
        this.manifest = EMBEDDED_DATA.manifest;
        this.tables = {
          table_1: this.normalizeTable1(EMBEDDED_DATA.tables.table_1 || []),
          table_2: this.normalizeTable2(EMBEDDED_DATA.tables.table_2 || []),
          table_3: this.normalizeTable3(EMBEDDED_DATA.tables.table_3 || []),
          table_4: this.normalizeTable4(EMBEDDED_DATA.tables.table_4 || []),
          table_5: this.normalizeTable5(EMBEDDED_DATA.tables.table_5 || []),
          table_6: this.normalizeTable6(EMBEDDED_DATA.tables.table_6 || [])
        };
        this.isLoaded = true;

        // If table_6 was missing in an older embedded cache, fetch it dynamically
        if (this.tables.table_6.length === 0) {
          await this.fetchSingleCsv('table_6', 'table_6_rollout_drift_details.csv');
        }

        console.log('Loaded dataset successfully:', {
          t1: this.tables.table_1.length,
          t2: this.tables.table_2.length,
          t3: this.tables.table_3.length,
          t4: this.tables.table_4.length,
          t5: this.tables.table_5.length,
          t6: this.tables.table_6.length
        });
        return true;
      }
    } catch (err) {
      console.warn('Failed loading embedded data, attempting CSV fetch fallback...', err);
    }

    // Fallback: try fetching CSVs directly if served via web server
    await this.fetchAndParseCsvs();
    return true;
  }

  async fetchSingleCsv(key, filename) {
    if (typeof Papa === 'undefined') return;
    try {
      const resp = await fetch(filename);
      if (!resp.ok) return;
      const text = await resp.text();
      const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
      if (parsed.data && key === 'table_6') {
        this.tables.table_6 = this.normalizeTable6(parsed.data);
      }
    } catch (e) {
      console.warn(`Could not fallback fetch ${filename}:`, e);
    }
  }

  async fetchAndParseCsvs() {
    if (typeof Papa === 'undefined') return false;
    const fileMap = {
      table_1: 'table_1_candidate_screening_and_rollout.csv',
      table_2: 'table_2_one_step_target_metrics.csv',
      table_3: 'table_3_safety_and_action_response.csv',
      table_4: 'table_4_resource_metrics.csv',
      table_5: 'table_5_quality_flag_ablation.csv',
      table_6: 'table_6_rollout_drift_details.csv'
    };

    for (const [key, filename] of Object.entries(fileMap)) {
      try {
        const resp = await fetch(filename);
        if (!resp.ok) continue;
        const text = await resp.text();
        const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
        if (parsed.data) {
          if (key === 'table_1') this.tables.table_1 = this.normalizeTable1(parsed.data);
          if (key === 'table_2') this.tables.table_2 = this.normalizeTable2(parsed.data);
          if (key === 'table_3') this.tables.table_3 = this.normalizeTable3(parsed.data);
          if (key === 'table_4') this.tables.table_4 = this.normalizeTable4(parsed.data);
          if (key === 'table_5') this.tables.table_5 = this.normalizeTable5(parsed.data);
          if (key === 'table_6') this.tables.table_6 = this.normalizeTable6(parsed.data);
        }
      } catch (e) {
        console.error(`Error fetching CSV ${filename}:`, e);
      }
    }
    this.isLoaded = true;
  }

  // Normalization Helpers
  toBool(val) {
    if (typeof val === 'boolean') return val;
    if (typeof val === 'string') {
      const lower = val.toLowerCase().trim();
      return lower === 'true' || lower === '1' || lower === 'yes';
    }
    return Boolean(val);
  }

  toFloat(val, fallback = null) {
    if (val === null || val === undefined || val === '') return fallback;
    const n = parseFloat(val);
    return isNaN(n) ? fallback : n;
  }

  normalizeTable1(rows) {
    return rows.map(r => ({
      ...r,
      baseline_only: this.toBool(r.baseline_only),
      deployment_eligible: this.toBool(r.deployment_eligible),
      validation_rollout_weighted_nrmse: this.toFloat(r.validation_rollout_weighted_nrmse),
      one_step_nrmse: this.toFloat(r.one_step_nrmse),
      rollout_15min_nrmse: this.toFloat(r.rollout_15min_nrmse),
      rollout_30min_nrmse: this.toFloat(r.rollout_30min_nrmse),
      rollout_60min_nrmse: this.toFloat(r.rollout_60min_nrmse),
      skill_vs_persistence_15min: this.toFloat(r.skill_vs_persistence_15min),
      skill_vs_persistence_30min: this.toFloat(r.skill_vs_persistence_30min),
      skill_vs_persistence_60min: this.toFloat(r.skill_vs_persistence_60min),
      physical_violation_rate_max: this.toFloat(r.physical_violation_rate_max, 0),
      nan_inf_count_total: parseInt(r.nan_inf_count_total || 0, 10)
    }));
  }

  normalizeTable2(rows) {
    return rows.map(r => ({
      ...r,
      deployment_eligible: this.toBool(r.deployment_eligible),
      mae: this.toFloat(r.mae),
      rmse: this.toFloat(r.rmse),
      nrmse: this.toFloat(r.nrmse),
      r2: this.toFloat(r.r2),
      absolute_error_q90: this.toFloat(r.absolute_error_q90),
      absolute_error_cvar90: this.toFloat(r.absolute_error_cvar90)
    }));
  }

  normalizeTable3(rows) {
    return rows.map(r => ({
      ...r,
      deployment_eligible: this.toBool(r.deployment_eligible),
      action_response_eligible: this.toBool(r.action_response_eligible),
      fully_action_insensitive: this.toBool(r.fully_action_insensitive),
      actions_total: parseInt(r.actions_total || 0, 10),
      actions_with_sufficient_support: parseInt(r.actions_with_sufficient_support || 0, 10),
      actions_insufficient_support: parseInt(r.actions_insufficient_support || 0, 10),
      actions_status_ok: parseInt(r.actions_status_ok || 0, 10),
      actions_marked_insensitive: parseInt(r.actions_marked_insensitive || 0, 10),
      minimum_counterfactual_sensitivity: this.toFloat(r.minimum_counterfactual_sensitivity, null),
      mean_counterfactual_sensitivity: this.toFloat(r.mean_counterfactual_sensitivity, null),
      physical_violation_rate_15min: this.toFloat(r.physical_violation_rate_15min, 0),
      physical_violation_rate_30min: this.toFloat(r.physical_violation_rate_30min, 0),
      physical_violation_rate_60min: this.toFloat(r.physical_violation_rate_60min, 0),
      nan_inf_count_total: parseInt(r.nan_inf_count_total || 0, 10),
      drift_slope_nmae: this.toFloat(r.drift_slope_nmae)
    }));
  }

  normalizeTable4(rows) {
    return rows.map(r => {
      let dev = {};
      try {
        if (typeof r.device === 'string' && r.device.startsWith('{')) {
          dev = JSON.parse(r.device);
        }
      } catch (e) {}
      return {
        ...r,
        deployment_eligible: this.toBool(r.deployment_eligible),
        training_time_seconds: this.toFloat(r.training_time_seconds),
        hpo_total_time_seconds: this.toFloat(r.hpo_total_time_seconds),
        inference_latency_median_ms: this.toFloat(r.inference_latency_median_ms),
        inference_latency_p95_ms: this.toFloat(r.inference_latency_p95_ms),
        peak_cpu_memory_mb: this.toFloat(r.peak_cpu_memory_mb),
        peak_gpu_memory_mb: this.toFloat(r.peak_gpu_memory_mb, null),
        serialized_model_size_mb: this.toFloat(r.serialized_model_size_mb),
        artifact_directory_size_mb: this.toFloat(r.artifact_directory_size_mb),
        device_info: dev
      };
    });
  }

  normalizeTable5(rows) {
    return rows.map(r => ({
      ...r,
      baseline_only: this.toBool(r.baseline_only),
      with_flags_deployment_eligible: this.toBool(r.with_flags_deployment_eligible),
      without_flags_deployment_eligible: this.toBool(r.without_flags_deployment_eligible),
      paired_validation_window_count: parseInt(r.paired_validation_window_count || 0, 10),
      with_flags_paired_rollout_nrmse: this.toFloat(r.with_flags_paired_rollout_nrmse),
      without_flags_paired_rollout_nrmse: this.toFloat(r.without_flags_paired_rollout_nrmse),
      difference_with_minus_without: this.toFloat(r.difference_with_minus_without),
      bootstrap_ci95_lower: this.toFloat(r.bootstrap_ci95_lower),
      bootstrap_ci95_upper: this.toFloat(r.bootstrap_ci95_upper),
      bootstrap_samples: parseInt(r.bootstrap_samples || 0, 10),
      bootstrap_seed: parseInt(r.bootstrap_seed || 0, 10)
    }));
  }

  normalizeTable6(rows) {
    if (!Array.isArray(rows)) return [];
    return rows.map(r => ({
      ...r,
      baseline_only: this.toBool(r.baseline_only),
      deployment_eligible: this.toBool(r.deployment_eligible),
      validation_window_count: parseInt(r.validation_window_count || 0, 10),
      physical_violation_rate: this.toFloat(r.physical_violation_rate, 0),
      nan_inf_count: parseInt(r.nan_inf_count || 0, 10),
      drift_slope_nmae: this.toFloat(r.drift_slope_nmae),
      drift_slope_nmae_median: this.toFloat(r.drift_slope_nmae_median),
      drift_slope_nmae_q90: this.toFloat(r.drift_slope_nmae_q90),
      drift_slope_nmae_cvar90: this.toFloat(r.drift_slope_nmae_cvar90)
    }));
  }

  // Accessors
  getTable1() { return this.tables.table_1; }
  getTable2() { return this.tables.table_2; }
  getTable3() { return this.tables.table_3; }
  getTable4() { return this.tables.table_4; }
  getTable5() { return this.tables.table_5; }
  getTable6() { return this.tables.table_6 || []; }

  // Unique Filter Options
  getUniqueCrops() {
    const crops = new Set(this.tables.table_1.map(r => r.crop).filter(Boolean));
    return Array.from(crops);
  }

  getUniqueModels() {
    const models = new Set(this.tables.table_1.map(r => r.model).filter(Boolean));
    return Array.from(models);
  }

  getUniqueFeatureVariants() {
    const variants = new Set(this.tables.table_1.map(r => r.feature_variant).filter(Boolean));
    return Array.from(variants);
  }
}

export const dataService = new DataService();
