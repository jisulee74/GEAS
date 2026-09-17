/**
 * GEAS Dashboard - Quality Data Service
 * Loads and manages artifacts from quality_panel_bundle/
 */

import { QUALITY_EMBEDDED_FALLBACK } from './quality-embedded-fallback.js';

export class QualityDataLoader {
  constructor() {
    this.basePath = './quality_panel_bundle';
    this.validationResults = [];
    this.testResults = [];
    this.reconstructionTable = [];
    this.anomalyDetectionTable = [];
    this.onlineBenchmarkTable = [];
    this.trainingHistories = {};       // key: `${crop}_${modelSlug}`
    this.bestConfigs = {};             // key: `${crop}_${modelSlug}`
    this.thresholdCalibrations = {};   // key: `${crop}_${modelSlug}`
    this.modelApplications = {};       // key: `${crop}_${modelSlug}`
    this.isLoaded = false;
  }

  static modelNameToSlug(name) {
    if (!name) return '';
    const clean = name.toLowerCase().replace(/[-_]/g, '');
    if (clean.includes('moderntcn')) return 'modern_tcn';
    if (clean.includes('timesnet')) return 'timesnet';
    if (clean.includes('patchtst')) return 'patch_tst';
    return name.toLowerCase().replace(/[^a-z0-9]/g, '_');
  }

  static slugToModelName(slug) {
    if (!slug) return '';
    const clean = slug.toLowerCase().replace(/[-_]/g, '');
    if (clean.includes('moderntcn')) return 'ModernTCN';
    if (clean.includes('timesnet')) return 'TimesNet';
    if (clean.includes('patchtst')) return 'PatchTST';
    return slug;
  }

  static formatCrop(crop) {
    const map = {
      cucumber: '오이 (Cucumber)',
      melon: '멜론 (Melon)',
      strawberry: '딸기 (Strawberry)'
    };
    return map[crop?.toLowerCase()] || crop;
  }

  static formatColumnName(col) {
    const map = {
      __all__: '전체 종합 (__all__)',
      in_temp: '실내온도 1 (in_temp, °C)',
      in_temp2: '실내온도 2 (in_temp2, °C)',
      in_hum: '실내습도 1 (in_hum, %)',
      in_hum2: '실내습도 2 (in_hum2, %)',
      in_co2: '실내 CO₂ 1 (in_co2, ppm)',
      in_co2_2: '실내 CO₂ 2 (in_co2_2, ppm)',
      in_medium_temp1: '배지온도 1 (in_medium_temp1, °C)',
      in_medium_hum1: '배지습도 1 (in_medium_hum1, %)'
    };
    return map[col] || col;
  }

  async initialize() {
    if (this.isLoaded) return true;
    try {
      console.log('Loading Data Quality Management artifacts from quality_panel_bundle...');

      // 1. Load summary validation and test results (with embedded fallback)
      let [valRes, testRes] = await Promise.all([
        this.fetchJson(`${this.basePath}/data/validation_results.json`),
        this.fetchJson(`${this.basePath}/data/test_results.json`)
      ]);

      if (!valRes || valRes.length === 0) {
        console.info('Using embedded fallback for quality validation results');
        valRes = QUALITY_EMBEDDED_FALLBACK.validation;
      }
      if (!testRes || testRes.length === 0) {
        console.info('Using embedded fallback for quality test results');
        testRes = QUALITY_EMBEDDED_FALLBACK.test;
      }

      this.validationResults = (valRes || []).map(row => this.normalizeSummaryRow(row, 'validation'));
      this.testResults = (testRes || []).map(row => this.normalizeSummaryRow(row, 'test'));

      // 2. Load crop-specific detailed tables (Cucumber, Melon, Strawberry)
      const crops = ['cucumber', 'melon', 'strawberry'];
      for (const crop of crops) {
        // Reconstruction table
        const reconCsv = await this.fetchCsv(`${this.basePath}/data/quality/${crop}/reconstruction_metric_table.csv`);
        if (reconCsv) {
          reconCsv.forEach(r => {
            if (!r.column && !r.mae && !r.rmse) return;
            this.reconstructionTable.push({
              crop,
              column: r.column,
              model: QualityDataLoader.slugToModelName(r.model_name),
              model_slug: QualityDataLoader.modelNameToSlug(r.model_name),
              split: r.split,
              mae: parseFloat(r.mae) || 0,
              rmse: parseFloat(r.rmse) || 0,
              masked_cells: parseInt(r.masked_cells, 10) || 0
            });
          });
        }

        // Anomaly Detection table
        const anomalyCsv = await this.fetchCsv(`${this.basePath}/data/quality/${crop}/anomaly_detection_metric_table.csv`);
        if (anomalyCsv) {
          anomalyCsv.forEach(r => {
            if (!r.column && !r.f1_score) return;
            this.anomalyDetectionTable.push({
              crop,
              column: r.column,
              model: QualityDataLoader.slugToModelName(r.model_name),
              model_slug: QualityDataLoader.modelNameToSlug(r.model_name),
              split: r.split,
              f1_score: parseFloat(r.f1_score) || 0,
              precision: parseFloat(r.precision) || 0,
              recall: parseFloat(r.recall) || 0,
              roc_auc: parseFloat(r.roc_auc) || 0,
              pr_auc: parseFloat(r.pr_auc) || 0,
              mcc: parseFloat(r.mcc) || null,
              fpr: parseFloat(r.fpr) || null,
              false_negative: parseInt(r.false_negative ?? r.confusion_matrix_false_negative, 10) || 0,
              false_positive: parseInt(r.false_positive ?? r.confusion_matrix_false_positive, 10) || 0,
              true_negative: parseInt(r.true_negative ?? r.confusion_matrix_true_negative, 10) || 0,
              true_positive: parseInt(r.true_positive ?? r.confusion_matrix_true_positive, 10) || 0,
              threshold_raw: r.threshold,
              injected_cells: parseInt(r.injected_cells, 10) || 0
            });
          });
        }

        // Online Benchmark table
        const benchCsv = await this.fetchCsv(`${this.basePath}/data/quality/${crop}/online_benchmark_table.csv`);
        if (benchCsv) {
          benchCsv.forEach(r => {
            if (!r.model_name && !r.inference_latency_ms_per_row) return;
            const modelSizeBytes = parseInt(r.model_size_bytes, 10) || 0;
            const peakMemBytes = parseInt(r.peak_memory_bytes, 10) || 0;
            this.onlineBenchmarkTable.push({
              crop,
              model: QualityDataLoader.slugToModelName(r.model_name),
              model_slug: QualityDataLoader.modelNameToSlug(r.model_name),
              split: r.split,
              inference_latency_ms_per_row: parseFloat(r.inference_latency_ms_per_row) || 0,
              measured_rows: parseInt(r.measured_rows, 10) || 0,
              repeats: parseInt(r.repeats, 10) || 1,
              model_size_bytes: modelSizeBytes,
              model_size_mb: modelSizeBytes ? +(modelSizeBytes / (1024 * 1024)).toFixed(4) : 0,
              peak_memory_bytes: peakMemBytes,
              peak_memory_mb: peakMemBytes ? +(peakMemBytes / (1024 * 1024)).toFixed(2) : 0
            });
          });
        }
      }

      this.isLoaded = true;
      console.log('Quality data successfully initialized:', {
        valRows: this.validationResults.length,
        testRows: this.testResults.length,
        reconRows: this.reconstructionTable.length,
        anomalyRows: this.anomalyDetectionTable.length,
        benchRows: this.onlineBenchmarkTable.length
      });
      return true;
    } catch (err) {
      console.error('Failed to initialize QualityDataLoader:', err);
      return false;
    }
  }

  normalizeSummaryRow(r, split) {
    return {
      crop: r.crop,
      crop_label: QualityDataLoader.formatCrop(r.crop),
      model: r.model,
      model_slug: QualityDataLoader.modelNameToSlug(r.model),
      split: split,
      masked_cells: parseInt(r.masked_cells, 10) || 0,
      rmse: parseFloat(r.rmse) || 0,
      mae: parseFloat(r.mae) || 0,
      injected_cells: parseInt(r.injected_cells, 10) || 0,
      f1_score: parseFloat(r.f1_score) || 0,
      precision: parseFloat(r.precision) || 0,
      recall: parseFloat(r.recall) || 0,
      mcc: parseFloat(r.mcc) || 0,
      fpr: parseFloat(r.fpr) || 0,
      roc_auc: parseFloat(r.roc_auc) || 0,
      pr_auc: parseFloat(r.pr_auc) || 0,
      inference_latency_ms_per_row: parseFloat(r.inference_latency_ms_per_row) || 0,
      peak_memory_mb: parseFloat(r.peak_memory_mb) || 0,
      model_size_mb: parseFloat(r.model_size_mb) || 0
    };
  }

  async fetchJson(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      return await res.json();
    } catch (e) {
      console.warn(`Could not load JSON: ${url}`, e);
      return null;
    }
  }

  async fetchCsv(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      const text = await res.text();
      return new Promise((resolve) => {
        if (window.Papa) {
          window.Papa.parse(text, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => resolve(results.data),
            error: () => resolve([])
          });
        } else {
          resolve([]);
        }
      });
    } catch (e) {
      console.warn(`Could not load CSV: ${url}`, e);
      return [];
    }
  }

  // Model-specific lazy loaders with in-memory caching
  async getTrainingHistory(crop, model) {
    const slug = QualityDataLoader.modelNameToSlug(model);
    const key = `${crop}_${slug}`;
    if (this.trainingHistories[key]) return this.trainingHistories[key];

    const url = `${this.basePath}/data/quality/${crop}/${slug}/training_history.csv`;
    const rows = await this.fetchCsv(url);
    const parsed = (rows || []).map(r => ({
      epoch: parseInt(r.epoch, 10) || 0,
      train_loss: parseFloat(r.train_reconstruction_loss) || 0,
      val_loss: parseFloat(r.validation_reconstruction_loss) || 0
    }));
    this.trainingHistories[key] = parsed;
    return parsed;
  }

  async getBestConfig(crop, model) {
    const slug = QualityDataLoader.modelNameToSlug(model);
    const key = `${crop}_${slug}`;
    if (this.bestConfigs[key]) return this.bestConfigs[key];

    const url = `${this.basePath}/data/quality/${crop}/${slug}/best_config.json`;
    const data = await this.fetchJson(url);
    this.bestConfigs[key] = data;
    return data;
  }

  async getThresholdCalibration(crop, model) {
    const slug = QualityDataLoader.modelNameToSlug(model);
    const key = `${crop}_${slug}`;
    if (this.thresholdCalibrations[key]) return this.thresholdCalibrations[key];

    const url = `${this.basePath}/data/quality/${crop}/${slug}/threshold_calibration.json`;
    const data = await this.fetchJson(url);
    this.thresholdCalibrations[key] = data;
    return data;
  }

  async getModelApplication(crop, model) {
    const slug = QualityDataLoader.modelNameToSlug(model);
    const key = `${crop}_${slug}`;
    if (this.modelApplications[key]) return this.modelApplications[key];

    const url = `${this.basePath}/data/quality/${crop}/${slug}/quality_model_application.json`;
    const data = await this.fetchJson(url);
    this.modelApplications[key] = data;
    return data;
  }

  getSummaryResults(filters = {}) {
    const { split = 'validation', crop = 'all', model = 'all' } = filters;
    const base = split === 'test' ? this.testResults : this.validationResults;
    return base.filter(item => {
      if (crop !== 'all' && item.crop.toLowerCase() !== crop.toLowerCase()) return false;
      if (model !== 'all' && item.model.toLowerCase() !== model.toLowerCase()) return false;
      return true;
    });
  }

  getReconstructionMetrics(filters = {}) {
    const { split = 'validation', crop = 'all', model = 'all', variable = 'all' } = filters;
    return this.reconstructionTable.filter(item => {
      if (item.split.toLowerCase() !== split.toLowerCase()) return false;
      if (crop !== 'all' && item.crop.toLowerCase() !== crop.toLowerCase()) return false;
      if (model !== 'all' && item.model.toLowerCase() !== model.toLowerCase()) return false;
      if (variable !== 'all' && item.column !== variable) return false;
      return true;
    });
  }

  getAnomalyDetectionMetrics(filters = {}) {
    const { split = 'validation', crop = 'all', model = 'all', variable = 'all' } = filters;
    return this.anomalyDetectionTable.filter(item => {
      if (item.split.toLowerCase() !== split.toLowerCase()) return false;
      if (crop !== 'all' && item.crop.toLowerCase() !== crop.toLowerCase()) return false;
      if (model !== 'all' && item.model.toLowerCase() !== model.toLowerCase()) return false;
      if (variable !== 'all' && item.column !== variable) return false;
      return true;
    });
  }

  getResourceMetrics(filters = {}) {
    const { split = 'validation', crop = 'all', model = 'all' } = filters;
    return this.onlineBenchmarkTable.filter(item => {
      if (item.split.toLowerCase() !== split.toLowerCase()) return false;
      if (crop !== 'all' && item.crop.toLowerCase() !== crop.toLowerCase()) return false;
      if (model !== 'all' && item.model.toLowerCase() !== model.toLowerCase()) return false;
      return true;
    });
  }

  getVariables() {
    return [
      { key: 'all', label: '전체 변수 (All Variables)' },
      { key: '__all__', label: '종합 평균/합계 (__all__)' },
      { key: 'in_temp', label: '실내온도 1 (in_temp)' },
      { key: 'in_temp2', label: '실내온도 2 (in_temp2)' },
      { key: 'in_hum', label: '실내습도 1 (in_hum)' },
      { key: 'in_hum2', label: '실내습도 2 (in_hum2)' },
      { key: 'in_co2', label: '실내 CO₂ 1 (in_co2)' },
      { key: 'in_co2_2', label: '실내 CO₂ 2 (in_co2_2)' },
      { key: 'in_medium_temp1', label: '배지온도 1 (in_medium_temp1)' },
      { key: 'in_medium_hum1', label: '배지습도 1 (in_medium_hum1)' }
    ];
  }
}

export const qualityDataLoader = new QualityDataLoader();
