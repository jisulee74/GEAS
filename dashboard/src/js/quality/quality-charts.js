/**
 * GEAS Dashboard - Data Quality Management Interactive Web Charts & Data Service
 * 
 * Replaces static PNG figures with real CSV/JSON interactive Chart.js & HTML5 Matrix visualizers.
 * Sources:
 * - validation_results.csv & test_results.csv (Global category performance & efficiency)
 * - training_history.csv (Epoch loss curves)
 * - hpo_results.json (Random search candidate progress)
 * - threshold_calibration.json (100 candidate per-variable curves, PR curve, ROC operating point)
 * - reconstruction_metric_table.csv (Model x Variable reconstruction error heatmap)
 */

import { ChartRenderer } from '../components/chart-renderer.js';

export class QualityCharts {
  static MODEL_NAMES = {
    modern_tcn: 'ModernTCN',
    timesnet: 'TimesNet',
    patch_tst: 'PatchTST'
  };

  static MODEL_COLORS = {
    modern_tcn: '#4C78A8',
    timesnet: '#F58518',
    patch_tst: '#54A24B'
  };

  static VARIABLE_INFO = {
    in_temp: { label: '실내온도 1 (in_temp)', unit: '°C' },
    in_temp2: { label: '실내온도 2 (in_temp2)', unit: '°C' },
    in_medium_temp1: { label: '배지온도 1 (in_medium_temp1)', unit: '°C' },
    in_hum: { label: '실내습도 1 (in_hum)', unit: '%' },
    in_hum2: { label: '실내습도 2 (in_hum2)', unit: '%' },
    in_medium_hum1: { label: '배지습도 1 (in_medium_hum1)', unit: '%' },
    in_co2: { label: '실내 CO₂ 1 (in_co2)', unit: 'ppm' },
    in_co2_2: { label: '실내 CO₂ 2 (in_co2_2)', unit: 'ppm' }
  };

  static cache = {
    validationResults: null,
    testResults: null,
    trainingHistory: {},
    hpoResults: {},
    thresholdCalibrations: {},
    reconstructionTables: {}
  };

  // =========================================================================
  // Data Fetching & Caching Helpers
  // =========================================================================

  static async fetchCsv(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      const text = await res.text();
      return new Promise((resolve) => {
        if (window.Papa) {
          window.Papa.parse(text, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            complete: (results) => resolve(results.data),
            error: () => resolve([])
          });
        } else {
          resolve([]);
        }
      });
    } catch (e) {
      console.warn(`QualityCharts: Failed to fetch CSV from ${url}`, e);
      return [];
    }
  }

  static async fetchJson(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      return await res.json();
    } catch (e) {
      console.warn(`QualityCharts: Failed to fetch JSON from ${url}`, e);
      return null;
    }
  }

  static async getSummaryResults(split = 'validation') {
    const isTest = split === 'test';
    const cacheKey = isTest ? 'testResults' : 'validationResults';
    if (this.cache[cacheKey]) return this.cache[cacheKey];

    const fileName = isTest ? 'test_results.csv' : 'validation_results.csv';
    const url = `quality_panel_bundle/data/quality/${fileName}`;
    const data = await this.fetchCsv(url);
    this.cache[cacheKey] = data;
    return data;
  }

  static async getValidationSummaryData() {
    return this.getSummaryResults('validation');
  }

  static async getTrainingHistory(crop) {
    if (this.cache.trainingHistory[crop]) return this.cache.trainingHistory[crop];
    const url = `quality_panel_bundle/data/quality/${crop}/training_history.csv`;
    const data = await this.fetchCsv(url);
    this.cache.trainingHistory[crop] = data;
    return data;
  }

  static async getHpoResults(crop) {
    if (this.cache.hpoResults[crop]) return this.cache.hpoResults[crop];
    const url = `quality_panel_bundle/data/quality/${crop}/hpo_results.json`;
    const data = await this.fetchJson(url);
    this.cache.hpoResults[crop] = data;
    return data;
  }

  static async getThresholdCalibration(crop) {
    if (this.cache.thresholdCalibrations[crop]) return this.cache.thresholdCalibrations[crop];
    const url = `quality_panel_bundle/data/quality/${crop}/threshold_calibration.json`;
    const data = await this.fetchJson(url);
    this.cache.thresholdCalibrations[crop] = data;
    return data;
  }

  static async getReconstructionTable(crop) {
    if (this.cache.reconstructionTables[crop]) return this.cache.reconstructionTables[crop];
    const url = `quality_panel_bundle/data/quality/${crop}/reconstruction_metric_table.csv`;
    const data = await this.fetchCsv(url);
    this.cache.reconstructionTables[crop] = data;
    return data;
  }

  // =========================================================================
  // 1. Performance & Computational Efficiency Charts (Overview, Resource, Detection)
  // =========================================================================

  /**
   * Render Multi-Crop Reconstruction Performance (RMSE & MAE)
   */
  static renderReconstructionPerformanceChart(canvasId, summaryData, activeCrop = 'all') {
    if (!summaryData || summaryData.length === 0) return;

    let filtered = summaryData;
    if (activeCrop !== 'all') {
      filtered = summaryData.filter(d => d.crop?.toLowerCase() === activeCrop.toLowerCase());
    }

    const crops = activeCrop !== 'all' ? [activeCrop] : ['cucumber', 'melon', 'strawberry'];
    const cropLabels = crops.map(c => c === 'cucumber' ? '오이' : (c === 'melon' ? '멜론' : '딸기'));
    const models = ['ModernTCN', 'TimesNet', 'PatchTST'];

    const datasets = models.map(m => {
      const slug = m === 'ModernTCN' ? 'modern_tcn' : (m === 'TimesNet' ? 'timesnet' : 'patch_tst');
      const color = this.MODEL_COLORS[slug];
      const data = crops.map(c => {
        const item = summaryData.find(d => d.crop?.toLowerCase() === c.toLowerCase() && d.model === m);
        return item ? item.rmse : null;
      });
      return {
        label: `${m} (RMSE ↓)`,
        data,
        backgroundColor: color,
        borderRadius: 4,
        barPercentage: 0.8
      };
    });

    ChartRenderer.getOrCreateChart(canvasId, {
      type: 'bar',
      data: {
        labels: cropLabels,
        datasets
      },
      options: {
        plugins: {
          title: {
            display: false
          },
          tooltip: {
            callbacks: {
              afterBody: (context) => {
                const item = summaryData.find(d => 
                  d.crop?.toLowerCase() === crops[context[0].dataIndex]?.toLowerCase() && 
                  d.model === models[context[0].datasetIndex]
                );
                return item ? `MAE: ${item.mae?.toFixed(3)} | 마스킹 셀: ${item.masked_cells?.toLocaleString()}개` : '';
              }
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: '마스킹 복원 RMSE (낮을수록 우수)', font: { size: 11, weight: 600 } }
          }
        }
      }
    });
  }

  /**
   * Render Multi-Crop Anomaly Detection Classification (F1-score & ROC-AUC)
   */
  static renderAnomalyClassificationChart(canvasId, summaryData, activeCrop = 'all') {
    if (!summaryData || summaryData.length === 0) return;

    const crops = activeCrop !== 'all' ? [activeCrop] : ['cucumber', 'melon', 'strawberry'];
    const cropLabels = crops.map(c => c === 'cucumber' ? '오이' : (c === 'melon' ? '멜론' : '딸기'));
    const models = ['ModernTCN', 'TimesNet', 'PatchTST'];

    const datasets = models.map(m => {
      const slug = m === 'ModernTCN' ? 'modern_tcn' : (m === 'TimesNet' ? 'timesnet' : 'patch_tst');
      const color = this.MODEL_COLORS[slug];
      const data = crops.map(c => {
        const item = summaryData.find(d => d.crop?.toLowerCase() === c.toLowerCase() && d.model === m);
        return item ? item.f1_score : null;
      });
      return {
        label: `${m} (F1 ↑)`,
        data,
        backgroundColor: color,
        borderRadius: 4,
        barPercentage: 0.8
      };
    });

    ChartRenderer.getOrCreateChart(canvasId, {
      type: 'bar',
      data: {
        labels: cropLabels,
        datasets
      },
      options: {
        scales: {
          y: {
            min: 0,
            max: 1.0,
            title: { display: true, text: '이상치 탐지 F1-Score (높을수록 우수)', font: { size: 11, weight: 600 } }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              afterBody: (context) => {
                const item = summaryData.find(d => 
                  d.crop?.toLowerCase() === crops[context[0].dataIndex]?.toLowerCase() && 
                  d.model === models[context[0].datasetIndex]
                );
                return item ? `Precision: ${item.precision?.toFixed(4)} | Recall: ${item.recall?.toFixed(4)} | ROC-AUC: ${item.roc_auc?.toFixed(4)}` : '';
              }
            }
          }
        }
      }
    });
  }

  /**
   * Render Computational Efficiency Multi-panel Chart (Latency, Peak RAM, Model Size)
   */
  static renderComputationalEfficiencyChart(canvasId, summaryData, metricKey = 'inference_latency_ms_per_row') {
    if (!summaryData || summaryData.length === 0) return;

    const crops = ['cucumber', 'melon', 'strawberry'];
    const cropLabels = ['오이', '멜론', '딸기'];
    const models = ['ModernTCN', 'TimesNet', 'PatchTST'];

    const metricLabels = {
      inference_latency_ms_per_row: '추론 지연시간 (ms/row ↓)',
      peak_memory_mb: '피크 RAM 사용량 (MB ↓)',
      model_size_mb: '모델 디스크 크기 (MB ↓)'
    };

    const datasets = models.map(m => {
      const slug = m === 'ModernTCN' ? 'modern_tcn' : (m === 'TimesNet' ? 'timesnet' : 'patch_tst');
      const color = this.MODEL_COLORS[slug];
      const data = crops.map(c => {
        const item = summaryData.find(d => d.crop?.toLowerCase() === c.toLowerCase() && d.model === m);
        return item ? item[metricKey] : null;
      });
      return {
        label: `${m}`,
        data,
        backgroundColor: color,
        borderRadius: 4,
        barPercentage: 0.8
      };
    });

    ChartRenderer.getOrCreateChart(canvasId, {
      type: 'bar',
      data: {
        labels: cropLabels,
        datasets
      },
      options: {
        scales: {
          y: {
            title: { display: true, text: metricLabels[metricKey] || metricKey, font: { size: 11, weight: 600 } }
          }
        }
      }
    });
  }

  // =========================================================================
  // 2. Training Loss Curve & HPO Search Progress (Quality HPO Section)
  // =========================================================================

  /**
   * Render Training Loss & Validation Loss over Epochs
   */
  static renderTrainingLossChart(canvasId, historyRows, activeModel = 'all') {
    if (!historyRows || historyRows.length === 0) return;

    const grouped = {};
    historyRows.forEach(r => {
      const model = r.model_name;
      if (!grouped[model]) grouped[model] = [];
      grouped[model].push(r);
    });

    const datasets = [];
    const models = activeModel === 'all' ? ['modern_tcn', 'timesnet', 'patch_tst'] : [activeModel];

    models.forEach(model => {
      const rows = grouped[model];
      if (!rows || rows.length === 0) return;

      const baseColor = this.MODEL_COLORS[model] || '#4f46e5';
      const label = this.MODEL_NAMES[model] || model;

      // Validation Loss
      datasets.push({
        label: `${label} (Validation Loss)`,
        data: rows.map(r => ({ x: r.epoch, y: r.validation_reconstruction_loss })),
        borderColor: baseColor,
        backgroundColor: baseColor,
        borderWidth: 2,
        tension: 0.2,
        pointRadius: 3,
        pointHoverRadius: 6
      });

      // Train Loss (Dashed)
      datasets.push({
        label: `${label} (Train Loss)`,
        data: rows.map(r => ({ x: r.epoch, y: r.train_reconstruction_loss })),
        borderColor: baseColor,
        borderDash: [5, 4],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        tension: 0.2,
        pointRadius: 2,
        pointHoverRadius: 5
      });
    });

    ChartRenderer.getOrCreateChart(canvasId, {
      type: 'line',
      data: { datasets },
      options: {
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: '학습 Epoch', font: { size: 11, weight: 600 } },
            ticks: { precision: 0 }
          },
          y: {
            title: { display: true, text: '재구성 손실 (Reconstruction Loss)', font: { size: 11, weight: 600 } }
          }
        },
        plugins: {
          tooltip: {
            mode: 'index',
            intersect: false
          }
        }
      }
    });
  }

  /**
   * Render HPO Progress across Trials
   */
  static renderHpoProgressChart(canvasId, hpoJson, activeModel = 'all') {
    if (!hpoJson || !hpoJson.models) return;

    const models = activeModel === 'all' ? ['modern_tcn', 'timesnet', 'patch_tst'] : [activeModel];
    const datasets = [];

    models.forEach(model => {
      const modelPayload = hpoJson.models[model];
      if (!modelPayload || !modelPayload.candidate_results) return;

      const candidates = modelPayload.candidate_results;
      const baseColor = this.MODEL_COLORS[model] || '#4f46e5';
      const label = this.MODEL_NAMES[model] || model;

      const points = candidates.map((cand, idx) => ({
        x: idx + 1,
        y: cand.validation_metrics?.validation_synthetic_masking_rmse ?? null,
        name: cand.name
      })).filter(p => p.y !== null);

      datasets.push({
        label: `${label} 탐색 궤적`,
        data: points,
        borderColor: baseColor,
        backgroundColor: baseColor,
        borderWidth: 1.8,
        pointRadius: 3.5,
        pointHoverRadius: 6,
        tension: 0.1
      });
    });

    ChartRenderer.getOrCreateChart(canvasId, {
      type: 'line',
      data: { datasets },
      options: {
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Random Search 탐색 후보 (Candidate Index)', font: { size: 11, weight: 600 } },
            ticks: { precision: 0 }
          },
          y: {
            title: { display: true, text: 'Validation Synthetic Masking RMSE ↓', font: { size: 11, weight: 600 } }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const pt = ctx.raw;
                return pt?.name ? `후보명: ${pt.name}` : '';
              }
            }
          }
        }
      }
    });
  }

  // =========================================================================
  // 3. Per-Variable Threshold Performance Curve (Quality HPO Section)
  // =========================================================================

  /**
   * Render Per-Variable Threshold vs F1-score Curve
   * Uses per_column_calibration[variable].candidate_results (100 candidates)
   */
  static renderThresholdCurveChart(canvasId, thresholdJson, { variable = 'in_temp', selectedRange = false, activeModel = 'all' }) {
    if (!thresholdJson || !thresholdJson.models) return;

    const models = activeModel === 'all' ? ['modern_tcn', 'timesnet', 'patch_tst'] : [activeModel];
    const datasets = [];
    let selectedThresholds = [];
    let candidateThresholds = [];

    models.forEach(model => {
      const modelPayload = thresholdJson.models[model];
      if (!modelPayload) return;

      const colCalibration = modelPayload.per_column_calibration?.[variable];
      if (!colCalibration || !colCalibration.candidate_results) return;

      const baseColor = this.MODEL_COLORS[model] || '#4f46e5';
      const label = this.MODEL_NAMES[model] || model;
      const bestX = colCalibration.best_threshold;
      const bestY = colCalibration.best_objective_value;

      if (bestX != null) selectedThresholds.push(bestX);

      const points = colCalibration.candidate_results
        .map(c => ({
          x: c.threshold,
          y: c.f1_score,
          precision: c.precision,
          recall: c.recall
        }))
        .filter(p => p.x != null && p.y != null)
        .sort((a, b) => a.x - b.x);

      points.forEach(p => candidateThresholds.push(p.x));

      datasets.push({
        label: `${label} (F1 탐색곡선)`,
        data: points,
        borderColor: baseColor,
        backgroundColor: baseColor,
        borderWidth: 1.8,
        pointRadius: 2.5,
        pointHoverRadius: 5,
        tension: 0.1
      });

      // Add Star marker for Selected Best Threshold
      if (bestX != null && bestY != null) {
        datasets.push({
          label: `${label} 선택 임계값 (${bestX.toFixed(2)})`,
          data: [{ x: bestX, y: bestY }],
          borderColor: '#0f172a',
          backgroundColor: baseColor,
          pointStyle: 'star',
          pointRadius: 9,
          pointHoverRadius: 12,
          showLine: false
        });
      }
    });

    // Calculate x-axis max if in selectedRange mode
    let xMax = null;
    if (selectedRange && selectedThresholds.length > 0 && candidateThresholds.length > 0) {
      const maxSelected = Math.max(...selectedThresholds);
      const maxCandidate = Math.max(...candidateThresholds);
      const rawLimit = maxSelected * 4.0;
      const legacyLimit = Math.max(10.0, Math.ceil(rawLimit / 10.0) * 10.0);
      xMax = Math.min(maxCandidate, legacyLimit);
    }

    const varMeta = this.VARIABLE_INFO[variable] || { label: variable, unit: '' };

    ChartRenderer.getOrCreateChart(canvasId, {
      type: 'line',
      data: { datasets },
      options: {
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: `이상치 판정 임계값 Threshold (${varMeta.unit})`, font: { size: 11, weight: 600 } },
            ...(xMax != null ? { max: xMax, min: 0 } : { min: 0 })
          },
          y: {
            min: 0,
            max: 1.05,
            title: { display: true, text: 'Validation F1-score (0 ~ 1.0)', font: { size: 11, weight: 600 } }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const pt = ctx.raw;
                return pt?.precision != null ? `Precision: ${pt.precision.toFixed(4)} | Recall: ${pt.recall.toFixed(4)}` : '';
              }
            }
          }
        }
      }
    });
  }

  // =========================================================================
  // 4. Reconstruction Error Matrix / Heatmap (Quality Reconstruction Section)
  // =========================================================================

  /**
   * Render Model x Variable Reconstruction Heatmap Grid
   * Excludes '__all__'. Includes unit badges & column-normalized colors.
   */
  static renderReconstructionHeatmap(containerId, reconRows, { metric = 'rmse', split = 'validation' }) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!reconRows || reconRows.length === 0) {
      container.innerHTML = `<div class="empty-notice">복원 오차 데이터가 존재하지 않습니다.</div>`;
      return;
    }

    // Filter by split and exclude '__all__'
    const filtered = reconRows.filter(r => r.split?.toLowerCase() === split.toLowerCase() && r.column !== '__all__');
    const variables = Object.keys(this.VARIABLE_INFO);
    const models = ['modern_tcn', 'timesnet', 'patch_tst'];

    // Pivot table: matrix[model][col] = value
    const matrix = {};
    const colValues = {};
    variables.forEach(v => colValues[v] = []);

    models.forEach(m => {
      matrix[m] = {};
      variables.forEach(v => {
        const item = filtered.find(r => (r.model_name === m || r.model_slug === m) && r.column === v);
        const val = item ? (metric === 'mae' ? item.mae : item.rmse) : null;
        matrix[m][v] = val;
        if (val != null) colValues[v].push(val);
      });
    });

    // Compute min and max per column for column-wise relative coloring
    const colMinMax = {};
    variables.forEach(v => {
      const vals = colValues[v];
      colMinMax[v] = {
        min: vals.length > 0 ? Math.min(...vals) : 0,
        max: vals.length > 0 ? Math.max(...vals) : 1
      };
    });

    // Helper for cell color interpolation: lowest error = crisp green, highest error = soft warm amber
    const getCellBg = (val, col) => {
      if (val == null) return '#f1f5f9';
      const { min, max } = colMinMax[col];
      const range = max - min;
      const ratio = range > 0 ? (val - min) / range : 0.5; // 0 = best (lowest error), 1 = worst
      // HSL: 150 (green) -> 45 (amber) -> 10 (reddish)
      const hue = Math.round(145 - ratio * 110);
      const lightness = Math.round(96 - ratio * 14);
      return `hsl(${hue}, 80%, ${lightness}%)`;
    };

    const getCellBorder = (val, col) => {
      if (val == null) return '#e2e8f0';
      const { min, max } = colMinMax[col];
      const range = max - min;
      const ratio = range > 0 ? (val - min) / range : 0.5;
      const hue = Math.round(145 - ratio * 110);
      return `hsl(${hue}, 75%, 65%)`;
    };

    let html = `
      <div class="reconstruction-heatmap-wrapper">
        <div class="heatmap-toolbar" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.75rem;">
          <div style="font-size: 0.88rem; font-weight: 700; color: #0f172a;">
            📊 모델 × 변수별 복원 정밀도 매트릭스 (${split.toUpperCase()} 분할)
          </div>
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">평가 지표:</span>
            <button class="qc-btn qc-btn-sm ${metric === 'rmse' ? 'qc-btn-primary' : 'qc-btn-outline'}" id="btn-heatmap-rmse" data-metric="rmse">RMSE</button>
            <button class="qc-btn qc-btn-sm ${metric === 'mae' ? 'qc-btn-primary' : 'qc-btn-outline'}" id="btn-heatmap-mae" data-metric="mae">MAE</button>
          </div>
        </div>

        <div class="qc-academic-callout" style="background: #eff6ff; border-left: 4px solid #3b82f6; padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.82rem; color: #1e3a8a; margin-bottom: 1rem; line-height: 1.55;">
          <strong>⚠️ 학술 단위 해석 주의:</strong> 온도(°C), 습도(%), CO₂(ppm)는 물리적 단위 및 스케일이 상이하므로 서로 다른 변수 간 오차 절대값의 직접 비교는 무의미합니다.
          본 매트릭스의 셀 배경색은 <strong>변수별(열 기준) 정규화</strong>를 적용하여, 동일 변수 내에서 모델 간(ModernTCN vs TimesNet vs PatchTST) 상대적 우열을 명확히 파악할 수 있도록 표기되었습니다.
        </div>

        <div style="overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 8px;">
          <table class="data-table" style="width: 100%; border-collapse: collapse; text-align: center;">
            <thead>
              <tr style="background: #f8fafc;">
                <th style="padding: 0.85rem 1rem; font-size: 0.82rem; text-align: left; border-bottom: 2px solid #cbd5e1;">모델</th>
                ${variables.map(v => {
                  const meta = this.VARIABLE_INFO[v];
                  return `
                    <th style="padding: 0.85rem 0.75rem; font-size: 0.8rem; border-bottom: 2px solid #cbd5e1; min-width: 105px;">
                      <div>${meta.label.split(' ')[0]}</div>
                      <span class="badge badge-neutral" style="font-size: 0.7rem; padding: 0.1rem 0.4rem; margin-top: 0.2rem;">[${meta.unit}]</span>
                    </th>
                  `;
                }).join('')}
              </tr>
            </thead>
            <tbody>
              ${models.map(m => {
                const modelName = this.MODEL_NAMES[m] || m;
                const modelColor = this.MODEL_COLORS[m];
                return `
                  <tr>
                    <td style="padding: 0.85rem 1rem; text-align: left; font-weight: 700; border-bottom: 1px solid #f1f5f9; white-space: nowrap;">
                      <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: ${modelColor}; margin-right: 6px;"></span>
                      ${modelName}
                    </td>
                    ${variables.map(v => {
                      const val = matrix[m][v];
                      const meta = this.VARIABLE_INFO[v];
                      const bg = getCellBg(val, v);
                      const border = getCellBorder(val, v);
                      const isBest = val != null && val === colMinMax[v].min;
                      return `
                        <td style="padding: 0.75rem; background: ${bg}; border-bottom: 1px solid #e2e8f0; border-left: 1px solid #f1f5f9; font-family: var(--font-family-mono); font-size: 0.84rem; font-weight: ${isBest ? '800' : '600'}; color: #0f172a; position: relative;" title="${modelName} ${meta.label}: ${val?.toFixed(3)} ${meta.unit}">
                          ${val != null ? val.toFixed(3) : '-'}
                          ${isBest ? '<span style="position: absolute; top: 2px; right: 4px; font-size: 0.65rem; color: #059669;">★</span>' : ''}
                        </td>
                      `;
                    }).join('')}
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

    container.innerHTML = html;
  }

  // =========================================================================
  // 5. PR Curve & ROC Operating Point Charts (Quality Detection Section)
  // =========================================================================

  /**
   * Render Validation Threshold-Candidate-Based PR Curve
   * Clearly labeled as Candidate-based, with Star markers at Selected Threshold
   */
  static renderPrCurveChart(canvasId, thresholdJson, { variable = 'in_temp', activeModel = 'all' }) {
    if (!thresholdJson || !thresholdJson.models) return;

    const models = activeModel === 'all' ? ['modern_tcn', 'timesnet', 'patch_tst'] : [activeModel];
    const datasets = [];

    models.forEach(model => {
      const modelPayload = thresholdJson.models[model];
      if (!modelPayload) return;

      const colCalibration = modelPayload.per_column_calibration?.[variable];
      if (!colCalibration || !colCalibration.candidate_results) return;

      const baseColor = this.MODEL_COLORS[model] || '#4f46e5';
      const label = this.MODEL_NAMES[model] || model;
      const bestThreshold = colCalibration.best_threshold;

      // Extract and sort (Recall, Precision) points
      const points = colCalibration.candidate_results
        .map(c => ({
          x: c.recall,
          y: c.precision,
          threshold: c.threshold,
          f1: c.f1_score
        }))
        .filter(p => p.x != null && p.y != null)
        .sort((a, b) => a.x - b.x);

      datasets.push({
        label: `${label} (Validation 후보 기반 PR)`,
        data: points,
        borderColor: baseColor,
        backgroundColor: baseColor,
        borderWidth: 2,
        pointRadius: 2.5,
        pointHoverRadius: 5,
        stepped: 'after',
        tension: 0
      });

      // Find closest or matching point for best_threshold
      if (bestThreshold != null && points.length > 0) {
        let bestPoint = points.reduce((prev, curr) => 
          Math.abs(curr.threshold - bestThreshold) < Math.abs(prev.threshold - bestThreshold) ? curr : prev
        );
        datasets.push({
          label: `${label} 선택 임계점 (Recall ${bestPoint.x.toFixed(2)}, Precision ${bestPoint.y.toFixed(2)})`,
          data: [{ x: bestPoint.x, y: bestPoint.y }],
          borderColor: '#0f172a',
          backgroundColor: baseColor,
          pointStyle: 'star',
          pointRadius: 9,
          pointHoverRadius: 12,
          showLine: false
        });
      }
    });

    ChartRenderer.getOrCreateChart(canvasId, {
      type: 'line',
      data: { datasets },
      options: {
        scales: {
          x: {
            type: 'linear',
            min: 0,
            max: 1.05,
            title: { display: true, text: '재현율 (Recall)', font: { size: 11, weight: 600 } }
          },
          y: {
            min: 0,
            max: 1.05,
            title: { display: true, text: '정밀도 (Precision)', font: { size: 11, weight: 600 } }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const pt = ctx.raw;
                return pt?.threshold != null ? `임계값: ${pt.threshold.toFixed(2)} | F1: ${pt.f1?.toFixed(4)}` : '';
              }
            }
          }
        }
      }
    });
  }

  /**
   * Render ROC Operating Point Chart
   * Honest representation: diagonal baseline + single operating point (FPR, TPR) computed from confusion matrix.
   * Does NOT connect artificial curved lines.
   */
  static renderRocOperatingPointChart(canvasId, thresholdJson, activeModel = 'all') {
    if (!thresholdJson || !thresholdJson.models) return;

    const models = activeModel === 'all' ? ['modern_tcn', 'timesnet', 'patch_tst'] : [activeModel];
    const datasets = [];

    // 1. Diagonal Baseline (Random Guess y = x)
    datasets.push({
      label: '기준선 (Random Guess, y = x)',
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: '#94a3b8',
      borderDash: [6, 4],
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false
    });

    // 2. Single Operating Point per model computed from model-level confusion_matrix
    models.forEach(model => {
      const modelPayload = thresholdJson.models[model];
      if (!modelPayload) return;

      const cm = modelPayload.confusion_matrix;
      const rocAuc = modelPayload.roc_auc;
      const baseColor = this.MODEL_COLORS[model] || '#4f46e5';
      const label = this.MODEL_NAMES[model] || model;

      if (cm) {
        const tp = cm.true_positive || 0;
        const fp = cm.false_positive || 0;
        const tn = cm.true_negative || 0;
        const fn = cm.false_negative || 0;

        const fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0;
        const tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0;

        datasets.push({
          label: `${label} 운영점 (FPR: ${fpr.toFixed(4)}, TPR: ${tpr.toFixed(4)}) [AUC: ${rocAuc?.toFixed(4)}]`,
          data: [{ x: fpr, y: tpr, tp, fp, tn, fn, rocAuc }],
          borderColor: '#0f172a',
          backgroundColor: baseColor,
          pointStyle: 'circle',
          pointRadius: 8,
          pointHoverRadius: 11,
          showLine: false
        });
      }
    });

    ChartRenderer.getOrCreateChart(canvasId, {
      type: 'scatter',
      data: { datasets },
      options: {
        scales: {
          x: {
            min: 0,
            max: 1.0,
            title: { display: true, text: 'False Positive Rate (FPR = FP / (FP + TN))', font: { size: 11, weight: 600 } }
          },
          y: {
            min: 0,
            max: 1.05,
            title: { display: true, text: 'True Positive Rate (TPR = TP / (TP + FN))', font: { size: 11, weight: 600 } }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const pt = ctx.raw;
                if (!pt || pt.tp == null) return '';
                return `TP: ${pt.tp.toLocaleString()} | FP: ${pt.fp.toLocaleString()} | TN: ${pt.tn.toLocaleString()} | FN: ${pt.fn.toLocaleString()} | 저장된 ROC-AUC: ${pt.rocAuc?.toFixed(4)}`;
              }
            }
          }
        }
      }
    });
  }
}
