/**
 * GEAS Dashboard - Quality HPO & Threshold Calibration Section
 */

import { qualityDataLoader, QualityDataLoader } from '../quality-data-loader.js';
import { qualityFilterStore } from '../quality-filter-store.js';
import { DataTable } from '../../components/data-table.js';
import { ChartRenderer } from '../../components/chart-renderer.js';
import { QualityCharts } from '../quality-charts.js';

export class QualityHpoThresholdSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.table = null;
    this.lossChart = null;
    this.selectedCrop = 'cucumber';
    this.selectedModel = 'ModernTCN';
    this.thresholdVariable = 'in_temp';
    this.thresholdSelectedRange = false;
  }

  async render() {
    if (!this.container) return;
    const filterState = qualityFilterStore.getState();
    if (filterState.crop !== 'all') this.selectedCrop = filterState.crop;
    if (filterState.model !== 'all') this.selectedModel = filterState.model;

    this.container.innerHTML = `
      <div class="section-container">
        <!-- 1. Header -->
        <div class="section-header-card">
          <div class="section-badge-row">
            <span class="badge badge-primary">HPO & Calibration</span>
            <span class="badge badge-success">Source: injected_validation_scores</span>
            <span class="badge badge-info">100 Equal-spaced Candidates</span>
          </div>
          <h2 class="section-title">🎛️ 하이퍼파라미터 최적화(HPO) 설정, 학습 곡선 및 임계값 보정</h2>
          <p class="section-description">
            각 작물 및 모델별 50회 HPO 탐색을 통해 확정된 최적 하이퍼파라미터 설정(<code>best_config.json</code>), 
            에포크별 재구성 손실 학습 곡선(<code>training_history.csv</code>), 그리고 
            합성 이상치 주입 검증 점수를 기반으로 도출된 변수별 최적 F1 임계값(<code>threshold_calibration.json</code>) 분석입니다.
          </p>
        </div>

        <!-- 2. Target Range Source Notice Banner -->
        <div class="disclaimer-banner" style="background: #f0fdf4; border-left: 4px solid #10b981; color: #065f46; margin-bottom: 1.25rem;">
          <div>
            <strong>📌 임계값 탐색 범위 명세 (injected_validation_scores):</strong><br>
            각 변수별 최소/최대 검증 오차(Clean 오차 상한 + Injected 점수 상한)를 100등분하여 균등 간격의 임계값 후보를 생성하고,
            검증 세트에서 F1-Score를 극대화하는 지점을 최적 임계값으로 확정하였습니다.
          </div>
        </div>

        <!-- 3. Dynamic Artifact Controls -->
        <div class="content-card" style="margin-bottom: 1.25rem;">
          <div style="display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">
            <div>
              <label style="font-size: 0.8rem; font-weight: 700; color: #475569; display: block; margin-bottom: 0.25rem;">작물 선택</label>
              <select id="hpo-crop-select" class="qc-select">
                <option value="cucumber" ${this.selectedCrop === 'cucumber' ? 'selected' : ''}>오이 (Cucumber)</option>
                <option value="melon" ${this.selectedCrop === 'melon' ? 'selected' : ''}>멜론 (Melon)</option>
                <option value="strawberry" ${this.selectedCrop === 'strawberry' ? 'selected' : ''}>딸기 (Strawberry)</option>
              </select>
            </div>
            <div>
              <label style="font-size: 0.8rem; font-weight: 700; color: #475569; display: block; margin-bottom: 0.25rem;">대상 모델</label>
              <select id="hpo-model-select" class="qc-select">
                <option value="ModernTCN" ${this.selectedModel === 'ModernTCN' ? 'selected' : ''}>ModernTCN (최우선 선정)</option>
                <option value="TimesNet" ${this.selectedModel === 'TimesNet' ? 'selected' : ''}>TimesNet</option>
                <option value="PatchTST" ${this.selectedModel === 'PatchTST' ? 'selected' : ''}>PatchTST</option>
              </select>
            </div>
            <div style="align-self: flex-end;">
              <button id="hpo-load-btn" class="qc-btn qc-btn-primary">아티팩트 조회</button>
            </div>
          </div>
        </div>

        <!-- 4. Interactive Training & HPO Charts (Requirement 2) -->
        <div class="two-column-grid">
          <div class="content-card">
            <h3 class="card-title">
              <span>📈</span> HPO 탐색 진도 곡선 (Search Progress)
            </h3>
            <p style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
              실제 <code>hpo_results.json</code>의 Trial 후보 순서별 Validation RMSE 수렴 궤적입니다.
            </p>
            <div style="height: 300px; position: relative;">
              <canvas id="qc-hpo-progress-chart"></canvas>
            </div>
          </div>

          <div class="content-card">
            <h3 class="card-title">
              <span>📉</span> 에포크별 재구성 손실 학습 곡선 (Loss Curve)
            </h3>
            <p style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
              실제 <code>training_history.csv</code>의 훈련(Train) 및 검증(Val) 손실 추이입니다.
            </p>
            <div style="height: 300px; position: relative;">
              <canvas id="qc-training-loss-chart"></canvas>
            </div>
          </div>
        </div>

        <!-- 5. Interactive Per-Variable Threshold Calibration Curve (Requirement 3) -->
        <div class="content-card" style="margin-top: 1.25rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.75rem;">
            <h3 class="card-title" style="margin: 0; border: none;">
              <span>🎚️</span> 변수별 임계값 보정 탐색 곡선 (Threshold Calibration Curve)
            </h3>
            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
              <label for="select-threshold-var" style="font-size: 0.8rem; font-weight: 600; color: #475569;">변수:</label>
              <select id="select-threshold-var" class="qc-select">
                ${Object.entries(QualityCharts.VARIABLE_INFO).map(([k, v]) => `
                  <option value="${k}" ${this.thresholdVariable === k ? 'selected' : ''}>${v.label} [${v.unit}]</option>
                `).join('')}
              </select>

              <div style="display: flex; gap: 0.25rem; margin-left: 0.5rem;">
                <button id="btn-threshold-full" class="qc-btn qc-btn-sm ${!this.thresholdSelectedRange ? 'qc-btn-primary' : 'qc-btn-outline'}">전체 범위</button>
                <button id="btn-threshold-selected" class="qc-btn qc-btn-sm ${this.thresholdSelectedRange ? 'qc-btn-primary' : 'qc-btn-outline'}">선택 지점 확대 (0~4x)</button>
              </div>
            </div>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.75rem;">
            <code>threshold_calibration.json</code>의 <code>per_column_calibration[변수].candidate_results</code> (100개 탐색 후보) 기준입니다. 별(★) 표시는 도출된 최적 임계값입니다.
          </p>
          <div style="height: 320px; position: relative;">
            <canvas id="qc-threshold-curve-chart"></canvas>
          </div>
        </div>

        <!-- 6. Best Hyperparameter JSON Display -->
        <div class="content-card" style="margin-top: 1.25rem;">
          <h3 class="card-title">
            <span>⚙️</span> 모델별 최적 하이퍼파라미터 설정 (best_config.json)
          </h3>
          <div id="hpo-best-config-container" style="max-height: 280px; overflow-y: auto; font-family: var(--font-family-mono); font-size: 0.8rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
            로딩 중...
          </div>
        </div>

        <!-- 7. Threshold Calibration Table -->
        <div id="threshold-table-container"></div>

        <!-- 8. Academic Reference Artifacts (Static Non-Interactive) -->
        <div class="content-card" style="opacity: 0.9;">
          <details>
            <summary style="cursor: pointer; font-size: 0.88rem; font-weight: 700; color: #475569; padding: 0.25rem 0;">
              🖼️ 논문/보고서 제출용 원본 정적 아티팩트 보기 (Static Reference Figures - Non-Interactive)
            </summary>
            <p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0.75rem 0;">
              ※ 아래 이미지는 실험 산출물 디렉터리에 보관된 정적 이미지이며 실시간 필터 적용 대상이 아닙니다.
            </p>
            <div class="reference-figures-grid">
              <div class="reference-figure-box">
                <div class="figure-label">HPO Optimization Progress</div>
                <img src="quality_panel_bundle/data/quality/${this.selectedCrop}/figures/hpo_progress.png" alt="HPO Progress" class="ref-img" loading="lazy">
                <div class="figure-caption">50회 Trial 동안의 목적함수 수렴 과정</div>
              </div>
              <div class="reference-figure-box">
                <div class="figure-label">Training vs Validation Loss</div>
                <img src="quality_panel_bundle/data/quality/${this.selectedCrop}/figures/training_loss.png" alt="Training Loss" class="ref-img" loading="lazy">
                <div class="figure-caption">에포크별 재구성 손실 및 조기종료(Early Stopping) 시점</div>
              </div>
              <div class="reference-figure-box">
                <div class="figure-label">Threshold Calibration Curve</div>
                <img src="quality_panel_bundle/data/quality/${this.selectedCrop}/figures/threshold_curve.png" alt="Threshold Curve" class="ref-img" loading="lazy">
                <div class="figure-caption">전체 점수 범위에서의 F1-Score 변화 추이</div>
              </div>
              <div class="reference-figure-box">
                <div class="figure-label">Selected Range Threshold Curve</div>
                <img src="quality_panel_bundle/data/quality/${this.selectedCrop}/figures/threshold_curve_selected_range.png" alt="Selected Range Curve" class="ref-img" loading="lazy">
                <div class="figure-caption">injected_validation_scores 선정 구간 내 최적 임계값 도출 상세</div>
              </div>
            </div>
          </details>
        </div>
      </div>
    `;

    this.bindEvents();
    await this.loadModelArtifacts();
  }

  bindEvents() {
    const cropSelect = document.getElementById('hpo-crop-select');
    const modelSelect = document.getElementById('hpo-model-select');
    const loadBtn = document.getElementById('hpo-load-btn');

    const updateSelection = async () => {
      if (cropSelect) this.selectedCrop = cropSelect.value;
      if (modelSelect) this.selectedModel = modelSelect.value;
      await this.loadModelArtifacts();
    };

    cropSelect?.addEventListener('change', updateSelection);
    modelSelect?.addEventListener('change', updateSelection);
    loadBtn?.addEventListener('click', updateSelection);
  }

  async loadModelArtifacts() {
    const crop = this.selectedCrop;
    const model = this.selectedModel;
    const modelSlug = QualityDataLoader.modelNameToSlug(model);

    // 1. Load best_config
    const bestConfig = await qualityDataLoader.getBestConfig(crop, model);
    const configContainer = document.getElementById('hpo-best-config-container');
    if (configContainer) {
      if (bestConfig) {
        configContainer.innerHTML = `<pre style="margin: 0; white-space: pre-wrap;">${JSON.stringify(bestConfig, null, 2)}</pre>`;
      } else {
        configContainer.innerHTML = `<span style="color: var(--text-muted);">설정 파일을 찾을 수 없습니다.</span>`;
      }
    }

    // 2. Load and render HPO Search Progress (hpo_results.json)
    const hpoJson = await QualityCharts.getHpoResults(crop);
    QualityCharts.renderHpoProgressChart('qc-hpo-progress-chart', hpoJson, modelSlug);

    // 3. Load and render Training History (training_history.csv)
    const cropHistory = await QualityCharts.getTrainingHistory(crop);
    QualityCharts.renderTrainingLossChart('qc-training-loss-chart', cropHistory, modelSlug);

    // 4. Load and render Threshold Calibration (threshold_calibration.json)
    const thresholdData = await QualityCharts.getThresholdCalibration(crop);
    this.renderThresholdCurve(thresholdData, modelSlug);
    this.renderThresholdTable(thresholdData);
  }

  renderThresholdCurve(thresholdData, modelSlug) {
    QualityCharts.renderThresholdCurveChart('qc-threshold-curve-chart', thresholdData, {
      variable: this.thresholdVariable,
      selectedRange: this.thresholdSelectedRange,
      activeModel: modelSlug
    });

    const varSelect = document.getElementById('select-threshold-var');
    if (varSelect) {
      varSelect.onchange = (e) => {
        this.thresholdVariable = e.target.value;
        this.renderThresholdCurve(thresholdData, modelSlug);
      };
    }

    const btnFull = document.getElementById('btn-threshold-full');
    const btnSelected = document.getElementById('btn-threshold-selected');
    if (btnFull) {
      btnFull.onclick = () => {
        this.thresholdSelectedRange = false;
        btnFull.className = 'qc-btn qc-btn-sm qc-btn-primary';
        if (btnSelected) btnSelected.className = 'qc-btn qc-btn-sm qc-btn-outline';
        this.renderThresholdCurve(thresholdData, modelSlug);
      };
    }
    if (btnSelected) {
      btnSelected.onclick = () => {
        this.thresholdSelectedRange = true;
        btnSelected.className = 'qc-btn qc-btn-sm qc-btn-primary';
        if (btnFull) btnFull.className = 'qc-btn qc-btn-sm qc-btn-outline';
        this.renderThresholdCurve(thresholdData, modelSlug);
      };
    }
  }

  renderThresholdTable(thresholdData) {
    if (!thresholdData) return;

    const source = thresholdData.candidate_range_source || 'injected_validation_scores';
    const cols = thresholdData.columns || {};

    const rows = Object.entries(cols).map(([colName, colData]) => {
      const cleanRange = colData.clean_validation_score_range || [];
      const injectedRange = colData.injected_validation_score_range || [];
      const bestCandidate = colData.best_candidate || {};

      return {
        column: colName,
        column_label: QualityDataLoader.formatColumnName(colName),
        selected_threshold: colData.selected_threshold ?? bestCandidate.threshold,
        validation_f1: bestCandidate.validation_f1 ?? bestCandidate.f1_score ?? 0,
        validation_precision: bestCandidate.validation_precision ?? bestCandidate.precision ?? 0,
        validation_recall: bestCandidate.validation_recall ?? bestCandidate.recall ?? 0,
        clean_min: cleanRange[0] ?? 0,
        clean_max: cleanRange[1] ?? 0,
        injected_min: injectedRange[0] ?? 0,
        injected_max: injectedRange[1] ?? 0,
        candidate_count: colData.candidates ? colData.candidates.length : 100,
        range_source: source
      };
    });

    const columns = [
      { key: 'column_label', label: '대상 변수', sortable: true },
      { key: 'selected_threshold', label: '최종 보정 임계값', sortable: true, format: v => v ? v.toFixed(5) : '-' },
      { key: 'validation_f1', label: '최적 F1-Score', sortable: true, format: v => v ? v.toFixed(4) : '-' },
      { key: 'validation_precision', label: '검증 Precision', sortable: true, format: v => v ? v.toFixed(4) : '-' },
      { key: 'validation_recall', label: '검증 Recall', sortable: true, format: v => v ? v.toFixed(4) : '-' },
      { key: 'clean_max', label: 'Clean 점수 상한', sortable: true, format: v => v.toFixed(4) },
      { key: 'injected_max', label: 'Injected 점수 상한', sortable: true, format: v => v.toFixed(4) },
      { key: 'range_source', label: '탐색 범위 출처', sortable: true }
    ];

    this.table = new DataTable({
      containerId: 'threshold-table-container',
      title: `변수별 임계값 보정 상세표 (${QualityDataLoader.formatCrop(this.selectedCrop)} · ${this.selectedModel})`,
      columns,
      data: rows,
      pageSize: 10,
      initialSort: { key: 'validation_f1', dir: 'desc' }
    });
  }

  update() {
    this.render();
  }
}
