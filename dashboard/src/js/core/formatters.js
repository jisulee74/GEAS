/**
 * GEAS Dashboard - Formatting, Labels & Academic Metadata Utilities
 */

export const CROP_LABELS = {
  strawberry: 'Strawberry',
  melon: 'Melon',
  cucumber: 'Cucumber'
};

export const MODEL_LABELS = {
  persistence: 'Persistence (Baseline)',
  linear_svr: 'Linear SVR',
  ridge: 'Ridge',
  knn: 'KNN',
  extra_trees: 'ExtraTrees',
  random_forest: 'Random Forest',
  gradient_boosting: 'Gradient Boosting',
  hist_gradient_boosting: 'Hist Gradient Boosting',
  lightgbm: 'LightGBM',
  xgboost: 'XGBoost',
  catboost: 'CatBoost',
  mlp: 'MLP',
  linear_regression: 'Linear Regression'
};

export const TARGET_LABELS = {
  indoor_temperature: 'Indoor Temperature (°C)',
  indoor_humidity: 'Indoor Relative Humidity (%)',
  indoor_co2: 'Indoor CO₂ (ppm)'
};

export const TARGET_UNITS = {
  indoor_temperature: '°C',
  indoor_humidity: '%',
  indoor_co2: 'ppm'
};

export const METRIC_DESCRIPTIONS = {
  validation_rollout_weighted_nrmse: {
    name: 'Validation Rollout Weighted NRMSE',
    unit: 'NRMSE',
    direction: 'lower',
    desc: 'Official HPO objective function: 0.10 × One-Step + 0.10 × 15min Rollout + 0.25 × 30min Rollout + 0.45 × 60min Rollout + 0.10 × Drift. Primary model selection metric.'
  },
  one_step_nrmse: {
    name: '5-min One-Step NRMSE',
    unit: 'NRMSE',
    direction: 'lower',
    desc: 'Normalized Root Mean Square Error for 5-minute single-step prediction.'
  },
  rollout_15min_nrmse: {
    name: '15-min Rollout NRMSE',
    unit: 'NRMSE',
    direction: 'lower',
    desc: 'Recursive 3-step rollout prediction error.'
  },
  rollout_30min_nrmse: {
    name: '30-min Rollout NRMSE',
    unit: 'NRMSE',
    direction: 'lower',
    desc: 'Recursive 6-step rollout prediction error.'
  },
  rollout_60min_nrmse: {
    name: '60-min Rollout NRMSE',
    unit: 'NRMSE',
    direction: 'lower',
    desc: 'Recursive 12-step rollout prediction error.'
  },
  skill_vs_persistence_60min: {
    name: 'Persistence Skill (60-min)',
    unit: 'Skill',
    direction: 'higher',
    desc: 'Skill = 1 - (RMSE_model / RMSE_persistence). Values > 0 indicate superior performance over persistence baseline.'
  },
  mae: { name: 'MAE', unit: 'raw unit', direction: 'lower', desc: 'Mean Absolute Error in native target unit.' },
  rmse: { name: 'RMSE', unit: 'raw unit', direction: 'lower', desc: 'Root Mean Square Error in native target unit.' },
  nrmse: { name: 'NRMSE', unit: 'ratio', direction: 'lower', desc: 'Normalized RMSE (RMSE / standard deviation of target), dimensionless and comparable across targets.' },
  r2: { name: 'R² (Coefficient of Determination)', unit: 'score', direction: 'higher', desc: 'Proportion of variance explained. 1.0 is perfect prediction.' },
  absolute_error_q90: { name: 'Q90 Absolute Error', unit: 'raw unit', direction: 'lower', desc: '90th percentile of absolute prediction errors.' },
  absolute_error_cvar90: { name: 'CVaR90 Absolute Error', unit: 'raw unit', direction: 'lower', desc: 'Conditional Value at Risk (expected tail error beyond 90th percentile).' },
  inference_latency_median_ms: { name: 'Median Latency', unit: 'ms', direction: 'lower', desc: 'Median single-step inference time in milliseconds.' },
  training_time_seconds: { name: 'Training Time', unit: 's', direction: 'lower', desc: 'Model training duration in seconds.' },
  serialized_model_size_mb: { name: 'Model Size', unit: 'MB', direction: 'lower', desc: 'Disk size of serialized model artifact.' }
};

/**
 * Deterministic Candidate Comparator with Parsimony Tie-Breaking
 */
export function compareCandidates(a, b) {
  // 1. Deployable candidates prioritized over ineligible/baseline
  const aElig = Boolean(a.deployment_eligible && !a.baseline_only && a.model !== 'persistence');
  const bElig = Boolean(b.deployment_eligible && !b.baseline_only && b.model !== 'persistence');
  if (aElig !== bElig) return aElig ? -1 : 1;

  // 2. Lowest weighted rollout NRMSE
  const diff = (a.validation_rollout_weighted_nrmse || 0) - (b.validation_rollout_weighted_nrmse || 0);
  if (Math.abs(diff) > 1e-6) {
    return diff;
  }

  // 3. Parsimony Tie-break: When performance is identical / tied, prefer without_quality_flags (lower pipeline complexity)
  if (a.feature_variant !== b.feature_variant) {
    if (a.feature_variant === 'without_quality_flags') return -1;
    if (b.feature_variant === 'without_quality_flags') return 1;
  }

  // 4. Secondary: Lower 60-min rollout NRMSE
  const r60Diff = (a.rollout_60min_nrmse || 0) - (b.rollout_60min_nrmse || 0);
  if (Math.abs(r60Diff) > 1e-6) return r60Diff;

  // 5. Secondary: Lower One-step NRMSE
  return (a.one_step_nrmse || 0) - (b.one_step_nrmse || 0);
}

/**
 * Format numeric value with specified decimal places or scientific notation
 */
export function formatNumber(val, decimals = 4, naText = 'N/A') {
  if (val === null || val === undefined || isNaN(val) || val === '') return naText;
  const num = Number(val);
  if (!isFinite(num)) return 'Non-finite (Inf/NaN)';
  if (Math.abs(num) > 0 && Math.abs(num) < 0.0001) {
    return num.toExponential(2);
  }
  return num.toLocaleString('en-US', {
    minimumFractionDigits: Math.min(decimals, 2),
    maximumFractionDigits: decimals
  });
}

/**
 * Format percentage
 */
export function formatPercent(val, decimals = 2) {
  if (val === null || val === undefined || isNaN(val)) return 'N/A';
  const num = Number(val);
  return `${(num * 100).toFixed(decimals)}%`;
}

/**
 * Format milliseconds / time
 */
export function formatLatency(val) {
  if (val === null || val === undefined || isNaN(val)) return 'N/A';
  const num = Number(val);
  if (num < 1) return `${(num * 1000).toFixed(1)} µs`;
  return `${num.toFixed(2)} ms`;
}

/**
 * Format Memory in MB / GB
 */
export function formatMemory(val, emptyIsCpu = false) {
  if (val === null || val === undefined || isNaN(val) || val === '') {
    return emptyIsCpu ? 'N/A (CPU execution)' : 'N/A';
  }
  const num = Number(val);
  if (num >= 1024) return `${(num / 1024).toFixed(2)} GB`;
  return `${num.toFixed(1)} MB`;
}

/**
 * Format Crop Name
 */
export function formatCrop(crop) {
  if (!crop) return 'All';
  return CROP_LABELS[crop.toLowerCase()] || crop.charAt(0).toUpperCase() + crop.slice(1);
}

/**
 * Format Model Name
 */
export function formatModel(model) {
  if (!model) return 'N/A';
  return MODEL_LABELS[model.toLowerCase()] || model;
}

/**
 * Format Feature Variant
 */
export function formatVariant(variant) {
  if (!variant) return 'N/A';
  if (variant === 'with_quality_flags') return 'With Flags (품질플래그 포함)';
  if (variant === 'without_quality_flags') return 'Without Flags (품질플래그 제외)';
  return variant;
}

/**
 * Format Target Environmental Variable Name
 */
export function formatTarget(target) {
  if (!target) return 'All Targets';
  return TARGET_LABELS[target] || target;
}

/**
 * Format Quality Flag Variant Effect
 */
export function formatVariantEffect(effect) {
  if (effect === 'with_flags_improved') return 'With Flags Improved';
  if (effect === 'with_flags_worsened') return 'Without Flags Better';
  return 'No Significant Diff';
}

/**
 * Generate Badge HTML for deployment eligibility
 */
export function renderEligibilityBadge(eligible, isBaseline = false) {
  if (isBaseline) {
    return `<span class="badge badge-baseline" title="Baseline benchmark only, not for deployment"><span class="badge-dot"></span>Baseline Only</span>`;
  }
  if (eligible === true || eligible === 'True' || eligible === 'true') {
    return `<span class="badge badge-eligible" title="Passed all validation, safety, and action-response gates"><span class="badge-dot"></span>Deployable Candidate</span>`;
  }
  return `<span class="badge badge-ineligible" title="Failed deployment criteria or safety gates"><span class="badge-dot"></span>Ineligible</span>`;
}

/**
 * Generate Badge HTML for Action Response Status
 */
export function renderActionResponseBadge(actionEligible, fullyInsensitive, insufficientCount = 0) {
  if (fullyInsensitive === true || fullyInsensitive === 'True' || fullyInsensitive === 'true') {
    return `<span class="badge badge-action-insensitive" title="Completely insensitive to control actions">⚠️ Action-Insensitive</span>`;
  }
  if (insufficientCount > 0) {
    return `<span class="badge badge-insufficient-support" title="${insufficientCount} action(s) lack sufficient observational support">ℹ️ Data-Sparse (${insufficientCount})</span>`;
  }
  if (actionEligible === true || actionEligible === 'True' || actionEligible === 'true') {
    return `<span class="badge badge-safe" title="Passed counterfactual action sensitivity criteria">✓ Action-Sensitive</span>`;
  }
  return `<span class="badge badge-ineligible">Gate Failed</span>`;
}

/**
 * Generate Badge HTML for Quality Flag Effect
 */
export function renderQualityEffectBadge(effect) {
  if (effect === 'with_flags_improved') {
    return `<span class="badge badge-eligible">🟢 With Flags Improved</span>`;
  }
  if (effect === 'with_flags_worsened') {
    return `<span class="badge badge-ineligible">🔴 Without Flags Better</span>`;
  }
  return `<span class="badge badge-baseline">⚪ No Significant Diff</span>`;
}

/**
 * Generate High-Visibility Eligibility & Ineligibility Gating Criteria Box
 */
export function renderEligibilityGuide() {
  return `
    <div class="eligibility-guide-card">
      <div class="eligibility-guide-title">
        <span>🛡️</span> 배포 적격 여부(Deployment Eligibility) 판정 기준 및 부적격(Ineligible) 사유 안내
      </div>
      <div class="eligibility-guide-grid">
        <div class="guide-badge-box eligible">
          <div>
            <span class="badge badge-eligible"><span class="badge-dot"></span>Deployable Candidate (배포 적격)</span>
          </div>
          <p>
            Validation 롤아웃 중 <strong>Safety Gate</strong>(물리적 제약 준수, NaN/Inf 무결성)와 
            <strong>Action-Response Gate</strong>(제어 행동 반응성)를 모두 통과하여 온실 강화학습 시뮬레이터 배포가 승인된 후보 모델입니다.
          </p>
        </div>

        <div class="guide-badge-box ineligible">
          <div>
            <span class="badge badge-ineligible"><span class="badge-dot"></span>Ineligible (배포 부적격) 발생 4대 조건</span>
          </div>
          <ul>
            <li><strong>1. 물리 한계 위반 (Physical Breach)</strong>: 롤아웃 중 온실 온도, 습도, CO₂의 물리적/열역학적 임계 범위를 초과한 경우</li>
            <li><strong>2. 수치 발산 및 결측 (NaN / Inf)</strong>: 재귀 롤아웃 계산 중 무한대(Inf) 또는 결측치(NaN)가 발생한 경우</li>
            <li><strong>3. 제어 무반응 (Action-Insensitive)</strong>: 환기/난방/CO₂ 등 제어 Action이 바뀌어도 예측 상태가 전혀 변하지 않는 경우</li>
            <li><strong>4. 비교 기준선 (Baseline Only)</strong>: <code>Persistence</code> 모델은 성능 비교를 위한 기준선일 뿐 배포 대상이 아님</li>
          </ul>
        </div>
      </div>
    </div>
  `;
}
