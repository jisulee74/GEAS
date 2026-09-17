/**
 * GEAS Dashboard - Quality Deployment Status & Caveats Section
 */

export class QualityDeploymentSection {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="section-container">
        <!-- 1. Header -->
        <div class="section-header-card">
          <div class="section-badge-row">
            <span class="badge badge-primary">Deployment Status</span>
            <span class="badge badge-warning">Audit & Provenance Rigor</span>
          </div>
          <h2 class="section-title">🚀 데이터 품질 관리 모델 적용 현황 및 데이터 감사(Audit) 공지</h2>
          <p class="section-description">
            GEAS 3.5 파이프라인의 <strong>코드상 기본 적용 모델(Code-level Defaults)</strong>과 
            <strong>실제 적용 완료 여부 및 감사 이력</strong>을 엄격히 구분하여 기술적 투명성을 보장합니다.
          </p>
        </div>

        <!-- 2. Dual Status Comparison Cards -->
        <div class="two-column-grid">
          <!-- Card A: Code Default Selected Models -->
          <div class="content-card" style="border-top: 4px solid #4f46e5;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
              <h3 class="card-title" style="margin: 0;">
                <span>💻</span> 1. 코드상 기본 적용 모델 (Code Defaults)
              </h3>
              <span class="badge badge-primary">DEFAULT_SELECTED_MODELS</span>
            </div>
            <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
              오프라인 품질 관리 실행 스크립트(<code>03_control_quality.py</code>)에 사전 정의된 작물별 기본 모델 매핑입니다.
            </p>
            <div class="table-responsive-wrapper">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>작물 (Crop)</th>
                    <th>기본 적용 모델</th>
                    <th>임계값 출처</th>
                    <th>자동선택 플래그</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>딸기 (Strawberry)</strong></td>
                    <td><span class="badge badge-success">ModernTCN</span></td>
                    <td>Validation Injected Scores</td>
                    <td><code>automatic_best_model_selection: false</code></td>
                  </tr>
                  <tr>
                    <td><strong>멜론 (Melon)</strong></td>
                    <td><span class="badge badge-success">ModernTCN</span></td>
                    <td>Validation Injected Scores</td>
                    <td><code>automatic_best_model_selection: false</code></td>
                  </tr>
                  <tr>
                    <td><strong>오이 (Cucumber)</strong></td>
                    <td><span class="badge badge-info">PatchTST</span></td>
                    <td>Validation Injected Scores</td>
                    <td><code>automatic_best_model_selection: false</code></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #64748b; line-height: 1.5;">
              * 출처: <code>reference/offline/scripts/03_control_quality.py</code> Line 40~44.<br>
              * 모델 산출물 내 <code>quality_model_application.json</code>은 후보군 적용 산출물이며 최종 인간 심사위원의 선정 완료 증빙이 아닙니다.
            </div>
          </div>

          <!-- Card B: Actual Deployment Audit Record -->
          <div class="content-card" style="border-top: 4px solid #dc2626;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
              <h3 class="card-title" style="margin: 0; color: #b91c1c;">
                <span>⚠️</span> 2. 실제 적용 결과 및 감사 이력 (Audit Status)
              </h3>
              <span class="badge badge-danger">자료 없음</span>
            </div>
            <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1rem;">
              연구 데이터 무결성 원칙에 따른 공식 선정 사유서 및 현장 적용 완료 이력 확인 상태입니다.
            </p>
            
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem;">
              <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 700; color: #991b1b; font-size: 0.95rem;">
                <span>🛑</span> 공식 선정 사유서 및 최종 현장 배포 이력 부재
              </div>
              <p style="font-size: 0.84rem; color: #7f1d1d; line-height: 1.55; margin: 0;">
                현재 제공된 공식 실험 전달 묶음(<code>quality_panel_bundle</code>)에는 별도의 <strong>심사자 서명, 선정 평가 회의록, 또는 실제 현장 온실 온디바이스 배포 완료를 증빙하는 감사 로그(Audit Log)</strong>가 포함되어 있지 않습니다.
                따라서 본 대시보드는 이를 '최종 배포 완료'로 왜곡하지 않고, <strong>'코드 베이스라인 기본값 설정 상태'</strong>로 객관적으로 표기합니다.
              </p>
            </div>
          </div>
        </div>

        <!-- 3. Missing Data Items (Before/After Timeseries & QC Correction Counts) -->
        <div class="content-card">
          <h3 class="card-title">
            <span>📋</span> 미보유 데이터 항목 및 엄격한 데이터 왜곡 방지 원칙
          </h3>
          
          <div class="setup-grid" style="margin-top: 1rem;">
            <!-- Box 1 -->
            <div class="setup-box" style="border-left: 4px solid #94a3b8;">
              <div class="setup-box-title" style="color: #475569;">
                <span>📉</span> 1. 보정 전후 시계열 (Before / After Timeseries)
              </div>
              <div class="setup-box-content">
                <div style="font-size: 1.1rem; font-weight: 800; color: #64748b; margin: 0.4rem 0;">
                  [ 자료 없음 ]
                </div>
                품질 관리 파이프라인 적용 전/후의 원시 시계열과 보정 시계열은 묶음에 포함되지 않았습니다. 
                임의의 합성 그래프나 목업 시계열을 생성하지 않고 정직하게 '자료 없음'으로 유지합니다.
              </div>
            </div>

            <!-- Box 2 -->
            <div class="setup-box" style="border-left: 4px solid #94a3b8;">
              <div class="setup-box-title" style="color: #475569;">
                <span>🔢</span> 2. 실제 QC 보정 건수 (QC Correction Counts)
              </div>
              <div class="setup-box-content">
                <div style="font-size: 1.1rem; font-weight: 800; color: #64748b; margin: 0.4rem 0;">
                  [ 자료 없음 ]
                </div>
                실제 오프라인 파이프라인 구동 시 발생한 결측/이상치 수정 건수 통계 자료는 포함되지 않았습니다.
              </div>
            </div>

            <!-- Box 3 -->
            <div class="setup-box" style="border-left: 4px solid #f59e0b;">
              <div class="setup-box-title" style="color: #d97706;">
                <span>⚠️</span> 3. data/preprocessing 요약의 왜곡 금지
              </div>
              <div class="setup-box-content">
                <div style="font-weight: 700; color: #92400e; margin: 0.4rem 0;">
                  리샘플링/분할 요약 ≠ QC 보정 통계
                </div>
                <code>data/preprocessing/</code> 폴더의 수치(01_resampled, 02_split)는 원천 데이터의 <strong>리샘플링 및 분할 단계 요약</strong>일 뿐입니다. 이를 최종 품질 관리 보정 결과로 표현하는 것은 명백한 데이터 왜곡이므로 금지됩니다.
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  update() {
    this.render();
  }
}
