# GEAS AI Quality Model Experiment Plan_v1.1

## Summary
ModernTCN, TimesNet, PatchTST를 동일 조건에서 비교하기 위한 실험 계획이다.  
Experiment는 최종 모델을 자동 선택하지 않고, 모델별 HPO 결과, threshold calibration 결과, reconstruction 성능, anomaly detection 성능, online deployment 성능을 계산하고 저장하는 데서 끝난다. 최종 의사결정은 사람이 논문 목적과 운영 목적에 따라 수행한다.

## Experiment Flow
1. **Dataset / Preprocessing 확인**
   - 기존 v1.4 preprocessing pipeline 결과를 사용한다.
   - Observation columns는 AI model input, reconstruction, anomaly detection, threshold calibration, evaluation 대상이다.
   - Action columns는 AI Quality Model 대상에서 제외한다.
   - 단, Action columns의 missing detection, `{action}_missing_flag`, Action Log 기반 restore, `{action}_restored_flag`는 v1.4 preprocessing 정책을 그대로 유지한다.
   - Action restore 결과는 데이터 품질 분석용 metadata로 보존하되, AI model metric 계산에는 포함하지 않는다.

2. **Train / Validation / Test 사용 원칙**
   - Train: model fitting 및 HPO candidate 학습에만 사용.
   - Validation: HPO best configuration 선택 및 threshold calibration에 사용.
   - Test: 확정된 best configuration + calibrated threshold를 적용한 최종 평가에만 사용.
   - Test 결과는 model comparison report에 포함하지만, HPO 또는 threshold calibration에는 절대 사용하지 않는다.

3. **Model별 HPO**
   - ModernTCN, TimesNet, PatchTST 각각에 대해 동일한 HPO budget과 seed policy를 적용한다.
   - HPO는 reconstruction 중심 validation metric으로 best configuration을 선택한다.
   - Threshold는 HPO search space에 포함하지 않는다.

4. **Threshold Calibration**
   - HPO로 선택된 best hyperparameter configuration의 학습 완료 모델에 대해 Validation에서 별도 수행한다.
   - Threshold는 hyperparameter가 아니라, 학습된 model의 anomaly score를 outlier decision으로 변환하기 위한 calibration parameter이다.
   - `ThresholdOptimizer`는 best model configuration이 고정된 뒤에만 실행한다.

5. **Final Experiment Report**
   - 각 모델의 Validation/Test 결과와 online benchmark 결과를 저장한다.
   - 자동 ranking이나 자동 selected model field는 생성하지 않는다.
   - Offline Batch Refinement용 후보와 Online Inference용 후보는 동일할 수도 있고 다를 수도 있음을 report에 명시한다.

## Hyperparameter Optimization
Common hyperparameters:
- Fixed: `lookback=288`, `epochs=100` (`max_epochs`)
- Search: `mask_fraction ∈ {0.30,0.40,0.50}`, `batch_size ∈ {32,64,128}`
- Search: `learning_rate ~ LogUniform(1e-4,1e-2)`, `weight_decay ~ LogUniform(1e-6,1e-2)`, `dropout ~ Uniform(0.0,0.3)`
- `random_state`, `expected_frequency`
- `early_stopping.patience=10`, `early_stopping.min_delta`

Model-specific hyperparameters:
- ModernTCN: `channel_width ∈ {32,64,128}`, `depth ∈ {[1,1,1],[2,2,2]}`, `kernel_size ∈ {13,31,51}`
- TimesNet: `temporal_blocks ∈ {1,2,3}`, `top_k_periods ∈ {2,3,5}`, `period_embedding_dim ∈ {32,64,128}`, derived `d_ff=2×d_model`
- PatchTST: `patch_length ∈ {8,16,24,32}`, derived `patch_stride=patch_length/2`, `transformer_depth ∈ {2,3,4}`, `attention_heads ∈ {4,8,16}`, `embedding_dim ∈ {32,64,128,256}`, derived `d_ff=2×d_model`

Search policy:
- 기본 search strategy는 Random Search.
- 모델별 동일 candidate budget 50을 사용한다.
- Seed는 동일 목록을 사용하고, 결과는 seed별 raw result와 mean/std로 저장한다.
- Early stopping은 `validation_reconstruction_loss` 기준으로 적용한다.
- HPO objective는 `validation_synthetic_masking_rmse`를 primary로 사용하고, `validation_synthetic_masking_mae`, `validation_anomaly_pr_auc`를 보조 지표로 기록한다.
- Precision/Recall/F1은 threshold calibration 이후 산출되는 지표이므로 HPO primary objective로 사용하지 않는다.

Artifact:
- `{run_id}/config.json`
- `{run_id}/{model}/hpo_results.json`
- `{run_id}/{model}/best_config.json`
- `{run_id}/{model}/training_history.csv`

## Threshold Calibration
Calibration 절차:
- 고정 threshold 후보 리스트를 사용하지 않는다.
- 합성 이상치를 주입한 Validation split에서 정상 셀과 주입 셀을 모두 포함한 reconstruction-error score의 유한 최솟값과 최댓값을 변수별로 계산한다.
- 변수별 최솟값부터 최댓값까지 정확히 100개 후보를 균등 간격으로 생성한다. 원본(clean) Validation score 범위도 진단용으로 함께 저장한다.
- 각 모델의 best hyperparameter configuration을 고정한다.
- Train으로 해당 configuration의 model을 학습한다.
- Validation anomaly score를 계산한다.
- `ThresholdOptimizer`로 모든 threshold 후보의 Validation synthetic-anomaly F1을 평가한다.
- Validation F1이 최대인 threshold를 선택하며, F1 동률이면 Precision이 높은 후보, 이후 더 높은 threshold 순으로 선택한다.
- 선택된 threshold와 calibration metrics를 저장한다.

Calibration artifact:
- `{run_id}/{model}/threshold_calibration.json`
- 포함 항목:
  - candidate thresholds
  - calibrated threshold
  - objective metric
  - Precision, Recall, F1-score
  - ROC-AUC, PR-AUC
  - Confusion Matrix
  - synthetic anomaly injection 설정
  - HPO와 별도 calibration 단계임을 나타내는 metadata

## Evaluation
Reconstruction Performance:
- Synthetic masking 기반 평가.
- Metrics:
  - MAE
  - RMSE
  - Per-column MAE/RMSE
  - masked cell count

Anomaly Detection Performance:
- Synthetic anomaly injection 기반 평가.
- Metrics:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC
  - Confusion Matrix
  - Per-column detection metrics

Online Deployment Performance:
- 동일 hardware, 동일 batch/repeat 조건에서 측정.
- Metrics:
  - inference latency
  - peak memory usage
  - model size
- Offline Batch Refinement에서는 reconstruction accuracy와 detection performance를 우선 검토할 수 있다.
- Online Inference에서는 latency, memory usage, model size를 반드시 함께 검토한다.
- Experiment는 두 목적에 필요한 metric을 모두 저장하며, 어떤 모델을 쓸지는 사람이 결정한다.

## Report Outputs
Experiment 산출물:
- Model Comparison CSV
- JSON Artifact
- Markdown Summary
- Reconstruction Metric Table
- Anomaly Detection Metric Table
- Online Benchmark Table
- Visualization Figure

Visualization:
- Reconstruction MAE/RMSE bar plot
- Anomaly F1/PR-AUC bar plot
- latency / memory / model size tradeoff plot
- per-column reconstruction error heatmap

Report 원칙:
- 자동 selected model을 쓰지 않는다.
- Validation과 Test 결과를 분리해서 표시한다.
- Offline Batch Refinement 관점과 Online Inference 관점의 해석 section을 분리한다.
- 최종 모델 결정은 report를 읽는 연구자/운영자가 수행한다.

## Required Implementation Work
기존 v1.4 구현은 변경하지 않고 실험용 코드만 추가한다.

추가 작업:
- Experiment config schema
- Experiment runner
- HPO candidate generator
- threshold calibration runner
- validation/test evaluation runner
- online benchmark runner
- artifact writer
- model comparison CSV generator
- Markdown summary generator
- visualization generator
- artifact integrity checker

주의:
- 기존 `BaseQualityModel`, preprocessing pipeline, evaluation framework, public interface는 변경하지 않는다.
- 기존 selection report utility 중 자동 selected model 의미가 들어가는 출력은 experiment의 최종 report로 사용하지 않는다.
- 필요하면 experiment 전용 no-selection comparison report schema를 새로 추가한다.

## Implementation Priority
1. Experiment config + artifact directory layout
2. Model별 HPO runner
3. Threshold calibration runner
4. Validation/Test evaluation runner
5. Online benchmark runner
6. CSV / JSON / Markdown report generator
7. Visualization generator
8. Artifact integrity checker
9. End-to-end experiment CLI

## Assumptions
- GEAS AI Quality Model v1.4 Step 1~8 구현은 완료된 상태로 둔다.
- Experiment code는 기존 기능을 변경하지 않고 신규 실험 계층에만 추가한다.
- Action columns는 AI model evaluation 대상이 아니지만, v1.4 preprocessing의 missing flag 및 Action Log restore 정책은 유지한다.
- Experiment는 평가 결과 생성까지만 수행하며, 최종 의사결정은 사람이 수행한다.
