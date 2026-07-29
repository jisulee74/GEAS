# 코드베이스 재구조화 plan v1.2

대상 파일명: `plan/코드베이스 재구조화 plan v1.2_20260727_131507.md`

## 1. 요약

v1.2의 완료 범위는 quality model 실험과 transition model 실험을 GPU 환경에서 실행할 수 있도록 만드는 최소 재구조화다. v1.1 작성 과정에서 코드베이스 조사와 blocker 확인은 완료된 사전 조사로 처리하고, 실제 구현 Step은 공통 데이터·config·artifact 계약 정리부터 시작한다.

online inference 전체 구조, live adapter, RL policy 학습·선정·추론은 후속 범위다. 단, 실제 policy artifact가 없을 때 dummy, zero, recorded action을 만들지 않는 원칙은 유지한다.

## 2. 현재 구조와 핵심 문제

확인된 사실:
- quality 실험 entrypoint는 `experiments/quality_control_model_selection/scripts/run.py --config <yaml>`이다.
- transition 실험 entrypoint는 `experiments/transition_model_selection/scripts/run.py --config <yaml>`이다.
- `03_control_quality.py --quality-model-artifact <quality_model_application.json>`는 선택된 quality artifact를 offline QC에 적용한다.
- `04_prepare_rl_dataset.py`는 `03_quality_controlled`에서 `5_rl_dataset`을 생성한다.
- transition runner는 RL-ready frame을 기대하지만 현재 기본 config는 `02_split`을 참조한다.
- `5_rl_dataset`은 아직 생성되어 있지 않다.
- 현재 Git에 포함된 자동 테스트 fixture와 smoke test가 없다.

핵심 문제:
- transition 실험 입력 단계가 실제 runner 계약과 맞지 않는다.
- quality 수동 선정 절차가 문서와 완료 기준에 충분히 고정되어 있지 않다.
- GPU 실행 전 로컬에서 결정론적으로 확인할 smoke test가 없다.
- config load와 `--help`는 데이터가 없어도 실패하지 않아야 하지만, 실제 실행은 필수 split/column 누락 시 명확히 fail-fast 해야 한다.

## 3. 목표 구조와 데이터 흐름

데이터 준비 흐름:
`02_split` -> quality experiment -> 사람이 선택한 `quality_model_application.json` -> `03_control_quality.py --quality-model-artifact` -> `03_quality_controlled` -> `04_prepare_rl_dataset.py` -> `5_rl_dataset`.

Quality 실험 흐름:
- train: HPO candidate 학습
- validation: best config, threshold calibration, 수동 선정 근거 metric 확인
- test: 선택 판단에 사용하지 않고 최종 평가 결과로만 기록
- output: model별 HPO, threshold, validation/test evaluation, benchmark, no-auto-selection comparison, `quality_model_application.json`, `quality_model.pkl`

Transition 실험 흐름:
- train: 후보 transition model 학습
- validation: 후보 비교 및 최종 selected model 결정
- test: validation에서 선정된 model을 한 번만 최종 평가
- output: candidate artifacts, `selected_transition_model.json`, selected model test metric artifact

## 4. 단계별 구현 계획

### Step 1. 공통 데이터·config·artifact 계약 정리

- 변경 목적: GPU와 로컬에서 동일 entrypoint/config schema로 실행 가능하게 한다.
- 실제 변경 대상 파일: experiment README, default/smoke config, 필요 시 config validation helper.
- 재사용할 기존 구현: quality/transition config loaders, `write_json()`, artifact helpers.
- 필요한 최소 변경: dataset path와 output dir는 config로 지정한다. config load와 `--help`는 입력 parquet가 없어도 실패하지 않게 유지한다.
- 테스트 방법과 완료 기준: quality/transition experiment와 `03_control_quality.py`, `04_prepare_rl_dataset.py`의 CLI `--help`가 baseline으로 통과한다. config loader는 path를 resolve하지만 실제 파일 존재 검증은 실행 단계에서 수행한다.
- 이번 Step에서 하지 않을 작업: config schema 전면 변경, 환경별 절대경로 hard-code, 실제 실험 실행.

### Step 2. Quality model 실험 경로와 수동 선정 절차 정리

- 변경 목적: no-auto-selection 원칙을 유지하면서 사람이 선택한 artifact를 offline QC에 적용하는 절차를 고정한다.
- 실제 변경 대상 파일: quality experiment README, smoke config/test, 필요 시 integrity check 보강.
- 재사용할 기존 구현: `run_quality_hpo()`, `run_quality_threshold_calibration()`, `run_quality_validation_test_evaluation()`, `run_quality_report_generation()`, `save_quality_model_application()`, `load_quality_model_application()`.
- 필요한 최소 변경: README에 validation 기준 수동 선정 절차를 명시한다. 선정 근거는 기존 `model_comparison.csv/json`, validation reconstruction metric, validation anomaly metric, online benchmark metric을 사용한다. test 결과는 참고용 최종 평가로만 기록하고 선택 근거로 사용하지 않는다.
- 테스트 방법과 완료 기준: synthetic fixture 기반 smoke에서 HPO budget 1, 모델 1개, figures off로 학습·threshold·evaluation·application artifact 저장이 통과한다. README에는 선택된 artifact 경로 지정 방법, 선정 근거 metadata 기록 방식, `03_control_quality.py --quality-model-artifact <path>` 적용 명령이 포함된다.
- 이번 Step에서 하지 않을 작업: 자동 선정 알고리즘, 새 모델 추가, 성능 개선, test 기반 selection.

### Step 3. Transition model 실험 경로 정리

- 변경 목적: transition 실험이 RL-ready dataset을 입력으로 받아 train/validation/test 계약을 지키며 artifact를 저장하게 한다.
- 실제 변경 대상 파일: transition default/smoke config, README, transition runner/evaluator의 최소 test metric artifact 저장 로직, 관련 test.
- 재사용할 기존 구현: `04_prepare_rl_dataset.py`, `prepare_rl_dataset_splits()`, `build_transition_dataset_from_rl_frame()`, `train_transition_candidates()`, `save_selected_transition_model()`, `load_selected_transition_model()`, transition evaluator.
- 필요한 최소 변경: default config 입력을 `5_rl_dataset/{crop}/{split}.parquet`로 정정한다. 실제 실행 시 train/validation/test split 파일과 `rl_valid_transition`, `next_*`, action/observation 필수 column을 fail-fast 검증한다. 오류 메시지에는 `03_control_quality.py`와 `04_prepare_rl_dataset.py` 선행 실행 명령을 포함한다.
- 테스트 방법과 완료 기준: synthetic RL-ready fixture로 후보 학습, validation selection, `selected_transition_model.json` 저장, selected model 재로딩, selected model test metric artifact 저장이 모두 통과한다. test metric은 재선정에 사용되지 않는다.
- 이번 Step에서 하지 않을 작업: transition 알고리즘 추가, reward 개선, policy 학습, test 기반 재선정.

### Step 4. 로컬 smoke test와 GPU 실행 전 검증

- 변경 목적: 대규모 GPU 실험 없이 실행 준비 상태를 확인한다.
- 실제 변경 대상 파일: Git에 포함 가능한 synthetic fixture 생성 helper 또는 pytest fixture, smoke configs, pytest 파일.
- 재사용할 기존 구현: experiment CLIs, config loaders, artifact loaders, pandas parquet I/O.
- 필요한 최소 변경: 자동 smoke test는 결정론적 synthetic fixture만 사용한다. 로컬 실제 데이터나 생성된 대용량 artifact는 Git에 포함하지 않는다. sliced parquet 검증은 사용자가 경로를 지정할 때만 실행하는 선택적 검증으로 분리한다.
- 테스트 방법과 완료 기준: quality smoke와 transition smoke가 임시 디렉터리에 artifact를 만들고 종료 후 repo-tracked artifact를 남기지 않는다.
- 이번 Step에서 하지 않을 작업: 실제 dataset slice를 Git에 추가, GPU 학습 실행, 대용량 artifact 생성.

### Step 5. 실험 관련 회귀 테스트 및 문서 정리

- 변경 목적: commit/push 가능한 최소 안정성 기준을 만든다.
- 실제 변경 대상 파일: experiment README, offline README, pytest 파일.
- 재사용할 기존 구현: 기존 CLI, config loader, artifact loader, preprocessing/RL dataset builder.
- 필요한 최소 변경: 실행 순서, 입력 dataset 단계, output artifact 위치, GPU 환경에서 바꿀 config field, 수동 quality 선정 절차, transition selected test metric 위치를 문서화한다.
- 테스트 방법과 완료 기준: `python -m compileall GEAS3.5/src/geas35`, CLI `--help`, synthetic smoke tests가 통과한다.
- 이번 Step에서 하지 않을 작업: unrelated 기술부채 정리, 숫자 preprocessing 파일명 변경, `selection_report.py` 이동.

### Step 6. 이후 작업으로 미룰 online inference·RL 연결 경계 기록

- 변경 목적: GPU 실험 준비 범위 밖의 online/RL 항목을 후속 작업으로 분리한다.
- 실제 변경 대상 파일: v1.2 plan 또는 README의 후속 작업 섹션.
- 재사용할 기존 구현: `PolicyActionProvider`, `load_selected_transition_model()`, `sanitize_for_controller()`.
- 필요한 최소 변경: policy artifact가 없으면 action 미생성/unsupported 처리 원칙만 기록한다.
- 테스트 방법과 완료 기준: 문서 기준 완료.
- 이번 Step에서 하지 않을 작업: `online_inference/` 생성, live adapter, replay runner, policy inference 구현.

## 5. 예정 파일 변경표

| 파일 | 변경 유형 | 목적 |
|---|---|---|
| `plan/코드베이스 재구조화 plan v1.2_*.md` | 생성 | v1.2 계획 저장 |
| `experiments/quality_control_model_selection/README.md` | 수정 | GPU 실행, 수동 선정, artifact 적용 절차 명확화 |
| `experiments/quality_control_model_selection/configs/smoke.yaml` | 생성 | synthetic smoke용 최소 quality config |
| `experiments/transition_model_selection/README.md` | 수정 | RL-ready 선행 단계, selected test metric 명확화 |
| `experiments/transition_model_selection/configs/default.yaml` | 수정 | `5_rl_dataset` 입력으로 정정 |
| `experiments/transition_model_selection/configs/smoke.yaml` | 생성 | synthetic smoke용 최소 transition config |
| `src/geas35/experiments/transition/runner.py` | 최소 수정 | selected model test metric artifact 저장 |
| `src/geas35/experiments/transition/config.py` | 선택적 수정 | 실행 단계 fail-fast 메시지 개선 |
| `offline_dataset_preparation/README.md` | 수정 | quality artifact 적용 후 RL dataset 생성 순서 |
| `GEAS3.5/tests/test_quality_experiment_smoke.py` | 생성 | quality synthetic smoke |
| `GEAS3.5/tests/test_transition_experiment_smoke.py` | 생성 | transition synthetic smoke |
| `GEAS3.5/tests/test_experiment_config_contracts.py` | 생성 | config/path/CLI 계약 검증 |

## 6. v1.1 Step 대응표

| v1.1 Step | v1.2 처리 |
|---|---|
| Step 1 조사와 blocker 확인 | 사전 조사 완료로 처리, baseline `--help` 검증만 Step 1에 포함 |
| Step 2 공통 계약 정리 | v1.2 Step 1로 승격 |
| Step 3 quality 실험 경로 | v1.2 Step 2로 유지, 수동 선정 절차와 `03_control_quality.py` 적용 명령 강화 |
| Step 4 transition 실험 경로 | v1.2 Step 3으로 유지, selected model test metric artifact 필수화 |
| Step 5 smoke/GPU 전 검증 | v1.2 Step 4로 유지, synthetic fixture만 자동화 |
| Step 6 회귀 테스트/문서 | v1.2 Step 5로 유지 |
| Step 7 online/RL 경계 | v1.2 Step 6으로 유지, 후속 범위 명시 |

## 7. 하위 호환성 전략

- 기존 entrypoint 경로와 CLI 인자는 유지한다.
- 기존 config schema key는 유지한다.
- 기존 quality artifact 파일명과 field는 유지한다.
- 기존 transition artifact layout과 `selected_transition_model.json`은 유지한다.
- `03_control_quality.py --quality-model-artifact` 계약은 유지한다.
- config load와 `--help`는 데이터 파일이 없어도 실패하지 않게 유지한다.
- 실제 실행 단계에서만 입력 split과 필수 column을 fail-fast 검증한다.

## 8. 위험 요소 및 미결정 사항

- `5_rl_dataset`은 아직 생성되어 있지 않으므로 transition 실험 전 선행 실행이 필요하다.
- GPU dependency 차이는 synthetic smoke만으로 완전히 검증할 수 없다.
- quality model 최종 선택은 자동화하지 않는다. 사람이 validation 중심 report를 보고 artifact 경로와 선정 metadata를 기록해야 한다.
- selected transition model의 test metric 저장은 필수 구현 대상이지만, test metric은 재선정에 사용하지 않는다.
- online inference와 RL policy 연결은 이번 완료 범위가 아니다.

## 9. 전체 완료 기준

- quality experiment와 transition experiment의 실행 명령 및 config가 명확하다.
- 두 실험 모두 train/validation/test 분리를 보존한다.
- dataset path와 output artifact path를 GPU 환경에서 config로 지정할 수 있다.
- quality 실험은 no-auto-selection을 유지하고, validation 중심 수동 선정 절차와 `03_control_quality.py --quality-model-artifact` 적용 명령이 문서화된다.
- transition 실험은 train 후보 학습, validation 선정, selected model test 최종 평가를 수행하며 test metric artifact를 저장한다.
- config load와 CLI `--help`는 입력 dataset 미존재 상태에서도 불필요하게 실패하지 않는다.
- 실제 실행은 필수 split/column 누락 시 선행 단계와 실행 명령을 포함한 오류로 fail-fast 한다.
- 자동 smoke test는 Git에 포함 가능한 결정론적 synthetic fixture만 사용한다.
- 실제 대규모 GPU 실험 없이도 config load, CLI import, artifact contract, synthetic smoke run으로 준비 상태를 확인할 수 있다.
- 실제 dataset 또는 생성된 대용량 artifact는 Git에 포함하지 않는다.
- 재구조화 완료 범위는 quality·transition GPU 실험 실행 준비에 한정된다.
- online inference 전체 구조와 RL policy 연결은 후속 범위로 분리된다.

구현 승인 대기 중
