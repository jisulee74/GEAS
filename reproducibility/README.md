# GEAS 3.5 재현 안내

기준: 2026-10-29. 원본 데이터를 공개하지 않고도 코드·설정·선정 근거를 확인할 수 있도록 구성했습니다.

## 현재 연구 상태

- 품질 모델: `results/quality_validation_results.csv`와 `quality_test_results.csv`를 구분합니다.
  Validation으로 선택하고 Test는 최종 평가에만 사용합니다. 작물/모델별 실제 best config는 `configs/quality/`에 있습니다.
- 최신 4채널 전이모델: `results/transition_4ch_comparison.csv`와 작물별 selected manifest에 선정 점수,
  검증 gate, 모델·스키마·scaler 해시 및 학습 당시 패키지 버전이 있습니다.
  `historical_v1_4_selection.json`은 이전 3채널 lineage이므로 최신 4채널 선정과 섞지 않습니다.
- 오이 Profile-A PPO: seed 42/43/44의 bounded main 학습은 완료됐고 Test는 열지 않았습니다.
  완료율은 seed별 0.76/0.76/0.72이며 배포 준비 완료를 뜻하지 않습니다.
  final-action 환경 이행은 기술 검증을 통과했지만 정책 적합성은 구조적 검토가 남아 있습니다.
- `configs/profile_a_ppo.json`은 실제 고정 하이퍼파라미터와 예산의 공개용 발췌입니다.
  원본 해시가 검증되는 실행 계약은 비공개 백업의 원래 경로로 복원해야 합니다.

## 환경

최신 전이모델은 **Python 3.12.3 / CPU PyTorch 2.13.0** 환경을 사용했습니다.
`environment/requirements-geas312.txt`는 그 환경의 전체 패키지 기록입니다.
기존 Python 3.11/CUDA 검증 환경은 `environment/requirements-observed.txt`로 별도 보존했습니다.
모든 실험이 같은 환경에서 생성됐다고 가정하지 마세요. selected manifest의 dependencies와 비교하세요.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# CPU torch는 기록된 버전에 맞는 공식 CPU wheel을 먼저 설치합니다.
python -m pip install -r GEAS3.5/requirements.txt
python scripts/reproduce.py smoke
```

위 일반 requirements는 설치 진입점입니다. 과거 수치 재현에는 해당 실험의 기록된 버전과
CPU/GPU 종류까지 맞춰야 합니다. 이번 작업에서는 새 환경 설치나 전체 GPU 재학습을 하지 않았습니다.

## 비공개 입력 복원

```bash
python scripts/reproduce.py verify-backup /backup/GEAS.tar
python scripts/reproduce.py restore /backup/GEAS.tar --destination "$PWD"
```

옆의 `GEAS.tar.sha256` 파일이 필요합니다. 복원 도구는 데이터셋·실험 아티팩트·수집 상태만 복원하며,
기존 파일은 동일한 경우 건너뛰고 다른 경우 중단합니다. `.env`, 인증키, 외부 인터뷰 원문은 자동 공개하지 않습니다.
인터뷰 근거와 민감한 registry/원천 데이터는 백업에서만 복원하고 `.gitignore`를 유지하세요.

실행 결과와 입력을 함께 보존한 복구 clone에서는 기존 output을 다시 만들지 마세요.
**새 실험용 clone/디렉터리**를 별도로 만들고 필요한 입력만 복사하여 실행해야 합니다.
절대 경로가 동결 계약에 포함돼 있으므로, 옮긴 경로를 무조건 치환하면 해시 계약이 깨집니다.
원래 경로 구조 유지 또는 별도 실험 버전·새 계약으로 재등록이 필요합니다.

## 주요 실행 순서

```bash
cd GEAS3.5
export PYTHONPATH="$PWD/src"
python offline_dataset_preparation/scripts/00_extract_raw.py --help
python offline_dataset_preparation/scripts/01_resample.py --help
python offline_dataset_preparation/scripts/02_split.py --help
python experiments/quality_control_model_selection/scripts/run.py --config experiments/quality_control_model_selection/configs/cucumber.yaml
```

위 config 경로는 GEAS3.5 작업 폴더 기준입니다. 실행 전 `--help`와 해당 README를 확인하세요.
학습 후 Validation 기반 품질 모델을 명시적으로 선택하여 QC와 RL 데이터 준비로 넘깁니다.
전이모델·RL 각 단계의 설정 및 원본 입력은 기존 실험 README와 복원된 manifest에 연결돼 있습니다.
최신 전이모델 재선정은 `src/geas35/models/transition/` 및 관련 실험 스크립트의
고정 contract와 입력 registry를 사용하며, 이전 step12 결과를 최신 결과로 대체하지 않습니다.

Profile-A는 새 실험 폴더에서 계약과 prerequisite 입력을 복원한 뒤
`python -m geas35.rl.profile_a_smoke_trainer_v1`,
`python -m geas35.rl.profile_a_main_trainer_v1` 순서로 실행합니다.
두 runner는 출력 폴더가 있으면 거부하며, 공개 발췌 JSON만으로 실행하지 않습니다.
신규 환경으로 정책을 이행하거나 실운영 제어하는 승인을 자동 부여하지 않습니다.

## 검증 범위

clone만으로 합성 QC 학습·전이모델·온라인 경계·4채널 테스트를 실행할 수 있습니다.
실제 농가 데이터로 같은 수치를 다시 얻는 전 과정은 비공개 데이터 복원과 동결 입력 검증이 필요합니다.
확률적 GPU 학습의 bitwise 동일성은 보장하지 않으며 원래 metric tolerance와 seed 규칙을 사용합니다.
전체 연구 suite에는 로컬 실험 결과 검증도 포함되므로 clone만으로 모두 통과하지는 않습니다.
