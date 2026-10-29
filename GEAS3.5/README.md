# GEAS3.5

최신 소스 정리 기준: 2026-10-29.

공유 구현은 `src/geas35/`에 있으며 오프라인 전처리, 품질 모델, 전이 모델,
RL 학습, 온라인 추론과 receive-only 수집 코드가 포함됩니다.
코드 존재가 실운영 검증 완료를 뜻하지는 않습니다. 실행 전 선택된 모델,
데이터 계약, 각 실험의 검증 결과와 설정을 확인하세요.

## 설치

검증에 사용한 환경은 Python 3.11입니다. 프로젝트 폴더에서:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# .env에 로컬 DB 연결 정보를 설정합니다.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

기존 서버 패키지 버전은 비공개 작업공간 백업의 `python311_packages.txt`에 보존합니다.
GPU 드라이버에 맞는 PyTorch 설치 및 CUDA 인식 확인이 필요합니다.

## 데이터 준비와 모델 실험

```bash
python offline_dataset_preparation/scripts/00_extract_raw.py --help
python offline_dataset_preparation/scripts/01_resample.py --help
python offline_dataset_preparation/scripts/02_split.py --help
python offline_dataset_preparation/scripts/03_control_quality.py --help
python offline_dataset_preparation/scripts/04_prepare_rl_dataset.py --help
python experiments/quality_control_model_selection/scripts/run.py --help
python experiments/transition_model_selection/scripts/run.py --help
```

실제 실행은 각 단계 설정과 입력 데이터 준비 후 수행합니다.
동결된 이행·품질·목표 프로파일·PPO 의미 계약 JSON 4개는
실행과 합성 테스트의 입력이므로 예외적으로 Git에 보존합니다.
`offline_dataset_preparation/datasets/`와 `experiments/**/artifacts/`는 Git에서 제외합니다.

| 코드 | 안내 |
|---|---|
| 품질 모델 비교 | [README](experiments/quality_control_model_selection/README.md) |
| 전이 모델 비교 | [README](experiments/transition_model_selection/README.md) |
| Random restart v1.3 | [README](experiments/transition_model_selection_random_restart_v1_3/README.md) |
| RL 학습 | `experiments/rl_policy_training/scripts/` 및 `configs/` |
| RL v2.1 | [README](experiments/rl_policy_training_v2_1/README.md) |
| 온라인 추론 | `src/geas35/realtime/` |
| Receive-only 수집 | `src/geas35/collection/receive_only.py` |

온라인 추론에는 실제 선택 모델과 RL 정책 아티팩트가 필요합니다.
모델이 없을 때 더미 정책으로 대체하지 않습니다. 수집·제어는 해당 설정과
권한 확인 후 별도로 재개해야 하며, 백업 복원 자체로 자동 시작하지 않습니다.

## 검증

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests -q
```

일부 통합 검사는 로컬 데이터·모델·manifest를 요구합니다. Git clone만으로는
해당 검사를 통과할 수 없으므로 [복구 안내](../docs/RESTORE.md)에 따라 복원하세요.
독립 Random restart 실험은 자체 `runtime_src/`와 테스트를 사용합니다.
