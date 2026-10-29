# GPU 컨테이너 재생성 후 복구

정리 기준: 2026-10-29. 안내된 작업: 2026-10-30 17:00–18:00 한국시간.

## 백업 위치와 영속성

`/NHNHOME/WORKSPACE/` 아래 기존 작업공간과 `maintenance_backups/`는 컨테이너 루트와
분리된 Lustre 네트워크 볼륨에 있습니다. 관리자가 안내한 보존 경로와 실제 mount를 확인했습니다.
`maintenance_backups/LATEST` 파일에 이번 백업 폴더명이 있습니다.
같은 스토리지의 별도 복사본이므로 독립된 외부 재해복구 백업은 아닙니다.

상세 목록과 경고는 비공개 백업의 `BACKUP_SUMMARY.md`를 확인하세요.
현재 저장소에 필요한 아카이브의 크기·SHA-256은 `reproducibility/backup_manifest.json`에 있습니다.

- `GEAS.tar`: 정리 전 GEAS 소스·데이터·가중치·실험 결과·로컬 설정.
- `AXData_focused.tar`: AX 원천/준비 데이터·핵심 결과·전체 최신 upper_truss 캠페인.
  과거 캠페인의 trial별 중간 파일은 원래 작업공간에 보존하고 집중 백업에서는 제외했습니다.
- `AXData_live_delta.tar`: 본 백업 도중 생성·변경된 중요 파일 12개.
- `deployment_worktrees.tar`: 정리 전 배포용 작업 폴더와 추가 대시보드 소스.
- `*.bundle`, `*_uncommitted.patch`: Git 이력과 원래 미커밋 변경.
- `private_runtime/`: 환경 파일. 외부에 업로드하지 마세요.
- `sqlite_consistent/`, `sqlite_consistent_manifest.json`: SQLite backup API로 생성하고 integrity_check를 통과한 평가 상태.
- `*_verification.json`: SHA-256, 구성 파일과 경고 확인 기록.

AX 본 백업의 tar 종료 코드는 2입니다. 실행 중 완료된 작업의 임시 `running.json` 두 개가
복사 직전에 사라진 것으로, 파일별 재검증에서 필수 누락은 없었습니다.
`AXData_focused_verification.json`에 허용한 정확한 두 경로가 기록돼 있습니다.

## 새 Clone에 복원

기존 폴더를 지우거나 덮어쓰지 말고 빈 경로에 GitHub 저장소를 Clone합니다.
표준 라이브러리만 사용하는 복원 도구로 필요한 데이터 경로만 복원하세요.

GEAS 저장소 루트:

```bash
python scripts/reproduce.py restore /backup/GEAS.tar --destination "$PWD"
```

SmartFarm-Crop-Prediction 저장소 루트:

```bash
python scripts/reproduce.py restore /backup/AXData_focused.tar \
  --overlay /backup/AXData_live_delta.tar --destination "$PWD"
```

각 아카이브 옆의 `.sha256` 파일이 필요합니다. 전체 해시를 확인한 후 복원하며,
동일한 기존 파일은 건너뛰고 다른 파일이 있으면 중단합니다. 증분을 먼저 별도로 풀지 마세요.
AX의 prepared 데이터 연결 두 개는 Clone 내부의 상대 링크로 재생성됩니다.
임의의 링크, 경로 이탈, `.git`, 실행 중 PID/lock/running 표식은 복원하지 않습니다.

`.env`와 서비스 접속키는 자동으로 복원하지 않습니다. 권한 있는 담당자가 비공개 설정에서
필요한 항목만 별도로 적용하고 `git status`/ignore 상태를 확인하세요.
SQLite 일관 백업을 사용할 때에는 `sqlite_consistent_manifest.json`의 원래 상대 경로를 확인하고
서비스를 시작하기 전에 새 복구 폴더에 적용하세요. 기존 DB를 덮어쓰지 마세요.

## 환경과 실행

GEAS 최신 모델은 기록된 Python 3.12/CPU 환경을, AX는 Python 3.11 환경을 기준으로 합니다.
저장소별 `reproducibility/README.md`와 환경 목록, 해당 모델 manifest의 dependencies를 따르세요.
환경 설치 후 `python scripts/reproduce.py smoke`로 핵심 합성 테스트를 확인합니다.
GPU 작업을 재개하기 전에는 `nvidia-smi`와 PyTorch CUDA 인식을 확인합니다.

복구한 기존 결과의 재개와 처음부터 재학습하는 작업은 별도 폴더를 사용하세요.
새 실험에는 입력만 복원하고 기존 결과·분할 기준은 비교용으로 보존합니다.
GEAS 동결 계약의 절대 경로를 무조건 치환하면 해시 계약이 깨질 수 있습니다.
AX `supervise`는 자동 게시도 포함하므로 단순 재현에는 사용하지 않습니다.

## 실행 중 상태와 남은 범위

AX 학습과 GEAS receive-only 수집을 중단하지 않고 백업했습니다.
전체 폴더가 한 시점의 원자적 스냅샷은 아니며, 백업 이후 변경분은 포함되지 않습니다.
점검 직전 담당자가 실행을 정상 종료한 뒤 마지막 체크포인트·수집 상태를 보존해야 합니다.
이번 Clone 검증은 핵심 테스트, GEAS 계약 사전검증, AX 분할 해시와 base 전처리 일치까지입니다.
새 패키지 설치, 전체 센서 집계 재계산, 전체 실데이터 GPU 재학습은 수행하지 않았습니다.
