# RL HPO Budget Calibration v2.1

## 목적

이 폴더는 기존 strict v2 결과를 삭제하거나 덮어쓰지 않고, 논문 작성에 필요한 재현성을
유지하면서 계산 종료 가능성을 보장하기 위한 **pre-Test protocol amendment**이다.

- Train에서만 PPO를 업데이트하고 고정 Validation episode에서 calibration한다.
- Test는 계속 잠근다.
- 대표 설정 5개와 seed `42, 43, 44`는 변경하지 않는다.
- strict v2의 완료 learning curve와 checkpoint를 복사해 이어서 학습한다.
- v2.1 DB와 산출물은 이 폴더 아래에 별도로 저장한다.

이 변경은 strict v2 Validation learning curve를 확인한 뒤 계산 종료 문제를 해결하기 위해
이루어졌다. 따라서 manifest에는 `prior_validation_curves_seen=true`,
`test_results_seen=false`를 기록한다. v2.1을 strict v2의 사전등록 결과처럼 표현하면 안 된다.

## strict v2와의 차이

| 항목 | strict v2 | v2.1 |
|---|---:|---:|
| Kendall rank correlation | `tau >= 0.8` | `tau >= 0.6` |
| 상위 설정 조건 | 상위 2개 완전 동일 | 상위 2개 중 최소 1개 유지 |
| 연속 안정 구간 | 3개 | 2개 |
| reward 변화 | 모든 설정의 최대 변화 <= IQR 5% | 설정별 변화 중앙값 <= robust scale 15% |
| slope CI·finite diagnostics·safety gate | 필수 | 유지 |
| 최대 budget | 없음 | 작물별 524,288 steps |
| 최대 budget 미수렴 | 계속 두 배 증가 | 마지막 두 관측 budget을 HPO rung으로 사용 |

`robust scale`은 현재 reward IQR과 `|median reward| x 5%` 중 큰 값이다.

## 실행 전 필수 절차

strict v2가 checkpoint를 쓰는 중에는 복사하면 안 된다. 딸기·오이·멜론의 `--run-crop`
터미널과 `--add-workers` 터미널에서 각각 `Ctrl+C`를 눌러 총 6개 실행을 먼저 중단한다.

```bash
cd /NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/GEAS/GEAS3.5
ps -ef | grep run_step19_5_calibration_v2.py | grep -v grep
```

출력이 없어야 한다. 그다음 checkpoint와 완료 learning curve를 승계한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
../../envs/jslee_py311/bin/python3.11 \
experiments/rl_policy_training_v2_1/scripts/run_step19_5_calibration_v2_1.py --prepare
```

strict coordinator PID가 살아 있으면 `--prepare`는 fail-closed로 중단된다.

## 작물별 실행 명령

각 명령은 별도 터미널에서 실행한다. v2.1 runner는 작물당 15개 job을 한 coordinator가 모두 기다리도록 `--workers 15`를 사용한다.

### 딸기

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
../../envs/jslee_py311/bin/python3.11 \
experiments/rl_policy_training_v2_1/scripts/run_step19_5_calibration_v2_1.py \
  --run-crop --crop strawberry --device cuda --transition-n-jobs 1 --workers 15
```

### 오이

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
../../envs/jslee_py311/bin/python3.11 \
experiments/rl_policy_training_v2_1/scripts/run_step19_5_calibration_v2_1.py \
  --run-crop --crop cucumber --device cuda --transition-n-jobs 1 --workers 15
```

### 멜론

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
../../envs/jslee_py311/bin/python3.11 \
experiments/rl_policy_training_v2_1/scripts/run_step19_5_calibration_v2_1.py \
  --run-crop --crop melon --device cuda --transition-n-jobs 1 --workers 15
```

## 최종화

세 작물의 상태가 모두 `passed`가 된 후 한 번 실행한다.

```bash
../../envs/jslee_py311/bin/python3.11 \
experiments/rl_policy_training_v2_1/scripts/run_step19_5_calibration_v2_1.py --finalize
```

최종 budget은 다음 경로에 생성된다.

```text
experiments/rl_policy_training_v2_1/artifacts/step19_5_v2_1/calibrated_budget.json
```

기존 Step 20 v2는 strict v2 budget 경로를 기대하므로, v2.1 budget schema와 경로를
사용하도록 별도 전환한 뒤 실행해야 한다.
