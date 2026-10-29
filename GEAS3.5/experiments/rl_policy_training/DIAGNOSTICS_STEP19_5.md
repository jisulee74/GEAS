# Step 19.5 원인 진단 실행 안내

Transition Model 확정(Step 14) → RL 환경·PPO 구현 검증(Step 15~19) → 학습량 calibration(Step 19.5) → HPO(Step 20) → 최종 학습·비교·Test·인계 순서다. 현재 세 작물은 Step 19.5 통과 조건을 충족하지 못했다. 딸기 15건의 1,048,576 step 학습 완료는 calibration 통과와 다르다.

이번 진단은 저장 정책 재평가와 기록 상태/기록 행동 비교로 원인을 구분하고 개선안을 작성한다. 정책 학습, 운영 환경 수정, 안전성 임계값 조정, HPO 및 Test 접근은 수행하지 않는다.

## 실행

프로젝트 루트 `GEAS/GEAS3.5`에서 프로젝트 Python 환경을 사용한다.

```bash
../../envs/jslee_py311/bin/python -u experiments/rl_policy_training/scripts/run_step19_5_diagnostics.py --stage all --workers 3 --device cpu
```

`all`은 입력 목록 동결 → 오이 평가기 비교(1 worker) → 멜론·딸기 평가기 비교 → 기록 행동/정책 rollout 진단 → 딸기 학습 분석 → 종합 보고서를 순서대로 실행한다. 개별 단계는 `--stage inventory|evaluator-parity|rollout-diagnosis|learning-analysis|report`, 작물은 `--crop strawberry|melon|cucumber|all`로 선택한다.

Validation 전체 에피소드를 사용하며, Train은 동결된 에피소드 순서에서 최대 24개를 균등 간격으로 선택한다. `--workers`는 1~3이다. 동일 출력 디렉터리에서 device를 변경하면 입력 fingerprint가 달라져 결과 재사용을 거부한다.

## 중간 저장·재개

결과는 `artifacts/step19_5_diagnostics_v1`에 저장한다. 단계별·작업별 잠금으로 중복 실행을 막는다. 에피소드별 gzip trace를 먼저 저장한 뒤 해시가 포함된 완료 JSON을 원자적으로 저장한다. 같은 명령을 다시 실행하면 입력/trace 해시가 일치하는 완료 에피소드를 재사용한다. 미완료 에피소드는 처음부터 다시 평가한다. 실행 중인 동일 단계를 중복 실행하지 않는다.

코드·모델·데이터·체크포인트가 바뀌면 기존 결과와 혼합하지 않고 새로운 진단 버전으로 분리한다. 입력 manifest는 기존 산출물의 변경을 허용하지 않는다. 단, 과거 공식 manifest에 기록된 구현 코드의 해시와 현재 코드가 이미 다른 경우에는 `source_code_drift`로 명시하고 현재 코드를 동결한다. 데이터/모델 해시 불일치는 실행 오류다.

## 결과 해석

- `input_manifest.json`: 입력 해시, 실제 보존 checkpoint step, 에피소드 선택, 코드 이력 차이.
- `parity/*.json`: 평가기 비교 및 반복 재현성 판정.
- `jobs/*/status.json`: 현재 에피소드 진행률.
- `jobs/*/episodes/*.jsonl.gz`: 상태, 행동, 보상 분해, OOD/거리 성분, 예측을 담은 trace.
- `report/diagnosis_report_ko.md`: 전체 흐름, 증거와 한계, 개선안.
- `report/evidence.json`: 필수 작업 누락 여부 및 기계 판독 가능한 근거.
- `report/*.csv`, `report/*.png`: seed/에피소드별 보상, PPO 지표, OOD와 예측 오차 비교.

`report`는 중간에도 생성할 수 있지만 필수 결과가 없으면 `incomplete`로 표시한다. 원래 학습 프로세스와 중단 없는 대조 실험의 완전 동일성은 입증하지 않는다. 기록 행동도 기존 환경의 투영/fallback을 적용받으므로 수정 횟수와 함께 해석한다. k-NN 거리 기여도는 인과적 중요도가 아니다.

## 검증

```bash
../../envs/jslee_py311/bin/python -m pytest -q tests/test_rl_step19_5_diagnostics.py tests/test_rl_step19_5_v2_2.py -k 'not prepare_freezes'
```

`prepare_freezes` 테스트는 실제 공식 산출물을 수정하는 기존 테스트이므로 진단 검증에서 제외한다. 현재 실제 코드와 Step 16 동결 manifest의 `causal_state.py` 해시 차이는 진단 입력에 보존되며, 기존 평가 결과 재현 여부와 별도로 해석한다.

## 추가 원인 분석과 최종 판단

현재 실행에는 기본 진단 완료 후 다음 후처리를 실행하는 별도 coordinator가 연결되어 있다. 수동 재실행 시 모든 기본 작업 완료 후 아래 명령을 실행한다. 미완료 기본 결과로는 최종 판정을 생성하지 않는다.

```bash
../../envs/jslee_py311/bin/python experiments/rl_policy_training/scripts/finalize_step19_5_diagnosis_review.py
```

이 명령은 support 점수의 누적 광량 좌표 민감도, 기록 행동 1-step의 persistence 비교, 딸기 에피소드별 변화 집중도와 PPO 지표 연관을 분석한다. 최종 파일은 `report/final_decision_ko.md`와 `report/diagnosis_decision.json`이다. 좌표 민감도는 저장 상태에 대한 점수 검사이며, 수정된 환경을 실행한 결과가 아니다.
