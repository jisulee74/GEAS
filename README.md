# GEAS Repository Layout

Last updated: 2026-04-06

This repository is organized into two top-level code roots and one technical document.

## Root Items

- `origin/`: preserved original GEAS 3.0 source tree and converted reference scripts
- `updated/`: reorganized working tree for the current GEAS 3.0 runtime and evaluation code
- `AI 자율제어기(GEAS Ver3.0) 기술문서.pdf`: technical document used as the main reference for code and methodology mapping

## Directory Structure

```text
origin/
  GEAS ver3.0/
  R_to_python/

updated/
  GEAS3.0/
    controller_layer/
    inner_layer/
    outer_layer/
  evaluation/
```

## origin/

`origin/` keeps the original materials separated from the reorganized runtime tree.

- `origin/GEAS ver3.0/`: original GEAS 3.0 runtime-oriented source code
- `origin/R_to_python/`: converted and reference scripts, including the `3_6_*` methodology modules

## updated/

`updated/` contains the actively reorganized structure.

- `updated/GEAS3.0/controller_layer/`: orchestration and controller entrypoints
- `updated/GEAS3.0/inner_layer/`: core control logic, feature generation, repository access, utilities
- `updated/GEAS3.0/outer_layer/`: daily modeling, policy update, and supporting outer-loop runtime logic
- `updated/evaluation/`: offline evaluation and validation scripts separated from the runtime layers

## evaluation/

The `3_6_*` modules were moved under `updated/evaluation/` because they function as offline validation or policy-comparison code rather than runtime control-layer logic.

Current evaluation files include:

- `3_6_3__AI_자율제어기_견고성_검증_방법론.py`
- `3_6_4__제한적_데이터_환경에서의_부분_검증_방법론.py`
- `3_6_6__외기_조건_고전_기반_정책_비교_시뮬레이션.py`
