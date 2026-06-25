# GEAS Repository Layout

Last updated: 2026-06-25

This repository is organized into preserved source materials, the active GEAS 3.0 working tree, an offline predictor benchmark workspace, and the main technical document.

## Root Items

- `GEAS3.0_incomplete/`: preserved GEAS 3.0 source tree with missing or incomplete implementation pieces
- `GEAS3.0/`: reorganized and completed working tree for the current GEAS 3.0 runtime and evaluation code
- `geas_predictor_benchmark/`: offline predictor comparison workspace for inner-layer temperature forecasting
- `AI 자율제어기(GEAS Ver3.0) 기술문서.pdf`: technical document used as the main reference for code and methodology mapping

## Directory Structure

```text
GEAS3.0_incomplete/
  source/
  R_to_python/

GEAS3.0/
  source/
    controller_layer/
    inner_layer/
    outer_layer/
  evaluation/

geas_predictor_benchmark/
  benchmark.py
  plot_from_csv.py
  results/
```

## GEAS3.0_incomplete/

`GEAS3.0_incomplete/` keeps the incomplete source materials separated from the reorganized runtime tree.

- `GEAS3.0_incomplete/source/`: GEAS 3.0 runtime-oriented source code with missing or incomplete implementation pieces
- `GEAS3.0_incomplete/R_to_python/`: converted and reference scripts, including the `3_6_*` methodology modules

## GEAS3.0/

`GEAS3.0/` contains the actively reorganized and completed structure.

- `GEAS3.0/source/controller_layer/`: orchestration and controller entrypoints
- `GEAS3.0/source/inner_layer/`: core control logic, feature generation, repository access, utilities
- `GEAS3.0/source/outer_layer/`: daily modeling, policy update, and supporting outer-loop runtime logic
- `GEAS3.0/evaluation/`: offline evaluation scripts, period-selection helpers, and generated evaluation outputs separated from the runtime layers

## evaluation/

`GEAS3.0/evaluation/` now centers on the current offline evaluation entrypoints rather than the earlier `3_6_*` filenames.

Current evaluation files include:

- `limited_data_eval.py`
- `no_data_eval.py`
- `sufficient_data_eval.py`

Additional evaluation-related directories include:

- `eval_period_selection/`: helper scripts and saved outputs for representative period selection
- `eval_results/`: generated evaluation artifacts for no-data, limited-data, and sufficient-data runs

## geas_predictor_benchmark/

`geas_predictor_benchmark/` is an offline experiment workspace for swapping only the inner-layer next-step temperature predictor while keeping the surrounding GEAS control flow fixed.

- `benchmark.py`: runs predictor comparison experiments against the shared replay pipeline
- `plot_from_csv.py`: regenerates plots from saved benchmark CSV outputs
- `results/`: saved reports, CSV summaries, JSON metadata, and PNG charts from benchmark runs

## Notes

- The repository was restructured so that `main` exposes the high-level layout directly.
- The previous root README can still be recovered from commit `9f22911` if needed.
- Local backup folders such as `_restructure_backup/` are intentionally not tracked in GitHub.
