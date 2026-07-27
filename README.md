# GEAS Repository

Last updated: 2026-07-27

This repository keeps the legacy GEAS 3.0 materials, the current GEAS 3.5
experiment-ready project, and reference archives used during GEAS 3.5
development.

The active project for current work is `GEAS3.5/`.

## Root Layout

```text
GEAS3.5/                         Active GEAS 3.5 project
archive/
  GEAS3.5_backup_20260708/        Reference backup from 2026-07-08
  GEAS3.5_backup_20260727/        Reference backup from 2026-07-27
GEAS3.0/                          Legacy GEAS 3.0 working tree
GEAS3.0_incomplete/               Preserved incomplete GEAS 3.0 source tree
geas_predictor_benchmark/         Legacy predictor benchmark workspace
AI 자율제어기(GEAS Ver3.0) 기술문서.pdf
```

## Active GEAS 3.5 Project

`GEAS3.5/` is organized around offline dataset preparation and the two GPU
experiment workflows that are currently ready to run:

1. quality model training, comparison, manual selection, and application
   artifact handoff;
2. transition model training, validation-based selection, and selected-model
   test evaluation.

Key directories:

```text
GEAS3.5/
  offline_dataset_preparation/      Dataset extraction, preprocessing, QC, and RL-ready dataset scripts
  experiments/
    quality_control_model_selection/
    transition_model_selection/
  src/geas35/                       Reusable importable implementation
  configs/                          Shared and experiment configuration
  tests/                            Smoke and contract tests
```

The intended data flow is:

```text
02_split
  -> quality experiment
  -> manually selected quality_model_application.json
  -> 03_control_quality.py --quality-model-artifact
  -> 03_quality_controlled
  -> 04_prepare_rl_dataset.py
  -> 5_rl_dataset
  -> transition experiment
  -> selected_transition_model.json
  -> selected_transition_model_test_metrics.json
```

See `GEAS3.5/README.md` for the GEAS 3.5 entrypoints, artifact contracts, and
the deferred online/RL boundary.

## Database Configuration

`GEAS3.5/offline_dataset_preparation/scripts/00_extract_raw.py` reads database
settings from `GEAS3.5/.env`.

`GEAS3.5/.env` is intentionally ignored by Git because it contains local
credentials. Use `GEAS3.5/.env.example` as the shareable template.

## Archives

The `archive/` directory stores historical GEAS 3.5 backup code for reference
while the active implementation continues in `GEAS3.5/`.

Generated artifacts, cache folders, and local `.env` files under `archive/` are
ignored by `archive/.gitignore`.

## Legacy Areas

`GEAS3.0/` and `GEAS3.0_incomplete/` are preserved for reference and are not the
current target of the GEAS 3.5 restructuring work.

`geas_predictor_benchmark/` remains as an older offline benchmark workspace for
inner-layer temperature predictor experiments.
