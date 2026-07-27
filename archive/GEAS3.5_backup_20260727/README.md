# GEAS3.5

This folder is the rebuilt GEAS3.5 codebase.

The previous implementation was backed up as:

```text
GEAS3.5_backup_20260708
```

## Design

GEAS3.5 separates shared processing logic from execution mode.

```text
src/geas35/core           Shared constants and schema definitions
src/geas35/data           Dataset and database input adapters
src/geas35/qc             Sensor QC and outlier detection
src/geas35/preprocessing  Runtime/model-input data cleaning before feature generation
src/geas35/features       Feature generation
src/geas35/models         Model definitions and model artifacts
src/geas35/pipelines      End-to-end orchestration
src/geas35/offline        Parquet-based experiment/evaluation entry points
src/geas35/realtime       DB-based greenhouse operation entry points
src/geas35/utils          Common utilities
configs                   Versioned configuration files
scripts                   Command-line helper scripts
tests                     Unit and integration tests
docs                      Design notes and technical documentation
artifacts                 Saved preprocessors, thresholds, scalers, and models
```

## Data Policy

Offline experiments and realtime operation should use the same core logic.

```text
fit: train split only
transform: train, validation, test, and realtime data
```

Domain-fixed QC rules can be applied directly to all data. Data-driven thresholds,
imputation fallback statistics, scalers, and model parameters must be fitted only
on train data and then reused for validation, test, and realtime inference.

Note: dataset construction steps such as 5-minute resampling, episode creation,
coverage filtering, and train/validation/test splitting live under `datasets/`.
The package-level `src/geas35/preprocessing` area currently means model-input
data cleaning. It may be renamed to `data_cleaning` once the pipeline stabilizes.

## Preprocessing Pipeline

The current offline dataset flow is:

```text
3_splits
-> 4_preprocessed/1_input_schema_prepared
-> 4_preprocessed/2_unit_canonicalized
-> 4_preprocessed/3_missing_outliers_handled
-> 5_rl_dataset
```

Quality Model training/selection is completed before RL dataset creation. See
`docs/preprocessing_pipeline.md` for stage responsibilities, artifacts, and
legacy compatibility notes.
