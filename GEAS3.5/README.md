# GEAS3.5

This directory combines the GEAS3.5 model-selection code and the iot97 historical
offline dataset into one project layout.

The source folders used for this restructure were preserved as:

```text
../GEAS3.5_backup_20260727
../datasets_iot97_backup_20260727
```

## Layout

```text
offline_dataset_preparation/      Dataset extraction, resampling, splitting, quality-control, and RL dataset scripts
offline_dataset_preparation/datasets/00_raw/                  Raw historical parquet files
offline_dataset_preparation/datasets/01_resampled/            5-minute resampled parquet files
offline_dataset_preparation/datasets/02_split/                Train/validation/test split parquet files
offline_dataset_preparation/datasets/03_quality_controlled/   Empty until quality control is executed
offline_dataset_preparation/datasets/5_rl_dataset/            Empty until RL dataset preparation is executed
experiments/quality_control_model_selection/
experiments/transition_model_selection/
src/geas35/                       Shared Python implementation used by the scripts
```

`offline_dataset_preparation/datasets/03_quality_controlled` intentionally contains only crop directories right now.
Quality and transition model selection are experiment-layer workflows. They do
not automatically choose production models; a human-selected quality artifact can
be passed to `03_control_quality.py` with `--quality-model-artifact`. Quality
experiments write one application-ready artifact per candidate model at
`{experiment_output}/{model}/quality_model_application.json`.

## Entry Points

```powershell
python offline_dataset_preparation/scripts/00_extract_raw.py
python offline_dataset_preparation/scripts/01_resample.py
python offline_dataset_preparation/scripts/02_split.py
python offline_dataset_preparation/scripts/03_control_quality.py
python offline_dataset_preparation/scripts/04_prepare_rl_dataset.py

python experiments/quality_control_model_selection/scripts/run.py
python experiments/transition_model_selection/scripts/run.py
```

## Current Completion Scope

This restructure is complete only for preparing the two GPU experiments:

1. quality model training, comparison, manual selection, and application artifact handoff;
2. transition model training, validation-based selection, and selected-model test evaluation.

The connected data flow is:

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

Local readiness checks use deterministic synthetic pytest fixtures and optional
user-specified sliced parquet configs. They do not require committing real
datasets or generated experiment artifacts.

## Deferred Online And RL Boundary

Full online inference, live adapters, parquet replay runners, RL policy training,
and RL policy inference are follow-up scope. Do not add a dummy, zero-action, or
recorded-action fallback to stand in for a missing RL policy artifact.

Current reusable boundaries that future work should preserve:

- `src/geas35/` remains the canonical home for reusable QC, feature, artifact,
  transition, RL-dataset, and safety helpers.
- `geas35.models.transition.load_selected_transition_model()` loads the selected
  transition model artifact for future inference.
- `geas35.models.transition.PolicyActionProvider` is an abstract placeholder for
  a future RL policy-backed action source; it does not generate actions by
  itself.
- `geas35.models.transition.LoggedActionProvider` may be used for transition
  rollout evaluation against logged data, but it is not a production policy
  substitute.
- `geas35.realtime.sanitize_for_controller()` is a reusable controller-output
  sanitization helper, not a live adapter or policy runner.

When online inference is implemented later, live operation and test parquet
replay should call the same one-timestep inference core. Only their input and
output adapters should differ: live adapters may receive external measurements
and send control commands, while replay adapters must read historical parquet
rows and save evaluation outputs without sending commands. Each timestep may use
only current and past observations, and state/history must reset at
episode/segment boundaries.
