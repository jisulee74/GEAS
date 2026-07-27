# Transition Model Selection

This experiment contains the transition model selection implementation.

Default execution:

```powershell
python scripts/run.py
```

The default config is a runnable template against the reorganized dataset paths.
It expects RL-ready train/validation/test parquet files under
`../../offline_dataset_preparation/datasets/5_rl_dataset/<crop>`.

Config path contract:

- `dataset.train`, `dataset.validation`, `dataset.test`, and optional `dataset.rollout` are resolved relative to the YAML file directory.
- `experiment.output_dir` is resolved relative to this experiment directory when the YAML lives under `configs/`.
- `--help` and config loading do not require the dataset parquet files to exist; dataset files are read only when the experiment runs.
- Absolute dataset and output paths are supported for GPU runs.

Data contract:

- Run quality control and RL dataset preparation before this experiment:

```powershell
python ../../offline_dataset_preparation/scripts/03_control_quality.py `
  --quality-model-artifact ../quality_control_model_selection/artifacts/default/<model_name>/quality_model_application.json
python ../../offline_dataset_preparation/scripts/04_prepare_rl_dataset.py
```

- The transition runner expects RL-ready columns such as `rl_valid_transition`,
  MDP v1 action columns, current `obs_*` columns, and matching `next_*` target columns.
- Missing files or non-RL-ready inputs fail fast with the prerequisite commands.

Selection contract:

- Candidate models are fitted on train.
- Validation metrics select the final transition model.
- Test metrics are computed once for the selected model only and are not used for re-selection.
- The selected manifest is written to `artifacts/default/<crop>/selected_transition_model.json`.
- The selected model test metrics are written to
  `artifacts/default/<crop>/selected_transition_model_test_metrics.json`.

Local validation before GPU execution:

```powershell
python -m pytest -q GEAS3.5/tests/test_transition_experiment_step3.py
python -m pytest -q GEAS3.5/tests/test_experiment_contracts.py
```

The automatic smoke test uses only deterministic synthetic RL-ready parquet
files created under pytest temporary directories. To validate a user-provided
sliced parquet config without creating experiment artifacts, opt in explicitly:

```powershell
$env:GEAS35_TRANSITION_SMOKE_CONFIG = "GEAS3.5/experiments/transition_model_selection/configs/default.yaml"
python -m pytest -q GEAS3.5/tests/test_optional_real_data_contracts.py
Remove-Item Env:\GEAS35_TRANSITION_SMOKE_CONFIG
```
