# Transition Model Candidate Evaluation

This experiment contains the transition model candidate evaluation implementation.

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

Candidate evaluation contract:

- Candidate models are fitted on train.
- HPO objective, validation metrics, and validation rollout metrics produce candidate ranking information.
- The framework does not choose the final transition model.
- Test metrics are computed for every candidate after validation ranking and are descriptive only.
- Test metrics are not used for HPO, validation ranking, or model selection.
- Every candidate artifact is written under `artifacts/default/<crop>/<model_name>/`.
- Downstream inference must be given an explicit candidate model name or artifact directory.

Output contract:

- `candidate_metrics.csv/json`
- `candidate_rollout_metrics.csv/json`
- `candidate_resource_metrics.csv/json`
- `candidate_test_metrics.csv/json`
- `transition_summary.md`
- `artifact_integrity.json`
- `<crop>/<model_name>/model.pkl`
- `<crop>/<model_name>/manifest.json`
- `<crop>/<model_name>/feature_schema.json`
- `<crop>/<model_name>/one_step_metrics.json`
- `<crop>/<model_name>/rollout_metrics.json`
- `<crop>/<model_name>/resource_metrics.json`
- `<crop>/<model_name>/hpo_results.json`
- `<crop>/<model_name>/best_config.json`
- `<crop>/<model_name>/training_summary.json`
- `<crop>/<model_name>/test_metrics.json`

The experiment intentionally does not write selected-model artifacts such as
`selection_report.json`, `selected_transition_model.json`, selected model
copies, winner fields, recommended-model fields, or automatic inference handoff
artifacts.

Local validation before GPU execution:

```powershell
python -m pytest -q GEAS3.5/tests/test_transition_experiment_step3.py
python -m pytest -q GEAS3.5/tests/test_experiment_contracts.py
python -m pytest -q GEAS3.5/tests/test_transition_step10_contract_migration.py
```

The automatic smoke test uses only deterministic synthetic RL-ready parquet
files created under pytest temporary directories. To validate a user-provided
sliced parquet config without creating experiment artifacts, opt in explicitly:

```powershell
$env:GEAS35_TRANSITION_SMOKE_CONFIG = "GEAS3.5/experiments/transition_model_selection/configs/default.yaml"
python -m pytest -q GEAS3.5/tests/test_optional_real_data_contracts.py
Remove-Item Env:\GEAS35_TRANSITION_SMOKE_CONFIG
```
