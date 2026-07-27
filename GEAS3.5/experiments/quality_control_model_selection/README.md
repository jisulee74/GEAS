# Quality Control Model Selection

This experiment selects the quality model and threshold used by
`offline_dataset_preparation/scripts/03_control_quality.py`.

Default execution:

```powershell
python scripts/run.py
```

The default config reads cucumber train/validation/test splits from
`../../offline_dataset_preparation/datasets/02_split/cucumber` and writes experiment artifacts under
`artifacts/default`.

Config path contract:

- `dataset.train`, `dataset.validation`, and `dataset.test` are resolved relative to the YAML file directory.
- `experiment.output_dir` is resolved relative to this experiment directory when the YAML lives under `configs/`.
- `--help` and config loading do not require the dataset parquet files to exist; dataset files are read only when the experiment runs.
- Absolute dataset and output paths are supported for GPU runs.

Manual selection contract:

- This experiment does not automatically choose the final quality model.
- Compare candidates with the validation columns in `model_comparison.csv` and `model_comparison.json`.
- Use validation reconstruction metrics, validation anomaly metrics, and validation online benchmark metrics as the selection basis.
- Treat test metrics as final evaluation only. Do not use test metrics to choose or re-rank the model.
- The application artifact to pass into offline QC is the selected model directory's `quality_model_application.json`.
- Record the decision beside the experiment output, for example as `manual_selection_decision.json`, with:
  - `selected_quality_model_artifact`
  - `selected_model_name`
  - validation metric values used for the decision
  - `test_used_for_selection: false`
  - reviewer/date/free-text rationale

Apply the manually selected artifact to offline quality control:

```powershell
python ../../offline_dataset_preparation/scripts/03_control_quality.py `
  --quality-model-artifact artifacts/default/<model_name>/quality_model_application.json
```

For GPU runs, keep the same entrypoint and set dataset paths and `experiment.output_dir`
in the YAML config. Do not edit scripts to inject local absolute paths.

Local validation before GPU execution:

```powershell
python -m pytest -q GEAS3.5/tests/test_quality_experiment_smoke.py
python -m pytest -q GEAS3.5/tests/test_experiment_contracts.py
```

The automatic smoke test uses only deterministic synthetic parquet files created
under pytest temporary directories. To validate a user-provided sliced parquet
config without creating artifacts, opt in explicitly:

```powershell
$env:GEAS35_QUALITY_SMOKE_CONFIG = "GEAS3.5/experiments/quality_control_model_selection/configs/default.yaml"
python -m pytest -q GEAS3.5/tests/test_optional_real_data_contracts.py
Remove-Item Env:\GEAS35_QUALITY_SMOKE_CONFIG
```
