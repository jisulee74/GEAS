# Quality Control Model Selection

This experiment selects the quality model and threshold used by
`offline_dataset_preparation/scripts/03_control_quality.py`.

Default execution:

```powershell
python scripts/run.py
```

The default `cucumber.yaml` config reads cucumber train/validation/test splits from
`../../offline_dataset_preparation/datasets/02_split/cucumber` and writes experiment artifacts under
`artifacts/cucumber`.

Config path contract:

- `dataset.train`, `dataset.validation`, and `dataset.test` are resolved relative to the YAML file directory.
- `experiment.output_dir` is resolved relative to this experiment directory when the YAML lives under `configs/`.
- `--help` and config loading do not require the dataset parquet files to exist; dataset files are read only when the experiment runs.
- Absolute dataset and output paths are supported for GPU runs.

HPO search-space contract:

- Random-search budget: 50 trials per model, using seed 42 for every model.
- Fixed common settings: `lookback=288`, `epochs=100` as the maximum epoch count, and early stopping with `patience=10`.
- Common search: `mask_fraction` in `{0.30, 0.40, 0.50}`, `batch_size` in `{32, 64, 128}`, log-uniform `learning_rate` in `[1e-4, 1e-2]`, log-uniform `weight_decay` in `[1e-6, 1e-2]`, and uniform `dropout` in `[0.0, 0.3]`.
- ModernTCN: `channel_width` in `{32, 64, 128}`, stage-depth vectors in `{[1,1,1], [2,2,2]}`, and `kernel_size` in `{13, 31, 51}`. The current single-width implementation realizes a depth vector as the sum of its stage depths.
- TimesNet: `temporal_blocks` in `{1,2,3}`, `top_k_periods` in `{2,3,5}`, and `period_embedding_dim` in `{32,64,128}`; `d_ff` is derived as `2 * period_embedding_dim`.
- PatchTST: `patch_length` in `{8,16,24,32}`, `transformer_depth` in `{2,3,4}`, `attention_heads` in `{4,8,16}`, and `embedding_dim` in `{32,64,128,256}`; stride is derived as `patch_length / 2` and `d_ff` as `2 * embedding_dim`.
- Figures are generated as PNG only.
- The simplified in-project ModernTCN and TimesNet implementations do not expose the original-repository `small_size`, `ffn_ratio`, or `num_kernels` arguments. They are not reported as applied parameters.

Threshold calibration contract:

- Thresholds are not fixed YAML values and are not HPO parameters.
- After the best HPO configuration is fixed and retrained, reconstruction-error scores are computed on the original Validation split.
- Exactly 100 thresholds are generated at equal intervals from the finite Validation error minimum to maximum (`linspace`).
- Every candidate is evaluated by Validation synthetic-anomaly F1, and the threshold with the maximum F1 is selected. Equal-F1 ties retain the first threshold in ascending order.
- The artifact records the generation method, Validation error range, requested/actual candidate counts, every candidate metric, and the selected threshold.

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
  --quality-model-artifact artifacts/cucumber/<model_name>/quality_model_application.json
```

For GPU runs, keep the same entrypoint and set dataset paths and `experiment.output_dir`
in the YAML config. Do not edit scripts to inject local absolute paths.

Terminal progress output:

- HPO prints model start/finish and every trial start/finish with `validation_rmse`.
- Threshold calibration prints model start/finish, progress every 10 candidates, Validation error range, selected threshold, and best F1.
- Validation/Test evaluation and online benchmark print model-level start/finish messages.
- Report, PNG figures, artifact integrity, and the full crop pipeline print stage-level status messages immediately with flushed output.

Local validation before GPU execution:

```powershell
python -m pytest -q GEAS3.5/tests/test_quality_experiment_smoke.py
python -m pytest -q GEAS3.5/tests/test_experiment_contracts.py
```

The automatic smoke test uses only deterministic synthetic parquet files created
under pytest temporary directories. To validate a user-provided sliced parquet
config without creating artifacts, opt in explicitly:

```powershell
$env:GEAS35_QUALITY_SMOKE_CONFIG = "GEAS3.5/experiments/quality_control_model_selection/configs/cucumber.yaml"
python -m pytest -q GEAS3.5/tests/test_optional_real_data_contracts.py
Remove-Item Env:\GEAS35_QUALITY_SMOKE_CONFIG
```
