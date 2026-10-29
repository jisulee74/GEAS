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
python scripts/finalize_step11_integrity.py
```

- `finalize_step11_integrity.py` is the Step 11.6 fail-closed gate. It writes
  `5_rl_dataset/step11_6_integrity_manifest.json` and `step12_handoff.json` only
  after all three crops and all Train/Validation/Test hashes, schemas, scalers,
  row counts, QC artifacts, and parent manifests agree.
- Step 12 consumes the immutable manifest path and SHA-256 from
  `step12_handoff.json`.
- The transition runner expects RL-ready columns such as `rl_valid_transition`,
  MDP v1 action columns, current `obs_*` columns, and matching `next_*` target columns.
- Missing files or non-RL-ready inputs fail fast with the prerequisite commands.

Candidate evaluation contract:

- Candidate models are fitted on train.
- HPO objective, validation metrics, and validation rollout metrics produce candidate ranking information.
- The framework does not choose the final transition model.
- Test metrics are not computed in Step 12; Test remains locked during HPO and candidate ranking.
- After Step 13 integrity review and explicit Step 14 researcher handoff, only the selected candidate may be evaluated on Test.
- Every candidate artifact is written under `artifacts/default/<crop>/<model_name>/`.
- Downstream inference must be given an explicit candidate model name or artifact directory.

Output contract:

- `candidate_metrics.csv/json`
- `candidate_rollout_metrics.csv/json`
- `candidate_resource_metrics.csv/json`
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

The experiment intentionally does not write selected-model artifacts such as
`selection_report.json`, `selected_transition_model.json`, selected model
copies, winner fields, recommended-model fields, or automatic inference handoff
artifacts.

Official Step 12 execution:

```powershell
python scripts/run_step12_real_data_benchmark.py
```

This validates each crop config and parquet hash against Step 11.6, then runs
the ordered eight candidates in independent `artifacts/step12/<crop>` roots.
Target-specific and aggregate Validation/Test ranks, 15/30/60-minute rollout
and drift, resource metrics, and Persistence improvements are retained. The
all-crop tables and `step12_benchmark_manifest.json` are written under
`artifacts/step12`. Test remains descriptive and no model is selected.

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

Step 13 reproducibility and integrity review:

```powershell
python scripts/run_step13_reproducibility_review.py
```

The review verifies the immutable Step 11/12 hash chain, crop configs and seeds,
dependency/device records, official three-target schemas, observation scalers,
recorded/forecast exogenous-provider contract, all candidate file hashes,
comparison reports against raw artifacts, and explicit cold-load
inference/reward for all 24 candidates. Outputs are written under
`artifacts/step13`. It does not create a Step 14 decision or deployment handoff.

Step 14 researcher decision and explicit deployment handoff:

```powershell
python scripts/run_step14_deployment_handoff.py `
  --researcher-decision <researcher-authored-decision.json>
```

The external decision must follow `configs/researcher_decision.schema.json` and
contain one explicit candidate, rationale, Validation evidence, resource
trade-off, and deployment constraints for every crop. The command validates
those values against Step 12/13 and writes references to the existing candidate
artifact directories under `artifacts/step14`. It never chooses a candidate,
creates the researcher decision, or copies a model artifact.


## Residual transition experiment v1.1

The v1.1 configs explicitly set `target_formulation: residual_physical_delta_v1` and write only to `artifacts/step12_v1_1`; the existing absolute-state Step 12 artifacts are not overwritten. Test remains unopened during HPO and Validation ranking.

Run one crop with `python scripts/run.py --config configs/strawberry_v1_1.yaml` (replace the crop name as needed). Run the all-crop publisher with three repeated `--config` options and `--output-root artifacts/step12_v1_1`. Nonlinear candidates use 20 broad trials followed by 20 local trials around the three best broad Validation configurations; other learned candidates use at most 20 unique trials.


## v1.2 deduplicated quality-flag ablation

Run the six Train/Validation-only experiments independently. Existing v1.1 artifacts are not reused because the feature schema hash differs.

```bash
for crop in strawberry melon cucumber; do
  python scripts/run.py --config configs/${crop}_v1_2_with_flags.yaml
  python scripts/run.py --config configs/${crop}_v1_2_without_flags.yaml
done
python scripts/compare_quality_flag_ablation.py
```

The two variants write under `artifacts/step12_v1_2/`. Test remains unopened. The comparison script must only run after all six experiments complete.
