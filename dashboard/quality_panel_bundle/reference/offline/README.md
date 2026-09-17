# Offline Dataset Preparation

The offline dataset pipeline is the canonical data preparation path for the
quality and transition experiments.

```text
00_extract_raw.py       Build offline_dataset_preparation/datasets/00_raw
01_resample.py          Build offline_dataset_preparation/datasets/01_resampled
02_split.py             Build offline_dataset_preparation/datasets/02_split
03_control_quality.py   Build offline_dataset_preparation/datasets/03_quality_controlled
04_prepare_rl_dataset.py Build offline_dataset_preparation/datasets/5_rl_dataset
```

The raw extraction stage writes two top-level manifests: `series_manifest.csv`
describes the crop/GEAS-version data files, while `crop_cycle_manifest.csv`
records `crop_info` transplant dates, original crop end dates, and effective
5-minute-grid end dates adjusted to avoid overlap with the next transplant.

Execution order:

```powershell
python scripts/00_extract_raw.py
python scripts/01_resample.py
python scripts/02_split.py
python scripts/03_control_quality.py `
  --quality-model-artifact ../experiments/quality_control_model_selection/artifacts/default/<model_name>/quality_model_application.json
python scripts/04_prepare_rl_dataset.py
```

`03_control_quality.py` can run without `--quality-model-artifact` using the
rule-only quality path, but GPU experiment preparation should pass the manually
selected `quality_model_application.json` from the quality model experiment.
Quality model selection is not performed in this script.

`04_prepare_rl_dataset.py` reads `03_quality_controlled` and writes RL-ready
train/validation/test parquet files under `5_rl_dataset/<crop>`. The transition
model experiment reads those files and validates required RL columns fail-fast at
runtime.

For GPU runs, keep these entrypoints unchanged and set dataset or artifact paths
through script arguments and experiment YAML files. Do not commit real datasets,
large sliced parquet files, or generated experiment artifacts.
