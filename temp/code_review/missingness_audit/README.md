# Missingness Audit for `GEAS3.0/source`

This folder contains standalone review code only. It does not add or modify files under `GEAS3.0/source/`.

Run from the repository root with the same Python environment used for GEAS:

```powershell
python code_review/missingness_audit/missingness_pipeline_report.py --start "2025-11-01 00:00:00" --end "2025-12-01 00:00:00"
```

Useful options:

```powershell
python code_review/missingness_audit/missingness_pipeline_report.py --all
python code_review/missingness_audit/missingness_pipeline_report.py --today
python code_review/missingness_audit/missingness_pipeline_report.py --csv code_review/missingness_audit/stage_summary.csv
```

The script uses the same DB environment variables as `GEAS3.0/source/inner_layer/core/config.py`.
