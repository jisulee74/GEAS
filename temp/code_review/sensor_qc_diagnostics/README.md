# Sensor QC Diagnostics

Standalone diagnostics for `GEAS3.0/source/inner_layer/core/sensor_QC.py`.

The script reads the production sensor QC module, runs DB-backed checks for a requested period, and writes CSV outputs under `outputs/`.

```powershell
python code_review/sensor_qc_diagnostics/sensor_qc_db_diagnostics.py --start "2025-11-01 00:00:00" --end "2025-12-01 00:00:00"
```
