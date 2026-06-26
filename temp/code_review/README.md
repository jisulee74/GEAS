# Code Review Workspace

Standalone review and diagnostics assets live here so the production code under `GEAS3.0/source/` can stay untouched.

## Folders

- `sensor_qc_diagnostics/`: DB-backed diagnostics for `GEAS3.0/source/inner_layer/core/sensor_QC.py`, plus generated CSV outputs.
- `db_structure_inspection/`: Notebook-based DB schema inspection.
- `missingness_audit/`: Standalone missing-value ratio audit through selected `GEAS3.0/source` preprocessing stages.

## Notes

Scripts in this folder may import or read code from `GEAS3.0/source/`, but they should not write there.
