# Module 3: Physical Parameter Store and Lookup

This folder implements the document-style Module 3 storage layer for physical parameters.

## Purpose
- Save physical parameters estimated by modeling ver3.0/offline scripts
- Look up the latest matching parameter set from the real-time controller side
- Provide deterministic fallback defaults when no stored record exists

## Current storage format
- JSON file: `physics_params_store.json`

## Key structure
Each record is stored under a composite key:
- `farm:<farm_sn>|stage:<stage_name>|case:<data_case>|model:<model_name>`

Examples:
- `farm:97|stage:stage3|case:sufficient|model:default`
- `farm:any|stage:any|case:no_data|model:default`

## Main API
- `PhysicsStore.save_record(...)`
- `PhysicsStore.get_record(...)`
- `PhysicsStore.resolve_params(...)`
- `PhysicsStore.list_records()`

## Intended flow
1. modeling ver3.0 scripts estimate physics parameters for a specific data case
2. modeling ver3.0 scripts write them into this store
3. GEAS real-time runtime reads them back and injects them into the controller as `derived_result` or controller-side physics defaults

## Note
This folder only implements storage/lookup. It does not by itself execute modeling ver3.0 estimation jobs.
