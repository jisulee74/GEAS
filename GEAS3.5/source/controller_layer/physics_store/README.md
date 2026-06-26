# Module 3: Physical Parameter Store and Lookup

This folder implements the document-style Module 3 storage layer for physical parameters in the controller layer.

## Purpose
- Save physical parameters estimated by outer_layer/offline scripts
- Look up the latest matching parameter set from the controller-layer runtime side
- Provide deterministic fallback defaults when no stored record exists

## Current storage format
- JSON file: `physics_params_store.json`

## Key structure
Each record is stored under a composite key:
- `farm:<farm_sn>|stage:<stage_name>|model:<model_name>|from:<valid_from>|to:<valid_to>`

Examples:
- `farm:97|stage:stage3|model:default|from:2026-02-01T00:00:00+09:00|to:2026-03-01T00:00:00+09:00`
- `farm:any|stage:any|model:default|from:any|to:any`

## Main API
- `PhysicsStore.save_record(...)`
- `PhysicsStore.get_record(...)`
- `PhysicsStore.resolve_params(...)`
- `PhysicsStore.list_records()`

## Intended flow
1. outer_layer scripts estimate physics parameters for a specific data case
2. outer_layer scripts write them into this store
3. The controller-layer-managed runtime reads them back and injects them into the controller as `derived_result` or controller-side physics defaults

## Note
This folder only implements storage/lookup for the shared controller layer. It does not by itself execute outer_layer estimation jobs.


- `data_case`는 저장/조회 key가 아니라 metadata로 보존되며, 물리파라미터 계산 방법의 출처를 나타냅니다.
- 조회는 `farm/stage/model`과 현재 `target_ts`에 가장 잘 맞는 기간 레코드를 우선 선택합니다.
