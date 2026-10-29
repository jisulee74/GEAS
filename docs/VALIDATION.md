# Source snapshot validation — 2026-10-29

- New/updated Python sources parse; changed JSON and shell scripts were checked.
- New/updated files were checked for known local credential values and common key formats.
- Frozen random-restart snapshot files match every recorded SHA-256 hash.
- AX core suite (in its repository): 59 passed.
- GEAS main suite before retaining the final target-profile contracts: 443 passed,
  53 failed, 56 setup errors, 2 skipped. Reported failures require workspace artifacts
  excluded from Git (datasets, trained models and experiment manifests).
- After retaining the target profiles and their frozen evidence document, the targeted
  cucumber-profile and final-action synthetic suite passed all 30 tests.

The entire GEAS suite is not claimed to pass in a clone without workspace artifacts.
Restore the local data/model directories before running integration checks. Existing
absolute paths inside frozen experiment contracts must also be reviewed after relocation.
GPU training, live control and new model-quality claims were outside this source cleanup.
Existing dashboard assets were retained from origin/main.
