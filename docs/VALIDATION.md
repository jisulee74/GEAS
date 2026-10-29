# Validation scope — 2026-10-29

- `python scripts/reproduce.py smoke`: 39 synthetic checks passed with the recorded Python 3.12 environment.
  Includes QC training, transition experiment contracts, realtime boundaries and four-channel behavior.
- Frozen random-restart snapshot: every recorded source SHA-256 matched; 4 standalone tests passed.
- Original Profile-A PPO preflight passed its frozen input/configuration hashes without training.
- The 30 targeted profile/final-action tests passed when the private interview evidence was available.
  The evidence text is excluded from the public tree; its hash test requires private restoration.
- Full research tests also validate historical local outputs, datasets and trained models. They do not
  all pass in a source-only clone; no full-suite pass or new real-data training result is claimed.
- Source syntax, JSON, shell syntax and Git whitespace were checked. New exports contain aggregate
  metrics and hashes, without facility memberships, observation rows or credential values.

Clone validation details and private-input checks are recorded in `reproducibility/clone_validation.json`
when completed. Fresh package installation and full GPU retraining are separate from using a fresh clone
with the recorded existing environment.

## Actual GitHub clone and private restore

- Fresh GitHub clone: 39 synthetic tests passed with the recorded existing environment.
- Restore safety: 7 checks passed (no overwrite, traversal/escape rejection, checksum, aliases, delta precedence).
- Profile-A preflight passed all authoritative/frozen input hashes after private restore.
- See `reproducibility/clone_validation.json` for the exact tested source revision and limits.
