# Transition Model Random Restart v1.3

This experiment is an isolated snapshot. It does not import the project's
shared `src/geas35` tree and never opens the Test split during selection.

Run all six finite-value preflights:

```bash
python scripts/run_all.py --preflight-only
```

Run all six experiments sequentially:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/run_all.py
```

Regenerate only the six-run aggregate and paired bootstrap report:

```bash
python scripts/run_all.py --aggregate-only
```

Run or resume one experiment:

```bash
python scripts/run.py --config configs/strawberry_with_flags_v1_3.yaml
```

The runner refuses to start while a legacy
`experiments/transition_model_selection/scripts/run.py` process is active.
