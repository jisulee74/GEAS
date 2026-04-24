# GEAS Predictor Benchmark Report

- Selected period: `2025-03-17 00:00:00` to `2025-03-24 00:00:00`
- Evaluation period: `2025-03-17 00:30:00` to `2025-03-23 23:55:00`
- Common theta objective: `0.0971`
- Theta fit success: `True`

## Common Theta

- `UA`: `500`
- `C`: `7.76168e+06`
- `k_heat`: `23333.4`
- `a0`: `0.21555`
- `a1`: `3.9669`
- `a2`: `0.462436`
- `k_evap`: `1.41422e-05`
- `k_photo`: `0.0005`
- `eta`: `0.0948165`
- `rho_cp`: `1206`

## Comparison Summary

```text
 temp_viol_rate  temp_viol_maxrun  cond_viol_rate  cond_viol_maxrun  rh_viol_rate  vpd_viol_rate  hv_ineff_rate  tv_vent  sw_heat                     model  Tin_next_MAE  Tin_next_RMSE  rollout_sec  mean_control_step_ms  p95_control_step_ms  model_bytes  fit_sec
         0.0000               0.0             0.0               0.0           0.0            0.0         0.0000    122.6    723.0                   physics        0.2289         0.2996      55.6885               25.4600              34.7309           47   0.0000
         0.0000               0.0             0.0               0.0           0.0            0.0         0.0000    127.4    743.0 physics_residual_tiny_ttm        0.1906         0.2363     595.0816              293.8313             419.9538        41504   0.1420
         0.0000               0.0             0.0               0.0           0.0            0.0         0.0000      2.6    955.0             random_forest        0.0522         0.0722    2148.2412             1066.8556            4155.0988      1036819   0.9307
         0.1364             128.0             0.0               0.0           0.0            0.0         0.0396    114.9    781.0                  tiny_ttm        0.1439         0.1896    1153.4882              571.7756            1366.9038        60410   0.1664
```

## Edge Notes

- `physics` is the lightest and most deterministic baseline.
- `tiny_ttm` uses temporal feature mixing with a small MLP head, so sequence context is reflected while remaining edge-friendly.
- `physics_residual_tiny_ttm` is usually the safest AI deployment path because it preserves physics priors and learns only correction residuals.
- `random_forest` is easy to fit and interpret, but serialized size and branch-heavy inference can grow quickly on edge devices.