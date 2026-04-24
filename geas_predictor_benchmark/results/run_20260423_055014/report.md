# GEAS Predictor Benchmark Report

- Selected period: `2025-11-15T00:00:00` to `2025-11-22T00:00:00`
- Common theta objective: `1.4472`
- Theta fit success: `False`

## Common Theta

- `UA`: `2723.31`
- `C`: `8e+07`
- `k_heat`: `7041.44`
- `a0`: `0.01`
- `a1`: `0.2`
- `a2`: `0.05`
- `k_evap`: `4.22862e-06`
- `k_photo`: `3.8737e-05`
- `eta`: `0.0179862`
- `rho_cp`: `1206`

## Comparison Summary

```text
 temp_viol_rate  temp_viol_maxrun  cond_viol_rate  cond_viol_maxrun  rh_viol_rate  vpd_viol_rate  hv_ineff_rate  tv_vent  sw_heat                     model  Tin_next_MAE  Tin_next_RMSE  rollout_sec  mean_control_step_ms  p95_control_step_ms  model_bytes  fit_sec
         0.0426               2.0          0.0426               2.0        0.0638         0.0851            0.0      0.3      1.0                   physics        0.2783         0.2789       2.5645               52.2336             113.8060           47   0.0000
         0.0426               2.0          0.0000               0.0        0.0426         0.0426            0.0      1.1      7.0 physics_residual_tiny_ttm        0.1756         0.2019      36.5187              774.5289            1395.8516        42008   0.1017
         0.0426               2.0          0.0000               0.0        0.0000         0.0000            0.0      1.6      1.0             random_forest        0.0255         0.0378     104.5015             2220.9122            4362.1820       313981   0.3222
         0.0426               2.0          0.0000               0.0        0.0000         0.0000            0.0      2.2     23.0                  tiny_ttm        0.4140         0.4171      35.7212              757.5771            1405.1396        60255   0.0786
```

## Edge Notes

- `physics` is the lightest and most deterministic baseline.
- `tiny_ttm` uses temporal feature mixing with a small MLP head, so sequence context is reflected while remaining edge-friendly.
- `physics_residual_tiny_ttm` preserves physics priors and learns only residual correction.
- `random_forest` achieved the best prediction accuracy in this short replay, but it is also the heaviest model.