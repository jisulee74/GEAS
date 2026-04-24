# GEAS Predictor Benchmark Report

- Selected period: `2025-03-17 00:00:00` to `2025-03-24 00:00:00`
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
         0.0000               0.0             0.0               0.0           0.0            0.0            0.0     15.0     44.0                   physics        0.1812         0.2569       7.7453               24.7183              34.7486           47   0.0000
         0.0000               0.0             0.0               0.0           0.0            0.0            0.0     14.8     44.0 physics_residual_tiny_ttm        0.1395         0.1731      81.6973              282.2570             421.9290        41504   0.1422
         0.0070               2.0             0.0               0.0           0.0            0.0            0.0      0.2     51.0             random_forest        0.0639         0.0892     647.3839             2253.2393            4214.5225      1036819   0.9343
         0.4077             117.0             0.0               0.0           0.0            0.0            0.0     12.4     69.0                  tiny_ttm        0.1105         0.1651     202.1051              701.7956            1374.8826        60410   0.1675
```

## Edge Notes

- `physics` is the lightest and most deterministic baseline.
- `tiny_ttm` uses temporal feature mixing with a small MLP head, so sequence context is reflected while remaining edge-friendly.
- `physics_residual_tiny_ttm` is usually the safest AI deployment path because it preserves physics priors and learns only correction residuals.
- `random_forest` is easy to fit and interpret, but serialized size and branch-heavy inference can grow quickly on edge devices.