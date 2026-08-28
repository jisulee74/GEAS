# GEAS Transition Candidate Comparison

## Experiment Config

- crop: melon
- output_root: /NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/GEAS/GEAS3.5/experiments/transition_model_selection/artifacts/step12/melon
- dataset: {'source_rows': {'train': 28923, 'validation': 6027, 'test': 6136, 'rollout': 6027}, 'transition_dataset_rows': {'train': 28923, 'validation': 6027, 'test': 6136}}

## Validation Ranking

| Rank | Candidate | Validation Rollout Weighted Score | One-step RMSE | One-step NRMSE | Persistence Improvement |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | extra_trees | 1.5123 | 1.8072 | 0.1115 | 0.8382 |
| 2 | xgboost | 1.5296 | 2.8541 | 0.2370 | 0.8364 |
| 3 | lightgbm | 1.9801 | 2.9259 | 0.2387 | 0.7882 |
| 4 | knn | 3.1382 | 5.6934 | 0.4182 | 0.6643 |
| 5 | persistence | 9.3473 | 157.4977 | 10.1231 | 0.0000 |
| 6 | mlp | 1690545824409731 | 3.9062 | 0.2555 | -180859823127367 |
| 7 | linear_regression | 6621740864819045 | 1.2680 | 0.0774 | -708414326493983 |
| 8 | linear_svr | 7231124279681664 | 1.5227 | 0.0908 | -773608049750368 |

## Rollout Metrics

| Candidate | 15min RMSE | 30min RMSE | 60min RMSE | Drift | Physical Violation Rate | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 234.8529 | 234.1191 | 233.4594 | -0.0062 | 0.0000 | 0.0000 |
| linear_regression | 104703 | 1554455599 | 486982527049593536 | 666224595997694 | 0.6759 | 0.0000 |
| linear_svr | 106483 | 1619356729 | 531622883390501888 | 727372770770710 | 0.8056 | 0.0000 |
| knn | 97.7597 | 109.8696 | 115.7521 | 0.8563 | 0.0000 | 0.0000 |
| extra_trees | 39.4671 | 44.7389 | 47.4910 | 0.4397 | 0.0000 | 0.0000 |
| lightgbm | 41.0223 | 46.1788 | 48.8605 | 0.5542 | 0.0000 | 0.0000 |
| xgboost | 26.3940 | 30.0803 | 32.0716 | 0.4240 | 0.0000 | 0.0000 |
| mlp | 75494 | 766508927 | 111596068062783312 | 170758152230920 | 0.7407 | 0.0000 |

## Resource Usage

| Candidate | Train s | HPO s | Latency median ms | Latency p95 ms | Model bytes | Artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 0.0000 | 0.0000 | 0.7830 | 0.9052 | 1843 | 22253 |
| linear_regression | 0.0835 | 3.0518 | 5.1702 | 5.3933 | 11301 | 199312 |
| linear_svr | 4.1937 | 42.6658 | 5.1102 | 5.2378 | 9803 | 198562 |
| knn | 0.0158 | 9.3434 | 12.9541 | 13.1134 | 45822330 | 46008274 |
| extra_trees | 0.7527 | 74.9545 | 63.2064 | 63.8840 | 4947637 | 5136239 |
| lightgbm | 0.7347 | 17.7286 | 5.5633 | 5.6450 | 448199 | 637013 |
| xgboost | 0.4895 | 22.0694 | 17.2827 | 18.1967 | 217508 | 406190 |
| mlp | 12.1795 | 121.1422 | 5.2027 | 5.3114 | 448799 | 640302 |

## Invalid or Failed Candidates

Candidates with failed resource benchmarks or invalid numeric metrics are retained with status fields rather than dropped.

## Deployment Trade-offs

Use the validation, rollout, drift, and resource tables together to judge deployment trade-offs. Resource metrics are reported separately and are not part of the validation score.

This framework intentionally does not perform automatic model selection.
Final transition-model selection is left to the researcher after considering rollout accuracy, resource usage, deployment constraints, and practical trade-offs.

## Test Evaluation

| Validation Rank | Candidate | Test RMSE | Test NRMSE | Test R2 | 60min Test Rollout RMSE | Test Descriptive Rank |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | linear_regression | 5.0180 | 0.2016 | 0.9096 | 462366514537011200 | 1 |
| 2 | linear_svr | 5.0357 | 0.2021 | 0.9090 | 506869575625015552 | 2 |
| 3 | extra_trees | 6.5681 | 0.2654 | 0.8436 | 52.7433 | 4 |
| 4 | xgboost | 7.5588 | 0.3891 | 0.8063 | 39.9415 | 6 |
| 5 | lightgbm | 7.4048 | 0.3833 | 0.8137 | 55.0324 | 5 |
| 6 | mlp | 5.5603 | 0.2385 | 0.8939 | 106198160586990960 | 3 |
| 7 | knn | 9.3623 | 0.5759 | 0.6380 | 122.8502 | 7 |
| 8 | persistence | 159.2718 | 8.3256 | -85.2155 | 225.7621 | 8 |

Test metrics are descriptive only. They were not used for HPO, validation ranking, or automatic selection.

This framework intentionally does not perform automatic model selection.
Final transition-model selection is left to the researcher after considering rollout accuracy, resource usage, deployment constraints, and practical trade-offs.
