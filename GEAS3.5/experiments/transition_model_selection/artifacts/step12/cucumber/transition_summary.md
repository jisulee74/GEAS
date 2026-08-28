# GEAS Transition Candidate Comparison

## Experiment Config

- crop: cucumber
- output_root: /NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/GEAS/GEAS3.5/experiments/transition_model_selection/artifacts/step12/cucumber
- dataset: {'source_rows': {'train': 22293, 'validation': 4858, 'test': 5430, 'rollout': 4858}, 'transition_dataset_rows': {'train': 22293, 'validation': 4858, 'test': 5430}}

## Validation Ranking

| Rank | Candidate | Validation Rollout Weighted Score | One-step RMSE | One-step NRMSE | Persistence Improvement |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | extra_trees | 0.8577 | 1.4543 | 0.0835 | 0.8903 |
| 2 | xgboost | 0.8903 | 4.1448 | 0.3050 | 0.8861 |
| 3 | lightgbm | 1.0601 | 3.9297 | 0.2844 | 0.8643 |
| 4 | knn | 1.0848 | 6.3887 | 0.4157 | 0.8612 |
| 5 | persistence | 7.8147 | 149.2780 | 8.4437 | 0.0000 |
| 6 | mlp | 2619155466000830 | 2.1639 | 0.1232 | -335157690297109 |
| 7 | linear_regression | 9152165027839090 | 1.2437 | 0.0680 | -1171147926026762 |
| 8 | linear_svr | 10618122859126154 | 2.2784 | 0.1163 | -1358737798863649 |

## Rollout Metrics

| Candidate | 15min RMSE | 30min RMSE | 60min RMSE | Drift | Physical Violation Rate | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 219.8658 | 220.3484 | 223.3148 | 0.0190 | 0.0000 | 0.0000 |
| linear_regression | 108457 | 1845010377 | 756918062171499648 | 919647986492699 | 0.8056 | 0.0000 |
| linear_svr | 108543 | 1938870373 | 876712853868003200 | 1066488503085184 | 0.7407 | 0.0000 |
| knn | 33.1188 | 36.4875 | 35.4083 | 0.1778 | 0.0000 | 0.0000 |
| extra_trees | 23.3645 | 25.7898 | 24.6196 | 0.2197 | 0.0000 | 0.0000 |
| lightgbm | 24.7721 | 27.3827 | 26.0662 | 0.2550 | 0.0000 | 0.0000 |
| xgboost | 19.7752 | 21.7869 | 20.2726 | 0.2053 | 0.0000 | 0.0000 |
| mlp | 80006 | 967070646 | 199324803955262944 | 264068297829133 | 0.7407 | 0.0000 |

## Resource Usage

| Candidate | Train s | HPO s | Latency median ms | Latency p95 ms | Model bytes | Artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 0.0000 | 0.0000 | 0.7928 | 0.8816 | 1843 | 22208 |
| linear_regression | 0.0559 | 3.0450 | 5.1916 | 5.4319 | 11291 | 199620 |
| linear_svr | 8.3746 | 58.1956 | 5.1490 | 5.2633 | 9803 | 198556 |
| knn | 0.0119 | 9.4377 | 10.6164 | 10.9854 | 35320410 | 35506773 |
| extra_trees | 1.8665 | 69.2147 | 65.2285 | 72.0675 | 349783923 | 349973138 |
| lightgbm | 0.5969 | 17.3687 | 5.5468 | 5.6543 | 447495 | 636638 |
| xgboost | 0.3813 | 22.2265 | 17.7941 | 18.6259 | 218534 | 407459 |
| mlp | 10.1159 | 118.5357 | 5.1722 | 5.3045 | 246934 | 438067 |

## Invalid or Failed Candidates

Candidates with failed resource benchmarks or invalid numeric metrics are retained with status fields rather than dropped.

## Deployment Trade-offs

Use the validation, rollout, drift, and resource tables together to judge deployment trade-offs. Resource metrics are reported separately and are not part of the validation score.

This framework intentionally does not perform automatic model selection.
Final transition-model selection is left to the researcher after considering rollout accuracy, resource usage, deployment constraints, and practical trade-offs.

## Test Evaluation

| Validation Rank | Candidate | Test RMSE | Test NRMSE | Test R2 | 60min Test Rollout RMSE | Test Descriptive Rank |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | linear_regression | 1.6903 | 0.1096 | 0.9834 | 724710181266099328 | 1 |
| 2 | extra_trees | 2.1220 | 0.1609 | 0.9703 | 38.4374 | 2 |
| 3 | mlp | 4.0252 | 0.2509 | 0.8988 | 196818335324162560 | 6 |
| 4 | linear_svr | 2.5937 | 0.1590 | 0.9539 | 833879702648381184 | 3 |
| 5 | lightgbm | 3.7876 | 0.3238 | 0.8881 | 37.7111 | 4 |
| 6 | xgboost | 3.9793 | 0.3495 | 0.8647 | 31.9344 | 5 |
| 7 | knn | 8.4577 | 0.6054 | 0.5770 | 47.3833 | 7 |
| 8 | persistence | 148.7307 | 9.4997 | -147.9965 | 211.1448 | 8 |

Test metrics are descriptive only. They were not used for HPO, validation ranking, or automatic selection.

This framework intentionally does not perform automatic model selection.
Final transition-model selection is left to the researcher after considering rollout accuracy, resource usage, deployment constraints, and practical trade-offs.
