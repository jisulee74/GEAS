# GEAS Transition Candidate Comparison

## Experiment Config

- crop: strawberry
- output_root: /NHNHOME/WORKSPACE/26mafra001_A/BASE/theimc/jslee/GEAS/GEAS3.5/experiments/transition_model_selection/artifacts/step12/strawberry
- dataset: {'source_rows': {'train': 90861, 'validation': 18015, 'test': 19335, 'rollout': 18015}, 'transition_dataset_rows': {'train': 90861, 'validation': 18015, 'test': 19335}}

## Validation Ranking

| Rank | Candidate | Validation Rollout Weighted Score | One-step RMSE | One-step NRMSE | Persistence Improvement |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | extra_trees | 2.0029 | 34.3806 | 0.2457 | 0.2325 |
| 2 | xgboost | 2.0042 | 49.0361 | 0.4471 | 0.2320 |
| 3 | knn | 2.3304 | 32.5250 | 0.4893 | 0.1069 |
| 4 | lightgbm | 2.3469 | 37.1568 | 0.3749 | 0.1006 |
| 5 | persistence | 2.6095 | 170.9706 | 2.9970 | 0.0000 |
| 6 | mlp | 180110390692047 | 83.2916 | 0.8955 | -69020292332565 |
| 7 | linear_regression | 5627269542566288 | 15.1924 | 0.1472 | -2156431882523378 |
| 8 | linear_svr | 12643840159967546 | 14.9762 | 0.1332 | -4845259291782419 |

## Rollout Metrics

| Candidate | 15min RMSE | 30min RMSE | 60min RMSE | Drift | Physical Violation Rate | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 240.4398 | 239.8969 | 239.7229 | 0.0011 | 0.3333 | 0.0000 |
| linear_regression | 154597 | 3940079753 | 3604955004105803264 | 563508204533733 | 0.7407 | 0.0000 |
| linear_svr | 178336 | 5698951815 | 8221616160253943808 | 1263896717345704 | 0.7407 | 0.0000 |
| knn | 591.9013 | 662.7956 | 695.4379 | 0.6694 | 0.0000 | 0.0000 |
| extra_trees | 388.9785 | 436.1322 | 456.0413 | 0.5843 | 0.0000 | 0.0000 |
| lightgbm | 320.9030 | 359.5148 | 377.5850 | 0.6576 | 0.0000 | 0.0000 |
| xgboost | 225.2709 | 252.5934 | 265.1007 | 0.5539 | 0.0000 | 0.0000 |
| mlp | 5188 | 107968271 | 82062314289094752 | 18049597099970 | 0.8056 | 0.0000 |

## Resource Usage

| Candidate | Train s | HPO s | Latency median ms | Latency p95 ms | Model bytes | Artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 0.0000 | 0.0000 | 0.7787 | 0.8687 | 1843 | 23467 |
| linear_regression | 0.4031 | 3.1982 | 5.1306 | 5.3410 | 11301 | 201397 |
| linear_svr | 63.8909 | 67.8646 | 5.0973 | 5.2234 | 9804 | 199818 |
| knn | 0.0545 | 9.0990 | 29.4575 | 29.6307 | 143932140 | 144119059 |
| extra_trees | 11.4063 | 77.8803 | 143.6983 | 198.8958 | 17382679 | 17572155 |
| lightgbm | 1.7188 | 18.4997 | 5.7385 | 5.8538 | 455354 | 644249 |
| xgboost | 1.3702 | 22.6739 | 17.1696 | 17.9831 | 217852 | 406268 |
| mlp | 49.4319 | 134.6838 | 5.2182 | 5.3459 | 449837 | 641117 |

## Invalid or Failed Candidates

Candidates with failed resource benchmarks or invalid numeric metrics are retained with status fields rather than dropped.

## Deployment Trade-offs

Use the validation, rollout, drift, and resource tables together to judge deployment trade-offs. Resource metrics are reported separately and are not part of the validation score.

This framework intentionally does not perform automatic model selection.
Final transition-model selection is left to the researcher after considering rollout accuracy, resource usage, deployment constraints, and practical trade-offs.

## Test Evaluation

| Validation Rank | Candidate | Test RMSE | Test NRMSE | Test R2 | 60min Test Rollout RMSE | Test Descriptive Rank |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | linear_svr | 4.4448 | 0.2326 | 0.8765 | 8124357155532256256 | 2 |
| 2 | linear_regression | 4.2342 | 0.2229 | 0.8888 | 3572989542491095040 | 1 |
| 3 | knn | 8.8908 | 0.6306 | 0.5535 | 697.6705 | 7 |
| 4 | extra_trees | 4.6869 | 0.2542 | 0.8661 | 400.6175 | 3 |
| 5 | lightgbm | 5.8748 | 0.4258 | 0.8034 | 375.0829 | 5 |
| 6 | xgboost | 6.2024 | 0.4475 | 0.7814 | 174.8422 | 6 |
| 7 | mlp | 4.9192 | 0.2686 | 0.8558 | 79918733403188848 | 4 |
| 8 | persistence | 149.3867 | 8.6932 | -126.4047 | 236.8184 | 8 |

Test metrics are descriptive only. They were not used for HPO, validation ranking, or automatic selection.

This framework intentionally does not perform automatic model selection.
Final transition-model selection is left to the researcher after considering rollout accuracy, resource usage, deployment constraints, and practical trade-offs.
