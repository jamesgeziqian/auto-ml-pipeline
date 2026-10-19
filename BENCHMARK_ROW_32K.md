# Row-wise 32k-column stress test

Measured on the local machine (macOS arm64) with Python 3.10.18, pandas 2.3.3,
PyArrow 25.0.1, LightGBM 4.7.0, and Optuna 5.0.0.

## Dataset

| Property | Value |
| --- | --- |
| Feature files | 4 CSV files with identical schemas |
| Rows per file | 50,000 |
| Columns per file | 32,000 |
| Concatenation | Row-wise (`concat_mode: row`) |
| Total rows | 200,000 |
| Total cells | 6,400,000,000 |
| CSV size | 12,811,824,280 bytes (12.81 GB / 11.93 GiB) |
| Data generation time | 6.60 s |
| Positive rate | 0.44497 |

Each file has 32,000 binary feature columns. The first 20 columns carry signal;
the remaining columns are varied noise from a 256-row pattern pool. The label
file contains the target plus three extra label-file features.

## Result

The pipeline completed successfully in **1,327.49 s (22.12 min)** with peak RSS
of **4,338.3 MB**.

## Stage timings

| Stage | Seconds | Minutes |
| --- | ---: | ---: |
| Load labels / task inference | 0.237 | 0.00 |
| Schema and type inference | 69.400 | 1.16 |
| Chunked CSV → Parquet cache | 651.933 | 10.87 |
| Train/valid/test split | 0.044 | 0.00 |
| Feature statistics + filtering | 561.251 | 9.35 |
| Preprocessing fit | 11.178 | 0.19 |
| Matrix build | 11.070 | 0.18 |
| Optuna tuning + final fit | 21.790 | 0.36 |
| Evaluation | 0.052 | 0.00 |
| Reporting / plots | 0.479 | 0.01 |
| **Total** | **1,327.49** | **22.12** |

The blocking cache used eight 4,096-column Parquet blocks per feature source.
The CSV cache is **375,536,673 bytes (358 MiB)** after Zstandard compression,
down from 12.81 GB of CSV. Filtering reduced **32,003 cached columns to 500**,
and the final model matrix is **200,000 × 500 float32**.

Filtering metrics were computed on a 5,000-row deterministic sample per column
block (`filtering.stats_sample_rows: 5000`) to bound the 32k-column scan. The
full 200,000 rows are used for the cache, preprocessing fit, matrix, and model.

## Metrics

| Metric | Valid | Test |
| --- | ---: | ---: |
| Accuracy | 0.5696 | 0.5707 |
| Precision | 0.5099 | 0.5106 |
| Recall | 0.8481 | 0.8500 |
| F1 | 0.6369 | 0.6379 |
| AUC | 0.6699 | 0.6735 |
| KS | 0.2614 | 0.2671 |

Best TPE parameters:

```text
num_leaves=24
learning_rate=0.128942
min_data_in_leaf=10
feature_fraction=0.958343
bagging_fraction=0.671315
lambda_l1=1.27e-07
lambda_l2=0.000375
best_iteration=41
```

All 3 Optuna trials completed; none were pruned.

## Cleanup

The generated 12.81 GB dataset was deleted after the run, along with the
Parquet cache and matrix artifacts. The small timing, filtering, metrics, and
prediction reports remain under `runs/stress_row_32k/`.
