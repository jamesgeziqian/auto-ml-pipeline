# Stress-test benchmark

Measured on the local machine (macOS arm64) with Python 3.10.18, pandas 2.3.3,
PyArrow 25.0.1, LightGBM 4.7.0, and Optuna 5.0.0.

For the separate row-wise test with four 50,000 × 32,000 CSVs, see
[BENCHMARK_ROW_32K.md](BENCHMARK_ROW_32K.md).

## Dataset

| Property | Value |
| --- | --- |
| Rows | 32,500 |
| Feature columns | 500 |
| Feature files | 4 CSV parts, horizontally concatenated by `customer_id` |
| Label-file extra features | 4 |
| Total cached logical features | 504 |
| CSV size | 3,506,632,160 bytes (3.51 GB / 3.27 GiB) |
| Data generation time | 28.72 s |
| Positive rate | 0.5004 |

The data contains 50 predictive numeric columns, 20 predictive categorical
columns, 30 short noise columns, 200 long numeric noise columns, and 200
long high-cardinality categorical noise columns. Missing values are injected
into several feature columns.

## End-to-end runs

Multiple full runs were executed against the same data and configuration
(15 Optuna TPE trials, train+valid refit, 70/15/15 train/valid/test split).

| Stage | Cold run (s) | Warm final run (s) |
| --- | ---: | ---: |
| Load labels / task inference | 0.070 | 0.070 |
| Schema and type inference | 2.751 | 2.787 |
| Chunked CSV → Parquet cache | 17.612 | 17.398 |
| Train/valid/test split | 0.007 | 0.007 |
| Feature statistics + filtering | 7.709 | 7.686 |
| Preprocessing fit | 0.312 | 0.319 |
| Matrix build | 0.293 | 0.294 |
| Optuna tuning + final fit | 21.718 | 20.758 |
| Evaluation | 0.029 | 0.029 |
| Reporting / plots | 9.955 | 0.390 |
| **Total** | **60.47** | **49.75** |

The cold run's reporting stage includes a one-time Matplotlib font-cache build.
The warm run reuses the cache (`runs/.matplotlib`) and is the more representative
pipeline-only timing. The 15 Optuna trials include 9 completed and 6 pruned
trials; the final model uses 292 boosting iterations. The exact peak RSS was
measured in the final run, which includes process-wide peak tracking.

## Resource usage

| Property | Value |
| --- | ---: |
| Peak observed RSS | 795.3 MB (832 MB max across repeated runs) |
| Parquet cache size | 111 MB |
| Selected / cached features | 41 / 504 |
| Matrix shape | 32,500 × 42 (dense float32) |

The 64 GB process limit is far above the observed peak; the row-chunked ingest,
column-block statistics, and selected-column matrix keep memory bounded.

## Metrics

| Metric | Valid (selection model) | Test (final model) |
| --- | ---: | ---: |
| Accuracy | 0.7241 | 0.7274 |
| Precision | 0.6817 | 0.6880 |
| Recall | 0.8418 | 0.8327 |
| F1 | 0.7533 | 0.7535 |
| AUC | 0.8224 | 0.8203 |
| KS | 0.4880 | 0.4864 |

Best TPE parameters:

```text
num_leaves=22
learning_rate=0.023993
min_data_in_leaf=15
feature_fraction=0.728035
bagging_fraction=0.914070
lambda_l1=6.27e-07
lambda_l2=0.000425
```

Top features by gain: `num_sig_000`, `num_sig_001`, `num_sig_002`,
`num_sig_003`, `cat_sig_000`, `cat_sig_001`.

## Artifacts

- Cold run: `runs/stress_500x32k/`
- Warm run: `runs/stress_500x32k_warm/`
- Final warm run with exact peak RSS: `runs/stress_500x32k_deliver/`
- Machine-readable timings: `runs/stress_500x32k_deliver/stage_timings.csv`
- Full metrics: `runs/stress_500x32k_deliver/reports/metrics.json`
- Filtering report: `runs/stress_500x32k_deliver/filter_report.csv`
- Trials: `runs/stress_500x32k_deliver/tuning/trials.csv`
- Predictions: `runs/stress_500x32k_deliver/reports/predictions_test.csv`
