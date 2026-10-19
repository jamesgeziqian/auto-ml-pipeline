# Auto ML Pipeline

A chunked, configuration-driven auto machine learning pipeline for wide tabular
data. It performs type inference, feature filtering, preprocessing, LightGBM
training with Optuna TPE tuning, and metrics reporting without loading the full
CSV dataset into memory.

The implementation follows `implementation_plan.md`.

## Pipeline stages

1. **Labels and task inference** – stream the label file, validate the join key,
   detect binary/multiclass classification, and build train/valid/test splits.
2. **Schema inference** – sample each CSV and infer numeric, categorical,
   datetime, or identifier columns.
3. **Chunked ingest** – read rows in bounded chunks, align each source by key,
   and write a column-prunable Parquet cache. No stage loads all CSV columns for
   all rows into memory.
4. **Feature filtering** – compute NaN ratio, unique count/ratio, Pearson
   correlation with the target, information value (IV), and population stability
   index (PSI), then apply configurable thresholds.
5. **Preprocessing fit** – fit numeric imputers, categorical imputers/encoders,
   missing indicators, and optional clipping on the training split only.
6. **Matrix build** – materialize only the selected columns into a dense float32
   memmap or a sparse CSR matrix.
7. **Tuning and training** – Optuna TPE over a YAML search space, with an
   interchangeable model-adapter layer. LightGBM is the built-in adapter.
8. **Evaluation and reports** – accuracy, precision, recall, F1, AUC, KS,
   log loss, Brier score, confusion matrix, feature importance, predictions,
   plots, and per-stage timings.

A background heartbeat logs RSS/CPU usage during long stages, and every stage
prints START/END timing.

## Install

```bash
uv sync --extra dev
```

On macOS, LightGBM also needs the OpenMP runtime:

```bash
brew install libomp
```

The commands below use `.venv/bin/auto-ml` directly, which avoids `uv run`
needing to refresh its global cache on every invocation.

## Quick start

Generate a tiny dataset and run the smoke configuration:

```bash
.venv/bin/auto-ml generate-data \
  --out data/smoke --rows 1200 --cols 40 --target-gb 0 \
  --parts 2 --pad-len 8

.venv/bin/auto-ml run --config configs/smoke.yaml
```

Run the tests:

```bash
.venv/bin/pytest
```

## Wide-data stress test

The `configs/stress_500x32k.yaml` example expects the 32,500-row × 500-column
dataset generated with:

```bash
.venv/bin/auto-ml generate-data \
  --out data/stress --rows 32500 --cols 500 --target-gb 3.5 \
  --parts 4 --heartbeat-seconds 10

.venv/bin/auto-ml run --config configs/stress_500x32k.yaml
```

The generator writes four feature CSVs that are horizontally concatenated on
`customer_id`, plus a label file containing the target and four extra label
features. The feature mix includes predictive numeric/categorical columns,
missing values, short noise, and long padded values that raise the on-disk size
to the requested 3–4 GB range. Long padded columns are deliberately weak so the
filtering stage has something realistic to remove.

Stage timings are printed to stdout and saved to:

```text
runs/<name>/stage_timings.csv
runs/<name>/run_summary.json
runs/<name>/run.log
```

The measured run for the committed `configs/stress_500x32k.yaml` used a
3.51 GB (3.27 GiB) CSV dataset with 32,500 rows and 500 features. It completed
in 60.5 s cold and 49.7 s warm, with peak RSS of 795.3 MB, filtering
504 cached columns down to 41, and test AUC 0.8203. Full stage-by-stage results
and artifact locations are in [BENCHMARK.md](BENCHMARK.md).

For row-wise concatenation with 32,000 columns per file, use
`configs/stress_row_32k.yaml`:

```bash
.venv/bin/auto-ml generate-wide \
  --out data/stress_row_32k \
  --rows-per-file 50000 --files 4 --cols 32000 \
  --signal-cols 20 --pattern-pool 256

.venv/bin/auto-ml run --config configs/stress_row_32k.yaml
```

That test produced a 12.81 GB dataset with 200,000 total rows and 6.4 billion
cells. It completed in 22.1 minutes with peak RSS of 4.34 GB; see
[BENCHMARK_ROW_32K.md](BENCHMARK_ROW_32K.md). Delete `data/stress_row_32k` and
the run's `cache/` and `matrix/` directories afterward to reclaim the space.

## Data configuration

```yaml
data:
  feature_paths:
    - data/stress/features_part_*.csv
  label_path: data/stress/labels.csv
  feature_key: customer_id
  label_key: customer_id
  label_column: target
  concat_mode: column        # column = horizontal join by key
                             # row    = append files with identical schemas
  include_label_features: true
  column_block_size: 32
  sample_rows: 2000
  max_rows: null
  train_fraction: 0.7
  valid_fraction: 0.15
  test_fraction: 0.15
```

`feature_paths` accepts paths and globs. Relative paths are resolved against the
config directory first, then the current working directory. The label file may
contain extra feature columns; they are included in the model unless
`include_label_features` is false.

### Column vs row concatenation

- `concat_mode: column` joins feature files horizontally by the configured key.
  Duplicate feature names are prefixed with the source name when
  `deduplicate_columns` is true.
- `concat_mode: row` appends feature files vertically. All non-label feature
  files must have identical schemas.

## Filtering configuration

```yaml
filtering:
  max_nan_ratio: 0.98
  min_unique: 2
  max_unique_ratio: 0.98
  min_abs_pearson: 0.01
  min_iv: 0.001
  max_psi: 0.5
  max_features: 500
  max_pairwise_corr: 0.98
  iv_bins: 10
  categorical_top_k: 50
  drop_identifiers: true
  keep_columns: []
```

- IV uses the training split and quantile bins for numeric columns or the top-K
  categories for categorical columns.
- PSI compares the training and test split distributions.
- `max_pairwise_corr` optionally removes redundant numeric columns greedily,
  keeping the higher-scoring column from each correlated pair.
- The full per-column report is written to `filter_report.csv`.

## Preprocessing configuration

```yaml
preprocessing:
  numeric_impute: median         # median | mean | zero | constant
  categorical_impute: constant   # mode | constant
  categorical_constant: "__MISSING__"
  add_missing_indicator: true
  categorical_encoding: native   # native | onehot | ordinal
  max_categories: 64
  min_category_frequency: 5
  clip_numeric: quantile
  clip_quantiles: [0.001, 0.999]
  max_output_features: 5000
```

`native` passes integer category codes to LightGBM's categorical feature
handling, avoiding an unbounded one-hot expansion. `onehot` builds a sparse CSR
matrix; `ordinal` uses ordered integer codes.

## Tuning and model configuration

```yaml
model:
  name: lightgbm
  search_space: search_spaces/lightgbm.yaml
  num_boost_round: 1500
  early_stopping_rounds: 60
  num_threads: 0

tuning:
  enabled: true
  n_trials: 15
  direction: auto
  metric: auto
  sampler:
    n_startup_trials: 8
    seed: 42
  pruning: true
  refit_on_train_valid: true
```

Search spaces are plain YAML:

```yaml
fixed:
  max_depth: -1
parameters:
  num_leaves: {type: int, low: 15, high: 255, log: true}
  learning_rate: {type: float, low: 0.01, high: 0.2, log: true}
  feature_fraction: {type: float, low: 0.5, high: 1.0}
```

Binary tasks optimize validation AUC by default; multiclass tasks optimize
multiclass log loss. The final model is refit on train+valid, and the selection
model remains available for an honest out-of-sample validation score.

### Adding a model adapter

Implement `BaseModelAdapter` in `src/auto_ml_pipeline/models/base.py` and
register it:

```python
from auto_ml_pipeline.models import register_model
from auto_ml_pipeline.models.base import BaseModelAdapter

@register_model("my_model")
class MyModelAdapter(BaseModelAdapter):
    ...
```

The rest of the pipeline is model-agnostic; select it with `model.name:
my_model` and provide a matching search space if tuning is enabled.

## Memory behavior

- CSV files are read with a bounded row chunksize.
- Each chunk is aligned by key and written to Parquet immediately.
- Statistics and preprocessing read only one column block at a time.
- Only selected columns are materialized into the model matrix. `max_features`
  and `max_output_features` put hard bounds on the final width.
- Dense matrices use a float32 memmap; one-hot matrices use CSR.
- `psutil` heartbeat logs RSS and CPU every `project.heartbeat_seconds`.

The target server rule of 64 GB is therefore respected by design; the stress
run's observed peak is recorded in `run_summary.json`.

For a layout with roughly 32,000 columns, keep the feature store split into
many CSVs (for example 64 files of ~500 columns each). The cache and filtering
stages operate on 32-column blocks, and `max_features` bounds the final model
matrix; a single extremely wide CSV is also supported through projected column
reads, but re-reading that file once per column block is intentionally slower.

## Output artifacts

```text
runs/<name>/
  run.log
  run_summary.json
  stage_timings.csv
  labels.npy / keys.parquet / splits.npz
  schema/sources.json
  cache/manifest.json
  filter_report.csv
  selected_features.json
  preprocess_state.json
  matrix/matrix_meta.json
  tuning/trials.csv
  tuning/best_params.json
  model/model.txt
  reports/metrics.json
  reports/predictions_test.csv
  reports/feature_importance_by_source.csv
  reports/roc_curve.png / ks_curve.png / feature_importance.png
```

