# Auto ML Pipeline

A chunked, configuration-driven auto machine learning pipeline for wide tabular
data. It performs type inference, feature filtering, preprocessing, LightGBM
or XGBoost training with Optuna TPE tuning, and metrics reporting without
loading the full CSV dataset into memory.

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
   interchangeable model-adapter layer. LightGBM and XGBoost are built in.
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

## Reusing data artifacts across experiments

When you only change model or tuning settings, reuse the data-side stages from
an earlier run instead of rebuilding the cache, statistics, preprocessing, and
matrix:

```yaml
project:
  work_dir: runs/experiment_2
  reuse_from: runs/experiment_1
  reuse_mode: link        # link (default) or copy
```

`reuse_from` must point to a run created by this version of the pipeline. Every
run writes `reuse_state.json`, recording a fingerprint and completion flag for
each reusable stage. The pipeline reuses:

- cache plus schema metadata
- train/valid/test splits
- temporal PSI masks, when enabled
- filtering report and selected columns
- preprocessing state
- model matrix

Fingerprints cover the relevant data, filtering, and preprocessing config, input
file paths plus size/mtime, package versions, and the pipeline source files for
those stages. Filtering has separate fingerprints for metric computation and
final feature selection.

Changing a selection threshold such as `min_iv`, `max_iv`, `max_psi`,
`max_features`, or `max_pairwise_corr` reuses the expensive per-column report
and reapplies the current thresholds without recalculating NaN, uniqueness,
Pearson, IV, or PSI. Preprocessing and the matrix are restarted when the
selected columns change. Metric settings including `stats_sample_rows`,
`iv_bins`, `psi_bins`, `categorical_top_k`, and temporal PSI windows still force
metric recomputation. A change to an input CSV invalidates the cache and every
stage derived from it. If a threshold removes every column, the completed
metrics remain reusable so a relaxed threshold can be tried without repeating
feature statistics.

`reuse_mode: link` symlinks the cache and matrix directories and copies small
metadata, split, and filter artifacts. Keep the source run directory as long as
the new run uses it. `reuse_mode: copy` creates independent copies, at the cost
of duplicating potentially multi-GB cache and matrix files.

The mode also works as a resume mechanism: `reuse_from` may point to an
interrupted run. Stages marked complete are reused; stages not yet completed are
recomputed. `run_summary.json` and `stage_timings.csv` record which stages were
reused and from where.

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
  drop_unmatched_rows: false # true = exclude labels absent from every feature file
  column_block_size: 32
  sample_rows: 2000
  max_rows: null
  train_fraction: 0.7
  valid_fraction: 0.15
  test_fraction: 0.15
  csv:
    chunksize: 2000          # hard cap on rows per ingest chunk
    engine: pandas           # pandas (default) or arrow
    arrow_block_size: 67108864
    arrow_use_threads: true
    sep: ","                 # one character or an escape such as "\\x03"
    quoting: minimal         # minimal (default) or none
    quotechar: '"'
    doublequote: true
    escapechar: null
```

`feature_paths` accepts paths and globs. Relative paths are resolved against the
config directory first, then the current working directory. The label file may
contain extra feature columns; they are included in the model unless
`include_label_features` is false.

By default, label rows with no matching row in any feature file are retained
with NaN feature values. Set `data.drop_unmatched_rows: true` to exclude those
rows from filtering and every train/valid/test split. Extra columns in the label
file do not count as a feature-file match. The full cache remains keyed by the
label row order for reuse, but unmatched rows never enter training or
evaluation.

`csv.engine: arrow` streams each file through `pyarrow.csv` and writes the
column-blocked Parquet cache directly, without building a wide pandas frame.
This is substantially faster and uses less memory on tables with thousands of
columns, and it avoids the quadratic column-assignment path in pandas 1.x/2.0.
It requires UTF-8 input and gzip/bz2/zstd/lz4/brotli compression; `.zip` and
`.xz` files automatically fall back to the pandas engine. Numeric values are
parsed directly at `cache_numeric_dtype`, so `float32` caches avoid a float64
intermediate. Both engines produce the same cache schema and row coverage.

For 32k-column files, keep `chunksize` at roughly 1,000–5,000 rows. The Arrow
engine buffers `chunksize` rows across all retained columns, so memory is
approximately `chunksize × columns × 8` bytes plus Parquet writer buffers.

`sep` accepts a single character or a Python-style escape such as `"\\x03"`
(ASCII ETX) or `"\\t"`. If the exporter never quotes fields — common with
control-character delimiters — set `quoting: none`. A stray `"` is then kept as
data instead of opening a quoted field that consumes the rest of the file, which
is the usual cause of `EOF inside string`. Only disable quoting if the delimiter
cannot occur inside a field; otherwise re-export with proper quoting.

### Column vs row concatenation

- `concat_mode: column` joins feature files horizontally by the configured key.
  Duplicate feature names are prefixed with the source name when
  `deduplicate_columns` is true.
- `concat_mode: row` appends feature files vertically. All non-label feature
  files must have identical schemas.

### Out-of-time split

The label file can also provide a date column for an out-of-time holdout:

```yaml
data:
  split_strategy: temporal
  temporal_split:
    time_column: as_of_date
    date_format: "%Y-%m-%d"
    timezone: Asia/Shanghai
    oot_start: null          # explicit date wins when set
    oot_quantile: 0.5        # used when oot_start is null
    valid_fraction: 0.25     # latest portion of pre-cutoff rows
    min_positive_count: null # nullable; applies to each split
    max_positive_count: null # nullable; applies to each split
```

The split is:

- `train`: earliest rows before the OOT cutoff
- `valid`: latest `valid_fraction` of the pre-cutoff rows
- `test` / `oot`: all rows at or after the cutoff

The OOT period is used only for final evaluation. Filtering is restricted to
train + valid, and PSI compares train with valid. Preprocessing and CV also use
only pre-cutoff rows; the final model refits on train + valid. The time column
is excluded from model features by default. `predictions_oot.csv`,
`split_metadata.json`, and `metrics.json["evaluation"]` describe the OOT result.
The old global train/valid/test fractions are used only by the random split.

Changing the cutoff invalidates splits, filtering, preprocessing, and matrix,
but the cache remains reusable. Changing the time column or split strategy
invalidates the cache as well because it changes the feature set.

If temporal PSI is enabled, its reference and comparison windows must end
before the OOT cutoff; the pipeline rejects windows that would use OOT data for
feature filtering.

## Filtering configuration

```yaml
filtering:
  max_nan_ratio: 0.98
  min_unique: 2
  max_unique_ratio: 0.98
  min_abs_pearson: 0.01
  min_iv: 0.001
  max_iv: null                    # Reject IV > max_iv; null disables the limit
  max_psi: 0.5
  max_features: 500
  max_pairwise_corr: 0.98
  iv_bins: 10
  psi_bins: 10
  categorical_top_k: 50
  drop_identifiers: true
  keep_columns: []
  temporal_psi:
    enabled: false
    time_path: null
    time_key: null
    time_column: null
    date_format: null
    timezone: null
    reference_start: null
    reference_end: null
    comparison_start: null
    comparison_end: null
    min_rows_per_period: 100
```

- IV uses the training split and quantile bins for numeric columns or the top-K
  categories for categorical columns.
- `max_iv` excludes features whose IV is strictly greater than the threshold.
  This keeps unusually strong signals out of the model so they can be handled
  as candidate rules downstream. Rejected features remain in
  `filter_report.csv` with reason `max_iv`; `null` disables the limit.
- With temporal PSI disabled, PSI compares the random training and test splits.
- With temporal PSI enabled, PSI compares two configured time windows. Numeric
  bin edges and categorical vocabularies are derived from the reference period
  and then applied to the comparison period.
- Temporal windows use half-open intervals: `[start, end)`.
- `time_path` defaults to `data.label_path`. `time_key` defaults to
  `data.label_key`. The file must contain one unique key per label row and may
  be a label file or a separate CSV.
- `date_format` accepts a Python/pandas datetime format such as
  `"%Y-%m-%d %H:%M:%S"`. Set it to `null` for pandas mixed/ISO inference.
- The temporal masks and resolved window metadata are saved as
  `temporal_psi_windows.npz` and `temporal_psi_windows.json`.
- `filter_report.csv` records `psi_mode`, `psi_reference_n`, and
  `psi_comparison_n` for each feature.
- `max_pairwise_corr` optionally removes redundant numeric columns greedily,
  keeping the higher-scoring column from each correlated pair.
- The full per-column report is written to `filter_report.csv`.
- If filtering removes every column, `filter_report.csv` is still written
  before the pipeline fails, so the rejection reason for each column remains
  available for diagnosis.

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

`native` passes integer category codes to the selected tree model's categorical
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

### KS and cross-validation

Binary classification can tune directly for KS:

```yaml
tuning:
  enabled: true
  metric: ks
  direction: auto          # maximize
  n_trials: 30
  pruning: true
  cv:
    enabled: true
    folds: 5
    seed: 42
    prune_after_folds: 2
```

Each CV trial trains one model per fold, scores the fold with the chosen metric,
and returns the mean fold score. `best_iteration` for the final refit is the
median of the fold best iterations. `trials.csv` records `fold_scores` and
`fold_best_iterations` for each trial. `folds` must not exceed the smallest
training class count.

With CV enabled, Optuna pruning is evaluated on the running mean after each
fold. `prune_after_folds` controls when pruning may begin;
`tuning.n_warmup_steps` is used only when CV is disabled.
`tuning.timeout_seconds` is checked between trials, so one CV trial can consume
up to the cost of `folds` model fits.

The folds are built inside the existing training split. Cache, split, filtering,
preprocessing, and matrix fingerprints are unchanged by `tuning.cv`, so those
artifacts remain reusable across CV settings. Preprocessing and feature
selection are fitted on the full training split rather than independently per
fold; this is practical for model selection but not a strictly nested CV. A
fully nested design would require per-fold preprocessing and fold-specific
matrices.

Early stopping inside each model still uses the adapter's native evaluation
metric (`auc` for binary or `mlogloss` for multiclass). Trial selection uses
KS when `metric: ks`. This distinction matters if the AUC-optimal and
KS-optimal iteration counts differ. Without cross-validation, standard
LightGBM/XGBoost eval logs do not contain KS, so per-iteration Optuna pruning
is unavailable; fold-level pruning works when `tuning.cv.enabled` is true.

XGBoost is included as a second built-in adapter:

```yaml
model:
  name: xgboost
  search_space: search_spaces/xgboost.yaml
  num_boost_round: 1000
  early_stopping_rounds: 50
  num_threads: 4
```

The XGBoost adapter uses `tree_method: hist`, maps `num_threads` to `nthread`,
and uses `softprob` for multiclass probability output. Its search space tunes
`max_depth`, `min_child_weight`, `eta`, `subsample`, `colsample_bytree`,
`gamma`, `alpha`, and `reg_lambda`, with `max_depth` capped at 8 for cost
control. XGBoost Optuna pruning is supported through its training-callback API.

For a pinned legacy XGBoost stack, use Python 3.10 and a separate environment:

```bash
uv venv .venv-legacy --python 3.10
uv pip install \
  --python .venv-legacy/bin/python \
  -c constraints/legacy-xgboost.txt \
  -e .
```

The legacy constraints pin `xgboost==1.5.2`, `pandas==2.0.3`, and
`numpy<2`. XGBoost 1.5.2 predates NumPy 2 support, and `uv sync` uses the
project’s modern lock rather than these constraints. The adapter uses
`iteration_range`, `num_boosted_rounds`, categorical `DMatrix` options, and
training callbacks, all of which are available in XGBoost 1.5.2.

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

