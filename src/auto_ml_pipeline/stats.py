"""Streaming feature statistics and configurable feature filtering."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cache import CacheStore
from .logging_utils import Heartbeat, rss_mb
from .temporal import TemporalPSIWindows

EPS = 1e-9


@dataclass
class FilterResult:
    selected: List[str]
    report: pd.DataFrame
    stats: Dict[str, Dict[str, Any]]
    elapsed_seconds: float


def _numeric_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([-0.5, 0.5], dtype=np.float64)
    if np.nanmin(values) == np.nanmax(values):
        return np.asarray([float(values[0]) - 0.5, float(values[0]) + 0.5], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    edges = np.unique(np.nanquantile(values, quantiles))
    if edges.size < 2:
        return np.asarray([float(np.nanmin(values)) - 0.5, float(np.nanmax(values)) + 0.5])
    return edges.astype(np.float64, copy=False)


def _numeric_bin_ids(values: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.float64)
    n_bins = len(edges) - 1
    if n_bins <= 1:
        ids = np.zeros(len(values), dtype=np.int64)
    else:
        ids = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    ids[~np.isfinite(values)] = n_bins
    return ids, n_bins + 1


def _categorical_vocabulary(values: pd.Series, top_k: int) -> List[str]:
    counts = values.dropna().astype(str).value_counts()
    return [str(value) for value in counts.head(max(1, int(top_k))).index]


def _categorical_bin_ids(values: pd.Series, vocabulary: Sequence[str]) -> Tuple[np.ndarray, int]:
    n_known = len(vocabulary)
    other_code = n_known
    missing_code = n_known + 1
    id_map = {value: index for index, value in enumerate(vocabulary)}
    text = values.astype("object")
    known = text.isin(set(vocabulary))
    ids = np.full(len(values), other_code, dtype=np.int64)
    ids[text.isna().to_numpy()] = missing_code
    if known.any():
        ids[known.to_numpy()] = text[known].map(id_map).to_numpy(dtype=np.int64)
    return ids, n_known + 2


def _information_value(bad_counts: np.ndarray, good_counts: np.ndarray) -> float:
    n_bins = len(bad_counts)
    bad_total = float(np.sum(bad_counts))
    good_total = float(np.sum(good_counts))
    if bad_total <= 0 or good_total <= 0:
        return float("nan")
    smoothing = 0.5
    bad_rate = (bad_counts.astype(np.float64) + smoothing) / (bad_total + smoothing * n_bins)
    good_rate = (good_counts.astype(np.float64) + smoothing) / (good_total + smoothing * n_bins)
    return float(abs(np.sum((bad_rate - good_rate) * np.log(bad_rate / good_rate))))


def _psi(reference_counts: np.ndarray, comparison_counts: np.ndarray) -> float:
    ref_total = float(np.sum(reference_counts))
    cmp_total = float(np.sum(comparison_counts))
    if ref_total <= 0 or cmp_total <= 0:
        return float("nan")
    reference = reference_counts.astype(np.float64) / ref_total
    comparison = comparison_counts.astype(np.float64) / cmp_total
    mask = (reference > 0) | (comparison > 0)
    if not mask.any():
        return float("nan")
    ref = reference[mask] + EPS
    cmp = comparison[mask] + EPS
    return float(np.sum((ref - cmp) * np.log(ref / cmp)))


def _evenly_spaced_indices(
    indices: np.ndarray,
    limit: Optional[int],
) -> np.ndarray:
    if limit is None or len(indices) <= int(limit):
        return indices
    positions = np.linspace(
        0,
        len(indices) - 1,
        max(1, int(limit)),
        dtype=np.int64,
    )
    return indices[positions]


def _pearson_numeric(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _correlation_with_target(
    values: np.ndarray,
    y: np.ndarray,
    n_classes: int,
) -> float:
    if n_classes == 2:
        return _pearson_numeric(values, y.astype(np.float64))
    correlations = [
        _pearson_numeric(values, (y == class_index).astype(np.float64)) for class_index in range(min(n_classes, 10))
    ]
    finite = [abs(value) for value in correlations if np.isfinite(value)]
    return float(max(finite)) if finite else float("nan")


def _iv_for_bins(
    bin_ids: np.ndarray,
    y_train: np.ndarray,
    n_bins: int,
    n_classes: int,
) -> float:
    if len(y_train) == 0:
        return float("nan")
    flat = bin_ids.astype(np.int64) * max(1, n_classes) + y_train.astype(np.int64)
    counts = np.bincount(flat, minlength=n_bins * n_classes).reshape(n_bins, n_classes)
    if n_classes == 2:
        return _information_value(counts[:, 1], counts[:, 0])
    values = []
    for class_index in range(n_classes):
        bad = counts[:, class_index]
        good = counts.sum(axis=1) - bad
        value = _information_value(bad, good)
        if np.isfinite(value):
            values.append(value)
    return float(max(values)) if values else float("nan")


def compute_stats_and_filter(
    store: CacheStore,
    y: np.ndarray,
    splits: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    logger: Any,
    heartbeat: Heartbeat,
    temporal_windows: Optional[TemporalPSIWindows] = None,
) -> FilterResult:
    started = time.perf_counter()
    fcfg = cfg["filtering"]
    data_cfg = cfg["data"]
    block_size = int(data_cfg.get("column_block_size", 32))
    sample_rows = int(data_cfg.get("sample_rows", 2000))
    n_rows = store.n_rows
    n_classes = int(cfg["_runtime"]["n_classes"])
    train_mask = np.zeros(n_rows, dtype=bool)
    test_mask = np.zeros(n_rows, dtype=bool)
    train_mask[splits["train"]] = True
    test_mask[splits["test"]] = True
    y_train = y[train_mask]
    y_test = y[test_mask]
    columns = store.columns()
    report_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, Any]] = {}
    if temporal_windows is not None:
        if len(temporal_windows.reference_mask) != n_rows:
            raise ValueError("Temporal PSI reference mask does not match cached rows.")
        if len(temporal_windows.comparison_mask) != n_rows:
            raise ValueError("Temporal PSI comparison mask does not match cached rows.")
        psi_limit = int(fcfg["stats_sample_rows"]) if fcfg.get("stats_sample_rows") is not None else None
        psi_reference_indices = _evenly_spaced_indices(
            np.flatnonzero(temporal_windows.reference_mask),
            psi_limit,
        )
        psi_comparison_indices = _evenly_spaced_indices(
            np.flatnonzero(temporal_windows.comparison_mask),
            psi_limit,
        )
    else:
        psi_reference_indices = np.asarray([], dtype=np.int64)
        psi_comparison_indices = np.asarray([], dtype=np.int64)

    logger.info(
        "[filter] computing stats for %s columns in blocks of %s (sample=%s rows)",
        len(columns),
        block_size,
        sample_rows,
    )
    for block_number, (block, frame) in enumerate(
        store.iter_column_blocks(columns=columns, block_size=block_size), start=1
    ):
        block_started = time.perf_counter()
        stats_limit = fcfg.get("stats_sample_rows")
        if stats_limit is not None and len(frame) > int(stats_limit):
            sample_indices = np.linspace(0, len(frame) - 1, int(stats_limit), dtype=np.int64)
            block_frame = frame.iloc[sample_indices].reset_index(drop=True)
            block_train_mask = train_mask[sample_indices]
            block_test_mask = test_mask[sample_indices]
            block_y_train = y[sample_indices][block_train_mask]
        else:
            block_frame = frame
            block_train_mask = train_mask
            block_test_mask = test_mask
            block_y_train = y_train
        temporal_reference_frame = frame.iloc[psi_reference_indices] if temporal_windows is not None else None
        temporal_comparison_frame = frame.iloc[psi_comparison_indices] if temporal_windows is not None else None
        temporal_reference_sample = (
            frame.iloc[_evenly_spaced_indices(psi_reference_indices, sample_rows)]
            if temporal_windows is not None
            else None
        )
        block_n_rows = len(block_frame)
        sample_frame = block_frame.iloc[: min(sample_rows, block_n_rows)]
        for column in block:
            series = block_frame[column]
            dtype = store.dtype(column)
            non_null = series.notna()
            n_non_null = int(non_null.sum())
            nan_ratio = float(1.0 - (n_non_null / max(1, block_n_rows)))
            nunique = int(series.nunique(dropna=True))
            unique_ratio = float(nunique / max(1, n_non_null))
            sample_values = sample_frame[column]
            pearson = float("nan")
            iv_value = float("nan")
            psi_value = float("nan")
            psi_reference_n = 0
            psi_comparison_n = 0
            n_used_bins = 0
            n_iv_bins = 0
            method = "none"

            if dtype in {"numeric", "datetime"}:
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
                sample_numeric = pd.to_numeric(sample_values, errors="coerce").to_numpy(dtype=np.float64)
                if temporal_windows is not None:
                    reference_values = pd.to_numeric(temporal_reference_frame[column], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    comparison_values = pd.to_numeric(temporal_comparison_frame[column], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    reference_sample_values = pd.to_numeric(
                        temporal_reference_sample[column], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    edges = _numeric_bin_edges(
                        reference_sample_values,
                        int(fcfg.get("psi_bins", 10)),
                    )
                    reference_bins, n_used_bins = _numeric_bin_ids(reference_values, edges)
                    comparison_bins, _ = _numeric_bin_ids(comparison_values, edges)
                    psi_reference_n = len(reference_values)
                    psi_comparison_n = len(comparison_values)
                    ref_counts = np.bincount(reference_bins, minlength=n_used_bins)
                    cmp_counts = np.bincount(comparison_bins, minlength=n_used_bins)
                    psi_value = _psi(ref_counts, cmp_counts)
                    iv_edges = _numeric_bin_edges(sample_numeric, int(fcfg.get("iv_bins", 10)))
                    bin_ids, n_iv_bins = _numeric_bin_ids(values, iv_edges)
                else:
                    edges = _numeric_bin_edges(sample_numeric, int(fcfg.get("iv_bins", 10)))
                    bin_ids, n_used_bins = _numeric_bin_ids(values, edges)
                    n_iv_bins = n_used_bins
                    psi_reference_n = int(block_train_mask.sum())
                    psi_comparison_n = int(block_test_mask.sum())
                    if block_test_mask.any():
                        ref_counts = np.bincount(bin_ids[block_train_mask], minlength=n_used_bins)
                        cmp_counts = np.bincount(bin_ids[block_test_mask], minlength=n_used_bins)
                        psi_value = _psi(ref_counts, cmp_counts)
                pearson = _correlation_with_target(values[block_train_mask], block_y_train, n_classes)
                iv_value = _iv_for_bins(bin_ids[block_train_mask], block_y_train, n_iv_bins, n_classes)
                method = "quantile_bins_temporal" if temporal_windows is not None else "quantile_bins"
            elif dtype in {"categorical", "identifier"}:
                if temporal_windows is not None:
                    reference_values = temporal_reference_frame[column]
                    comparison_values = temporal_comparison_frame[column]
                    vocabulary = _categorical_vocabulary(
                        temporal_reference_sample[column],
                        int(fcfg.get("categorical_top_k", 50)),
                    )
                    reference_bins, n_used_bins = _categorical_bin_ids(reference_values, vocabulary)
                    comparison_bins, _ = _categorical_bin_ids(comparison_values, vocabulary)
                    psi_reference_n = len(reference_values)
                    psi_comparison_n = len(comparison_values)
                    ref_counts = np.bincount(reference_bins, minlength=n_used_bins)
                    cmp_counts = np.bincount(comparison_bins, minlength=n_used_bins)
                    psi_value = _psi(ref_counts, cmp_counts)
                    iv_vocabulary = _categorical_vocabulary(sample_values, int(fcfg.get("categorical_top_k", 50)))
                    bin_ids, n_iv_bins = _categorical_bin_ids(series, iv_vocabulary)
                else:
                    vocabulary = _categorical_vocabulary(sample_values, int(fcfg.get("categorical_top_k", 50)))
                    bin_ids, n_used_bins = _categorical_bin_ids(series, vocabulary)
                    n_iv_bins = n_used_bins
                    psi_reference_n = int(block_train_mask.sum())
                    psi_comparison_n = int(block_test_mask.sum())
                    if block_test_mask.any():
                        ref_counts = np.bincount(bin_ids[block_train_mask], minlength=n_used_bins)
                        cmp_counts = np.bincount(bin_ids[block_test_mask], minlength=n_used_bins)
                        psi_value = _psi(ref_counts, cmp_counts)
                iv_value = _iv_for_bins(bin_ids[block_train_mask], block_y_train, n_iv_bins, n_classes)
                method = "category_top_k_temporal" if temporal_windows is not None else "category_top_k"

            row = {
                "column": column,
                "dtype": dtype,
                "nan_ratio": nan_ratio,
                "n_unique": nunique,
                "unique_ratio": unique_ratio,
                "pearson": pearson,
                "abs_pearson": abs(pearson) if np.isfinite(pearson) else float("nan"),
                "iv": iv_value,
                "psi": psi_value,
                "psi_mode": ("temporal" if temporal_windows is not None else "train_test"),
                "psi_reference_n": psi_reference_n,
                "psi_comparison_n": psi_comparison_n,
                "bins": n_used_bins,
                "iv_bins": n_iv_bins,
                "binning": method,
            }
            report_rows.append(row)
            stats[column] = row

        elapsed_block = time.perf_counter() - block_started
        logger.info(
            "[filter] block %s/%s: %s columns in %.2fs rss=%.0fMiB",
            block_number,
            math.ceil(len(columns) / block_size),
            len(block),
            elapsed_block,
            rss_mb(),
        )
        heartbeat.detail(
            f"filter block={block_number}/{math.ceil(len(columns) / block_size)} "
            f"columns_done={min(block_number * block_size, len(columns))}"
        )

    report = pd.DataFrame(report_rows)
    keep_columns = {str(column) for column in fcfg.get("keep_columns", [])}
    selected: List[str] = []
    reasons: Dict[str, str] = {}

    for _, row in report.iterrows():
        column = str(row["column"])
        dtype = str(row["dtype"])
        fail_reasons: List[str] = []
        if column in keep_columns:
            selected.append(column)
            reasons[column] = "kept_by_config"
            continue
        if dtype == "identifier" and fcfg.get("drop_identifiers", True):
            fail_reasons.append("identifier")
        if row["nan_ratio"] > float(fcfg.get("max_nan_ratio", 0.98)):
            fail_reasons.append("nan_ratio")
        if row["n_unique"] < int(fcfg.get("min_unique", 2)):
            fail_reasons.append("min_unique")
        if dtype in {"categorical", "identifier"} and row["unique_ratio"] > float(fcfg.get("max_unique_ratio", 0.98)):
            fail_reasons.append("unique_ratio")
        min_abs_pearson = float(fcfg.get("min_abs_pearson", 0.0))
        if min_abs_pearson > 0 and np.isfinite(row["abs_pearson"]) and row["abs_pearson"] < min_abs_pearson:
            fail_reasons.append("pearson")
        min_iv = float(fcfg.get("min_iv", 0.0))
        if min_iv > 0 and np.isfinite(row["iv"]) and row["iv"] < min_iv:
            fail_reasons.append("iv")
        max_psi = float(fcfg.get("max_psi", 0.5))
        if np.isfinite(row["psi"]) and row["psi"] > max_psi:
            fail_reasons.append("psi")
        if fail_reasons:
            reasons[column] = ",".join(fail_reasons)
        else:
            selected.append(column)
            reasons[column] = "passed"

    max_features = fcfg.get("max_features")
    if max_features is not None and int(max_features) > 0 and len(selected) > int(max_features):
        ranked = sorted(
            selected,
            key=lambda column: _selection_score(stats[column]),
            reverse=True,
        )
        dropped = set(ranked[int(max_features) :])
        for column in dropped:
            reasons[column] = "max_features"
        selected = ranked[: int(max_features)]
        logger.info(
            "[filter] max_features=%s truncated selection from %s to %s columns",
            max_features,
            len(ranked),
            len(selected),
        )

    max_pairwise = fcfg.get("max_pairwise_corr")
    if max_pairwise is not None and float(max_pairwise) > 0 and len(selected) > 1:
        selected, pairwise_dropped = _deduplicate_correlated(
            store,
            selected,
            stats,
            float(max_pairwise),
            sample_rows,
            logger,
        )
        reasons.update(pairwise_dropped)

    report["selected"] = report["column"].map(lambda column: column in set(selected))
    report["reason"] = report["column"].map(lambda column: reasons.get(column, "not_evaluated"))
    report["selection_score"] = report["column"].map(lambda column: _selection_score(stats.get(str(column), {})))
    report = report.sort_values(["selected", "selection_score"], ascending=[False, False]).reset_index(drop=True)

    elapsed = time.perf_counter() - started
    logger.info(
        "[filter] selected %s/%s columns (%.1f%%) in %.2fs",
        len(selected),
        len(report),
        100.0 * len(selected) / max(1, len(report)),
        elapsed,
    )
    if len(selected) == 0:
        raise ValueError("Feature filtering removed every column. Relax filtering thresholds.")
    return FilterResult(selected=selected, report=report, stats=stats, elapsed_seconds=elapsed)


def _selection_score(row: Dict[str, Any]) -> float:
    pearson = row.get("abs_pearson", float("nan"))
    iv = row.get("iv", float("nan"))
    psi = row.get("psi", float("nan"))
    score = 0.0
    if pearson is not None and np.isfinite(pearson):
        score = max(score, float(pearson))
    if iv is not None and np.isfinite(iv):
        score = max(score, float(iv))
    if score == 0.0 and psi is not None and np.isfinite(psi):
        score = 1.0 / (1.0 + float(psi))
    return score


def _deduplicate_correlated(
    store: CacheStore,
    selected: List[str],
    stats: Dict[str, Dict[str, Any]],
    threshold: float,
    sample_rows: int,
    logger: Any,
) -> Tuple[List[str], Dict[str, str]]:
    numeric = [column for column in selected if store.dtype(column) in {"numeric", "datetime"}]
    if len(numeric) <= 1:
        return selected, {}
    sample = store.sample(numeric, n_rows=min(sample_rows, 5000))
    sample = sample.apply(pd.to_numeric, errors="coerce")
    corr = sample.corr().abs()
    ordered = sorted(numeric, key=lambda column: _selection_score(stats[column]), reverse=True)
    kept_numeric: List[str] = []
    dropped: Dict[str, str] = {}
    for column in ordered:
        if kept_numeric:
            max_corr = float(corr.loc[column, kept_numeric].max())
            if max_corr > threshold:
                dropped[column] = f"pairwise_correlation({max_corr:.3f})"
                continue
        kept_numeric.append(column)
    kept_set = set(kept_numeric)
    result = [column for column in selected if store.dtype(column) not in {"numeric", "datetime"} or column in kept_set]
    logger.info(
        "[filter] pairwise correlation > %.3f removed %s numeric columns",
        threshold,
        len(numeric) - len(kept_numeric),
    )
    return result, dropped
