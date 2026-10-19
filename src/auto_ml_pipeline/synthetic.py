"""Generate a large synthetic multi-file dataset for pipeline stress testing."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging_utils import Heartbeat, configure_logging, format_bytes


@dataclass
class FeatureSpec:
    name: str
    kind: str
    index: int


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _numeric_strings(values: np.ndarray) -> List[str]:
    array = np.asarray(values, dtype=np.float64)
    return ["" if not np.isfinite(value) else f"{value:.6f}" for value in array]


def _categorical_strings(codes: np.ndarray, labels: Sequence[str]) -> List[str]:
    result: List[str] = []
    for code in np.asarray(codes):
        if code < 0:
            result.append("")
        else:
            result.append(labels[int(code)])
    return result


def generate_synthetic_dataset(
    out_dir: str | Path,
    rows: int = 32_500,
    n_features: int = 500,
    target_gb: float = 3.5,
    n_feature_files: int = 4,
    seed: int = 42,
    chunk_rows: int = 1000,
    pad_len: int | None = None,
    heartbeat_seconds: float = 15.0,
    logger: Any = None,
) -> Dict[str, Any]:
    """Write feature CSVs that horizontally concatenate to one wide table.

    The file size is inflated with long numeric/categorical values so the stress
    test exercises real multigigabyte IO while keeping a subset of predictive
    columns for model training.
    """

    logger = logger or configure_logging("INFO")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    n_signal_numeric = min(50, max(5, n_features // 10))
    n_signal_categorical = min(20, max(3, n_features // 25))
    n_short_noise = min(30, max(3, n_features // 16))
    remaining = n_features - n_signal_numeric - n_signal_categorical - n_short_noise
    if remaining < 0:
        n_short_noise = max(2, n_short_noise + remaining)
        remaining = n_features - n_signal_numeric - n_signal_categorical - n_short_noise
    n_wide_numeric = remaining // 2
    n_wide_categorical = remaining - n_wide_numeric

    if pad_len is None:
        if target_gb and target_gb > 0 and remaining > 0:
            bytes_per_cell = (float(target_gb) * 1_000_000_000) / max(1, rows * remaining)
            pad_len = max(10, int(bytes_per_cell) - 18)
        else:
            pad_len = 12

    logger.info(
        "[generate] target: %s rows x %s features (~%.2f GB, pad=%s)",
        f"{rows:,}",
        n_features,
        target_gb,
        pad_len,
    )
    logger.info(
        "[generate] feature mix: %s signal-numeric, %s signal-categorical, "
        "%s short-noise, %s wide-numeric, %s wide-categorical",
        n_signal_numeric,
        n_signal_categorical,
        n_short_noise,
        n_wide_numeric,
        n_wide_categorical,
    )

    signal_numeric = rng.normal(size=(n_signal_numeric, rows)).astype(np.float64)
    signal_categorical = rng.integers(
        0, 12, size=(n_signal_categorical, rows), dtype=np.int16
    )
    category_labels = [
        [f"cat_{column:02d}_v{value:02d}" for value in range(12)]
        for column in range(n_signal_categorical)
    ]

    # Missing values on a couple of otherwise predictive columns.
    if n_signal_numeric > 6:
        signal_numeric[5, rng.random(rows) < 0.015] = np.nan
        signal_numeric[6, rng.random(rows) < 0.010] = np.nan
    if n_signal_categorical > 5:
        signal_categorical[5, rng.random(rows) < 0.010] = -1

    logits = (
        1.15 * signal_numeric[0]
        - 0.95 * signal_numeric[1]
        + 0.70 * signal_numeric[2]
        + 0.45 * signal_numeric[3]
        + 0.30 * (signal_categorical[0] % 4 - 1.5)
        - 0.22 * (signal_categorical[1] % 3 - 1.0)
        + rng.normal(scale=0.65, size=rows)
    )
    probability = _sigmoid(logits)
    target = rng.binomial(1, probability).astype(np.int8)
    wide_ids = rng.permutation(rows)

    specs: List[FeatureSpec] = []
    for index in range(n_signal_numeric):
        specs.append(FeatureSpec(f"num_sig_{index:03d}", "signal_numeric", index))
    for index in range(n_signal_categorical):
        specs.append(
            FeatureSpec(f"cat_sig_{index:03d}", "signal_categorical", index)
        )
    for index in range(n_short_noise):
        specs.append(FeatureSpec(f"num_noise_{index:03d}", "short_noise", index))
    for index in range(n_wide_numeric):
        specs.append(FeatureSpec(f"wide_num_{index:03d}", "wide_numeric", index))
    for index in range(n_wide_categorical):
        specs.append(
            FeatureSpec(f"wide_cat_{index:03d}", "wide_categorical", index)
        )
    specs = specs[:n_features]

    n_files = max(1, min(int(n_feature_files), len(specs)))
    file_specs: List[List[FeatureSpec]] = [[] for _ in range(n_files)]
    for index, spec in enumerate(specs):
        file_specs[index % n_files].append(spec)

    key_column = "customer_id"
    keys = np.asarray([f"cust_{index:09d}" for index in range(rows)], dtype=object)
    feature_paths: List[str] = []
    total_bytes = 0
    heartbeat = Heartbeat(logger, heartbeat_seconds)
    with heartbeat:
        heartbeat.set_stage("generate_features", f"rows={rows}")
        for file_index, assigned in enumerate(file_specs):
            path = out / f"features_part_{file_index:03d}.csv"
            feature_paths.append(str(path))
            file_started = time.perf_counter()
            with path.open("w", encoding="utf-8", newline="") as handle:
                header = [key_column] + [spec.name for spec in assigned]
                handle.write(",".join(header) + "\n")
                for start in range(0, rows, chunk_rows):
                    stop = min(rows, start + chunk_rows)
                    chunk_size = stop - start
                    data: Dict[str, Any] = {
                        key_column: keys[start:stop].tolist(),
                    }
                    row_numbers = np.arange(start, stop)
                    for spec in assigned:
                        if spec.kind == "signal_numeric":
                            data[spec.name] = _numeric_strings(
                                signal_numeric[spec.index, start:stop]
                            )
                        elif spec.kind == "signal_categorical":
                            data[spec.name] = _categorical_strings(
                                signal_categorical[spec.index, start:stop],
                                category_labels[spec.index],
                            )
                        elif spec.kind == "short_noise":
                            values = rng.normal(size=chunk_size)
                            data[spec.name] = _numeric_strings(values)
                        elif spec.kind == "wide_numeric":
                            prefix = f"0.{spec.index:04d}"
                            filler = "7" * pad_len
                            data[spec.name] = [
                                f"{prefix}{wide_ids[row]:08d}{filler}"
                                for row in row_numbers
                            ]
                        else:
                            prefix = f"id{spec.index:04d}_"
                            filler = "z" * pad_len
                            data[spec.name] = [
                                f"{prefix}{wide_ids[row]:08d}_{filler}"
                                for row in row_numbers
                            ]
                    frame = pd.DataFrame(data)
                    frame.to_csv(
                        handle, header=False, index=False, lineterminator="\n"
                    )
                    if (start // chunk_rows) % 5 == 0:
                        elapsed = max(1e-9, time.perf_counter() - file_started)
                        logger.info(
                            "[generate] file %s/%s rows=%s/%s written=%s speed=%.0f rows/s",
                            file_index + 1,
                            n_files,
                            f"{stop:,}",
                            f"{rows:,}",
                            format_bytes(handle.tell()),
                            stop / elapsed,
                        )
                        heartbeat.detail(
                            f"file={file_index + 1}/{n_files} rows={stop}/{rows}"
                        )
            size = path.stat().st_size
            total_bytes += size
            logger.info(
                "[generate] wrote %s (%s, %s columns) in %.1fs",
                path.name,
                format_bytes(size),
                len(assigned),
                time.perf_counter() - file_started,
            )

        heartbeat.set_stage("generate_labels", f"rows={rows}")
        labels_path = out / "labels.csv"
        label_started = time.perf_counter()
        with labels_path.open("w", encoding="utf-8", newline="") as handle:
            extra_categories = category_labels[2] if n_signal_categorical > 2 else category_labels[0]
            header = [
                key_column,
                "target",
                "label_feat_signal",
                "label_feat_category",
                "label_feat_noise_0",
                "label_feat_noise_1",
            ]
            handle.write(",".join(header) + "\n")
            for start in range(0, rows, chunk_rows):
                stop = min(rows, start + chunk_rows)
                label_signal = (
                    signal_numeric[4, start:stop] + rng.normal(scale=0.08, size=stop - start)
                    if n_signal_numeric > 4
                    else rng.normal(size=stop - start)
                )
                label_codes = (
                    signal_categorical[4, start:stop]
                    if n_signal_categorical > 4
                    else signal_categorical[0, start:stop]
                )
                frame = pd.DataFrame(
                    {
                        key_column: keys[start:stop],
                        "target": target[start:stop],
                        "label_feat_signal": _numeric_strings(label_signal),
                        "label_feat_category": _categorical_strings(
                            label_codes, extra_categories
                        ),
                        "label_feat_noise_0": _numeric_strings(
                            rng.normal(size=stop - start)
                        ),
                        "label_feat_noise_1": _numeric_strings(
                            rng.normal(size=stop - start)
                        ),
                    }
                )
                frame.to_csv(handle, header=False, index=False, lineterminator="\n")
        label_size = labels_path.stat().st_size
        total_bytes += label_size
        logger.info(
            "[generate] wrote labels.csv (%s) in %.1fs",
            format_bytes(label_size),
            time.perf_counter() - label_started,
        )

    info = {
        "rows": rows,
        "n_features": len(specs),
        "n_feature_files": n_files,
        "feature_files": feature_paths,
        "label_file": str(labels_path),
        "key_column": key_column,
        "label_column": "target",
        "target_gb_requested": target_gb,
        "actual_bytes": total_bytes,
        "actual_gb": total_bytes / 1_000_000_000,
        "pad_len": pad_len,
        "positive_rate": float(target.mean()),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "feature_mix": {
            "signal_numeric": n_signal_numeric,
            "signal_categorical": n_signal_categorical,
            "short_noise": n_short_noise,
            "wide_numeric": n_wide_numeric,
            "wide_categorical": n_wide_categorical,
        },
    }
    with (out / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    logger.info(
        "[generate] dataset complete: %s rows x %s features, %s on disk in %.1fs -> %s",
        f"{rows:,}",
        len(specs),
        format_bytes(total_bytes),
        time.perf_counter() - started,
        out,
    )
    return info


def generate_wide_rowwise_dataset(
    out_dir: str | Path,
    rows_per_file: int = 50_000,
    n_files: int = 4,
    n_features: int = 32_000,
    n_signal: int = 20,
    pattern_pool: int = 256,
    chunk_rows: int = 1000,
    seed: int = 42,
    heartbeat_seconds: float = 15.0,
    logger: Any = None,
) -> Dict[str, Any]:
    """Generate wide CSVs with identical schemas for row-wise concatenation.

    Each row contains ``n_features`` binary values, so the resulting CSV size is
    approximately ``2 * rows * columns`` bytes (value + comma). A small pattern
    pool keeps generation fast while preserving varied non-signal columns.
    """

    if n_signal >= n_features:
        raise ValueError("n_signal must be smaller than n_features")
    logger = logger or configure_logging("INFO")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total_rows = int(rows_per_file) * int(n_files)
    started = time.perf_counter()
    logger.info(
        "[generate-wide] %s files x %s rows x %s columns = %s rows / %s cells",
        n_files,
        f"{rows_per_file:,}",
        f"{n_features:,}",
        f"{total_rows:,}",
        f"{total_rows * n_features:,}",
    )

    latent = rng.normal(size=total_rows)
    signal_bits = (
        latent[:, None] + rng.normal(size=(total_rows, n_signal)) > 0.0
    ).astype(np.int8)
    weights = rng.normal(0.0, 0.55, size=n_signal)
    logits = (
        (signal_bits * weights).sum(axis=1)
        + 0.55 * latent
        + rng.normal(scale=0.75, size=total_rows)
    )
    target = rng.binomial(1, _sigmoid(logits)).astype(np.int8)

    patterns = rng.integers(0, 2, size=(pattern_pool, n_features), dtype=np.int8)
    pattern_rows = [
        ",".join("1" if value else "0" for value in row) for row in patterns
    ]
    suffixes = [
        row.split(",", n_signal)[n_signal] for row in pattern_rows
    ]
    feature_names = [f"f_{index:05d}" for index in range(n_features)]
    header = "customer_id," + ",".join(feature_names) + "\n"
    keys = np.asarray([f"cust_{index:09d}" for index in range(total_rows)], dtype=object)
    feature_paths: List[str] = []
    total_bytes = 0
    heartbeat = Heartbeat(logger, heartbeat_seconds)

    with heartbeat:
        for file_index in range(int(n_files)):
            path = out / f"features_part_{file_index:03d}.csv"
            feature_paths.append(str(path))
            file_start = file_index * rows_per_file
            file_stop = file_start + rows_per_file
            file_started = time.perf_counter()
            heartbeat.set_stage(
                "generate_wide_features",
                f"file={file_index + 1}/{n_files}",
            )
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(header)
                for chunk_index, chunk_start in enumerate(
                    range(file_start, file_stop, chunk_rows), start=1
                ):
                    chunk_stop = min(file_stop, chunk_start + chunk_rows)
                    indices = np.arange(chunk_start, chunk_stop)
                    lines = []
                    for row_index in indices:
                        prefix = ",".join(
                            "1" if value else "0"
                            for value in signal_bits[row_index, :n_signal]
                        )
                        suffix = suffixes[row_index % pattern_pool]
                        lines.append(f"{keys[row_index]},{prefix},{suffix}")
                    handle.write("\n".join(lines) + "\n")
                    if chunk_index % 10 == 0:
                        elapsed = max(1e-9, time.perf_counter() - file_started)
                        logger.info(
                            "[generate-wide] file %s/%s rows=%s/%s written=%s speed=%.0f rows/s",
                            file_index + 1,
                            n_files,
                            f"{chunk_stop - file_start:,}",
                            f"{rows_per_file:,}",
                            format_bytes(handle.tell()),
                            (chunk_stop - file_start) / elapsed,
                        )
                        heartbeat.detail(
                            f"file={file_index + 1}/{n_files} "
                            f"rows={chunk_stop - file_start}/{rows_per_file}"
                        )
            size = path.stat().st_size
            total_bytes += size
            logger.info(
                "[generate-wide] wrote %s (%s) in %.1fs",
                path.name,
                format_bytes(size),
                time.perf_counter() - file_started,
            )

        heartbeat.set_stage("generate_wide_labels", f"rows={total_rows}")
        labels_path = out / "labels.csv"
        label_started = time.perf_counter()
        with labels_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(
                "customer_id,target,label_feat_signal,"
                "label_feat_category,label_feat_noise\n"
            )
            for start in range(0, total_rows, chunk_rows):
                stop = min(total_rows, start + chunk_rows)
                categories = (
                    signal_bits[start:stop, 0].astype(np.int16) * 2
                    + signal_bits[start:stop, 1]
                )
                frame = pd.DataFrame(
                    {
                        "customer_id": keys[start:stop],
                        "target": target[start:stop],
                        "label_feat_signal": _numeric_strings(
                            latent[start:stop] + rng.normal(scale=0.1, size=stop - start)
                        ),
                        "label_feat_category": [f"c{value}" for value in categories],
                        "label_feat_noise": _numeric_strings(
                            rng.normal(size=stop - start)
                        ),
                    }
                )
                frame.to_csv(handle, header=False, index=False, lineterminator="\n")
        label_size = labels_path.stat().st_size
        total_bytes += label_size
        logger.info(
            "[generate-wide] wrote %s (%s) in %.1fs",
            labels_path.name,
            format_bytes(label_size),
            time.perf_counter() - label_started,
        )

    info = {
        "rows_per_file": int(rows_per_file),
        "n_files": int(n_files),
        "total_rows": total_rows,
        "n_features": int(n_features),
        "n_cells": int(total_rows * n_features),
        "n_signal_columns": int(n_signal),
        "pattern_pool": int(pattern_pool),
        "feature_files": feature_paths,
        "label_file": str(labels_path),
        "key_column": "customer_id",
        "label_column": "target",
        "actual_bytes": total_bytes,
        "actual_gb": total_bytes / 1_000_000_000,
        "positive_rate": float(target.mean()),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "concat_mode": "row",
    }
    with (out / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    logger.info(
        "[generate-wide] complete: %s rows x %s columns, %s on disk in %.1fs -> %s",
        f"{total_rows:,}",
        f"{n_features:,}",
        format_bytes(total_bytes),
        time.perf_counter() - started,
        out,
    )
    return info
