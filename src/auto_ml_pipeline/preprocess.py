"""Fit preprocessing state and materialize a bounded training matrix."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .cache import CacheStore
from .logging_utils import Heartbeat, rss_mb


@dataclass
class MatrixResult:
    X: Any
    y: np.ndarray
    feature_names: List[str]
    output_specs: List[Dict[str, Any]]
    categorical_indices: List[int]
    sparse: bool
    elapsed_seconds: float


def fit_preprocessing(
    store: CacheStore,
    selected: Sequence[str],
    y: np.ndarray,
    splits: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    logger: Any,
    heartbeat: Heartbeat,
) -> Dict[str, Any]:
    started = time.perf_counter()
    pre = cfg["preprocessing"]
    block_size = int(cfg["data"].get("column_block_size", 32))
    train_rows = splits["train"]
    encoding = pre.get("categorical_encoding", "native")
    add_indicator = bool(pre.get("add_missing_indicator", True))
    max_categories = int(pre.get("max_categories", 64))
    min_frequency = int(pre.get("min_category_frequency", 5))
    n_rows = store.n_rows
    selected_set = set(selected)
    ordered = [column for column in store.columns() if column in selected_set]
    state: Dict[str, Any] = {
        "encoding": encoding,
        "columns": ordered,
        "numeric": {},
        "categorical": {},
        "output_specs": [],
        "column_outputs": {},
        "add_missing_indicator": add_indicator,
    }
    output_specs: List[Dict[str, Any]] = state["output_specs"]
    column_outputs: Dict[str, Dict[str, Any]] = state["column_outputs"]

    logger.info(
        "[preprocess] fitting imputers and encoders on %s selected columns / %s train rows",
        len(ordered),
        len(train_rows),
    )
    for block_number, (block, frame) in enumerate(
        store.iter_column_blocks(columns=ordered, block_size=block_size), start=1
    ):
        train_frame = frame.iloc[train_rows]
        for column in block:
            dtype = store.dtype(column)
            series = train_frame[column]
            has_missing = bool(series.isna().any())
            if dtype in {"numeric", "datetime"}:
                values = pd.to_numeric(series, errors="coerce")
                policy = pre.get("numeric_impute", "median")
                if policy == "median":
                    fill_value = float(values.median()) if values.notna().any() else 0.0
                elif policy == "mean":
                    fill_value = float(values.mean()) if values.notna().any() else 0.0
                elif policy == "zero":
                    fill_value = 0.0
                else:
                    fill_value = float(pre.get("numeric_constant", 0.0))
                if not np.isfinite(fill_value):
                    fill_value = 0.0
                clip_low, clip_high = _clip_bounds(values, pre)
                offset = len(output_specs)
                output_specs.append(
                    {
                        "name": column,
                        "source": column,
                        "kind": "numeric",
                        "category": None,
                    }
                )
                column_outputs[column] = {
                    "kind": "numeric",
                    "offset": offset,
                    "count": 1,
                    "has_missing": has_missing,
                }
                state["numeric"][column] = {
                    "fill_value": fill_value,
                    "clip_low": clip_low,
                    "clip_high": clip_high,
                    "has_missing": has_missing,
                }
            else:
                text = series.astype("string")
                policy = pre.get("categorical_impute", "constant")
                if policy == "mode":
                    modes = text.dropna().mode()
                    fill_value = str(modes.iloc[0]) if not modes.empty else str(
                        pre.get("categorical_constant", "__MISSING__")
                    )
                else:
                    fill_value = str(pre.get("categorical_constant", "__MISSING__"))
                counts = text.dropna().value_counts()
                categories = [
                    str(value)
                    for value, count in counts.items()
                    if int(count) >= min_frequency
                ][:max_categories]
                if fill_value not in categories:
                    categories.insert(0, fill_value)
                categories = list(dict.fromkeys(categories))
                code_map = {value: index for index, value in enumerate(categories)}

                if encoding == "onehot":
                    offset = len(output_specs)
                    for category in categories:
                        output_specs.append(
                            {
                                "name": f"{column}=={category}",
                                "source": column,
                                "kind": "category",
                                "category": category,
                            }
                        )
                    column_outputs[column] = {
                        "kind": "onehot",
                        "offset": offset,
                        "count": len(categories),
                        "has_missing": has_missing,
                        "categories": categories,
                    }
                else:
                    offset = len(output_specs)
                    output_specs.append(
                        {
                            "name": column,
                            "source": column,
                            "kind": "categorical_code",
                            "category": None,
                        }
                    )
                    column_outputs[column] = {
                        "kind": "categorical_code",
                        "offset": offset,
                        "count": 1,
                        "has_missing": has_missing,
                        "categories": categories,
                    }
                state["categorical"][column] = {
                    "fill_value": fill_value,
                    "categories": categories,
                    "code_map": code_map,
                    "has_missing": has_missing,
                }

        logger.info(
            "[preprocess] fitted block %s (%s columns) rss=%.0fMiB",
            block_number,
            len(block),
            rss_mb(),
        )
        heartbeat.detail(f"preprocess fit block={block_number}")

    # Append missing indicators after all regular outputs.
    for column in ordered:
        info = column_outputs[column]
        if info["has_missing"] and add_indicator:
            offset = len(output_specs)
            output_specs.append(
                {
                    "name": f"{column}__isna",
                    "source": column,
                    "kind": "missing_indicator",
                    "category": None,
                }
            )
            info["indicator_offset"] = offset

    max_output = int(pre.get("max_output_features", 5000))
    if len(output_specs) > max_output:
        raise ValueError(
            f"Preprocessing would create {len(output_specs)} output features, above "
            f"preprocessing.max_output_features={max_output}. Reduce max_categories or "
            "use categorical_encoding: native."
        )
    state["n_output_features"] = len(output_specs)
    state["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    logger.info(
        "[preprocess] state fitted: %s outputs from %s columns in %.2fs",
        len(output_specs),
        len(ordered),
        state["elapsed_seconds"],
    )
    return state


def _clip_bounds(values: pd.Series, pre: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    clip = pre.get("clip_numeric")
    if clip is None:
        return None, None
    if clip == "quantile":
        quantiles = pre.get("clip_quantiles", [0.001, 0.999])
        clean = values.dropna()
        if clean.empty:
            return None, None
        low, high = np.nanquantile(clean.to_numpy(dtype=np.float64), quantiles)
        return float(low), float(high)
    if isinstance(clip, (list, tuple)) and len(clip) == 2:
        return float(clip[0]), float(clip[1])
    return None, None


def build_matrix(
    store: CacheStore,
    state: Dict[str, Any],
    y: np.ndarray,
    work_dir: str | Path,
    cfg: Dict[str, Any],
    logger: Any,
    heartbeat: Heartbeat,
) -> MatrixResult:
    started = time.perf_counter()
    work_dir = Path(work_dir)
    matrix_dir = work_dir / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    pre = cfg["preprocessing"]
    block_size = int(cfg["data"].get("column_block_size", 32))
    ordered = list(state["columns"])
    output_specs = state["output_specs"]
    column_outputs = state["column_outputs"]
    n_rows = store.n_rows
    n_outputs = len(output_specs)
    encoding = state["encoding"]
    sparse = encoding == "onehot"
    logger.info(
        "[matrix] building %s x %s matrix (%s)",
        n_rows,
        n_outputs,
        "sparse CSR" if sparse else "dense float32 memmap",
    )

    if sparse:
        row_parts: List[np.ndarray] = []
        col_parts: List[np.ndarray] = []
        data_parts: List[np.ndarray] = []
    else:
        matrix_path = matrix_dir / "X.npy"
        X = np.memmap(matrix_path, dtype=np.float32, mode="w+", shape=(n_rows, n_outputs))
        X[:] = np.nan

    for block_number, (block, frame) in enumerate(
        store.iter_column_blocks(columns=ordered, block_size=block_size), start=1
    ):
        for column in block:
            dtype = store.dtype(column)
            info = column_outputs[column]
            series = frame[column]
            if dtype in {"numeric", "datetime"}:
                config = state["numeric"][column]
                values = pd.to_numeric(series, errors="coerce").astype(np.float64)
                values = values.fillna(config["fill_value"])
                if config["clip_low"] is not None and config["clip_high"] is not None:
                    values = values.clip(config["clip_low"], config["clip_high"])
                if sparse:
                    row_parts.append(np.arange(n_rows, dtype=np.int64))
                    col_parts.append(np.full(n_rows, info["offset"], dtype=np.int64))
                    data_parts.append(values.to_numpy(dtype=np.float32))
                else:
                    X[:, info["offset"]] = values.to_numpy(dtype=np.float32)
            else:
                config = state["categorical"][column]
                text = series.astype("string").fillna(config["fill_value"])
                if info["kind"] == "onehot":
                    categories = config["categories"]
                    for category_index, category in enumerate(categories):
                        rows = np.flatnonzero((text == category).to_numpy())
                        if rows.size:
                            row_parts.append(rows.astype(np.int64))
                            col_parts.append(
                                np.full(rows.size, info["offset"] + category_index, dtype=np.int64)
                            )
                            data_parts.append(np.ones(rows.size, dtype=np.float32))
                else:
                    codes = text.map(config["code_map"]).astype(np.float32)
                    if sparse:
                        valid = np.isfinite(codes.to_numpy(dtype=np.float32))
                        rows = np.flatnonzero(valid)
                        if rows.size:
                            row_parts.append(rows.astype(np.int64))
                            col_parts.append(np.full(rows.size, info["offset"], dtype=np.int64))
                            data_parts.append(codes.to_numpy(dtype=np.float32)[valid])
                    else:
                        X[:, info["offset"]] = codes.to_numpy(dtype=np.float32)

            if "indicator_offset" in info:
                rows = np.flatnonzero(series.isna().to_numpy())
                if sparse and rows.size:
                    row_parts.append(rows.astype(np.int64))
                    col_parts.append(
                        np.full(rows.size, int(info["indicator_offset"]), dtype=np.int64)
                    )
                    data_parts.append(np.ones(rows.size, dtype=np.float32))
                elif not sparse:
                    X[:, int(info["indicator_offset"])] = series.isna().to_numpy(
                        dtype=np.float32
                    )

        logger.info(
            "[matrix] built block %s/%s columns rss=%.0fMiB",
            block_number,
            math.ceil(len(ordered) / block_size),
            rss_mb(),
        )
        heartbeat.detail(f"matrix block={block_number}")

    if sparse:
        if row_parts:
            rows = np.concatenate(row_parts)
            cols = np.concatenate(col_parts)
            data = np.concatenate(data_parts)
            X = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_outputs)).tocsr()
        else:
            X = sp.csr_matrix((n_rows, n_outputs), dtype=np.float32)
        sp.save_npz(matrix_dir / "X.npz", X)
        matrix_path = matrix_dir / "X.npz"
    else:
        X.flush()
        np.save(
            matrix_dir / "matrix_meta.npy",
            np.asarray([n_rows, n_outputs], dtype=np.int64),
        )
        matrix_path = matrix_dir / "X.npy"

    feature_names = [spec["name"] for spec in output_specs]
    categorical_indices = [
        index
        for index, spec in enumerate(output_specs)
        if spec["kind"] == "categorical_code"
    ]
    meta = {
        "shape": [n_rows, n_outputs],
        "sparse": sparse,
        "matrix_path": str(matrix_path),
        "encoding": encoding,
        "feature_names": feature_names,
        "output_specs": output_specs,
        "categorical_indices": categorical_indices,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    with (matrix_dir / "matrix_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    logger.info(
        "[matrix] saved %s (%s) in %.2fs",
        "sparse matrix" if sparse else "dense matrix",
        str(matrix_path),
        meta["elapsed_seconds"],
    )
    return MatrixResult(
        X=X,
        y=y,
        feature_names=feature_names,
        output_specs=output_specs,
        categorical_indices=categorical_indices,
        sparse=sparse,
        elapsed_seconds=meta["elapsed_seconds"],
    )
