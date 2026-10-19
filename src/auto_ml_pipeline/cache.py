"""Chunked CSV ingestion into a column-prunable Parquet cache."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .logging_utils import Heartbeat, format_bytes, rss_mb
from .schema import LabelInfo, SourceSpec


ROW_INDEX = "__row_index__"


def _partitions(columns: Sequence[str], size: int) -> List[List[str]]:
    return [list(columns[index : index + size]) for index in range(0, len(columns), size)]


def _estimate_chunk_rows(
    sample_bytes: float,
    sample_rows: int,
    n_columns: int,
    configured: int,
    target_bytes: int = 256 * 1024 * 1024,
) -> int:
    if sample_rows <= 0 or n_columns <= 0:
        return max(16, configured)
    avg_cell = max(2.0, sample_bytes / (sample_rows * n_columns))
    adaptive = int(target_bytes / (n_columns * avg_cell))
    return int(max(16, min(configured, adaptive)))


def _align_chunk(
    chunk: pd.DataFrame,
    key_col: str,
    key_index: pd.Index,
    source_id: str,
) -> Tuple[pd.DataFrame, int]:
    keys = chunk[key_col]
    null_mask = keys.isna()
    if null_mask.any():
        chunk = chunk.loc[~null_mask]
        keys = keys.loc[~null_mask]
    keys = keys.astype(str).str.strip()
    duplicated = keys.duplicated(keep=False)
    if duplicated.any():
        examples = keys.loc[duplicated].head(5).tolist()
        raise ValueError(
            f"Duplicate join keys in source {source_id!r} (sample: {examples}). "
            "Each row must resolve to one label row."
        )
    positions = key_index.get_indexer(keys)
    missing = int((positions < 0).sum())
    if missing:
        keep = positions >= 0
        chunk = chunk.loc[keep]
        positions = positions[keep]
    chunk = chunk.copy()
    chunk[ROW_INDEX] = positions.astype(np.int64)
    return chunk, missing


def _convert_columns(
    chunk: pd.DataFrame,
    dtypes: Dict[str, str],
    numeric_dtype: Any = np.float64,
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column, dtype in dtypes.items()
        if dtype == "numeric" and column in chunk.columns
    ]
    datetime_columns = [
        column
        for column, dtype in dtypes.items()
        if dtype == "datetime" and column in chunk.columns
    ]
    categorical_columns = [
        column
        for column, dtype in dtypes.items()
        if dtype not in {"numeric", "datetime"} and column in chunk.columns
    ]
    if numeric_columns:
        values = chunk[numeric_columns]
        if not bool(values.dtypes.map(pd.api.types.is_numeric_dtype).all()):
            try:
                values = values.astype(numeric_dtype)
            except (TypeError, ValueError):
                values = values.apply(pd.to_numeric, errors="coerce").astype(numeric_dtype)
        else:
            values = values.astype(numeric_dtype, copy=False)
        chunk[numeric_columns] = values
    if datetime_columns:
        for column in datetime_columns:
            parsed = pd.to_datetime(chunk[column], errors="coerce", format="mixed")
            values = parsed.astype("int64").astype(np.float64) / 1e9
            values[parsed.isna()] = np.nan
            chunk[column] = values
    if categorical_columns:
        values = chunk[categorical_columns].astype("string")
        chunk[categorical_columns] = values.mask(values.apply(lambda column: column.str.len() == 0))
    return chunk


def _write_part(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return int(path.stat().st_size)


def _ingest_blocked_wide(
    source: SourceSpec,
    logical_columns: List[str],
    dtype_by_logical: Dict[str, str],
    source_to_logical: Dict[str, str],
    key_index: pd.Index,
    source_dir: Path,
    work_dir: Path,
    chunk_rows: int,
    block_size: int,
    row_group_size: int,
    numeric_dtype: Any,
    csv_cfg: Dict[str, Any],
    n_rows: int,
    coverage: np.ndarray,
    logger: Any,
    heartbeat: Heartbeat,
) -> Dict[str, Any]:
    """One-pass ingest for very wide sources, partitioned by column block.

    Keeping one writer per column block avoids creating a Parquet file with tens
    of thousands of columns while still reading each CSV only once.
    """

    groups = _partitions(logical_columns, block_size)
    logical_to_source_col = {
        logical: source_col for source_col, logical in source_to_logical.items()
    }
    all_source_columns = [logical_to_source_col[column] for column in logical_columns]
    paths: Dict[int, Path] = {}
    schemas: Dict[int, pa.Schema] = {}
    writers: Dict[int, pq.ParquetWriter] = {}
    for group_index, group in enumerate(groups):
        path = source_dir / f"block_{group_index:04d}.parquet"
        fields = [(ROW_INDEX, pa.int64())]
        for column in group:
            dtype = dtype_by_logical[column]
            if dtype in {"numeric", "datetime"}:
                arrow_type = pa.float32() if numeric_dtype == np.float32 else pa.float64()
            else:
                arrow_type = pa.string()
            fields.append((column, arrow_type))
        schema = pa.schema(fields)
        paths[group_index] = path
        schemas[group_index] = schema
        writers[group_index] = pq.ParquetWriter(
            path, schema, compression="zstd", use_dictionary=True
        )

    rows_seen = 0
    rows_written = 0
    missing_keys = 0
    duplicate_rows = 0
    source_seen = np.zeros(n_rows, dtype=bool)
    source_started = time.perf_counter()
    read_dtype = {source.key_col: "string"}
    for column in logical_columns:
        if dtype_by_logical[column] in {"categorical", "identifier", "datetime"}:
            read_dtype[logical_to_source_col[column]] = "string"
    try:
        reader = pd.read_csv(
            source.path,
            sep=csv_cfg.get("sep", ","),
            encoding=csv_cfg.get("encoding", "utf-8"),
            compression=csv_cfg.get("compression", "infer"),
            usecols=[source.key_col] + all_source_columns,
            dtype=read_dtype,
            chunksize=chunk_rows,
            keep_default_na=True,
            na_values=csv_cfg.get("na_values"),
            low_memory=False,
        )
        for chunk_number, chunk in enumerate(reader, start=1):
            rows_seen += len(chunk)
            chunk.columns = [str(column) for column in chunk.columns]
            missing_keys += int(chunk[source.key_col].isna().sum())
            chunk, missing = _align_chunk(
                chunk, source.key_col, key_index, source.source_id
            )
            missing_keys += missing
            if chunk.empty:
                continue

            rename_map = {
                source_col: logical for source_col, logical in source_to_logical.items()
            }
            chunk = chunk.drop(columns=[source.key_col]).rename(columns=rename_map)
            chunk = _convert_columns(chunk, dtype_by_logical, numeric_dtype)
            chunk = chunk[[ROW_INDEX] + logical_columns]
            row_indices = chunk[ROW_INDEX].to_numpy(dtype=np.int64)
            duplicate_rows += int(source_seen[row_indices].sum())
            source_seen[row_indices] = True
            coverage[row_indices] = True
            rows_written += len(chunk)

            for group_index, group in enumerate(groups):
                frame = chunk[[ROW_INDEX] + group]
                table = pa.Table.from_pandas(frame, preserve_index=False)
                table = table.cast(schemas[group_index])
                writers[group_index].write_table(
                    table, row_group_size=row_group_size
                )

            if chunk_number % 10 == 0:
                elapsed = max(1e-9, time.perf_counter() - source_started)
                logger.info(
                    "[ingest] %s wide-chunk=%s rows=%s written=%s missing_keys=%s "
                    "speed=%.0f rows/s blocks=%s",
                    source.source_id,
                    chunk_number,
                    rows_seen,
                    rows_written,
                    missing_keys,
                    rows_written / elapsed,
                    len(groups),
                )
                heartbeat.detail(
                    f"wide source={source.source_id} chunk={chunk_number} "
                    f"rows={rows_written} blocks={len(groups)}"
                )
    finally:
        for writer in writers.values():
            writer.close()

    parts_meta = [
        {
            "columns": group,
            "parts": [str(paths[index].relative_to(work_dir))],
        }
        for index, group in enumerate(groups)
    ]
    bytes_written = int(sum(path.stat().st_size for path in paths.values()))
    return {
        "parts_meta": parts_meta,
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "missing_keys": missing_keys,
        "duplicate_rows": duplicate_rows,
        "bytes_written": bytes_written,
        "blocks": len(groups),
    }


def build_cache(
    cfg: Dict[str, Any],
    sources: List[SourceSpec],
    label_info: LabelInfo,
    work_dir: str | Path,
    logger: Any,
    heartbeat: Heartbeat,
) -> Dict[str, Any]:
    """Stream every source once and write row-chunked Parquet parts."""

    work_dir = Path(work_dir)
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = cfg["data"]
    csv_cfg = data["csv"]
    target_bytes = int(data.get("ingest_target_chunk_bytes", 256 * 1024 * 1024))
    max_in_memory = int(data.get("max_source_columns_in_memory", 4000))
    block_size = int(data.get("column_block_size", 32))
    key_index = pd.Index(label_info.keys)
    n_rows = label_info.n_rows
    coverage = np.zeros(n_rows, dtype=bool)
    key_set = set(label_info.keys.tolist())
    manifest_sources: Dict[str, Any] = {}
    total_cache_bytes = 0
    total_rows_written = 0
    total_missing_keys = 0
    started = time.perf_counter()

    for source_number, source in enumerate(sources, start=1):
        logical_columns = [column for column in source.feature_columns if column in source.source_to_logical]
        if not logical_columns:
            logger.info(
                "[ingest] %s has no usable feature columns; skipping cache write",
                source.source_id,
            )
            continue

        source_dir = cache_dir / "sources" / source.source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        source_started = time.perf_counter()
        sample_rows = max(1, int(data.get("sample_rows", 2000)))
        configured_chunk = int(csv_cfg.get("chunksize", 2000))
        n_read_columns = len(logical_columns) + 1
        chunk_rows = _estimate_chunk_rows(
            source.sample_bytes,
            min(sample_rows, 2000),
            n_read_columns,
            configured_chunk,
            target_bytes,
        )
        dtype_by_logical = {column: source.dtypes[column] for column in logical_columns}
        source_to_logical = {
            column: source.source_to_logical[column]
            for column in logical_columns
        }
        logical_to_source_col = {
            logical: source_col for source_col, logical in source_to_logical.items()
        }

        projected = n_read_columns > max_in_memory
        numeric_dtype = (
            np.float32
            if str(data.get("cache_numeric_dtype", "float64")).lower() == "float32"
            else np.float64
        )
        parts_meta: List[Dict[str, Any]] = []
        rows_written = 0
        rows_seen = 0
        missing_keys = 0
        duplicate_rows = 0
        bytes_written = 0

        if projected:
            wide_block_size = int(data.get("wide_block_size", 512))
            row_group_size = int(data.get("wide_row_group_size", 5000))
            logger.warning(
                "[ingest] %s is very wide (%s columns); using one-pass blocked "
                "ingest with %s-column blocks (row_group=%s)",
                source.source_id,
                n_read_columns,
                wide_block_size,
                row_group_size,
            )
            blocked = _ingest_blocked_wide(
                source=source,
                logical_columns=logical_columns,
                dtype_by_logical=dtype_by_logical,
                source_to_logical=source_to_logical,
                key_index=key_index,
                source_dir=source_dir,
                work_dir=work_dir,
                chunk_rows=chunk_rows,
                block_size=wide_block_size,
                row_group_size=row_group_size,
                numeric_dtype=numeric_dtype,
                csv_cfg=csv_cfg,
                n_rows=n_rows,
                coverage=coverage,
                logger=logger,
                heartbeat=heartbeat,
            )
            parts_meta = blocked["parts_meta"]
            rows_seen = blocked["rows_seen"]
            rows_written = blocked["rows_written"]
            missing_keys = blocked["missing_keys"]
            duplicate_rows = blocked["duplicate_rows"]
            bytes_written = blocked["bytes_written"]
        else:
            group = logical_columns
            source_columns = [logical_to_source_col[column] for column in group]
            usecols = [source.key_col] + source_columns
            part_paths: List[str] = []
            group_seen = np.zeros(n_rows, dtype=bool)
            reader = pd.read_csv(
                source.path,
                sep=csv_cfg.get("sep", ","),
                encoding=csv_cfg.get("encoding", "utf-8"),
                compression=csv_cfg.get("compression", "infer"),
                usecols=usecols,
                dtype={source.key_col: "string"},
                chunksize=chunk_rows,
                keep_default_na=True,
                na_values=csv_cfg.get("na_values"),
                low_memory=False,
            )
            for chunk_number, chunk in enumerate(reader, start=1):
                rows_seen += len(chunk)
                chunk.columns = [str(column) for column in chunk.columns]
                missing_key_values = int(chunk[source.key_col].isna().sum())
                missing_keys += missing_key_values
                chunk, missing = _align_chunk(
                    chunk, source.key_col, key_index, source.source_id
                )
                missing_keys += missing
                if chunk.empty:
                    continue

                rename_map = {
                    source_col: logical
                    for source_col, logical in zip(source_columns, group)
                }
                chunk = chunk.drop(columns=[source.key_col]).rename(columns=rename_map)
                chunk = _convert_columns(chunk, dtype_by_logical, numeric_dtype)
                chunk = chunk[[ROW_INDEX] + group]
                row_indices = chunk[ROW_INDEX].to_numpy(dtype=np.int64)
                duplicate_rows += int(group_seen[row_indices].sum())
                group_seen[row_indices] = True
                coverage[row_indices] = True
                rows_written += len(chunk)

                part_path = source_dir / f"part_{chunk_number:06d}.parquet"
                bytes_written += _write_part(chunk, part_path)
                part_paths.append(str(part_path.relative_to(work_dir)))
                if chunk_number % 10 == 0:
                    elapsed = max(1e-9, time.perf_counter() - source_started)
                    logger.info(
                        "[ingest] %s chunk=%s rows=%s written=%s missing_keys=%s speed=%.0f rows/s cache=%s",
                        source.source_id,
                        chunk_number,
                        rows_seen,
                        rows_written,
                        missing_keys,
                        rows_written / elapsed,
                        format_bytes(bytes_written),
                    )
                    heartbeat.detail(
                        f"source={source.source_id} chunk={chunk_number} "
                        f"rows={rows_written}"
                    )

            parts_meta.append({"columns": group, "parts": part_paths})

        elapsed = time.perf_counter() - source_started
        total_cache_bytes += bytes_written
        total_rows_written += rows_written
        total_missing_keys += missing_keys
        manifest_sources[source.source_id] = {
            "source_id": source.source_id,
            "path": source.path,
            "kind": source.kind,
            "key_col": source.key_col,
            "label_col": source.label_col,
            "logical_columns": logical_columns,
            "dtypes": dtype_by_logical,
            "part_groups": parts_meta,
            "rows_seen": rows_seen,
            "rows_written": rows_written,
            "missing_keys": missing_keys,
            "duplicate_row_indices": duplicate_rows,
            "cache_bytes": bytes_written,
            "elapsed_seconds": round(elapsed, 4),
            "chunk_rows": chunk_rows,
            "projected": projected,
            "wide_blocked": projected,
        }
        logger.info(
            "[ingest] %s/%s %s: %s rows -> %s cached (%s) in %.1fs missing_keys=%s",
            source_number,
            len(sources),
            source.source_id,
            f"{rows_written:,}",
            format_bytes(bytes_written),
            f"{rows_written / max(elapsed, 1e-9):,.0f} rows/s",
            elapsed,
            missing_keys,
        )
        if duplicate_rows:
            logger.warning(
                "[ingest] %s contains %s rows whose keys were seen more than once; "
                "the first cached row wins during reads",
                source.source_id,
                duplicate_rows,
            )
        heartbeat.detail(f"ingested {source_number}/{len(sources)} sources")

    column_sources: Dict[str, List[str]] = {}
    column_dtypes: Dict[str, str] = {}
    for source_id, meta in manifest_sources.items():
        for column in meta["logical_columns"]:
            column_sources.setdefault(column, []).append(source_id)
            column_dtypes.setdefault(column, meta["dtypes"][column])

    coverage_ratio = float(coverage.mean()) if n_rows else 0.0
    min_coverage = float(cfg["filtering"].get("min_row_coverage", 0.0))
    if coverage_ratio < min_coverage:
        raise ValueError(
            f"Only {coverage_ratio:.1%} of label rows have feature data, below "
            f"filtering.min_row_coverage={min_coverage:.1%}."
        )
    if coverage_ratio < 1.0:
        logger.warning(
            "[ingest] %.2f%% of label rows have no cached feature row; they will contain NaNs",
            (1.0 - coverage_ratio) * 100.0,
        )

    np.save(work_dir / "cache" / "row_coverage.npy", coverage)
    manifest = {
        "version": 1,
        "n_rows": n_rows,
        "key_column": data["feature_key"],
        "label_key": data["label_key"],
        "label_column": data["label_column"],
        "concat_mode": data.get("concat_mode", "column"),
        "task": label_info.task,
        "classes": [str(value) for value in label_info.classes],
        "sources": manifest_sources,
        "column_sources": column_sources,
        "column_dtypes": column_dtypes,
        "n_logical_columns": len(column_sources),
        "total_cache_bytes": total_cache_bytes,
        "total_rows_written": total_rows_written,
        "total_missing_keys": total_missing_keys,
        "coverage_ratio": coverage_ratio,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info(
        "[ingest] cache complete: %s logical columns, %s cache on disk, %s rows written",
        manifest["n_logical_columns"],
        format_bytes(total_cache_bytes),
        f"{total_rows_written:,}",
    )
    return manifest


class CacheStore:
    """Read selected logical columns from the row-chunked Parquet cache."""

    def __init__(self, work_dir: str | Path, manifest: Dict[str, Any]):
        self.work_dir = Path(work_dir)
        self.manifest = manifest
        self.n_rows = int(manifest["n_rows"])
        self.column_dtypes: Dict[str, str] = dict(manifest["column_dtypes"])
        self.column_sources: Dict[str, List[str]] = {
            column: list(source_ids)
            for column, source_ids in manifest["column_sources"].items()
        }

    @classmethod
    def load(cls, work_dir: str | Path) -> "CacheStore":
        manifest_path = Path(work_dir) / "cache" / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return cls(work_dir, manifest)

    def columns(self) -> List[str]:
        return list(self.column_dtypes.keys())

    def dtype(self, column: str) -> str:
        return self.column_dtypes[column]

    def _read_source_columns(
        self,
        source_id: str,
        selected: Dict[str, str],
        row_limit: Optional[int],
    ) -> pd.DataFrame:
        source_meta = self.manifest["sources"][source_id]
        logical_columns = set(source_meta["logical_columns"])
        wanted = {
            logical: source_col
            for logical, source_col in selected.items()
            if logical in logical_columns
        }
        if not wanted:
            return pd.DataFrame()

        logical_to_source = {}
        for group in source_meta["part_groups"]:
            for logical in group["columns"]:
                # Recover source column name from the source spec mapping stored in schema JSON.
                logical_to_source[logical] = logical

        group_frames: List[pd.DataFrame] = []
        for group in source_meta["part_groups"]:
            group_columns = set(group["columns"])
            group_wanted = {logical: source_col for logical, source_col in wanted.items() if logical in group_columns}
            if not group_wanted:
                continue
            source_columns = sorted(set(group_wanted.values()))
            read_columns = [ROW_INDEX] + source_columns
            pieces: List[pd.DataFrame] = []
            collected = 0
            for relative_path in group["parts"]:
                path = self.work_dir / relative_path
                piece = pd.read_parquet(path, columns=read_columns, engine="pyarrow")
                rename = {source_col: logical for logical, source_col in group_wanted.items()}
                piece = piece.rename(columns=rename)
                pieces.append(piece)
                collected += len(piece)
                if row_limit is not None and collected >= row_limit:
                    break
            if pieces:
                piece = pd.concat(pieces, ignore_index=True)
                piece = piece.drop_duplicates(subset=[ROW_INDEX], keep="first")
                piece = piece.set_index(ROW_INDEX)
                if row_limit is not None:
                    piece = piece.head(int(row_limit))
                group_frames.append(piece)

        if not group_frames:
            return pd.DataFrame()
        frame = group_frames[0]
        for group_frame in group_frames[1:]:
            frame = frame.combine_first(group_frame)
        return frame

    def read_columns(
        self,
        columns: Sequence[str],
        row_limit: Optional[int] = None,
    ) -> pd.DataFrame:
        columns = list(dict.fromkeys(str(column) for column in columns))
        missing = [column for column in columns if column not in self.column_dtypes]
        if missing:
            raise KeyError(f"Columns not present in cache: {missing[:10]}")

        by_source: Dict[str, Dict[str, str]] = {}
        for column in columns:
            for source_id in self.column_sources[column]:
                by_source.setdefault(source_id, {})[column] = column

        result = pd.DataFrame(index=pd.RangeIndex(self.n_rows))
        for source_id, selected in by_source.items():
            frame = self._read_source_columns(source_id, selected, row_limit)
            if frame.empty:
                continue
            for column in selected:
                if column not in frame.columns:
                    continue
                if column not in result.columns:
                    result[column] = pd.Series(index=result.index, dtype=object)
                incoming = frame[column].reindex(result.index)
                if result[column].notna().any() and incoming.notna().any():
                    result[column] = result[column].combine_first(incoming)
                else:
                    result[column] = incoming
        if row_limit is not None:
            result = result.head(int(row_limit))

        for column in columns:
            dtype = self.column_dtypes[column]
            if dtype in {"numeric", "datetime"}:
                result[column] = pd.to_numeric(result[column], errors="coerce").astype(np.float64)
            else:
                result[column] = result[column].astype("string")
        return result

    def sample(
        self,
        columns: Sequence[str],
        n_rows: int = 2000,
    ) -> pd.DataFrame:
        return self.read_columns(columns, row_limit=n_rows)

    def iter_column_blocks(
        self,
        columns: Optional[Sequence[str]] = None,
        block_size: int = 32,
    ) -> Iterator[Tuple[List[str], pd.DataFrame]]:
        selected = list(columns) if columns is not None else self.columns()
        for start in range(0, len(selected), block_size):
            block = selected[start : start + block_size]
            yield block, self.read_columns(block)

    def read_all_columns_one_by_one(self) -> Iterator[Tuple[str, pd.Series]]:
        for column in self.columns():
            frame = self.read_columns([column])
            yield column, frame[column]
