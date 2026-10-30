"""Source discovery, type inference, label loading, and train/valid/test splits."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ConfigError, pandas_csv_options
from .logging_utils import format_bytes

NUMERIC_TYPES = {"numeric", "datetime"}
CATEGORICAL_TYPES = {"categorical", "identifier"}


def _merge_row_types(previous_type: str, current_type: str) -> str:
    """Return a common row-concat type for compatible partition inferences."""
    if previous_type == "unknown":
        return current_type
    if current_type == "unknown":
        return previous_type
    if previous_type == current_type:
        return previous_type
    # CSV has no declared dtype. A string-backed categorical column preserves
    # every partition's values and avoids rejecting a shared schema solely
    # because independent samples crossed different inference thresholds.
    return "categorical"


@dataclass
class SourceSpec:
    source_id: str
    path: str
    kind: str
    key_col: str
    label_col: Optional[str]
    feature_columns: List[str] = field(default_factory=list)
    source_to_logical: Dict[str, str] = field(default_factory=dict)
    logical_to_source: Dict[str, str] = field(default_factory=dict)
    dtypes: Dict[str, str] = field(default_factory=dict)
    dropped_columns: Dict[str, str] = field(default_factory=dict)
    sample_bytes: float = 0.0

    def logical_columns(self) -> List[str]:
        return list(self.logical_to_source.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "kind": self.kind,
            "key_col": self.key_col,
            "label_col": self.label_col,
            "feature_columns": self.feature_columns,
            "source_to_logical": self.source_to_logical,
            "logical_to_source": self.logical_to_source,
            "dtypes": self.dtypes,
            "dropped_columns": self.dropped_columns,
            "sample_bytes": round(self.sample_bytes, 1),
        }


@dataclass
class LabelInfo:
    keys: np.ndarray
    raw_labels: np.ndarray
    y: np.ndarray
    classes: List[Any]
    task: str
    label_column: str
    label_key: str
    class_mapping: Dict[str, int]

    @property
    def n_rows(self) -> int:
        return len(self.y)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_classes": self.n_classes,
            "task": self.task,
            "label_column": self.label_column,
            "label_key": self.label_key,
            "classes": [str(value) for value in self.classes],
            "class_mapping": self.class_mapping,
        }


def _sanitize_id(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return value or "source"


def read_header(path: str | Path, csv_cfg: Dict[str, Any]) -> List[str]:
    try:
        header = pd.read_csv(
            path, nrows=0, low_memory=False, **pandas_csv_options(csv_cfg)
        )
    except Exception as exc:  # pragma: no cover - surfaced with a better message
        raise ConfigError(f"Unable to read CSV header from {path}: {exc}") from exc
    columns = [str(column) for column in header.columns]
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ConfigError(f"Duplicate column names in {path}: {duplicates[:10]}")
    return columns


def _sample_nrows(n_columns: int, configured: int) -> int:
    # Keep type-inference memory bounded even for extremely wide files.
    adaptive = max(100, int(4_000_000 / max(1, n_columns)))
    return max(50, min(int(configured), adaptive))


def sample_csv(
    path: str | Path,
    csv_cfg: Dict[str, Any],
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    header = read_header(path, csv_cfg)
    if n_rows is None:
        n_rows = _sample_nrows(len(header), 2000)
    frame = pd.read_csv(
        path,
        encoding=csv_cfg.get("encoding", "utf-8"),
        compression=csv_cfg.get("compression", "infer"),
        nrows=n_rows,
        dtype=str,
        keep_default_na=True,
        na_values=csv_cfg.get("na_values"),
        low_memory=False,
        **pandas_csv_options(csv_cfg),
    )
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _numeric_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    converted = pd.to_numeric(non_null, errors="coerce")
    return float(converted.notna().mean())


def _datetime_ratio(series: pd.Series) -> float:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return 0.0
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def infer_column_type(
    series: pd.Series,
    type_cfg: Dict[str, Any],
    *,
    row_concat_mode: bool = False,
) -> str:
    non_null = series.dropna()
    if non_null.empty:
        # A row partition can be entirely null for a column that has values in
        # another partition. Treat it as neutral so schema reconciliation can
        # adopt the populated partition's type.
        return "unknown" if row_concat_mode else "constant"
    if not row_concat_mode and non_null.nunique(dropna=True) <= 1:
        return "constant"

    unique_ratio = float(non_null.nunique(dropna=True) / max(1, len(non_null)))
    identifier_ratio = float(type_cfg.get("identifier_unique_ratio", 0.98))
    identifier_min = int(type_cfg.get("identifier_min_unique", 100))

    numeric_ratio = _numeric_ratio(series)
    if numeric_ratio >= float(type_cfg.get("numeric_threshold", 0.95)):
        numeric_values = pd.to_numeric(non_null, errors="coerce").dropna()
        if unique_ratio >= identifier_ratio and non_null.nunique() >= identifier_min:
            integer_like = float(np.isclose(np.mod(numeric_values.to_numpy(dtype=np.float64), 1.0), 0.0).mean())
            median_magnitude = float(np.median(np.abs(numeric_values.to_numpy(dtype=np.float64))))
            # Continuous measurements are naturally unique; only large integer-like
            # sequences are treated as identifiers.
            if not row_concat_mode and integer_like >= 0.98 and median_magnitude >= 1000.0:
                return "identifier"
        return "numeric"

    if type_cfg.get("detect_datetime") and _datetime_ratio(series) >= float(type_cfg.get("datetime_threshold", 0.98)):
        return "datetime"

    if unique_ratio >= identifier_ratio and non_null.nunique() >= identifier_min:
        return "identifier"
    return "categorical"


def discover_sources(
    cfg: Dict[str, Any],
    label_info: LabelInfo,
    logger: Any,
) -> Tuple[List[SourceSpec], Dict[str, Any]]:
    """Build source specs, infer per-column types, and resolve duplicate names."""

    data = cfg["data"]
    csv_cfg = data["csv"]
    feature_paths = [str(Path(path).resolve()) for path in data["feature_paths"]]
    label_path = str(Path(data["label_path"]).resolve())
    if label_path not in feature_paths:
        feature_paths.append(label_path)

    drop_columns = {str(column) for column in data.get("drop_columns", [])}
    split_strategy = str(data.get("split_strategy", "random")).lower()
    split_time_column = (
        str((data.get("temporal_split") or {}).get("time_column") or "")
        if split_strategy == "temporal"
        else None
    )
    used_ids: Dict[str, int] = {}
    sources: List[SourceSpec] = []
    total_sample_bytes = 0.0
    row_mode = data.get("concat_mode", "column") == "row"

    for index, path in enumerate(feature_paths):
        path_obj = Path(path)
        header = read_header(path_obj, csv_cfg)
        is_label_source = path == label_path
        key_col = data["label_key"] if is_label_source else data["feature_key"]
        if key_col not in header:
            raise ConfigError(f"Key column {key_col!r} not found in {path}")
        label_col = data["label_column"] if is_label_source else None
        if is_label_source and label_col not in header:
            raise ConfigError(f"Label column {label_col!r} not found in {path}")
        if (
            is_label_source
            and split_time_column
            and split_time_column not in header
        ):
            raise ConfigError(
                f"Temporal split time column {split_time_column!r} "
                f"not found in {path}"
            )

        base_id = _sanitize_id(path_obj.stem)
        if base_id in used_ids:
            used_ids[base_id] += 1
            source_id = f"{base_id}_{used_ids[base_id]}"
        else:
            used_ids[base_id] = 0
            source_id = base_id

        sample_n = _sample_nrows(len(header), int(data.get("sample_rows", 2000)))
        sample = sample_csv(path_obj, csv_cfg, n_rows=sample_n)
        total_sample_bytes += float(sample.memory_usage(deep=True).sum())
        if is_label_source:
            logger.info(
                "[schema] %s: sampled %s rows x %s columns (%s)",
                source_id,
                len(sample),
                len(header),
                format_bytes(sample.memory_usage(deep=True).sum()),
            )
        else:
            logger.info(
                "[schema] %s: sampled %s rows x %s columns (%s)",
                source_id,
                len(sample),
                len(header),
                format_bytes(sample.memory_usage(deep=True).sum()),
            )

        spec = SourceSpec(
            source_id=source_id,
            path=path,
            kind="labels" if is_label_source else "features",
            key_col=key_col,
            label_col=label_col,
        )

        for column in header:
            if column == key_col or column == label_col:
                continue
            if split_time_column and column == split_time_column:
                spec.dropped_columns[column] = "temporal split time column"
                continue
            if column in drop_columns:
                spec.dropped_columns[column] = "configured drop"
                continue
            if is_label_source and not data.get("include_label_features", True):
                spec.dropped_columns[column] = "label features disabled"
                continue
            # Cardinality and scalar type can depend on which rows happen to be
            # sampled in each row partition. Defer constant-column removal to
            # global filtering and keep numeric-backed columns numeric so a shared
            # schema is not rejected by partition-local inference.
            inferred = infer_column_type(
                sample[column],
                cfg["type_inference"],
                row_concat_mode=row_mode,
            )
            if inferred == "constant":
                spec.dropped_columns[column] = "constant"
                continue
            spec.feature_columns.append(column)
            spec.dtypes[column] = inferred

        sources.append(spec)

    # Resolve logical column names across sources.
    seen: Dict[str, SourceSpec] = {}
    for spec in sources:
        for source_column in spec.feature_columns:
            logical = source_column
            if source_column in seen:
                previous = seen[source_column]
                if row_mode:
                    previous_type = previous.dtypes.get(source_column)
                    current_type = spec.dtypes.get(source_column)
                    if previous_type != current_type:
                        merged_type = _merge_row_types(previous_type, current_type)
                        logger.warning(
                            "[schema] row-concat column %r inferred as %s in %s and %s " "in %s; using %s",
                            source_column,
                            previous_type,
                            previous.source_id,
                            current_type,
                            spec.source_id,
                            merged_type,
                        )
                elif data.get("deduplicate_columns", True):
                    logical = f"{spec.source_id}__{source_column}"
                    logger.warning(
                        "[schema] duplicate column %r in %s renamed to %r",
                        source_column,
                        spec.source_id,
                        logical,
                    )
                else:
                    raise ConfigError(
                        f"Duplicate feature column {source_column!r} in column concat mode. "
                        "Enable data.deduplicate_columns or rename the source column."
                    )
            else:
                seen[source_column] = spec
            spec.source_to_logical[source_column] = logical
            spec.logical_to_source[logical] = source_column
            if row_mode and logical != source_column:
                spec.dtypes[logical] = spec.dtypes.pop(source_column)

    if row_mode:
        resolved_types: Dict[str, str] = {}
        for spec in sources:
            for column, dtype in spec.dtypes.items():
                resolved_types[column] = _merge_row_types(
                    resolved_types.get(column, "unknown"),
                    dtype,
                )
        for spec in sources:
            for column in spec.dtypes:
                resolved = resolved_types[column]
                spec.dtypes[column] = "categorical" if resolved == "unknown" else resolved

    # In row mode, feature files should describe the same table schema.
    if row_mode:
        feature_sources = [spec for spec in sources if spec.kind == "features"]
        if feature_sources:
            reference = set(feature_sources[0].logical_columns())
            for spec in feature_sources[1:]:
                current = set(spec.logical_columns())
                if current != reference:
                    only_ref = sorted(reference - current)[:10]
                    only_cur = sorted(current - reference)[:10]
                    raise ConfigError(
                        "Row concat requires identical feature schemas. "
                        f"{feature_sources[0].source_id} vs {spec.source_id}: "
                        f"missing={only_ref}, extra={only_cur}"
                    )

    stats = {
        "n_sources": len(sources),
        "feature_columns": len(seen),
        "sample_bytes": total_sample_bytes,
        "row_mode": row_mode,
    }
    logger.info(
        "[schema] discovered %s sources, %s logical feature columns, sample memory=%s",
        len(sources),
        len(seen),
        format_bytes(total_sample_bytes),
    )
    return sources, stats


def _looks_numeric(values: List[str]) -> bool:
    converted = pd.to_numeric(pd.Series(values), errors="coerce")
    return bool(converted.notna().all())


def load_labels(cfg: Dict[str, Any], logger: Any) -> LabelInfo:
    data = cfg["data"]
    csv_cfg = data["csv"]
    path = data["label_path"]
    key_col = data["label_key"]
    label_col = data["label_column"]
    header = read_header(path, csv_cfg)
    if key_col not in header:
        raise ConfigError(f"Label key column {key_col!r} not found in {path}")
    if label_col not in header:
        raise ConfigError(f"Label column {label_col!r} not found in {path}")

    keys: List[str] = []
    labels: List[str] = []
    rows_seen = 0
    max_rows = data.get("max_rows")
    reader = pd.read_csv(
        path,
        encoding=csv_cfg.get("encoding", "utf-8"),
        compression=csv_cfg.get("compression", "infer"),
        usecols=[key_col, label_col],
        dtype=str,
        chunksize=int(csv_cfg.get("chunksize", 2000)),
        keep_default_na=True,
        na_values=csv_cfg.get("na_values"),
        low_memory=False,
        **pandas_csv_options(csv_cfg),
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        chunk.columns = [str(column) for column in chunk.columns]
        chunk[key_col] = chunk[key_col].astype("string").str.strip()
        chunk[label_col] = chunk[label_col].astype("string").str.strip()
        chunk = chunk[chunk[label_col].notna() & (chunk[label_col] != "")]
        if chunk.empty:
            continue
        duplicate_mask = chunk[key_col].duplicated(keep=False)
        if duplicate_mask.any():
            duplicates = chunk.loc[duplicate_mask, key_col].head(5).tolist()
            raise ConfigError(
                f"Label file contains duplicate keys (sample: {duplicates}). "
                "The join key must be unique in the label file."
            )
        keys.extend(chunk[key_col].astype(str).tolist())
        labels.extend(chunk[label_col].astype(str).tolist())
        rows_seen += len(chunk)
        if chunk_number % 20 == 0:
            logger.info("[labels] read %s labelled rows", rows_seen)
        if max_rows is not None and rows_seen >= int(max_rows):
            keys = keys[: int(max_rows)]
            labels = labels[: int(max_rows)]
            break

    if not keys:
        raise ConfigError("No labelled rows were found in the label file.")

    keys_array = np.asarray(keys, dtype=object)
    if len(np.unique(keys_array)) != len(keys_array):
        raise ConfigError("Label keys are not unique after stripping whitespace.")

    unique_labels = sorted(set(labels), key=lambda value: (not _is_number(value), value))
    if data.get("task") == "regression":
        raise ConfigError("Regression is not implemented; choose binary or multiclass classification.")

    numeric_labels = _looks_numeric(unique_labels)
    if numeric_labels:
        numeric_values = sorted(float(value) for value in unique_labels)
        classes: List[Any] = [int(value) if float(value).is_integer() else float(value) for value in numeric_values]
        mapping = {str(value): index for index, value in enumerate(classes)}
    else:
        classes = list(unique_labels)
        mapping = {str(value): index for index, value in enumerate(classes)}

    y = np.asarray([mapping[str(value)] for value in labels], dtype=np.int32)
    task = data.get("task", "auto")
    if task == "auto":
        if len(classes) == 2:
            task = "binary"
        elif 2 < len(classes) <= 50:
            task = "multiclass"
        else:
            raise ConfigError(
                f"Detected {len(classes)} label classes. Classification supports 2-50 classes; "
                "regression is not implemented."
            )
    elif task not in {"binary", "multiclass"}:
        raise ConfigError("data.task must be auto, binary, or multiclass.")
    if task == "binary" and len(classes) != 2:
        raise ConfigError(f"Binary task requires exactly 2 classes, found {len(classes)}.")
    if task == "multiclass" and len(classes) < 3:
        raise ConfigError(f"Multiclass task requires at least 3 classes, found {len(classes)}.")

    logger.info(
        "[labels] loaded %s rows, task=%s, %s classes: %s",
        len(y),
        task,
        len(classes),
        ", ".join(str(value) for value in classes[:10]),
    )
    return LabelInfo(
        keys=keys_array,
        raw_labels=np.asarray(labels, dtype=object),
        y=y,
        classes=classes,
        task=task,
        label_column=label_col,
        label_key=key_col,
        class_mapping=mapping,
    )


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _load_label_timestamps(
    label_info: LabelInfo,
    cfg: Dict[str, Any],
) -> pd.DatetimeIndex:
    data = cfg["data"]
    tcfg = data.get("temporal_split") or {}
    csv_cfg = data["csv"]
    path = data["label_path"]
    key_col = data["label_key"]
    time_col = str(tcfg["time_column"])
    try:
        frame = pd.read_csv(
            path,
            usecols=[key_col, time_col],
            dtype=str,
            keep_default_na=True,
            na_values=csv_cfg.get("na_values"),
            low_memory=False,
            **pandas_csv_options(csv_cfg),
        )
    except Exception as exc:
        raise ConfigError(
            f"Unable to read temporal split columns {key_col!r}, "
            f"{time_col!r} from {path}: {exc}"
        ) from exc
    frame.columns = [str(column) for column in frame.columns]
    keys = frame[key_col].astype("string").str.strip()
    null_keys = keys.isna() | (keys == "")
    if null_keys.any():
        raise ConfigError(
            f"Label file contains {int(null_keys.sum())} rows with an empty "
            f"{key_col!r} key while loading temporal splits."
        )

    label_index = pd.Index(label_info.keys)
    positions = label_index.get_indexer(pd.Index(keys.astype(str), dtype=object))
    matched_positions = positions[positions >= 0]
    if pd.Series(matched_positions).duplicated(keep=False).any():
        examples = keys[positions >= 0][
            pd.Series(matched_positions).duplicated(keep=False).to_numpy()
        ].tolist()[:5]
        raise ConfigError(
            f"Label file contains duplicate keys while loading temporal splits "
            f"(sample: {examples})."
        )

    values = np.empty(label_info.n_rows, dtype=object)
    filled = np.zeros(label_info.n_rows, dtype=bool)
    matched = positions >= 0
    values[positions[matched]] = frame.loc[matched, time_col].astype("string").to_numpy(
        dtype=object
    )
    filled[positions[matched]] = True
    missing = ~filled
    if missing.any():
        examples = label_index[missing].tolist()[:5]
        raise ConfigError(
            f"{int(missing.sum())} label keys have no temporal split timestamp "
            f"(sample: {examples})."
        )

    from .temporal import _parse_timestamp_values

    timestamps = _parse_timestamp_values(
        values,
        tcfg.get("date_format"),
        tcfg.get("timezone"),
    )
    invalid = timestamps.isna()
    if invalid.any():
        examples = label_index[invalid].tolist()[:5]
        raise ConfigError(
            f"{int(invalid.sum())} temporal split timestamps could not be parsed "
            f"(sample keys: {examples})."
        )
    return timestamps


def _split_metadata_for_indices(
    timestamps: Optional[pd.DatetimeIndex],
    y: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, Any]:
    labels = np.asarray(y)[indices]
    positives = int(labels.sum())
    n_rows = int(len(indices))
    subset = timestamps[indices] if timestamps is not None else None
    return {
        "n": n_rows,
        "n_positive": positives,
        "n_negative": n_rows - positives,
        "date_min": (
            subset.min().isoformat() if subset is not None and n_rows else None
        ),
        "date_max": (
            subset.max().isoformat() if subset is not None and n_rows else None
        ),
    }


def _validate_temporal_positive_counts(
    name: str,
    indices: np.ndarray,
    y: np.ndarray,
    min_positive: Optional[int],
    max_positive: Optional[int],
) -> None:
    positives = int(np.asarray(y)[indices].sum())
    n_negative = int(len(indices) - positives)
    if positives == 0 or n_negative == 0:
        raise ConfigError(
            f"Temporal split {name!r} has only one class "
            f"(positive={positives}, negative={n_negative}); "
            "AUC and KS cannot be computed."
        )
    if min_positive is not None and positives < min_positive:
        raise ConfigError(
            f"Temporal split {name!r} has {positives} positive rows, below "
            f"data.temporal_split.min_positive_count={min_positive}."
        )
    if max_positive is not None and positives > max_positive:
        raise ConfigError(
            f"Temporal split {name!r} has {positives} positive rows, above "
            f"data.temporal_split.max_positive_count={max_positive}."
        )


def _make_temporal_splits(
    label_info: LabelInfo,
    cfg: Dict[str, Any],
    logger: Any,
) -> Dict[str, np.ndarray]:
    data = cfg["data"]
    tcfg = data.get("temporal_split") or {}
    timestamps = _load_label_timestamps(label_info, cfg)
    timestamp_ns = timestamps.asi8
    if tcfg.get("oot_start") is not None:
        from .temporal import _parse_boundary

        cutoff = _parse_boundary(
            tcfg["oot_start"],
            tcfg.get("timezone"),
            "oot_start",
        )
        cutoff_ns = int(cutoff.value)
        cutoff_source = "explicit"
        oot_quantile = None
    else:
        oot_quantile = float(tcfg.get("oot_quantile", 0.5))
        cutoff_ns = int(np.quantile(timestamp_ns, oot_quantile))
        cutoff = pd.Timestamp(cutoff_ns, unit="ns", tz="UTC")
        cutoff_source = f"quantile:{oot_quantile}"

    pre_mask = timestamp_ns < cutoff_ns
    oot_mask = timestamp_ns >= cutoff_ns
    if not pre_mask.any() or not oot_mask.any():
        raise ConfigError(
            "Temporal split produced an empty pre-cutoff or OOT period. "
            "Adjust data.temporal_split.oot_start or oot_quantile."
        )

    valid_fraction = float(tcfg.get("valid_fraction", 0.25))
    pre_times = timestamp_ns[pre_mask]
    target_valid_rows = max(
        1,
        min(
            int(pre_mask.sum()) - 1,
            int(np.floor(valid_fraction * int(pre_mask.sum()))),
        ),
    )
    unique_pre_times = np.unique(pre_times)
    candidate_index = max(
        0,
        min(
            len(unique_pre_times) - 1,
            int(np.searchsorted(
                unique_pre_times,
                np.sort(pre_times)[-target_valid_rows],
                side="left",
            )),
        ),
    )
    valid_start_ns = int(unique_pre_times[candidate_index])
    while (
        candidate_index + 1 < len(unique_pre_times)
        and int(np.count_nonzero(pre_times >= valid_start_ns)) > target_valid_rows
    ):
        candidate_index += 1
        valid_start_ns = int(unique_pre_times[candidate_index])
    valid_mask = pre_mask & (timestamp_ns >= valid_start_ns)
    train_mask = pre_mask & (timestamp_ns < valid_start_ns)
    if not train_mask.any() or not valid_mask.any():
        raise ConfigError(
            "Temporal validation split is empty. Reduce "
            "data.temporal_split.valid_fraction or check the date distribution."
        )

    train_idx = np.sort(np.flatnonzero(train_mask).astype(np.int64))
    valid_idx = np.sort(np.flatnonzero(valid_mask).astype(np.int64))
    oot_idx = np.sort(np.flatnonzero(oot_mask).astype(np.int64))
    min_positive = tcfg.get("min_positive_count")
    max_positive = tcfg.get("max_positive_count")
    for name, indices in (
        ("train", train_idx),
        ("valid", valid_idx),
        ("oot", oot_idx),
    ):
        _validate_temporal_positive_counts(
            name,
            indices,
            label_info.y,
            None if min_positive is None else int(min_positive),
            None if max_positive is None else int(max_positive),
        )

    temporal_psi = cfg["filtering"].get("temporal_psi") or {}
    if temporal_psi.get("enabled"):
        from .temporal import _parse_boundary

        for key in (
            "reference_start",
            "reference_end",
            "comparison_start",
            "comparison_end",
        ):
            boundary = _parse_boundary(
                temporal_psi[key],
                temporal_psi.get("timezone"),
                key,
            )
            if boundary.value > cutoff_ns or (
                key.endswith("_start") and boundary.value >= cutoff_ns
            ):
                raise ConfigError(
                    "filtering.temporal_psi windows must remain within the "
                    "pre-cutoff period so OOT data is not used for feature filtering."
                )

    actual_valid_fraction = float(len(valid_idx) / max(1, int(pre_mask.sum())))
    if not 0.2 <= actual_valid_fraction <= 0.3:
        logger.warning(
            "[split] temporal valid fraction is %.3f (requested %.3f); "
            "date ties can move the boundary",
            actual_valid_fraction,
            valid_fraction,
        )

    metadata = {
        "strategy": "temporal",
        "evaluation_split": "oot",
        "time_column": tcfg.get("time_column"),
        "timezone": tcfg.get("timezone"),
        "date_format": tcfg.get("date_format"),
        "oot_start": cutoff.isoformat(),
        "oot_cutoff_source": cutoff_source,
        "oot_quantile": oot_quantile,
        "valid_start": pd.Timestamp(
            valid_start_ns, unit="ns", tz="UTC"
        ).isoformat(),
        "valid_fraction_requested": valid_fraction,
        "valid_fraction_actual": actual_valid_fraction,
        "positive_count_limits": {
            "min": min_positive,
            "max": max_positive,
        },
        "splits": {
            "train": _split_metadata_for_indices(
                timestamps, label_info.y, train_idx
            ),
            "valid": _split_metadata_for_indices(
                timestamps, label_info.y, valid_idx
            ),
            "oot": _split_metadata_for_indices(
                timestamps, label_info.y, oot_idx
            ),
        },
    }
    cfg.setdefault("_runtime", {})["split_metadata"] = metadata
    logger.info(
        "[split] temporal train=%s valid=%s oot=%s (cutoff=%s, valid_start=%s, source=%s, seed=%s)",
        len(train_idx),
        len(valid_idx),
        len(oot_idx),
        metadata["oot_start"],
        metadata["valid_start"],
        cutoff_source,
        cfg["project"]["random_seed"],
    )
    return {
        "train": train_idx,
        "valid": valid_idx,
        "test": oot_idx,
        "filter_test": valid_idx,
        "filter_indices": np.sort(
            np.concatenate([train_idx, valid_idx]).astype(np.int64)
        ),
    }


def _subset_label_info(
    label_info: LabelInfo,
    indices: np.ndarray,
) -> LabelInfo:
    return LabelInfo(
        keys=label_info.keys[indices],
        raw_labels=label_info.raw_labels[indices],
        y=label_info.y[indices],
        classes=label_info.classes,
        task=label_info.task,
        label_column=label_info.label_column,
        label_key=label_info.label_key,
        class_mapping=label_info.class_mapping,
    )


def make_splits(
    label_info: LabelInfo,
    cfg: Dict[str, Any],
    logger: Any,
    eligible_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    data = cfg["data"]
    restricted = eligible_indices is not None
    if eligible_indices is None:
        eligible = np.arange(label_info.n_rows, dtype=np.int64)
    else:
        eligible = np.asarray(eligible_indices, dtype=np.int64)
        if eligible.ndim != 1:
            raise ConfigError("Eligible row indices must be one-dimensional.")
        if (
            len(eligible)
            and (
                eligible.min() < 0
                or eligible.max() >= label_info.n_rows
            )
        ):
            raise ConfigError("Eligible row indices contain out-of-range values.")
        if len(np.unique(eligible)) != len(eligible):
            raise ConfigError("Eligible row indices must be unique.")
        eligible = np.sort(eligible)
    if len(eligible) == 0:
        raise ConfigError(
            "data.drop_unmatched_rows removed every label row because no "
            "feature-file keys matched."
        )
    split_info = (
        label_info if len(eligible) == label_info.n_rows else _subset_label_info(label_info, eligible)
    )
    if str(data.get("split_strategy", "random")).lower() == "temporal":
        splits = _make_temporal_splits(split_info, cfg, logger)
    else:
        splits = _make_random_splits(split_info, cfg, logger)

    if restricted:
        splits = {
            name: eligible[indices]
            for name, indices in splits.items()
        }
        if "filter_indices" not in splits:
            splits["filter_indices"] = eligible

    metadata = cfg.setdefault("_runtime", {}).setdefault(
        "split_metadata", {}
    )
    metadata["eligible_rows"] = int(len(eligible))
    metadata["dropped_unmatched_rows"] = int(
        label_info.n_rows - len(eligible)
    )
    metadata["drop_unmatched_rows"] = bool(restricted)
    split_names = ("train", "valid", "oot") if "oot" in splits else ("train", "valid", "test")
    timestamps = (
        _load_label_timestamps(label_info, cfg)
        if metadata.get("strategy") == "temporal"
        else None
    )
    metadata["splits"] = {
        name: _split_metadata_for_indices(
            timestamps,
            label_info.y,
            np.asarray(splits[name], dtype=np.int64),
        )
        for name in split_names
    }
    return splits


def _make_random_splits(
    label_info: LabelInfo,
    cfg: Dict[str, Any],
    logger: Any,
) -> Dict[str, np.ndarray]:
    data = cfg["data"]
    n_rows = label_info.n_rows
    seed = int(cfg["project"]["random_seed"])
    train_fraction = float(data["train_fraction"])
    valid_fraction = float(data["valid_fraction"])
    test_fraction = float(data["test_fraction"])
    indices = np.arange(n_rows)
    stratify = label_info.y if data.get("stratify") else None

    if stratify is not None:
        counts = np.bincount(label_info.y, minlength=label_info.n_classes)
        if counts.min() < 4:
            logger.warning("[split] too few rows per class for stratified split; using random split")
            stratify = None

    try:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=valid_fraction + test_fraction,
            random_state=seed,
            stratify=None if stratify is None else stratify,
        )
        temp_stratify = None if stratify is None else stratify[temp_idx]
        relative_test = test_fraction / (valid_fraction + test_fraction)
        valid_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test,
            random_state=seed + 1,
            stratify=temp_stratify,
        )
    except ValueError as exc:
        logger.warning("[split] stratified split failed (%s); falling back to permutation", exc)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        n_train = round(n_rows * train_fraction)
        n_valid = round(n_rows * valid_fraction)
        train_idx = shuffled[:n_train]
        valid_idx = shuffled[n_train : n_train + n_valid]
        test_idx = shuffled[n_train + n_valid :]

    splits = {
        "train": np.sort(train_idx.astype(np.int64)),
        "valid": np.sort(valid_idx.astype(np.int64)),
        "test": np.sort(test_idx.astype(np.int64)),
    }
    logger.info(
        "[split] train=%s valid=%s test=%s (seed=%s)",
        len(splits["train"]),
        len(splits["valid"]),
        len(splits["test"]),
        seed,
    )
    cfg.setdefault("_runtime", {})["split_metadata"] = {
        "strategy": "random",
        "evaluation_split": "test",
    }
    return splits


def save_json(path: str | Path, payload: Any) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
