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


def make_splits(
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
    return splits


def save_json(path: str | Path, payload: Any) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
