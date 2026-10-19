"""Temporal reference and comparison windows for PSI filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ConfigError
from .schema import LabelInfo


@dataclass
class TemporalPSIWindows:
    reference_mask: np.ndarray
    comparison_mask: np.ndarray
    metadata: Dict[str, Any]


def _parse_boundary(value: Any, timezone: Optional[str], name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid filtering.temporal_psi.{name} timestamp: {value!r}"
        ) from exc
    try:
        if parsed.tzinfo is None:
            parsed = (
                parsed.tz_localize(timezone)
                if timezone
                else parsed.tz_localize("UTC")
            )
        return parsed.tz_convert("UTC")
    except Exception as exc:
        raise ConfigError(
            f"Unable to apply timezone {timezone!r} to temporal PSI boundary "
            f"{name}={value!r}: {exc}"
        ) from exc


def _parse_timestamp_values(
    values: np.ndarray,
    date_format: Optional[str],
    timezone: Optional[str],
) -> pd.DatetimeIndex:
    text = pd.Series(values, dtype="string")
    format_value = date_format or "mixed"
    has_offset = bool(
        text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False).any()
    )
    try:
        if has_offset:
            parsed = pd.to_datetime(
                text, format=format_value, errors="coerce", utc=True
            )
        else:
            parsed = pd.to_datetime(text, format=format_value, errors="coerce")
            if isinstance(parsed.dtype, pd.DatetimeTZDtype):
                parsed = parsed.dt.tz_convert("UTC")
            elif timezone:
                parsed = parsed.dt.tz_localize(timezone).dt.tz_convert("UTC")
            else:
                parsed = parsed.dt.tz_localize("UTC")
    except Exception as exc:
        raise ConfigError(
            "Unable to parse filtering.temporal_psi.time_column with "
            f"date_format={date_format!r}: {exc}"
        ) from exc
    return pd.DatetimeIndex(parsed)


def _load_aligned_timestamps(
    cfg: Dict[str, Any],
    label_info: LabelInfo,
) -> Tuple[pd.DatetimeIndex, Dict[str, Any]]:
    data = cfg["data"]
    csv_cfg = data["csv"]
    tcfg = cfg["filtering"]["temporal_psi"]
    time_path = str(
        Path(tcfg.get("time_path") or data["label_path"]).expanduser().resolve()
    )
    time_key = str(tcfg.get("time_key") or data["label_key"])
    time_column = str(tcfg["time_column"])

    label_index = pd.Index(label_info.keys, dtype=object)
    aligned_values = np.full(len(label_index), np.nan, dtype=object)
    filled = np.zeros(len(label_index), dtype=bool)
    n_time_rows = 0
    try:
        reader = pd.read_csv(
            time_path,
            sep=csv_cfg.get("sep", ","),
            encoding=csv_cfg.get("encoding", "utf-8"),
            compression=csv_cfg.get("compression", "infer"),
            usecols=[time_key, time_column],
            dtype=str,
            chunksize=int(csv_cfg.get("chunksize", 2000)),
            keep_default_na=True,
            na_values=csv_cfg.get("na_values"),
            low_memory=False,
        )
    except Exception as exc:
        raise ConfigError(
            f"Unable to read temporal PSI time columns {time_key!r}, "
            f"{time_column!r} from {time_path}: {exc}"
        ) from exc

    for chunk in reader:
        chunk.columns = [str(column) for column in chunk.columns]
        n_time_rows += len(chunk)
        chunk_keys = chunk[time_key].astype("string").str.strip()
        null_keys = chunk_keys.isna() | (chunk_keys == "")
        if null_keys.any():
            raise ConfigError(
                f"Temporal PSI time file contains {int(null_keys.sum())} rows "
                f"with an empty {time_key!r} key."
            )
        positions = label_index.get_indexer(
            pd.Index(chunk_keys.astype(str), dtype=object)
        )
        matched = positions >= 0
        if not matched.any():
            continue
        relevant_positions = positions[matched]
        duplicate = filled[relevant_positions] | pd.Series(
            relevant_positions
        ).duplicated(keep=False).to_numpy()
        if duplicate.any():
            examples = chunk_keys[matched][duplicate].tolist()[:5]
            raise ConfigError(
                "Temporal PSI time file contains duplicate keys "
                f"(sample: {examples})."
            )
        aligned_values[relevant_positions] = chunk.loc[
            matched, time_column
        ].astype("string").to_numpy(dtype=object)
        filled[relevant_positions] = True

    if n_time_rows == 0:
        raise ConfigError(f"Temporal PSI time file has no rows: {time_path}")

    missing = ~filled
    if missing.any():
        examples = label_index[missing].tolist()[:5]
        raise ConfigError(
            f"{int(missing.sum())} label keys have no timestamp in {time_path} "
            f"(sample: {examples})."
        )

    timestamps = _parse_timestamp_values(
        aligned_values,
        tcfg.get("date_format"),
        tcfg.get("timezone"),
    )
    invalid = timestamps.isna()
    if invalid.any():
        examples = label_index[invalid].tolist()[:5]
        raise ConfigError(
            f"{int(invalid.sum())} timestamps could not be parsed "
            f"(sample keys: {examples})."
        )

    metadata = {
        "time_path": time_path,
        "time_key": time_key,
        "time_column": time_column,
        "date_format": tcfg.get("date_format"),
        "timezone": tcfg.get("timezone"),
        "n_timestamp_rows": n_time_rows,
    }
    return timestamps, metadata


def load_temporal_psi_windows(
    cfg: Dict[str, Any],
    label_info: LabelInfo,
    logger: Any,
) -> Optional[TemporalPSIWindows]:
    tcfg = cfg["filtering"]["temporal_psi"]
    if not tcfg.get("enabled"):
        return None

    timestamps, metadata = _load_aligned_timestamps(cfg, label_info)
    reference_start = _parse_boundary(
        tcfg["reference_start"], tcfg.get("timezone"), "reference_start"
    )
    reference_end = _parse_boundary(
        tcfg["reference_end"], tcfg.get("timezone"), "reference_end"
    )
    comparison_start = _parse_boundary(
        tcfg["comparison_start"], tcfg.get("timezone"), "comparison_start"
    )
    comparison_end = _parse_boundary(
        tcfg["comparison_end"], tcfg.get("timezone"), "comparison_end"
    )

    if reference_start >= reference_end:
        raise ConfigError(
            "filtering.temporal_psi.reference_start must be before reference_end."
        )
    if comparison_start >= comparison_end:
        raise ConfigError(
            "filtering.temporal_psi.comparison_start must be before comparison_end."
        )
    if reference_end > comparison_start:
        raise ConfigError(
            "Temporal PSI reference and comparison windows must not overlap, "
            "and the reference window must end no later than the comparison starts."
        )

    reference_mask = np.asarray(
        (timestamps >= reference_start) & (timestamps < reference_end),
        dtype=bool,
    )
    comparison_mask = np.asarray(
        (timestamps >= comparison_start) & (timestamps < comparison_end),
        dtype=bool,
    )
    min_rows = int(tcfg.get("min_rows_per_period", 1))
    reference_rows = int(reference_mask.sum())
    comparison_rows = int(comparison_mask.sum())
    if reference_rows < min_rows:
        raise ConfigError(
            f"Temporal PSI reference period has {reference_rows} rows, below "
            f"min_rows_per_period={min_rows}."
        )
    if comparison_rows < min_rows:
        raise ConfigError(
            f"Temporal PSI comparison period has {comparison_rows} rows, below "
            f"min_rows_per_period={min_rows}."
        )

    metadata.update(
        {
            "reference_start": reference_start.isoformat(),
            "reference_end": reference_end.isoformat(),
            "comparison_start": comparison_start.isoformat(),
            "comparison_end": comparison_end.isoformat(),
            "reference_rows": reference_rows,
            "comparison_rows": comparison_rows,
            "min_rows_per_period": min_rows,
        }
    )
    logger.info(
        "[temporal-psi] reference=[%s, %s) rows=%s comparison=[%s, %s) rows=%s",
        reference_start.isoformat(),
        reference_end.isoformat(),
        reference_rows,
        comparison_start.isoformat(),
        comparison_end.isoformat(),
        comparison_rows,
    )
    return TemporalPSIWindows(
        reference_mask=reference_mask,
        comparison_mask=comparison_mask,
        metadata=metadata,
    )
