from __future__ import annotations

import copy
import csv as csv_module
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_ml_pipeline.cache import CacheStore, build_cache
from auto_ml_pipeline.config import DEFAULTS, ConfigError, validate_config
from auto_ml_pipeline.logging_utils import Heartbeat
from auto_ml_pipeline.schema import LabelInfo, SourceSpec

FEATURES = ["num_a", "num_b", "cat_a", "dt_a", "id_a"]
DTYPES = {
    "num_a": "numeric",
    "num_b": "numeric",
    "cat_a": "categorical",
    "dt_a": "datetime",
    "id_a": "identifier",
}


def _features_frame(rows: int, duplicate: bool = False) -> pd.DataFrame:
    values = {
        "num_a": ["1.5", "2", "", "NA", "3.25"],
        "num_b": ["0", "1", "0", "1", "1"],
        "cat_a": ["a", "b", "", " c ", "dd"],
        "dt_a": [
            "2024-01-01",
            "2024-02-03 04:05:06",
            "",
            "2024-03-04T05:06:07",
            "2024-04-05",
        ],
    }
    records = []
    for index in range(rows):
        key = "" if index == 7 else f"k{index:03d}"
        if index == 10:
            key = " k010 "
        if duplicate and index in {3, 4}:
            key = "k003"
        record = {"customer_id": key}
        for column in FEATURES[:-1]:
            record[column] = values[column][index % len(values[column])]
        record["id_a"] = f"id_{index}"
        records.append(record)
    return pd.DataFrame(records)


def _write_features(path: Path, rows: int, duplicate: bool = False) -> None:
    _features_frame(rows, duplicate).to_csv(path, index=False)


def _labels(n_rows: int) -> LabelInfo:
    keys = np.asarray([f"k{index:03d}" for index in range(n_rows)], dtype=object)
    y = np.asarray([index % 2 for index in range(n_rows)], dtype=np.int32)
    return LabelInfo(
        keys=keys,
        raw_labels=y.astype(str),
        y=y,
        classes=[0, 1],
        task="binary",
        label_column="target",
        label_key="customer_id",
        class_mapping={"0": 0, "1": 1},
    )


def _spec(path: Path, rename: bool = False) -> SourceSpec:
    logical = {
        column: (f"src__{column}" if rename else column) for column in FEATURES
    }
    return SourceSpec(
        source_id="features",
        path=str(path),
        kind="features",
        key_col="customer_id",
        label_col=None,
        feature_columns=list(FEATURES),
        source_to_logical=dict(logical),
        logical_to_source={value: key for key, value in logical.items()},
        dtypes=dict(DTYPES),
        sample_bytes=1e6,
    )


def _config(
    engine: str,
    numeric_dtype: str = "float32",
    sep: str = ",",
    quoting: str = "minimal",
) -> dict:
    return {
        "data": {
            "feature_key": "customer_id",
            "label_key": "customer_id",
            "label_column": "target",
            "concat_mode": "row",
            "column_block_size": 2,
            "max_source_columns_in_memory": 1,
            "wide_block_size": 2,
            "wide_row_group_size": 5,
            "cache_numeric_dtype": numeric_dtype,
            "ingest_target_chunk_bytes": 1 << 20,
            "sample_rows": 100,
            "csv": {
                "sep": sep,
                "quoting": quoting,
                "encoding": "utf-8",
                "chunksize": 17,
                "engine": engine,
                "na_values": ["", "NA", "NaN", "nan", "null", "None"],
            },
        },
        "filtering": {"min_row_coverage": 0.0},
    }


def _run(
    tmp_path: Path,
    engine: str,
    rows: int = 60,
    label_rows: int = 40,
    path: Path | None = None,
    name: str | None = None,
    rename: bool = False,
    numeric_dtype: str = "float32",
    sep: str = ",",
    quoting: str = "minimal",
):
    if path is None:
        data_dir = tmp_path / f"data_{engine}"
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "features.csv"
        _write_features(path, rows)
    work_dir = tmp_path / f"run_{name or engine}"
    logger = logging.getLogger(f"arrow_test_{engine}")
    heartbeat = Heartbeat(logger, 60)
    manifest = build_cache(
        _config(engine, numeric_dtype, sep, quoting),
        [_spec(path, rename=rename)],
        _labels(label_rows),
        work_dir,
        logger,
        heartbeat,
    )
    return manifest, CacheStore.load(work_dir), work_dir


@pytest.mark.parametrize("numeric_dtype", ["float32", "float64"])
def test_arrow_ingest_matches_pandas(tmp_path: Path, numeric_dtype: str) -> None:
    pandas_manifest, pandas_store, pandas_dir = _run(
        tmp_path, "pandas", numeric_dtype=numeric_dtype
    )
    arrow_manifest, arrow_store, arrow_dir = _run(
        tmp_path, "arrow", numeric_dtype=numeric_dtype
    )

    pandas_source = pandas_manifest["sources"]["features"]
    arrow_source = arrow_manifest["sources"]["features"]
    assert pandas_source["csv_engine"] == "pandas"
    assert arrow_source["csv_engine"] == "arrow"
    for key in ["rows_seen", "rows_written", "missing_keys", "duplicate_row_indices"]:
        assert pandas_source[key] == arrow_source[key], key
    assert arrow_source["rows_seen"] == 60
    assert arrow_source["rows_written"] == 39
    assert arrow_source["missing_keys"] == 21
    assert len(arrow_source["part_groups"]) == 3

    columns = pandas_store.columns()
    assert columns == arrow_store.columns()
    pd.testing.assert_frame_equal(
        pandas_store.read_columns(columns),
        arrow_store.read_columns(columns),
    )
    np.testing.assert_array_equal(
        np.load(pandas_dir / "cache" / "row_coverage.npy"),
        np.load(arrow_dir / "cache" / "row_coverage.npy"),
    )


def test_arrow_ingest_renames_duplicate_columns(tmp_path: Path) -> None:
    _, pandas_store, _ = _run(tmp_path, "pandas", rename=True, name="pandas_renamed")
    manifest, arrow_store, _ = _run(tmp_path, "arrow", rename=True, name="arrow_renamed")
    assert "src__num_a" in manifest["sources"]["features"]["logical_columns"]
    pd.testing.assert_frame_equal(
        pandas_store.read_columns(pandas_store.columns()),
        arrow_store.read_columns(arrow_store.columns()),
    )


def test_arrow_ingest_rejects_duplicate_keys(tmp_path: Path) -> None:
    data_dir = tmp_path / "dup"
    data_dir.mkdir()
    path = data_dir / "features.csv"
    _write_features(path, rows=10, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate join keys"):
        build_cache(
            _config("arrow"),
            [_spec(path)],
            _labels(10),
            tmp_path / "run_dup",
            logging.getLogger("arrow_dup"),
            Heartbeat(logging.getLogger("arrow_dup"), 60),
        )


def test_config_accepts_only_known_csv_engines(tmp_path: Path) -> None:
    data_dir = tmp_path / "cfg"
    data_dir.mkdir()
    feature = data_dir / "features.csv"
    labels = data_dir / "labels.csv"
    feature.write_text("customer_id,num_a\nk000,1\n", encoding="utf-8")
    labels.write_text("customer_id,target\nk000,1\n", encoding="utf-8")
    cfg = copy.deepcopy(DEFAULTS)
    cfg["data"]["feature_paths"] = [str(feature)]
    cfg["data"]["label_path"] = str(labels)
    cfg["data"]["csv"]["engine"] = "arrow"
    validate_config(cfg)
    cfg["data"]["csv"]["engine"] = "spark"
    with pytest.raises(ConfigError, match="data.csv.engine"):
        validate_config(cfg)


def test_arrow_ingest_reads_gzip(tmp_path: Path) -> None:
    frame = _features_frame(30)
    plain = tmp_path / "features_plain.csv"
    frame.to_csv(plain, index=False)
    compressed = tmp_path / "features.csv.gz"
    frame.to_csv(compressed, index=False, compression="gzip")

    _, plain_store, _ = _run(tmp_path, "arrow", rows=30, label_rows=30, path=plain, name="plain")
    manifest, gzip_store, _ = _run(
        tmp_path, "arrow", rows=30, label_rows=30, path=compressed, name="gzip"
    )
    assert manifest["sources"]["features"]["rows_seen"] == 30
    pd.testing.assert_frame_equal(
        plain_store.read_columns(plain_store.columns()),
        gzip_store.read_columns(gzip_store.columns()),
    )


def test_arrow_ingest_falls_back_on_unparseable_numeric(tmp_path: Path) -> None:
    frame = _features_frame(20)
    frame.loc[3, "num_a"] = "not-a-number"
    path = tmp_path / "unparseable.csv"
    frame.to_csv(path, index=False)

    manifest, store, _ = _run(
        tmp_path, "arrow", rows=20, label_rows=20, path=path, name="fallback"
    )
    assert manifest["sources"]["features"]["csv_engine"] == "pandas"
    values = store.read_columns(["num_a"])["num_a"]
    assert pd.isna(values.iloc[3])


def test_arrow_ingest_honours_separator(tmp_path: Path) -> None:
    frame = _features_frame(30)
    comma = tmp_path / "comma.csv"
    frame.to_csv(comma, index=False)
    tab = tmp_path / "tab.tsv"
    frame.to_csv(tab, index=False, sep="\t")

    _, comma_store, _ = _run(
        tmp_path, "arrow", rows=30, label_rows=30, path=comma, name="comma"
    )
    _, tab_store, _ = _run(
        tmp_path, "arrow", rows=30, label_rows=30, path=tab, name="tab", sep="\t"
    )
    pd.testing.assert_frame_equal(
        comma_store.read_columns(comma_store.columns()),
        tab_store.read_columns(tab_store.columns()),
    )


def test_etx_separator_with_unquoted_quote_character(tmp_path: Path) -> None:
    frame = _features_frame(30)
    frame.loc[5, "cat_a"] = '"unterminated'
    path = tmp_path / "features_etx.csv"
    frame.to_csv(path, index=False, sep="\x03", quoting=csv_module.QUOTE_NONE)

    with pytest.raises(ValueError, match="Malformed CSV"):
        _run(
            tmp_path,
            "arrow",
            rows=30,
            label_rows=30,
            path=path,
            name="etx_minimal",
            sep="\\x03",
            quoting="minimal",
        )

    manifest, store, _ = _run(
        tmp_path,
        "arrow",
        rows=30,
        label_rows=30,
        path=path,
        name="etx_none",
        sep="\\x03",
        quoting="none",
    )
    assert manifest["sources"]["features"]["rows_seen"] == 30
    assert store.read_columns(["cat_a"])["cat_a"].iloc[5] == '"unterminated'
    _, pandas_store, _ = _run(
        tmp_path,
        "pandas",
        rows=30,
        label_rows=30,
        path=path,
        name="etx_pandas",
        sep="\\x03",
        quoting="none",
    )
    pd.testing.assert_frame_equal(
        pandas_store.read_columns(pandas_store.columns()),
        store.read_columns(store.columns()),
    )
