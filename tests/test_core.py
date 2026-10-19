from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_ml_pipeline.config import ConfigError, load_config
from auto_ml_pipeline.logging_utils import configure_logging
from auto_ml_pipeline.metrics import (
    ks_statistic,
    multiclass_metrics,
    select_binary_threshold,
)
from auto_ml_pipeline.schema import discover_sources, load_labels
from auto_ml_pipeline.stats import _information_value, _psi
from auto_ml_pipeline.synthetic import generate_synthetic_dataset


def test_information_value_and_psi() -> None:
    # Perfect separation between classes should produce a large IV.
    iv = _information_value(
        bad_counts=np.asarray([100, 0]),
        good_counts=np.asarray([0, 100]),
    )
    assert iv > 1.0
    # Identical distributions have zero PSI; a shifted distribution does not.
    reference = np.asarray([100, 200, 300, 400])
    assert _psi(reference, reference) < 1e-6
    assert _psi(reference, np.asarray([400, 300, 200, 100])) > 0.1


def test_ks_and_threshold() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(labels, scores) == pytest.approx(1.0)
    selection = select_binary_threshold(labels, scores)
    assert 0.0 <= selection["threshold"] <= 1.0
    assert selection["f1"] == pytest.approx(1.0)


def test_multiclass_metrics() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    metrics = multiclass_metrics(labels, probabilities)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert "auc_ovr_macro" in metrics


def test_load_config_and_overrides(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    features = tmp_path / "features.csv"
    labels.write_text("id,target\n1,0\n2,1\n", encoding="utf-8")
    features.write_text("id,x\n1,0.1\n2,0.2\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  feature_paths: [features.csv]",
                "  label_path: labels.csv",
                "  feature_key: id",
                "  label_key: id",
                "  label_column: target",
                "project:",
                f"  work_dir: {tmp_path / 'run'}",
                "tuning:",
                "  n_trials: 3",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_path, overrides=["tuning.n_trials=7"])
    assert cfg["tuning"]["n_trials"] == 7
    assert Path(cfg["data"]["feature_paths"][0]).exists()


def test_config_rejects_bad_fractions(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    features = tmp_path / "features.csv"
    labels.write_text("id,target\n1,0\n", encoding="utf-8")
    features.write_text("id,x\n1,0.1\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  feature_paths: [features.csv]",
                "  label_path: labels.csv",
                "  train_fraction: 0.8",
                "  valid_fraction: 0.2",
                "  test_fraction: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_synthetic_generator(tmp_path: Path) -> None:
    info = generate_synthetic_dataset(
        tmp_path / "data",
        rows=300,
        n_features=18,
        target_gb=0,
        n_feature_files=2,
        seed=3,
        chunk_rows=100,
        pad_len=5,
        heartbeat_seconds=1,
    )
    assert info["rows"] == 300
    assert info["n_features"] == 18
    assert info["actual_bytes"] > 0
    for path in info["feature_files"]:
        assert Path(path).exists()
    assert Path(info["label_file"]).exists()
    with (tmp_path / "data" / "dataset_info.json").open(encoding="utf-8") as handle:
        assert json.load(handle)["rows"] == 300


def test_row_concat_partition_sampling_dtype_conflicts(tmp_path: Path) -> None:
    n_parts = 3
    rows_per_part = 200
    n_rows = n_parts * rows_per_part
    feature_paths = []
    for part in range(n_parts):
        start = part * rows_per_part
        stop = start + rows_per_part
        identifiers = [f"id_{index}" for index in range(start, stop)]
        values = (
            [1000 + index % 10 for index in range(start, stop)] if part == 0 else list(range(1000, 1000 + stop - start))
        )
        categories = (
            [f"category_{index % 20}" for index in range(start, stop)]
            if part == 0
            else [f"category_{index}" for index in range(start, stop)]
        )
        constant_partition = [7] * (stop - start) if part == 0 else list(range(start, stop))
        null_partition = [None] * (stop - start) if part == 0 else [value + 0.5 for value in range(start, stop)]
        all_null = [None] * (stop - start)
        path = tmp_path / f"features_part_{part}.csv"
        pd.DataFrame(
            {
                "id": identifiers,
                "col4": values,
                "col13": categories,
                "stable": [index % 2 for index in range(start, stop)],
                "constant_partition": constant_partition,
                "null_partition": null_partition,
                "all_null": all_null,
            }
        ).to_csv(path, index=False)
        feature_paths.append(str(path))

    pd.DataFrame(
        {
            "id": [f"id_{index}" for index in range(n_rows)],
            "target": [index % 2 for index in range(n_rows)],
        }
    ).to_csv(tmp_path / "labels.csv", index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                f"  work_dir: {tmp_path / 'run'}",
                "data:",
                "  feature_paths:",
                f"    - {tmp_path / 'features_part_*.csv'}",
                f"  label_path: {tmp_path / 'labels.csv'}",
                "  feature_key: id",
                "  label_key: id",
                "  label_column: target",
                "  concat_mode: row",
                "  include_label_features: false",
                "  sample_rows: 200",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    logger = configure_logging("WARNING")
    label_info = load_labels(cfg, logger)
    sources, _ = discover_sources(cfg, label_info, logger)
    feature_sources = [source for source in sources if source.kind == "features"]

    assert [source.dtypes["col4"] for source in feature_sources] == [
        "numeric",
        "numeric",
        "numeric",
    ]
    assert [source.dtypes["col13"] for source in feature_sources] == [
        "categorical",
        "categorical",
        "categorical",
    ]
    assert [source.dtypes["constant_partition"] for source in feature_sources] == [
        "numeric",
        "numeric",
        "numeric",
    ]
    assert [source.dtypes["null_partition"] for source in feature_sources] == [
        "numeric",
        "numeric",
        "numeric",
    ]
    assert [source.dtypes["all_null"] for source in feature_sources] == [
        "categorical",
        "categorical",
        "categorical",
    ]
