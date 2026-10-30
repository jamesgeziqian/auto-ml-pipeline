from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from auto_ml_pipeline.config import load_config
from auto_ml_pipeline.pipeline import run_pipeline


def _write_dataset(tmp_path: Path, n_labels: int, n_matched: int) -> tuple[Path, Path]:
    labels = pd.DataFrame(
        {
            "customer_id": np.arange(n_labels),
            "target": np.arange(n_labels) % 2,
            "label_extra": (np.arange(n_labels) % 3).astype(float),
        }
    )
    features = pd.DataFrame(
        {
            "customer_id": np.arange(n_matched),
            "signal": np.arange(n_matched) % 7,
        }
    )
    labels_path = tmp_path / "labels.csv"
    features_path = tmp_path / "features.csv"
    labels.to_csv(labels_path, index=False)
    features.to_csv(features_path, index=False)
    return labels_path, features_path


def _config(
    tmp_path: Path,
    labels_path: Path,
    features_path: Path,
) -> Path:
    config = {
        "project": {
            "work_dir": str(tmp_path / "run"),
            "log_level": "ERROR",
            "heartbeat_seconds": 1,
        },
        "data": {
            "feature_paths": [str(features_path)],
            "label_path": str(labels_path),
            "feature_key": "customer_id",
            "label_key": "customer_id",
            "label_column": "target",
            "concat_mode": "column",
            "drop_unmatched_rows": True,
            "column_block_size": 8,
            "sample_rows": 100,
            "train_fraction": 0.6,
            "valid_fraction": 0.2,
            "test_fraction": 0.2,
        },
        "filtering": {
            "max_nan_ratio": 1.0,
            "min_unique": 2,
            "min_abs_pearson": 0.0,
            "min_iv": 0.0,
            "max_iv": None,
            "max_psi": 10.0,
        },
        "preprocessing": {
            "categorical_encoding": "native",
            "max_categories": 16,
            "min_category_frequency": 1,
            "add_missing_indicator": False,
        },
        "model": {
            "num_boost_round": 10,
            "early_stopping_rounds": 3,
            "num_threads": 1,
            "log_period": 100,
        },
        "tuning": {"enabled": False},
        "metrics": {"threshold": "auto", "plots": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_drop_unmatched_rows_excludes_label_only_rows_from_splits(
    tmp_path: Path,
) -> None:
    labels_path, features_path = _write_dataset(
        tmp_path,
        n_labels=120,
        n_matched=90,
    )
    summary = run_pipeline(
        load_config(_config(tmp_path, labels_path, features_path))
    )
    run_dir = Path(summary["work_dir"])
    splits = np.load(run_dir / "splits.npz")

    assert len(splits["train"]) + len(splits["valid"]) + len(splits["test"]) == 90
    assert np.all(splits["filter_indices"] < 90)
    assert np.all(splits["train"] < 90)
    assert np.all(splits["valid"] < 90)
    assert np.all(splits["test"] < 90)

    metadata = json.loads((run_dir / "split_metadata.json").read_text())
    assert metadata["drop_unmatched_rows"] is True
    assert metadata["eligible_rows"] == 90
    assert metadata["dropped_unmatched_rows"] == 30

    timings = pd.read_csv(run_dir / "stage_timings.csv")
    split_details = json.loads(
        timings.loc[timings["stage"] == "split", "details"].iloc[0]
    )
    assert split_details["eligible_rows"] == 90
    assert split_details["dropped_unmatched_rows"] == 30


def test_drop_unmatched_rows_fails_when_no_feature_keys_match(
    tmp_path: Path,
) -> None:
    labels_path, features_path = _write_dataset(
        tmp_path,
        n_labels=20,
        n_matched=0,
    )
    with pytest.raises(ValueError, match="removed every label row"):
        run_pipeline(load_config(_config(tmp_path, labels_path, features_path)))
