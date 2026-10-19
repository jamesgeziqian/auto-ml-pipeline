from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from auto_ml_pipeline.config import ConfigError, load_config
from auto_ml_pipeline.logging_utils import configure_logging
from auto_ml_pipeline.pipeline import run_pipeline
from auto_ml_pipeline.schema import load_labels
from auto_ml_pipeline.stats import compute_stats_and_filter
from auto_ml_pipeline.synthetic import generate_synthetic_dataset
from auto_ml_pipeline.temporal import load_temporal_psi_windows


def _write_temporal_config(tmp_path: Path, **overrides: object) -> Path:
    labels = tmp_path / "labels.csv"
    features = tmp_path / "features.csv"
    rows = []
    for index in range(12):
        date = "01/01/2025" if index < 6 else "02/01/2025"
        rows.append({"id": f"id_{index}", "target": index % 2, "event_date": date})
    pd.DataFrame(rows).to_csv(labels, index=False)
    pd.DataFrame(
        {"id": [row["id"] for row in rows], "unused": np.arange(len(rows))}
    ).to_csv(features, index=False)

    temporal_psi = {
        "enabled": True,
        "time_path": str(labels),
        "time_key": "id",
        "time_column": "event_date",
        "date_format": "%m/%d/%Y",
        "timezone": "Asia/Shanghai",
        "reference_start": "2025-01-01",
        "reference_end": "2025-01-02",
        "comparison_start": "2025-02-01",
        "comparison_end": "2025-02-02",
        "min_rows_per_period": 1,
    }
    temporal_psi.update(overrides)
    config = {
        "project": {"work_dir": str(tmp_path / "run")},
        "data": {
            "feature_paths": [str(features)],
            "label_path": str(labels),
            "feature_key": "id",
            "label_key": "id",
            "label_column": "target",
        },
        "filtering": {"temporal_psi": temporal_psi},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class _FakeStore:
    def __init__(self, frame: pd.DataFrame, dtypes: dict[str, str]):
        self.frame = frame
        self.dtypes = dtypes
        self.n_rows = len(frame)

    def columns(self) -> list[str]:
        return list(self.frame.columns)

    def dtype(self, column: str) -> str:
        return self.dtypes[column]

    def iter_column_blocks(self, columns, block_size):
        yield list(columns), self.frame.copy()


class _SilentHeartbeat:
    def detail(self, value: str) -> None:
        pass


def test_temporal_psi_windows_align_to_label_rows(tmp_path: Path) -> None:
    cfg = load_config(_write_temporal_config(tmp_path))
    logger = configure_logging("ERROR")
    label_info = load_labels(cfg, logger)

    windows = load_temporal_psi_windows(cfg, label_info, logger)

    assert windows is not None
    assert windows.reference_mask.tolist() == [True] * 6 + [False] * 6
    assert windows.comparison_mask.tolist() == [False] * 6 + [True] * 6
    assert windows.metadata["reference_rows"] == 6
    assert windows.metadata["comparison_rows"] == 6
    assert windows.metadata["date_format"] == "%m/%d/%Y"


def test_temporal_psi_rejects_overlapping_windows(tmp_path: Path) -> None:
    cfg = load_config(
        _write_temporal_config(
            tmp_path,
            reference_end="2025-02-02",
            comparison_start="2025-02-01",
        )
    )
    logger = configure_logging("ERROR")
    label_info = load_labels(cfg, logger)

    with pytest.raises(ConfigError, match="must not overlap"):
        load_temporal_psi_windows(cfg, label_info, logger)


def test_temporal_psi_filters_shifted_features(tmp_path: Path) -> None:
    cfg = load_config(_write_temporal_config(tmp_path))
    cfg["filtering"].update(
        {
            "max_nan_ratio": 1.0,
            "min_unique": 2,
            "max_unique_ratio": 1.0,
            "min_abs_pearson": 0.0,
            "min_iv": 0.0,
            "max_psi": 0.5,
            "max_features": 10,
            "psi_bins": 5,
        }
    )
    cfg["_runtime"] = {"n_classes": 2}
    logger = configure_logging("ERROR")
    label_info = load_labels(cfg, logger)
    windows = load_temporal_psi_windows(cfg, label_info, logger)
    assert windows is not None

    frame = pd.DataFrame(
        {
            "stable": [0, 1] * 6,
            "drift": [0, 1, 2, 3, 4, 5] + [100, 101, 102, 103, 104, 105],
            "category": ["a", "b", "a", "b", "a", "b"]
            + ["x", "x", "y", "y", "z", "z"],
        }
    )
    store = _FakeStore(
        frame,
        {"stable": "numeric", "drift": "numeric", "category": "categorical"},
    )
    splits = {
        "train": np.arange(0, 8, dtype=np.int64),
        "valid": np.asarray([8], dtype=np.int64),
        "test": np.asarray([9, 10, 11], dtype=np.int64),
    }

    result = compute_stats_and_filter(
        store,
        label_info.y,
        splits,
        cfg,
        logger,
        _SilentHeartbeat(),
        temporal_windows=windows,
    )
    report = result.report.set_index("column")

    assert report.loc["stable", "psi"] < 0.05
    assert report.loc["drift", "psi"] > 0.5
    assert report.loc["category", "psi"] > 0.5
    assert report.loc["drift", "psi_mode"] == "temporal"
    assert "psi" in report.loc["drift", "reason"].split(",")
    assert "psi" in report.loc["category", "reason"].split(",")


def test_temporal_psi_pipeline_integration(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    info = generate_synthetic_dataset(
        data_dir,
        rows=300,
        n_features=12,
        target_gb=0,
        n_feature_files=1,
        seed=19,
        chunk_rows=100,
        pad_len=5,
        heartbeat_seconds=1,
    )
    labels_path = Path(info["label_file"])
    labels = pd.read_csv(labels_path)
    labels["event_date"] = [
        "2025-01-01" if index < 150 else "2025-02-01"
        for index in range(len(labels))
    ]
    labels.to_csv(labels_path, index=False)

    config = {
        "project": {
            "name": "temporal_psi",
            "work_dir": str(tmp_path / "run"),
            "random_seed": 19,
            "heartbeat_seconds": 1,
        },
        "data": {
            "feature_paths": info["feature_files"],
            "label_path": str(labels_path),
            "feature_key": "customer_id",
            "label_key": "customer_id",
            "label_column": "target",
            "concat_mode": "column",
            "include_label_features": False,
            "column_block_size": 8,
            "sample_rows": 200,
            "chunksize": 100,
            "train_fraction": 0.6,
            "valid_fraction": 0.2,
            "test_fraction": 0.2,
        },
        "filtering": {
            "max_nan_ratio": 0.99,
            "min_unique": 2,
            "max_unique_ratio": 0.99,
            "min_abs_pearson": 0.0,
            "min_iv": 0.0,
            "max_psi": 2.0,
            "max_features": 50,
            "psi_bins": 5,
            "temporal_psi": {
                "enabled": True,
                "time_column": "event_date",
                "reference_start": "2025-01-01",
                "reference_end": "2025-01-02",
                "comparison_start": "2025-02-01",
                "comparison_end": "2025-02-02",
                "min_rows_per_period": 50,
            },
        },
        "preprocessing": {
            "categorical_encoding": "native",
            "max_categories": 16,
            "min_category_frequency": 1,
        },
        "model": {
            "search_space": str(
                Path(__file__).resolve().parents[1] / "search_spaces" / "lightgbm.yaml"
            ),
            "num_boost_round": 40,
            "early_stopping_rounds": 5,
            "log_period": 100,
        },
        "tuning": {
            "enabled": True,
            "n_trials": 1,
            "sampler": {"n_startup_trials": 1, "seed": 19},
            "pruning": False,
            "refit_on_train_valid": True,
        },
        "metrics": {"threshold": "auto", "plots": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run_pipeline(load_config(config_path))

    assert summary["status"] == "success"
    assert summary["data"]["psi_mode"] == "temporal"
    work_dir = Path(summary["work_dir"])
    assert (work_dir / "temporal_psi_windows.json").exists()
    report = pd.read_csv(work_dir / "filter_report.csv")
    assert set(report["psi_mode"]) == {"temporal"}
