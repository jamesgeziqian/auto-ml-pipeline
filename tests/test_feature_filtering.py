from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from auto_ml_pipeline.config import load_config
from auto_ml_pipeline.logging_utils import configure_logging
from auto_ml_pipeline.pipeline import run_pipeline
from auto_ml_pipeline.stats import compute_stats_and_filter


class _FakeStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.n_rows = len(frame)

    def columns(self) -> list[str]:
        return list(self.frame.columns)

    def dtype(self, column: str) -> str:
        return "numeric"

    def iter_column_blocks(self, columns: list[str], block_size: int) -> Any:
        yield list(columns), self.frame.copy()


class _SilentHeartbeat:
    def detail(self, value: str) -> None:
        pass


@pytest.mark.parametrize(
    ("max_iv", "expected_selected", "expected_reason"),
    [
        (None, {"high_signal", "stable"}, "passed"),
        (1.0, {"stable"}, "max_iv"),
    ],
)
def test_max_iv_filters_high_signal_features(
    max_iv: float | None,
    expected_selected: set[str],
    expected_reason: str,
) -> None:
    labels = np.tile([0, 1], 50)
    stable_signal = np.random.default_rng(42).normal(size=len(labels))
    frame = pd.DataFrame(
        {
            "high_signal": labels,
            "stable": stable_signal,
        }
    )
    cfg: dict[str, Any] = {
        "data": {"column_block_size": 8, "sample_rows": 100},
        "filtering": {
            "max_nan_ratio": 1.0,
            "min_unique": 2,
            "max_unique_ratio": 1.0,
            "min_abs_pearson": 0.0,
            "min_iv": 0.0,
            "max_iv": max_iv,
            "max_psi": 10.0,
            "iv_bins": 10,
            "psi_bins": 10,
        },
        "_runtime": {"n_classes": 2},
    }
    splits = {
        "train": np.arange(60, dtype=np.int64),
        "test": np.arange(60, 100, dtype=np.int64),
    }

    result = compute_stats_and_filter(
        _FakeStore(frame),
        labels,
        splits,
        cfg,
        configure_logging("ERROR"),
        _SilentHeartbeat(),
    )
    report = result.report.set_index("column")

    assert set(result.selected) == expected_selected
    assert report.loc["high_signal", "iv"] > 1.0
    assert report.loc["high_signal", "selected"] == (max_iv is None)
    assert report.loc["high_signal", "reason"] == expected_reason


def test_pipeline_persists_filter_report_when_all_columns_are_rejected(
    tmp_path: Path,
) -> None:
    labels = np.tile([0, 1], 50)
    labels_path = tmp_path / "labels.csv"
    features_path = tmp_path / "features.csv"
    pd.DataFrame(
        {"customer_id": np.arange(len(labels)), "target": labels}
    ).to_csv(labels_path, index=False)
    pd.DataFrame(
        {"customer_id": np.arange(len(labels)), "high_signal": labels}
    ).to_csv(features_path, index=False)
    config = {
        "project": {
            "work_dir": str(tmp_path / "run"),
            "heartbeat_seconds": 1,
        },
        "data": {
            "feature_paths": [str(features_path)],
            "label_path": str(labels_path),
            "feature_key": "customer_id",
            "label_key": "customer_id",
            "label_column": "target",
            "include_label_features": False,
            "train_fraction": 0.6,
            "valid_fraction": 0.2,
            "test_fraction": 0.2,
        },
        "filtering": {
            "max_nan_ratio": 1.0,
            "min_unique": 2,
            "min_abs_pearson": 0.0,
            "min_iv": 0.0,
            "max_iv": 1.0,
            "max_psi": 10.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="filter_report.csv"):
        run_pipeline(load_config(config_path))

    report = pd.read_csv(tmp_path / "run" / "filter_report.csv")
    rejected = report.set_index("column").loc["high_signal"]
    assert not rejected["selected"]
    assert rejected["reason"] == "max_iv"
    assert rejected["iv"] > 1.0
    state = json.loads((tmp_path / "run" / "reuse_state.json").read_text())
    assert state["filter_metrics_completed"] is True
    assert state["stages"]["filtering"] is False

    config["project"]["work_dir"] = str(tmp_path / "relaxed")
    config["project"]["reuse_from"] = str(tmp_path / "run")
    config["filtering"]["max_iv"] = None
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = run_pipeline(load_config(config_path))
    assert summary["status"] == "success"

    timings = pd.read_csv(tmp_path / "relaxed" / "stage_timings.csv")
    details = {
        str(row.stage): json.loads(row.details)
        for row in timings.itertuples()
    }
    assert details["filtering_stats"]["metrics_reused"] is True
    assert details["filtering_stats"]["thresholds_reapplied"] is True
