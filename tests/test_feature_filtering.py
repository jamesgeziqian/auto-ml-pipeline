from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from auto_ml_pipeline.logging_utils import configure_logging
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
