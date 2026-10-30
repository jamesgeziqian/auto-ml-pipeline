from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from auto_ml_pipeline.config import ConfigError, load_config
from auto_ml_pipeline.pipeline import run_pipeline
from auto_ml_pipeline.synthetic import generate_synthetic_dataset


def _dataset(tmp_path: Path) -> dict:
    data_dir = tmp_path / "data"
    return generate_synthetic_dataset(
        data_dir,
        rows=400,
        n_features=26,
        target_gb=0,
        n_feature_files=2,
        seed=13,
        chunk_rows=100,
        pad_len=5,
        heartbeat_seconds=1,
    )


def _run(
    tmp_path: Path,
    info: dict,
    name: str,
    *,
    reuse_from: str | None = None,
    reuse_mode: str = "link",
    max_features: int = 20,
    add_missing_indicator: bool = False,
    num_boost_round: int = 20,
    filtering_overrides: dict[str, Any] | None = None,
) -> Path:
    filtering = {
        "max_nan_ratio": 0.99,
        "min_unique": 2,
        "max_unique_ratio": 0.99,
        "min_abs_pearson": 0.0,
        "min_iv": 0.0,
        "max_psi": 2.0,
        "max_features": max_features,
        "max_pairwise_corr": None,
    }
    filtering.update(filtering_overrides or {})
    config = {
        "project": {
            "name": name,
            "work_dir": str(tmp_path / name),
            "random_seed": 13,
            "heartbeat_seconds": 1,
            "reuse_from": reuse_from,
            "reuse_mode": reuse_mode,
        },
        "data": {
            "feature_paths": info["feature_files"],
            "label_path": info["label_file"],
            "feature_key": "customer_id",
            "label_key": "customer_id",
            "label_column": "target",
            "concat_mode": "column",
            "column_block_size": 8,
            "sample_rows": 200,
            "train_fraction": 0.7,
            "valid_fraction": 0.15,
            "test_fraction": 0.15,
            "stratify": True,
            "csv": {"chunksize": 100, "engine": "arrow"},
        },
        "filtering": filtering,
        "preprocessing": {
            "categorical_encoding": "native",
            "max_categories": 16,
            "min_category_frequency": 1,
            "max_output_features": 100,
            "add_missing_indicator": add_missing_indicator,
        },
        "model": {
            "search_space": str(
                Path(__file__).resolve().parents[1] / "search_spaces" / "lightgbm.yaml"
            ),
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": 5,
            "num_threads": 1,
            "log_period": 20,
            "params": {
                "num_leaves": 18,
                "learning_rate": 0.1,
                "min_data_in_leaf": 19,
                "feature_fraction": 0.86,
                "bagging_fraction": 0.99,
                "bagging_freq": 1,
                "lambda_l1": 0.0007,
                "lambda_l2": 0.0003,
            },
        },
        "tuning": {
            "enabled": False,
            "n_trials": 1,
            "refit_on_train_valid": False,
        },
        "metrics": {"threshold": "auto", "plots": False},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = run_pipeline(load_config(path))
    assert summary["status"] == "success"
    return Path(summary["work_dir"])


def _details(run: Path) -> dict[str, dict]:
    frame = pd.read_csv(run / "stage_timings.csv")
    return {
        str(row.stage): json.loads(row.details)
        for row in frame.itertuples()
    }


def test_reuse_full_and_partial_stages(tmp_path: Path) -> None:
    info = _dataset(tmp_path)
    baseline = _run(tmp_path, info, "baseline")
    reused = _run(
        tmp_path,
        info,
        "reused",
        reuse_from=str(baseline),
        num_boost_round=25,
    )
    details = _details(reused)
    for stage in [
        "schema_inference",
        "cache_ingest",
        "split",
        "filtering_stats",
        "preprocess_fit",
        "matrix_build",
    ]:
        assert "reused_from" in details[stage], stage
    assert (reused / "cache").is_symlink()
    assert (reused / "matrix").is_symlink()
    assert json.loads((baseline / "selected_features.json").read_text())["selected"] == json.loads(
        (reused / "selected_features.json").read_text()
    )["selected"]
    assert json.loads((baseline / "matrix" / "matrix_meta.json").read_text())["shape"] == json.loads(
        (reused / "matrix" / "matrix_meta.json").read_text()
    )["shape"]
    summary = json.loads((reused / "run_summary.json").read_text())
    assert set(summary["reuse"]["reused_stages"]) >= {
        "cache",
        "splits",
        "filtering",
        "preprocessing",
        "matrix",
    }

    filtered = _run(
        tmp_path,
        info,
        "filter_changed",
        reuse_from=str(baseline),
        max_features=15,
    )
    filtered_details = _details(filtered)
    assert "reused_from" in filtered_details["cache_ingest"]
    assert "reused_from" in filtered_details["split"]
    assert "reused_from" in filtered_details["filtering_stats"]
    assert filtered_details["filtering_stats"]["metrics_reused"] is True
    assert filtered_details["filtering_stats"]["thresholds_reapplied"] is True
    assert filtered_details["filtering_stats"]["n_selected"] == 15
    assert "reused_from" not in filtered_details["preprocess_fit"]
    assert "reused_from" not in filtered_details["matrix_build"]

    baseline_report = pd.read_csv(baseline / "filter_report.csv").set_index("column")
    filtered_report = pd.read_csv(filtered / "filter_report.csv").set_index("column")
    pd.testing.assert_series_equal(
        baseline_report.loc[filtered_report.index, "iv"],
        filtered_report["iv"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        baseline_report.loc[filtered_report.index, "psi"],
        filtered_report["psi"],
        check_names=False,
    )

    metrics_changed = _run(
        tmp_path,
        info,
        "metrics_changed",
        reuse_from=str(baseline),
        filtering_overrides={"iv_bins": 4},
    )
    metrics_changed_details = _details(metrics_changed)
    assert "reused_from" in metrics_changed_details["cache_ingest"]
    assert "reused_from" in metrics_changed_details["split"]
    assert "reused_from" not in metrics_changed_details["filtering_stats"]
    assert metrics_changed_details["filtering_stats"]["metrics_reused"] is False

    preprocessed = _run(
        tmp_path,
        info,
        "preprocess_changed",
        reuse_from=str(baseline),
        add_missing_indicator=True,
    )
    preprocessed_details = _details(preprocessed)
    assert "reused_from" in preprocessed_details["cache_ingest"]
    assert "reused_from" in preprocessed_details["split"]
    assert "reused_from" in preprocessed_details["filtering_stats"]
    assert "reused_from" not in preprocessed_details["preprocess_fit"]
    assert "reused_from" not in preprocessed_details["matrix_build"]


def test_reuse_invalidated_by_input_change(tmp_path: Path) -> None:
    info = _dataset(tmp_path)
    baseline = _run(tmp_path, info, "baseline_input")
    os.utime(info["feature_files"][0], None)
    changed = _run(
        tmp_path,
        info,
        "input_changed",
        reuse_from=str(baseline),
    )
    details = _details(changed)
    assert "reused_from" not in details["cache_ingest"]
    assert "reused_from" not in details["split"]


def test_reuse_metrics_from_legacy_report_without_metric_fingerprint(
    tmp_path: Path,
) -> None:
    info = _dataset(tmp_path)
    baseline = _run(tmp_path, info, "baseline_legacy_metrics")
    state_path = baseline / "reuse_state.json"
    state = json.loads(state_path.read_text())
    state.pop("filter_metrics")
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    filtered = _run(
        tmp_path,
        info,
        "legacy_metrics_reused",
        reuse_from=str(baseline),
        max_features=15,
    )
    details = _details(filtered)
    assert details["filtering_stats"]["metrics_reused"] is True
    assert details["filtering_stats"]["thresholds_reapplied"] is True
    assert "reused_from" not in details["preprocess_fit"]


def test_reuse_resumes_after_preprocessing(tmp_path: Path) -> None:
    info = _dataset(tmp_path)
    baseline = _run(tmp_path, info, "baseline_partial")
    state_path = baseline / "reuse_state.json"
    state = json.loads(state_path.read_text())
    state["stages"]["matrix"] = False
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    resumed = _run(
        tmp_path,
        info,
        "resumed",
        reuse_from=str(baseline),
    )
    details = _details(resumed)
    assert "reused_from" in details["cache_ingest"]
    assert "reused_from" in details["split"]
    assert "reused_from" in details["filtering_stats"]
    assert "reused_from" in details["preprocess_fit"]
    assert "reused_from" not in details["matrix_build"]


def test_reuse_copy_mode(tmp_path: Path) -> None:
    info = _dataset(tmp_path)
    baseline = _run(tmp_path, info, "baseline_copy")
    copied = _run(
        tmp_path,
        info,
        "copied",
        reuse_from=str(baseline),
        reuse_mode="copy",
    )
    assert (copied / "cache").is_dir()
    assert not (copied / "cache").is_symlink()
    assert (copied / "matrix").is_dir()
    assert not (copied / "matrix").is_symlink()


def test_reuse_requires_source_state(tmp_path: Path) -> None:
    info = _dataset(tmp_path)
    empty = tmp_path / "empty_source"
    empty.mkdir()
    with pytest.raises(ConfigError, match="reuse_state.json"):
        _run(
            tmp_path,
            info,
            "missing_state",
            reuse_from=str(empty),
        )
