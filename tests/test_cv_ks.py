from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from auto_ml_pipeline.config import DEFAULTS, ConfigError, load_config, validate_config
from auto_ml_pipeline.metrics import ks_statistic
from auto_ml_pipeline.pipeline import run_pipeline
from auto_ml_pipeline.reuse import stage_fingerprints
from auto_ml_pipeline.synthetic import generate_synthetic_dataset
from auto_ml_pipeline.tuning import _validation_metric


def test_ks_validation_metric() -> None:
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int32)
    positive = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    probabilities = np.column_stack([1.0 - positive, positive])
    assert _validation_metric("binary", y_true, probabilities, "ks") == pytest.approx(
        ks_statistic(y_true, positive)
    )
    assert _validation_metric("binary", y_true, probabilities, "ks") == pytest.approx(
        1.0
    )
    with pytest.raises(ValueError, match="Unsupported multiclass"):
        _validation_metric(
            "multiclass",
            np.asarray([0, 1, 2], dtype=np.int32),
            np.full((3, 3), 1.0 / 3.0),
            "ks",
        )


def test_cv_settings_do_not_change_data_fingerprints() -> None:
    cfg = {
        "project": {"random_seed": 7},
        "data": {"feature_paths": [], "label_path": None, "csv": {}},
        "filtering": {"temporal_psi": {"enabled": False}},
        "preprocessing": {},
        "tuning": {"cv": {"enabled": False}},
    }
    before = stage_fingerprints(cfg)
    cfg["tuning"]["cv"] = {
        "enabled": True,
        "folds": 5,
        "seed": 13,
        "prune_after_folds": 2,
    }
    assert stage_fingerprints(cfg) == before


def test_cv_config_validation(tmp_path: Path) -> None:
    feature = tmp_path / "features.csv"
    labels = tmp_path / "labels.csv"
    feature.write_text("customer_id,f\nk000,1\n", encoding="utf-8")
    labels.write_text("customer_id,target\nk000,1\n", encoding="utf-8")
    cfg = copy.deepcopy(DEFAULTS)
    cfg["data"]["feature_paths"] = [str(feature)]
    cfg["data"]["label_path"] = str(labels)
    cfg["tuning"]["cv"] = {
        "enabled": True,
        "folds": 1,
        "seed": 1,
        "prune_after_folds": 1,
    }
    with pytest.raises(ConfigError, match="folds"):
        validate_config(cfg)


def test_xgboost_ks_cross_validation_pipeline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    info = generate_synthetic_dataset(
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
    config = {
        "project": {
            "name": "cv_ks_test",
            "work_dir": str(tmp_path / "run"),
            "random_seed": 13,
            "heartbeat_seconds": 1,
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
        "filtering": {
            "max_nan_ratio": 0.99,
            "min_unique": 2,
            "max_unique_ratio": 0.99,
            "min_abs_pearson": 0.0,
            "min_iv": 0.0,
            "max_psi": 2.0,
            "max_features": 20,
        },
        "preprocessing": {
            "categorical_encoding": "native",
            "max_categories": 16,
            "min_category_frequency": 1,
            "max_output_features": 100,
            "add_missing_indicator": False,
        },
        "model": {
            "name": "xgboost",
            "search_space": str(
                Path(__file__).resolve().parents[1]
                / "search_spaces"
                / "xgboost.yaml"
            ),
            "num_boost_round": 20,
            "early_stopping_rounds": 5,
            "num_threads": 2,
        },
        "tuning": {
            "enabled": True,
            "n_trials": 2,
            "metric": "ks",
            "direction": "auto",
            "pruning": True,
            "refit_on_train_valid": True,
            "sampler": {"n_startup_trials": 1, "seed": 13},
            "cv": {
                "enabled": True,
                "folds": 3,
                "seed": 13,
                "prune_after_folds": 1,
            },
        },
        "metrics": {"threshold": "auto", "plots": False},
    }
    path = tmp_path / "cv_ks.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = run_pipeline(load_config(path))
    assert summary["status"] == "success"
    work_dir = Path(summary["work_dir"])
    best = json.loads((work_dir / "tuning" / "best_params.json").read_text())
    assert best["metric"] == "ks"
    assert best["direction"] == "maximize"
    assert best["cv_enabled"] is True
    assert best["cv_folds"] == 3

    trials = pd.read_csv(work_dir / "tuning" / "trials.csv")
    assert len(trials) == 2
    assert (trials["state"] == "COMPLETE").any()
    assert trials["fold_scores"].notna().any()
    completed = trials[trials["state"] == "COMPLETE"].iloc[0]
    fold_scores = json.loads(completed["fold_scores"])
    assert len(fold_scores) == 3
    assert int(completed["best_iteration"]) > 0

    metrics = json.loads((work_dir / "reports" / "metrics.json").read_text())
    assert "ks" in metrics["test"]
    assert metrics["test"]["n"] > 0
