from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import yaml

from auto_ml_pipeline.config import load_config
from auto_ml_pipeline.models import available_models, create_adapter
from auto_ml_pipeline.pipeline import run_pipeline
from auto_ml_pipeline.synthetic import generate_synthetic_dataset
from auto_ml_pipeline.tuning import XGBoostPruningCallback


def _training_data(
    rows: int = 300,
    features: int = 6,
    classes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(rows, features)).astype(np.float32)
    signal = X[:, 0] + 0.4 * X[:, 1] + rng.normal(scale=0.3, size=rows)
    if classes == 2:
        y = (signal > np.median(signal)).astype(np.int32)
    else:
        edges = np.quantile(signal, np.linspace(0, 1, classes + 1)[1:-1])
        y = np.digitize(signal, edges).astype(np.int32)
    return X, y


def test_xgboost_binary_adapter(tmp_path: Path) -> None:
    assert "xgboost" in available_models()
    X, y = _training_data()
    adapter = create_adapter(
        "xgboost",
        task="binary",
        n_classes=2,
        params={
            "num_threads": 2,
            "max_depth": 4,
            "eta": 0.1,
            "feature_fraction": 0.8,
            "lambda_l2": 0.01,
        },
        seed=11,
        feature_names=[f"feature_{index}" for index in range(X.shape[1])],
    )
    resolved = adapter._resolved_params()
    assert resolved["nthread"] == 2
    assert resolved["colsample_bytree"] == 0.8
    assert resolved["reg_lambda"] == 0.01

    adapter.fit(
        X[:240],
        y[:240],
        X[240:],
        y[240:],
        num_boost_round=30,
    )
    probabilities = adapter.predict_proba(X[240:])
    assert probabilities.shape == (60, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert 0 < adapter.best_iteration <= 30

    importance = adapter.feature_importance()
    assert list(importance.columns) == ["feature", "gain", "split", "gain_pct"]
    assert len(importance) == X.shape[1]
    assert np.isclose(importance["gain_pct"].sum(), 1.0)

    model_path = tmp_path / "model.txt"
    adapter.save(model_path)
    assert model_path.exists()
    assert model_path.stat().st_size > 0


def test_xgboost_multiclass_adapter() -> None:
    X, y = _training_data(rows=240, classes=3)
    X[:, -1] = np.random.default_rng(3).integers(0, 4, size=len(X))
    adapter = create_adapter(
        "xgboost",
        task="multiclass",
        n_classes=3,
        params={"num_threads": 2, "max_depth": 4, "eta": 0.1},
        seed=3,
        feature_names=[f"feature_{index}" for index in range(X.shape[1])],
        categorical_indices=[X.shape[1] - 1],
    )
    adapter.fit(
        X[:200],
        y[:200],
        X[200:],
        y[200:],
        num_boost_round=20,
    )
    probabilities = adapter.predict_proba(X[200:])
    assert probabilities.shape == (40, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert adapter._use_categorical


def test_xgboost_pruning_callback_metric_alias() -> None:
    class Trial:
        def __init__(self, prune: bool = False):
            self.prune = prune
            self.reports: list[tuple[float, int]] = []

        def report(self, value: float, step: int) -> None:
            self.reports.append((value, step))

        def should_prune(self) -> bool:
            return self.prune

    trial = Trial()
    callback = XGBoostPruningCallback(trial, "multi_logloss")
    callback.after_iteration(None, 4, {"valid": {"mlogloss": [0.9, 0.7]}})
    assert trial.reports == [(0.7, 4)]

    with pytest.raises(optuna.TrialPruned):
        XGBoostPruningCallback(Trial(prune=True), "auc").after_iteration(
            None, 1, {"valid": {"auc": [0.6]}}
        )


def test_xgboost_tuning_pipeline(tmp_path: Path) -> None:
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
            "name": "xgboost_test",
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
            "num_boost_round": 30,
            "early_stopping_rounds": 5,
            "num_threads": 2,
        },
        "tuning": {
            "enabled": True,
            "n_trials": 2,
            "pruning": True,
            "n_warmup_steps": 1,
            "sampler": {"n_startup_trials": 1, "seed": 13},
            "refit_on_train_valid": False,
        },
        "metrics": {"threshold": "auto", "plots": False},
    }
    path = tmp_path / "xgboost.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = run_pipeline(load_config(path))
    assert summary["status"] == "success"
    work_dir = Path(summary["work_dir"])
    trials = pd.read_csv(work_dir / "tuning" / "trials.csv")
    assert len(trials) == 2
    assert (trials["state"] == "COMPLETE").any()
    assert (work_dir / "model" / "model.txt").exists()
    best = json.loads((work_dir / "tuning" / "best_params.json").read_text())
    assert best["model"] == "xgboost"
    metrics = json.loads((work_dir / "reports" / "metrics.json").read_text())
    assert metrics["test"]["n"] > 0
