from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from auto_ml_pipeline.config import ConfigError, load_config
from auto_ml_pipeline.pipeline import run_pipeline
from auto_ml_pipeline.reuse import stage_fingerprints
from auto_ml_pipeline.synthetic import generate_synthetic_dataset


def _dataset(tmp_path: Path) -> tuple[dict, pd.DataFrame]:
    data_dir = tmp_path / "data"
    info = generate_synthetic_dataset(
        data_dir,
        rows=400,
        n_features=26,
        target_gb=0,
        n_feature_files=2,
        seed=17,
        chunk_rows=100,
        pad_len=5,
        heartbeat_seconds=1,
    )
    labels = pd.read_csv(info["label_file"])
    labels["as_of_date"] = pd.date_range(
        "2024-01-01", periods=len(labels), freq="D"
    ).strftime("%Y-%m-%d")
    labels.to_csv(info["label_file"], index=False)
    return info, labels


def _config(
    tmp_path: Path,
    info: dict,
    name: str,
    *,
    oot_start: str | None = None,
    oot_quantile: float | None = 0.75,
    valid_fraction: float = 0.25,
    min_positive_count: int | None = None,
    max_positive_count: int | None = None,
) -> Path:
    config = {
        "project": {
            "name": name,
            "work_dir": str(tmp_path / name),
            "random_seed": 17,
            "heartbeat_seconds": 1,
        },
        "data": {
            "feature_paths": info["feature_files"],
            "label_path": info["label_file"],
            "feature_key": "customer_id",
            "label_key": "customer_id",
            "label_column": "target",
            "concat_mode": "column",
            "split_strategy": "temporal",
            "temporal_split": {
                "time_column": "as_of_date",
                "date_format": "%Y-%m-%d",
                "timezone": "UTC",
                "oot_start": oot_start,
                "oot_quantile": oot_quantile,
                "valid_fraction": valid_fraction,
                "min_positive_count": min_positive_count,
                "max_positive_count": max_positive_count,
            },
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
            "max_pairwise_corr": 0.99,
            "stats_sample_rows": 200,
        },
        "preprocessing": {
            "categorical_encoding": "native",
            "max_categories": 16,
            "min_category_frequency": 1,
            "max_output_features": 100,
            "add_missing_indicator": False,
        },
        "model": {
            "name": "lightgbm",
            "search_space": str(
                Path(__file__).resolve().parents[1]
                / "search_spaces"
                / "lightgbm.yaml"
            ),
            "num_boost_round": 20,
            "early_stopping_rounds": 5,
            "num_threads": 1,
        },
        "tuning": {
            "enabled": False,
            "refit_on_train_valid": False,
        },
        "metrics": {"threshold": "auto", "plots": False},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_temporal_split_holdout_pipeline(tmp_path: Path) -> None:
    info, labels = _dataset(tmp_path)
    summary = run_pipeline(load_config(_config(tmp_path, info, "temporal_quantile")))
    assert summary["status"] == "success"
    run = Path(summary["work_dir"])
    meta = json.loads((run / "split_metadata.json").read_text())
    assert meta["strategy"] == "temporal"
    assert meta["evaluation_split"] == "oot"
    assert meta["oot_cutoff_source"] == "quantile:0.75"

    splits = np.load(run / "splits.npz")
    dates = pd.to_datetime(labels["as_of_date"], utc=True)
    cutoff = pd.Timestamp(meta["oot_start"])
    assert dates.iloc[splits["train"]].max() < cutoff
    assert dates.iloc[splits["valid"]].max() < cutoff
    assert dates.iloc[splits["test"]].min() >= cutoff
    np.testing.assert_array_equal(splits["filter_test"], splits["valid"])
    np.testing.assert_array_equal(
        splits["filter_indices"],
        np.sort(np.concatenate([splits["train"], splits["valid"]])),
    )
    assert len(np.intersect1d(splits["train"], splits["valid"])) == 0
    assert len(np.intersect1d(splits["test"], splits["filter_indices"])) == 0

    report = pd.read_csv(run / "filter_report.csv")
    assert set(report["psi_mode"]) == {"train_valid"}
    matrix_meta = json.loads((run / "matrix" / "matrix_meta.json").read_text())
    assert "as_of_date" not in matrix_meta["feature_names"]
    assert (run / "reports" / "predictions_oot.csv").exists()
    metrics = json.loads((run / "reports" / "metrics.json").read_text())
    assert metrics["evaluation"]["strategy"] == "temporal"
    assert metrics["oot"]["n"] == len(splits["test"])


def test_temporal_split_explicit_cutoff_and_positive_limits(tmp_path: Path) -> None:
    info, _ = _dataset(tmp_path)
    explicit = run_pipeline(
        load_config(
            _config(
                tmp_path,
                info,
                "temporal_explicit",
                oot_start="2024-10-15",
                oot_quantile=None,
                valid_fraction=0.3,
            )
        )
    )
    meta = json.loads(
        (Path(explicit["work_dir"]) / "split_metadata.json").read_text()
    )
    assert meta["oot_cutoff_source"] == "explicit"
    assert meta["oot_start"].startswith("2024-10-15")
    assert 0.2 <= meta["valid_fraction_actual"] <= 0.3

    with pytest.raises(ConfigError, match="max_positive_count"):
        run_pipeline(
            load_config(
                _config(
                    tmp_path,
                    info,
                    "temporal_limit",
                    max_positive_count=1,
                )
            )
        )


def test_temporal_split_with_random_cv(tmp_path: Path) -> None:
    info, _ = _dataset(tmp_path)
    path = _config(tmp_path, info, "temporal_cv")
    config = yaml.safe_load(path.read_text())
    config["tuning"] = {
        "enabled": True,
        "n_trials": 1,
        "metric": "ks",
        "pruning": True,
        "refit_on_train_valid": False,
        "sampler": {"n_startup_trials": 1, "seed": 17},
        "cv": {
            "enabled": True,
            "folds": 2,
            "seed": 17,
            "prune_after_folds": 1,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = run_pipeline(load_config(path))
    run = Path(summary["work_dir"])
    best = json.loads((run / "tuning" / "best_params.json").read_text())
    assert best["cv_enabled"] is True
    assert best["cv_folds"] == 2
    trials = pd.read_csv(run / "tuning" / "trials.csv")
    fold_scores = json.loads(trials.iloc[0]["fold_scores"])
    assert len(fold_scores) == 2
    meta = json.loads((run / "split_metadata.json").read_text())
    assert meta["strategy"] == "temporal"


def test_temporal_cutoff_change_reuses_cache_not_splits(tmp_path: Path) -> None:
    info, _ = _dataset(tmp_path)
    baseline = run_pipeline(
        load_config(_config(tmp_path, info, "temporal_reuse_base"))
    )
    baseline_run = Path(baseline["work_dir"])
    path = _config(
        tmp_path,
        info,
        "temporal_reuse_changed",
        oot_quantile=0.6,
    )
    config = yaml.safe_load(path.read_text())
    config["project"]["reuse_from"] = str(baseline_run)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    changed = run_pipeline(load_config(path))
    run = Path(changed["work_dir"])
    timings = pd.read_csv(run / "stage_timings.csv")
    details = {
        row.stage: json.loads(row.details)
        for row in timings.itertuples()
    }
    assert "reused_from" in details["cache_ingest"]
    assert "reused_from" not in details["split"]
    assert "reused_from" not in details["filtering_stats"]


def test_temporal_split_fingerprints() -> None:
    cfg = {
        "project": {"random_seed": 7},
        "data": {
            "feature_paths": [],
            "label_path": None,
            "csv": {},
            "split_strategy": "temporal",
            "temporal_split": {
                "time_column": "as_of_date",
                "oot_quantile": 0.5,
                "valid_fraction": 0.25,
            },
        },
        "filtering": {"temporal_psi": {"enabled": False}},
        "preprocessing": {},
    }
    baseline = stage_fingerprints(cfg)
    cfg["data"]["temporal_split"]["oot_quantile"] = 0.7
    changed_cutoff = stage_fingerprints(cfg)
    assert changed_cutoff["cache"] == baseline["cache"]
    assert changed_cutoff["splits"] != baseline["splits"]
    assert changed_cutoff["filtering"] != baseline["filtering"]

    cfg["data"]["temporal_split"]["time_column"] = "other_date"
    changed_column = stage_fingerprints(cfg)
    assert changed_column["cache"] != baseline["cache"]
