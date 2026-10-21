from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from auto_ml_pipeline.config import load_config
from auto_ml_pipeline.pipeline import run_pipeline
from auto_ml_pipeline.synthetic import generate_wide_rowwise_dataset


@pytest.mark.parametrize("engine", ["pandas", "arrow"])
def test_wide_blocked_row_concat(tmp_path: Path, engine: str) -> None:
    data_dir = tmp_path / "wide"
    info = generate_wide_rowwise_dataset(
        data_dir,
        rows_per_file=250,
        n_files=2,
        n_features=5000,
        n_signal=10,
        pattern_pool=32,
        chunk_rows=100,
        seed=5,
        heartbeat_seconds=1,
    )
    assert info["total_rows"] == 500
    config = {
        "project": {
            "name": "wide_blocked_test",
            "work_dir": str(tmp_path / "run"),
            "random_seed": 5,
            "heartbeat_seconds": 1,
        },
        "data": {
            "feature_paths": info["feature_files"],
            "label_path": info["label_file"],
            "feature_key": "customer_id",
            "label_key": "customer_id",
            "label_column": "target",
            "concat_mode": "row",
            "column_block_size": 32,
            "max_source_columns_in_memory": 4000,
            "wide_block_size": 512,
            "wide_row_group_size": 100,
            "cache_numeric_dtype": "float32",
            "ingest_target_chunk_bytes": 8 * 1024 * 1024,
            "sample_rows": 200,
            "chunksize": 100,
            "csv": {"engine": engine, "chunksize": 100},
            "train_fraction": 0.6,
            "valid_fraction": 0.2,
            "test_fraction": 0.2,
        },
        "filtering": {
            "max_nan_ratio": 1.0,
            "min_unique": 2,
            "max_unique_ratio": 0.99,
            "min_abs_pearson": 0.0,
            "min_iv": 0.0,
            "max_psi": 2.0,
            "max_features": 50,
            "max_pairwise_corr": None,
        },
        "preprocessing": {
            "categorical_encoding": "native",
            "max_categories": 16,
            "min_category_frequency": 1,
            "max_output_features": 200,
        },
        "model": {
            "search_space": str(
                Path(__file__).resolve().parents[1] / "search_spaces" / "lightgbm.yaml"
            ),
            "num_boost_round": 50,
            "early_stopping_rounds": 10,
            "log_period": 100,
        },
        "tuning": {
            "enabled": True,
            "n_trials": 1,
            "sampler": {"n_startup_trials": 1, "seed": 5},
            "pruning": False,
            "refit_on_train_valid": True,
        },
        "metrics": {"threshold": "auto", "plots": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = run_pipeline(load_config(config_path))
    assert summary["status"] == "success"
    work_dir = Path(summary["work_dir"])
    manifest = json.loads((work_dir / "cache" / "manifest.json").read_text())
    assert manifest["n_rows"] == 500
    assert manifest["csv_engine"] == engine
    feature_sources = [
        source
        for source in manifest["sources"].values()
        if source["kind"] == "features"
    ]
    assert len(feature_sources) == 2
    assert all(source["wide_blocked"] for source in feature_sources)
    assert all(source["csv_engine"] == engine for source in feature_sources)
    assert all(len(source["part_groups"]) == 10 for source in feature_sources)
    matrix_meta = json.loads((work_dir / "matrix" / "matrix_meta.json").read_text())
    assert matrix_meta["shape"] == [500, 50]
    state = json.loads((work_dir / "preprocess_state.json").read_text())
    assert not any(
        config.get("has_missing") for config in state["numeric"].values()
    )
    assert (work_dir / "reports" / "metrics.json").exists()
