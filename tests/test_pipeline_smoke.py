from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from auto_ml_pipeline.config import load_config
from auto_ml_pipeline.pipeline import run_pipeline
from auto_ml_pipeline.synthetic import generate_synthetic_dataset


def _run(tmp_path: Path, concat_mode: str = "column") -> dict:
    data_dir = tmp_path / "data"
    info = generate_synthetic_dataset(
        data_dir,
        rows=500,
        n_features=26,
        target_gb=0,
        n_feature_files=2,
        seed=11,
        chunk_rows=100,
        pad_len=5,
        heartbeat_seconds=1,
    )
    feature_paths = info["feature_files"]
    if concat_mode == "row":
        frames = [pd.read_csv(path) for path in feature_paths]
        combined = pd.concat(frames, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        row_paths = []
        for index, chunk in enumerate(
            [combined.iloc[:250], combined.iloc[250:]]
        ):
            path = data_dir / f"row_part_{index}.csv"
            chunk.to_csv(path, index=False)
            row_paths.append(str(path))
        feature_paths = row_paths

    config = {
        "project": {
            "name": f"test_{concat_mode}",
            "work_dir": str(tmp_path / f"run_{concat_mode}"),
            "random_seed": 11,
            "heartbeat_seconds": 1,
        },
        "data": {
            "feature_paths": feature_paths,
            "label_path": info["label_file"],
            "feature_key": "customer_id",
            "label_key": "customer_id",
            "label_column": "target",
            "concat_mode": concat_mode,
            "column_block_size": 8,
            "sample_rows": 300,
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
            "max_features": 200,
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
            "num_boost_round": 80,
            "early_stopping_rounds": 10,
            "log_period": 100,
        },
        "tuning": {
            "enabled": True,
            "n_trials": 1,
            "sampler": {"n_startup_trials": 1, "seed": 11},
            "pruning": False,
            "refit_on_train_valid": True,
        },
        "metrics": {"threshold": "auto", "plots": False},
    }
    config_path = tmp_path / f"config_{concat_mode}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = load_config(config_path)
    summary = run_pipeline(cfg)
    assert summary["status"] == "success"
    work_dir = Path(summary["work_dir"])
    for relative in [
        "run_summary.json",
        "stage_timings.csv",
        "filter_report.csv",
        "selected_features.json",
        "preprocess_state.json",
        "matrix/matrix_meta.json",
        "model/model.txt",
        "reports/metrics.json",
        "reports/predictions_test.csv",
        "reports/feature_importance.csv",
    ]:
        assert (work_dir / relative).exists(), relative
    metrics = json.loads((work_dir / "reports/metrics.json").read_text())
    for key in ["accuracy", "precision", "recall", "f1", "auc", "ks"]:
        assert key in metrics["test"]
    assert metrics["test"]["n"] > 0
    summary_data = json.loads((work_dir / "run_summary.json").read_text())
    for stage in [
        "load_labels",
        "schema_inference",
        "cache_ingest",
        "filtering_stats",
        "preprocess_fit",
        "matrix_build",
        "tuning_training",
        "evaluation",
        "reporting",
    ]:
        assert stage in summary_data["stage_timings"]
    return summary


def test_column_concat_pipeline(tmp_path: Path) -> None:
    _run(tmp_path, concat_mode="column")


def test_row_concat_pipeline(tmp_path: Path) -> None:
    _run(tmp_path, concat_mode="row")
