"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .cache import ROW_INDEX, CacheStore, build_cache
from .config import config_to_yaml
from .logging_utils import (
    Heartbeat,
    StageRecorder,
    configure_logging,
    peak_rss_mb,
    rss_mb,
)
from .metrics import (
    aggregate_importance,
    binary_metrics,
    build_predictions,
    multiclass_metrics,
    save_metric_plots,
    save_metrics_json,
    select_binary_threshold,
)
from .preprocess import build_matrix, fit_preprocessing
from .reuse import ReuseState
from .schema import (
    discover_sources,
    load_labels,
    make_splits,
    save_json,
)
from .stats import (
    FeatureFilteringError,
    apply_filter_thresholds,
    compute_stats_and_filter,
)
from .temporal import load_temporal_psi_windows
from .tuning import tune_and_train


class AutoMLPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.logger = configure_logging(cfg["project"].get("log_level", "INFO"))
        self.work_dir = self._prepare_work_dir()
        self._add_file_handler()
        self.reuse = ReuseState(self.work_dir, self.cfg, self.logger)
        mpl_config = Path(self.work_dir).parent / ".matplotlib"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        os.environ.setdefault("XDG_CACHE_HOME", str(Path(self.work_dir).parent / ".cache"))
        self.report_dir = Path(self.work_dir) / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.cfg["output"]["report_dir"] = str(self.report_dir)
        self.cfg["project"]["work_dir"] = str(self.work_dir)
        self.heartbeat = Heartbeat(
            self.logger, float(cfg["project"].get("heartbeat_seconds", 15))
        )
        self.recorder = StageRecorder(self.logger, self.heartbeat)
        self.status = "running"
        self.error: Optional[str] = None

    def _prepare_work_dir(self) -> Path:
        requested = Path(self.cfg["project"]["work_dir"]).resolve()
        if not requested.exists() or not any(requested.iterdir()):
            requested.mkdir(parents=True, exist_ok=True)
            return requested
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg["project"].get("overwrite"):
            archived = requested.with_name(f"{requested.name}.old.{stamp}")
            shutil.move(str(requested), str(archived))
            requested.mkdir(parents=True, exist_ok=True)
            self.logger.warning(
                "[setup] existing work_dir archived to %s (recoverable)", archived
            )
            return requested
        new_dir = requested.with_name(f"{requested.name}_{stamp}")
        new_dir.mkdir(parents=True, exist_ok=True)
        self.logger.warning(
            "[setup] work_dir %s already exists; starting a new run at %s",
            requested,
            new_dir,
        )
        return new_dir

    def _add_file_handler(self) -> None:
        log_path = self.work_dir / "run.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(handler)

    def run(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": self.cfg.get("_meta", {}).get("config_path"),
            "work_dir": str(self.work_dir),
            "reuse": self.reuse.as_dict(),
        }
        self.logger.info(
            "[setup] AutoML run starting: work_dir=%s pid=%s rss=%.1fMiB",
            self.work_dir,
            os.getpid(),
            rss_mb(),
        )
        self.logger.info(
            "[setup] feature files=%s label_file=%s concat_mode=%s",
            len(self.cfg["data"]["feature_paths"]),
            self.cfg["data"]["label_path"],
            self.cfg["data"]["concat_mode"],
        )
        with (self.work_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            handle.write(config_to_yaml(self.cfg))

        try:
            with self.heartbeat:
                with self.recorder.stage("load_labels", "labels and task inference") as stage:
                    label_info = load_labels(self.cfg, self.logger)
                    self.cfg["_runtime"] = {
                        "task": label_info.task,
                        "n_classes": label_info.n_classes,
                    }
                    stage.update(label_info.to_dict())
                    np.save(self.work_dir / "labels.npy", label_info.y)
                    np.save(self.work_dir / "raw_labels.npy", label_info.raw_labels, allow_pickle=True)
                    pd.DataFrame(
                        {
                            ROW_INDEX: np.arange(label_info.n_rows, dtype=np.int64),
                            self.cfg["data"]["label_key"]: label_info.keys,
                        }
                    ).to_parquet(
                        self.work_dir / "keys.parquet", index=False, compression="zstd"
                    )
                    save_json(
                        self.work_dir / "class_map.json",
                        {
                            "task": label_info.task,
                            "classes": [str(value) for value in label_info.classes],
                            "class_mapping": label_info.class_mapping,
                        },
                    )

                sources = None
                schema_stats: Dict[str, Any] = {}
                with self.recorder.stage("schema_inference", "sample and infer column types") as stage:
                    if self.reuse.can_reuse("cache"):
                        schema_stats = self.reuse.load_schema_stats()
                        stage.update(
                            {
                                **schema_stats,
                                "reused_from": str(self.reuse.source),
                            }
                        )
                        self.logger.info(
                            "[reuse] schema_inference from %s", self.reuse.source
                        )
                    else:
                        sources, schema_stats = discover_sources(
                            self.cfg, label_info, self.logger
                        )
                        stage.update(schema_stats)
                        save_json(
                            self.work_dir / "schema" / "sources.json",
                            {
                                "sources": [source.to_dict() for source in sources],
                                "stats": schema_stats,
                            },
                        )

                with self.recorder.stage("cache_ingest", "chunked CSV -> parquet cache") as stage:
                    if self.reuse.can_reuse("cache"):
                        manifest = self.reuse.load_manifest()
                        stage.update(
                            {
                                "n_rows": manifest["n_rows"],
                                "n_columns": manifest["n_logical_columns"],
                                "cache_bytes": manifest["total_cache_bytes"],
                                "missing_keys": manifest["total_missing_keys"],
                                "coverage_ratio": manifest["coverage_ratio"],
                                "reused_from": str(self.reuse.source),
                            }
                        )
                        self.logger.info(
                            "[reuse] cache_ingest from %s", self.reuse.source
                        )
                    else:
                        if sources is None:
                            raise RuntimeError(
                                "Cache reuse was not available but source discovery "
                                "was skipped."
                            )
                        manifest = build_cache(
                            self.cfg,
                            sources,
                            label_info,
                            self.work_dir,
                            self.logger,
                            self.heartbeat,
                        )
                        self.reuse.mark("cache")
                        stage.update(
                            {
                                "n_rows": manifest["n_rows"],
                                "n_columns": manifest["n_logical_columns"],
                                "cache_bytes": manifest["total_cache_bytes"],
                                "missing_keys": manifest["total_missing_keys"],
                                "coverage_ratio": manifest["coverage_ratio"],
                            }
                        )

                with self.recorder.stage("split", "stratified train/valid/test") as stage:
                    if self.reuse.can_reuse("splits"):
                        splits = self.reuse.load_splits()
                        split_metadata_path = self.work_dir / "split_metadata.json"
                        split_metadata = (
                            json.loads(split_metadata_path.read_text(encoding="utf-8"))
                            if split_metadata_path.exists()
                            else {}
                        )
                        self.cfg.setdefault("_runtime", {})["split_metadata"] = split_metadata
                        stage.update(
                            {
                                **{
                                    name: len(indices)
                                    for name, indices in splits.items()
                                },
                                "eligible_rows": split_metadata.get(
                                    "eligible_rows"
                                ),
                                "dropped_unmatched_rows": split_metadata.get(
                                    "dropped_unmatched_rows"
                                ),
                                "reused_from": str(self.reuse.source),
                                "strategy": split_metadata.get(
                                    "strategy", "random"
                                ),
                                "evaluation_split": split_metadata.get(
                                    "evaluation_split", "test"
                                ),
                            }
                        )
                        self.logger.info("[reuse] split from %s", self.reuse.source)
                    else:
                        eligible_indices = None
                        if self.cfg["data"].get("drop_unmatched_rows", False):
                            coverage = np.load(
                                self.work_dir / "cache" / "row_coverage.npy"
                            )
                            eligible_indices = np.flatnonzero(coverage)
                            self.logger.info(
                                "[split] drop_unmatched_rows=true: "
                                "keeping %s/%s label rows with feature matches",
                                len(eligible_indices),
                                label_info.n_rows,
                            )
                        splits = make_splits(
                            label_info,
                            self.cfg,
                            self.logger,
                            eligible_indices=eligible_indices,
                        )
                        np.savez(self.work_dir / "splits.npz", **splits)
                        split_metadata = self.cfg.get("_runtime", {}).get(
                            "split_metadata", {}
                        )
                        save_json(
                            self.work_dir / "split_metadata.json",
                            split_metadata,
                        )
                        self.reuse.mark("splits")
                        stage.update(
                            {
                                **{
                                    name: len(indices)
                                    for name, indices in splits.items()
                                },
                                "eligible_rows": split_metadata.get(
                                    "eligible_rows"
                                ),
                                "dropped_unmatched_rows": split_metadata.get(
                                    "dropped_unmatched_rows"
                                ),
                                "strategy": split_metadata.get(
                                    "strategy", "random"
                                ),
                                "evaluation_split": split_metadata.get(
                                    "evaluation_split", "test"
                                ),
                            }
                        )

                with self.recorder.stage(
                    "temporal_psi_windows", "resolve temporal reference/comparison periods"
                ) as stage:
                    if self.reuse.can_reuse("temporal"):
                        temporal_windows = self.reuse.load_temporal()
                        stage.update(
                            {
                                "enabled": True,
                                "psi_mode": "temporal",
                                **temporal_windows.metadata,
                                "reused_from": str(self.reuse.source),
                            }
                        )
                        self.logger.info(
                            "[reuse] temporal_psi_windows from %s", self.reuse.source
                        )
                    else:
                        temporal_windows = load_temporal_psi_windows(
                            self.cfg, label_info, self.logger
                        )
                    if temporal_windows is None:
                        stage.update({"enabled": False, "psi_mode": "train_test"})
                    elif not self.reuse.can_reuse("temporal"):
                        np.savez(
                            self.work_dir / "temporal_psi_windows.npz",
                            reference_mask=temporal_windows.reference_mask,
                            comparison_mask=temporal_windows.comparison_mask,
                        )
                        save_json(
                            self.work_dir / "temporal_psi_windows.json",
                            temporal_windows.metadata,
                        )
                        self.reuse.mark("temporal")
                        stage.update(
                            {
                                "enabled": True,
                                "psi_mode": "temporal",
                                **temporal_windows.metadata,
                            }
                        )

                psi_mode = (
                    "temporal"
                    if temporal_windows is not None
                    else (
                        "train_valid"
                        if "filter_test" in splits
                        else "train_test"
                    )
                )
                store = CacheStore(self.work_dir, manifest)
                with self.recorder.stage("filtering_stats", "nan/nunique/pearson/IV/PSI") as stage:
                    metrics_reused = self.reuse.can_reuse("filtering")
                    try:
                        if metrics_reused:
                            cached_result = self.reuse.load_filter_result()
                            filter_result = apply_filter_thresholds(
                                cached_result.report,
                                self.cfg["filtering"],
                                store,
                                self.logger,
                                int(self.cfg["data"].get("sample_rows", 2000)),
                                filter_indices=splits.get("filter_indices"),
                            )
                        else:
                            filter_result = compute_stats_and_filter(
                                store,
                                label_info.y,
                                splits,
                                self.cfg,
                                self.logger,
                                self.heartbeat,
                                temporal_windows=temporal_windows,
                            )
                    except FeatureFilteringError as exc:
                        report_path = self.work_dir / "filter_report.csv"
                        exc.result.report.to_csv(report_path, index=False)
                        self.reuse.mark_filter_metrics()
                        stage.update(
                            {
                                "n_columns": len(exc.result.report),
                                "n_selected": 0,
                                "report_path": str(report_path),
                            }
                        )
                        self.logger.error(
                            "[filter] removed every column; report saved to %s",
                            report_path,
                        )
                        raise ValueError(
                            f"{exc} Filter report saved to {report_path}."
                        ) from exc
                    filter_result.report.to_csv(
                        self.work_dir / "filter_report.csv", index=False
                    )
                    save_json(
                        self.work_dir / "selected_features.json",
                        {
                            "selected": filter_result.selected,
                            "n_selected": len(filter_result.selected),
                            "n_total": len(filter_result.report),
                            "elapsed_seconds": filter_result.elapsed_seconds,
                            "psi_mode": psi_mode,
                        },
                    )
                    self.reuse.mark("filtering")
                    stage.update(
                        {
                            "n_columns": len(filter_result.report),
                            "n_selected": len(filter_result.selected),
                            "psi_mode": psi_mode,
                            "metrics_reused": metrics_reused,
                        }
                    )
                    if metrics_reused:
                        stage.update(
                            {
                                "reused_from": str(self.reuse.source),
                                "thresholds_reapplied": True,
                            }
                        )
                        self.logger.info(
                            "[reuse] filtering metrics from %s; "
                            "reapplied current thresholds",
                            self.reuse.source,
                        )

                with self.recorder.stage("preprocess_fit", "imputers and encoders") as stage:
                    if self.reuse.can_reuse("preprocessing"):
                        preprocess_state = self.reuse.load_preprocess_state()
                        stage.update(
                            {
                                "n_input_columns": len(preprocess_state["columns"]),
                                "n_output_features": preprocess_state["n_output_features"],
                                "encoding": preprocess_state["encoding"],
                                "reused_from": str(self.reuse.source),
                            }
                        )
                        self.logger.info(
                            "[reuse] preprocess_fit from %s", self.reuse.source
                        )
                    else:
                        preprocess_state = fit_preprocessing(
                            store,
                            filter_result.selected,
                            label_info.y,
                            splits,
                            self.cfg,
                            self.logger,
                            self.heartbeat,
                        )
                        save_json(
                            self.work_dir / "preprocess_state.json", preprocess_state
                        )
                        self.reuse.mark("preprocessing")
                        stage.update(
                            {
                                "n_input_columns": len(preprocess_state["columns"]),
                                "n_output_features": preprocess_state["n_output_features"],
                                "encoding": preprocess_state["encoding"],
                            }
                        )

                with self.recorder.stage("matrix_build", "materialize bounded model matrix") as stage:
                    if self.reuse.can_reuse("matrix"):
                        matrix = self.reuse.load_matrix(label_info.y)
                        self.logger.info(
                            "[reuse] matrix_build from %s", self.reuse.source
                        )
                    else:
                        matrix = build_matrix(
                            store,
                            preprocess_state,
                            label_info.y,
                            self.work_dir,
                            self.cfg,
                            self.logger,
                            self.heartbeat,
                        )
                        self.reuse.mark("matrix")
                    stage.update(
                        {
                            "n_rows": int(matrix.X.shape[0]),
                            "n_features": int(matrix.X.shape[1]),
                            "sparse": matrix.sparse,
                            **(
                                {"reused_from": str(self.reuse.source)}
                                if self.reuse.can_reuse("matrix")
                                else {}
                            ),
                        }
                    )

                X = matrix.X
                y = label_info.y
                X_train = X[splits["train"]]
                X_valid = X[splits["valid"]]
                X_test = X[splits["test"]]
                y_train = y[splits["train"]]
                y_valid = y[splits["valid"]]
                y_test = y[splits["test"]]

                with self.recorder.stage("tuning_training", "Optuna TPE + final fit") as stage:
                    tuning_result = tune_and_train(
                        X_train,
                        y_train,
                        X_valid,
                        y_valid,
                        matrix.feature_names,
                        matrix.categorical_indices,
                        self.cfg,
                        self.logger,
                        self.heartbeat,
                        self.work_dir,
                    )
                    stage.update(
                        {
                            "metric": tuning_result.metric,
                            "direction": tuning_result.direction,
                            "best_iteration": tuning_result.best_iteration,
                            "n_trials_run": int(len(tuning_result.trials)),
                            "n_trials_complete": _trial_count(
                                tuning_result.trials, "COMPLETE"
                            ),
                            "n_trials_pruned": _trial_count(
                                tuning_result.trials, "PRUNED"
                            ),
                        }
                    )

                with self.recorder.stage("evaluation", "valid/test metrics") as stage:
                    metrics, predictions = self._evaluate(
                        tuning_result,
                        X_valid,
                        y_valid,
                        X_test,
                        y_test,
                        label_info,
                        splits,
                        matrix,
                    )
                    metrics["evaluation"] = split_metadata
                    if split_metadata.get("strategy") == "temporal":
                        metrics["oot"] = metrics.get("test")
                    stage.update(
                        {
                            "primary_valid": metrics.get("primary_valid"),
                            "primary_test": metrics.get("primary_test"),
                        }
                    )

                with self.recorder.stage("reporting", "artifacts and summary") as stage:
                    self._write_reports(
                        tuning_result, metrics, predictions, matrix, filter_result, manifest
                    )
                    stage.update({"report_dir": str(self.report_dir)})

                self.status = "success"
                summary.update(
                    {
                        "status": "success",
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_seconds": round(self.recorder.total_seconds(), 3),
                        "data": {
                            "n_rows": manifest["n_rows"],
                            "n_cached_columns": manifest["n_logical_columns"],
                            "selected_columns": len(filter_result.selected),
                            "n_output_features": int(matrix.X.shape[1]),
                            "cache_bytes": manifest["total_cache_bytes"],
                            "psi_mode": psi_mode,
                            "split_strategy": split_metadata.get(
                                "strategy", "random"
                            ),
                            "evaluation_split": split_metadata.get(
                                "evaluation_split", "test"
                            ),
                            "split_metadata": split_metadata,
                            "temporal_psi": (
                                temporal_windows.metadata
                                if temporal_windows is not None
                                else None
                            ),
                        },
                        "tuning": {
                            "metric": tuning_result.metric,
                            "direction": tuning_result.direction,
                            "best_params": tuning_result.params,
                            "best_iteration": tuning_result.best_iteration,
                            "n_trials_run": int(len(tuning_result.trials)),
                            "n_trials_complete": _trial_count(
                                tuning_result.trials, "COMPLETE"
                            ),
                            "n_trials_pruned": _trial_count(
                                tuning_result.trials, "PRUNED"
                            ),
                        },
                        "metrics": metrics,
                        "stage_timings": self.recorder.as_dict(),
                        "artifacts": {
                            "report_dir": str(self.report_dir),
                            "model": str(self.work_dir / "model" / "model.txt"),
                            "filter_report": str(self.work_dir / "filter_report.csv"),
                            "split_metadata": str(
                                self.work_dir / "split_metadata.json"
                            ),
                            "temporal_psi_windows": (
                                str(self.work_dir / "temporal_psi_windows.json")
                                if temporal_windows is not None
                                else None
                            ),
                            "temporal_psi_masks": (
                                str(self.work_dir / "temporal_psi_windows.npz")
                                if temporal_windows is not None
                                else None
                            ),
                            "predictions_test": str(
                                self.report_dir / "predictions_test.csv"
                            ),
                            "predictions_oot": (
                                str(self.report_dir / "predictions_oot.csv")
                                if split_metadata.get("strategy") == "temporal"
                                else None
                            ),
                        },
                    }
                )
        except Exception as exc:
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self.logger.error("[failure] %s", self.error)
            self.logger.error("%s", traceback.format_exc())
            summary.update(
                {
                    "status": "failed",
                    "error": self.error,
                    "elapsed_seconds": round(self.recorder.total_seconds(), 3),
                    "stage_timings": self.recorder.as_dict(),
                }
            )
            raise
        finally:
            self.recorder.log_table()
            summary["peak_rss_mb"] = round(peak_rss_mb(), 1)
            summary["reuse"] = self.reuse.as_dict()
            save_json(self.work_dir / "run_summary.json", summary)
            pd.DataFrame(
                [
                    {
                        "stage": name,
                        "elapsed_seconds": round(timing.elapsed_seconds, 4),
                        "elapsed_minutes": round(timing.elapsed_seconds / 60.0, 4),
                        "start_rss_mb": round(timing.start_rss_mb, 1),
                        "end_rss_mb": round(timing.end_rss_mb, 1),
                        "details": json.dumps(timing.details, default=str),
                    }
                    for name, timing in self.recorder.timings.items()
                ]
            ).to_csv(self.work_dir / "stage_timings.csv", index=False)
        return summary

    def _evaluate(
        self,
        tuning_result: Any,
        X_valid: Any,
        y_valid: np.ndarray,
        X_test: Any,
        y_test: np.ndarray,
        label_info: Any,
        splits: Dict[str, np.ndarray],
        matrix: Any,
    ) -> Dict[str, Any]:
        task = label_info.task
        valid_prob = (
            tuning_result.selection_valid_probabilities
            if tuning_result.selection_valid_probabilities is not None
            else tuning_result.adapter.predict_proba(X_valid)
        )
        test_prob = tuning_result.adapter.predict_proba(X_test)
        keys = np.asarray(label_info.keys, dtype=object)
        if task == "binary":
            threshold_setting = self.cfg["metrics"].get("threshold", "auto")
            if threshold_setting in {None, "auto"}:
                selection = select_binary_threshold(y_valid, valid_prob[:, 1])
                threshold = float(selection["threshold"])
                self.logger.info(
                    "[evaluate] threshold selected on valid: %.6f (valid F1=%.4f)",
                    threshold,
                    selection["f1"],
                )
            else:
                threshold = float(threshold_setting)
            valid_metrics = binary_metrics(y_valid, valid_prob[:, 1], threshold)
            test_metrics = binary_metrics(y_test, test_prob[:, 1], threshold)
            primary_valid = valid_metrics.get("auc")
            primary_test = test_metrics.get("auc")
        else:
            threshold = None
            valid_metrics = multiclass_metrics(y_valid, valid_prob)
            test_metrics = multiclass_metrics(y_test, test_prob)
            primary_valid = valid_metrics.get("f1_macro")
            primary_test = test_metrics.get("f1_macro")

        valid_predictions = build_predictions(
            keys[splits["valid"]],
            y_valid,
            valid_prob[:, 1] if task == "binary" else valid_prob,
            task,
            threshold if threshold is not None else 0.5,
            "valid",
        )
        test_predictions = build_predictions(
            keys[splits["test"]],
            y_test,
            test_prob[:, 1] if task == "binary" else test_prob,
            task,
            threshold if threshold is not None else 0.5,
            "test",
        )
        predictions = pd.concat([valid_predictions, test_predictions], ignore_index=True)
        metrics = {
            "task": task,
            "classes": [str(value) for value in label_info.classes],
            "threshold": threshold,
            "primary_metric": "auc" if task == "binary" else "f1_macro",
            "primary_valid": primary_valid,
            "primary_test": primary_test,
            "valid": valid_metrics,
            "test": test_metrics,
            "tuning": {
                "metric": tuning_result.metric,
                "direction": tuning_result.direction,
                "best_iteration": tuning_result.best_iteration,
                "best_params": tuning_result.params,
                "n_trials_run": int(len(tuning_result.trials)),
                "n_trials_complete": _trial_count(tuning_result.trials, "COMPLETE"),
                "n_trials_pruned": _trial_count(tuning_result.trials, "PRUNED"),
            },
        }
        self._log_metrics("valid", valid_metrics, task)
        self._log_metrics("test", test_metrics, task)
        return metrics, predictions

    def _log_metrics(self, split: str, metrics: Dict[str, Any], task: str) -> None:
        if task == "binary":
            self.logger.info(
                "[metrics] %-5s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f ks=%.4f",
                split,
                _metric(metrics, "accuracy"),
                _metric(metrics, "precision"),
                _metric(metrics, "recall"),
                _metric(metrics, "f1"),
                _metric(metrics, "auc"),
                _metric(metrics, "ks"),
            )
        else:
            self.logger.info(
                "[metrics] %-5s accuracy=%.4f f1_macro=%.4f f1_weighted=%.4f auc_ovr=%.4f",
                split,
                _metric(metrics, "accuracy"),
                _metric(metrics, "f1_macro"),
                _metric(metrics, "f1_weighted"),
                _metric(metrics, "auc_ovr_macro"),
            )

    def _write_reports(
        self,
        tuning_result: Any,
        metrics: Dict[str, Any],
        predictions: pd.DataFrame,
        matrix: Any,
        filter_result: Any,
        manifest: Dict[str, Any],
    ) -> None:
        importance = tuning_result.adapter.feature_importance()
        spec_map = {
            spec["name"]: spec.get("source") or spec["name"]
            for spec in matrix.output_specs
        }
        importance["source_feature"] = importance["feature"].map(spec_map).fillna(
            importance["feature"]
        )
        importance.to_csv(self.report_dir / "feature_importance.csv", index=False)
        source_importance = aggregate_importance(
            tuning_result.adapter.feature_importance(), matrix.output_specs
        )
        source_importance.to_csv(
            self.report_dir / "feature_importance_by_source.csv", index=False
        )
        predictions.to_csv(self.report_dir / "predictions.csv", index=False)
        test_predictions = predictions[predictions["split"] == "test"]
        test_predictions.to_csv(
            self.report_dir / "predictions_test.csv", index=False
        )
        if (
            self.cfg.get("_runtime", {})
            .get("split_metadata", {})
            .get("strategy")
            == "temporal"
        ):
            test_predictions.to_csv(
                self.report_dir / "predictions_oot.csv", index=False
            )
        save_metrics_json(self.report_dir / "metrics.json", metrics)
        if self.cfg["output"].get("save_model", True):
            model_path = self.work_dir / "model" / "model.txt"
            tuning_result.adapter.save(model_path)
            self.logger.info("[report] model saved to %s", model_path)
        if self.cfg["metrics"].get("plots", True):
            plot_paths = save_metric_plots(
                metrics["task"],
                test_predictions["y_true"].to_numpy(),
                self._probability_matrix(test_predictions, metrics["task"]),
                float(metrics["threshold"] or 0.5),
                source_importance,
                self.report_dir,
                int(self.cfg["metrics"].get("top_importance", 30)),
                metrics["classes"],
            )
            self.logger.info("[report] wrote %s plots", len(plot_paths))
        self.logger.info(
            "[report] top features: %s",
            ", ".join(
                f"{row.source_feature}={row.gain_pct:.3f}"
                for row in source_importance.head(10).itertuples()
            ),
        )

    @staticmethod
    def _probability_matrix(predictions: pd.DataFrame, task: str) -> np.ndarray:
        if task == "binary":
            return predictions["y_prob"].to_numpy()
        columns = [column for column in predictions.columns if column.startswith("prob_")]
        return predictions[columns].to_numpy()


def _metric(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _trial_count(trials: pd.DataFrame, state: str) -> int:
    if trials.empty or "state" not in trials.columns:
        return 0
    return int((trials["state"] == state).sum())


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return AutoMLPipeline(cfg).run()
