"""Optuna TPE hyperparameter tuning around interchangeable model adapters."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
import yaml
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from .logging_utils import Heartbeat
from .models import create_adapter


@dataclass
class TuningResult:
    adapter: Any
    params: Dict[str, Any]
    best_iteration: int
    trials: pd.DataFrame
    metric: str
    direction: str
    elapsed_seconds: float
    selection_valid_probabilities: Optional[np.ndarray] = None


def load_search_space(path: str | Path, logger: Any) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning("[tune] search space file %s does not exist; using fixed params only", path)
        return {"parameters": {}, "fixed": {}}
    with path_obj.open("r", encoding="utf-8") as handle:
        space = yaml.safe_load(handle) or {}
    if "parameters" not in space:
        space["parameters"] = {}
    if "fixed" not in space:
        space["fixed"] = {}
    return space


def suggest_params(trial: optuna.Trial, parameters: Dict[str, Any]) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}
    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Search space entry {name!r} must be a mapping.")
        kind = spec.get("type")
        if kind is None:
            kind = "float" if isinstance(spec.get("low"), float) or isinstance(spec.get("high"), float) else "int"
        if kind == "int":
            sampled[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif kind == "float":
            sampled[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step"),
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            sampled[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported search-space type {kind!r} for {name!r}.")
    return sampled


class LightGBMPruningCallback:
    """Report intermediate validation metrics to Optuna and prune poor trials."""

    def __init__(self, trial: optuna.Trial, metric: str, valid_name: str = "valid"):
        self.trial = trial
        self.metric = metric
        self.valid_name = valid_name

    def __call__(self, env: Any) -> None:
        for result in env.evaluation_result_list:
            data_name, metric_name, value = result[0], result[1], result[2]
            if data_name == self.valid_name and metric_name == self.metric:
                self.trial.report(float(value), step=int(env.iteration))
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
                break


class XGBoostPruningCallback(xgb.callback.TrainingCallback):
    """Report XGBoost validation metrics to Optuna after each boosting round."""

    _METRIC_ALIASES = {  # noqa: RUF012
        "multi_logloss": "mlogloss",
        "binary_logloss": "logloss",
    }

    def __init__(self, trial: optuna.Trial, metric: str, valid_name: str = "valid"):
        super().__init__()
        self.trial = trial
        self.metric = metric
        self.valid_name = valid_name

    def after_iteration(self, model: Any, epoch: int, evals_log: Dict[str, Any]) -> bool:
        candidates = {self.metric, self._METRIC_ALIASES.get(self.metric, self.metric)}
        for data_name, metrics in evals_log.items():
            if data_name != self.valid_name:
                continue
            for metric_name, values in metrics.items():
                if metric_name in candidates and values:
                    self.trial.report(float(values[-1]), step=int(epoch))
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
                    return False
        return False


def _validation_metric(
    task: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: str,
) -> float:
    if task == "binary":
        positive = probabilities[:, 1]
        if metric == "auc":
            return float(roc_auc_score(y_true, positive))
        if metric == "accuracy":
            return float(accuracy_score(y_true, (positive >= 0.5).astype(int)))
        if metric in {"f1", "f1_macro"}:
            return float(f1_score(y_true, (positive >= 0.5).astype(int), average="binary"))
        raise ValueError(f"Unsupported binary tuning metric {metric!r}.")
    if metric in {"multi_logloss", "logloss"}:
        labels = list(range(probabilities.shape[1]))
        return float(log_loss(y_true, probabilities, labels=labels))
    if metric == "accuracy":
        return float(accuracy_score(y_true, probabilities.argmax(axis=1)))
    if metric in {"f1", "f1_macro", "f1_weighted"}:
        return float(
            f1_score(
                y_true,
                probabilities.argmax(axis=1),
                average="macro" if metric in {"f1", "f1_macro"} else "weighted",
            )
        )
    raise ValueError(f"Unsupported multiclass tuning metric {metric!r}.")


def tune_and_train(
    X_train: Any,
    y_train: np.ndarray,
    X_valid: Any,
    y_valid: np.ndarray,
    feature_names: Sequence[str],
    categorical_indices: Sequence[int],
    cfg: Dict[str, Any],
    logger: Any,
    heartbeat: Heartbeat,
    work_dir: str | Path,
) -> TuningResult:
    started = time.perf_counter()
    model_cfg = cfg["model"]
    tuning_cfg = cfg["tuning"]
    task = cfg["_runtime"]["task"]
    n_classes = int(cfg["_runtime"]["n_classes"])
    seed = int(cfg["project"]["random_seed"])
    model_name = model_cfg.get("name", "lightgbm")
    work_dir = Path(work_dir)
    tuning_dir = work_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    space = load_search_space(model_cfg.get("search_space"), logger)
    fixed = dict(space.get("fixed", {}))
    parameters = dict(space.get("parameters", {}))
    configured_params = dict(model_cfg.get("params", {}))

    if task == "binary":
        default_metric = "auc"
        default_direction = "maximize"
    else:
        default_metric = "multi_logloss"
        default_direction = "minimize"
    metric = tuning_cfg.get("metric", "auto")
    if metric in {None, "auto"}:
        metric = default_metric
    direction = tuning_cfg.get("direction", "auto")
    if direction in {None, "auto"}:
        direction = "maximize" if metric in {"auc", "accuracy", "f1", "f1_macro", "f1_weighted"} else "minimize"

    base_params = {
        **fixed,
        **configured_params,
        "early_stopping_rounds": int(model_cfg.get("early_stopping_rounds", 80)),
        "log_period": int(model_cfg.get("log_period", 100)),
        "num_threads": int(model_cfg.get("num_threads", 0)),
    }
    num_boost_round = int(model_cfg.get("num_boost_round", 2000))

    def make_adapter(params: Dict[str, Any]) -> Any:
        return create_adapter(
            model_name,
            task=task,
            n_classes=n_classes,
            params=params,
            seed=int(model_cfg.get("seed", seed)),
            feature_names=feature_names,
            categorical_indices=categorical_indices,
        )

    study: Optional[optuna.Study] = None
    best_iteration = num_boost_round
    trials_rows: List[Dict[str, Any]] = []

    if tuning_cfg.get("enabled", True) and parameters:
        sampler_cfg = tuning_cfg.get("sampler", {})
        sampler = optuna.samplers.TPESampler(
            seed=int(sampler_cfg.get("seed", seed)),
            n_startup_trials=int(sampler_cfg.get("n_startup_trials", 10)),
        )
        pruner = (
            optuna.pruners.MedianPruner(
                n_startup_trials=int(sampler_cfg.get("n_startup_trials", 10)),
                n_warmup_steps=int(tuning_cfg.get("n_warmup_steps", 20)),
            )
            if tuning_cfg.get("pruning", True)
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        def objective(trial: optuna.Trial) -> float:
            params = {**base_params, **suggest_params(trial, parameters)}
            logger.info("[tune] trial %s start params=%s", trial.number, _short_params(params))
            callbacks = []
            if tuning_cfg.get("pruning", True):
                if model_name == "lightgbm":
                    callbacks.append(LightGBMPruningCallback(trial, metric))
                elif model_name == "xgboost":
                    callbacks.append(XGBoostPruningCallback(trial, metric))
                else:
                    logger.warning(
                        "[tune] pruning is not implemented for model=%s; " "running without intermediate pruning",
                        model_name,
                    )
            adapter = make_adapter(params)
            adapter.fit(
                X_train,
                y_train,
                X_valid,
                y_valid,
                num_boost_round=num_boost_round,
                callbacks=callbacks,
            )
            probabilities = adapter.predict_proba(X_valid)
            score = _validation_metric(task, y_valid, probabilities, metric)
            trial.set_user_attr("best_iteration", adapter.best_iteration)
            trial.set_user_attr("score", score)
            logger.info(
                "[tune] trial %s done %s=%.6f best_iteration=%s",
                trial.number,
                metric,
                score,
                adapter.best_iteration,
            )
            return score

        def log_trial(study_obj: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            try:
                duration = float(trial.duration.total_seconds()) if trial.duration else 0.0
            except Exception:
                duration = 0.0
            trials_rows.append(
                {
                    "number": trial.number,
                    "state": str(trial.state.name),
                    "value": trial.value,
                    "duration_seconds": round(duration, 3),
                    "best_iteration": trial.user_attrs.get("best_iteration"),
                    "params": json.dumps(trial.params, sort_keys=True),
                }
            )
            try:
                best_value = study_obj.best_value
            except ValueError:
                best_value = None
            logger.info(
                "[tune] %s/%s trials complete: trial=%s state=%s value=%s best=%s",
                len(study_obj.trials),
                int(tuning_cfg.get("n_trials", 20)),
                trial.number,
                trial.state.name,
                trial.value,
                best_value if best_value is not None else "n/a",
            )
            heartbeat.detail(
                f"tuning trial={trial.number} state={trial.state.name} best={best_value if best_value is not None else 'n/a'}"
            )

        logger.info(
            "[tune] starting TPE: model=%s trials=%s timeout=%ss direction=%s metric=%s",
            model_name,
            tuning_cfg.get("n_trials"),
            tuning_cfg.get("timeout_seconds"),
            direction,
            metric,
        )
        study.optimize(
            objective,
            n_trials=int(tuning_cfg.get("n_trials", 20)),
            timeout=tuning_cfg.get("timeout_seconds"),
            callbacks=[log_trial],
        )
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise RuntimeError("All Optuna trials failed or were pruned; no model can be trained.")
        best_trial = study.best_trial
        best_params = {**base_params, **best_trial.params}
        best_iteration = int(best_trial.user_attrs.get("best_iteration", 0) or 0)
        logger.info(
            "[tune] best trial=%s %s=%.6f params=%s",
            best_trial.number,
            metric,
            best_trial.value,
            _short_params(best_params),
        )
    else:
        best_params = dict(base_params)
        logger.info(
            "[tune] tuning disabled or search space empty; training once with params=%s",
            _short_params(best_params),
        )

    trials = pd.DataFrame(trials_rows)
    if not trials.empty:
        trials = trials.sort_values("number").reset_index(drop=True)
        trials.to_csv(tuning_dir / "trials.csv", index=False)

    refit_on_train_valid = bool(tuning_cfg.get("refit_on_train_valid", True))
    final_adapter = make_adapter(best_params)
    selection_valid_probabilities: Optional[np.ndarray] = None
    if refit_on_train_valid and X_valid is not None:
        # Keep an honest out-of-sample valid score: train the selected
        # configuration on train only, then refit on train+valid for the final model.
        selection_adapter = make_adapter(best_params)
        selection_rounds = max(1, int(best_iteration)) if best_iteration else num_boost_round
        selection_adapter.fit(
            X_train,
            y_train,
            X_valid=None,
            y_valid=None,
            num_boost_round=selection_rounds,
            callbacks=[],
        )
        selection_valid_probabilities = selection_adapter.predict_proba(X_valid)
        logger.info(
            "[train] refitting best configuration on train+valid (%s + %s rows)",
            len(y_train),
            len(y_valid),
        )
        if sp.issparse(X_train):
            X_fit = sp.vstack([X_train, X_valid], format="csr")
        else:
            X_fit = np.vstack([np.asarray(X_train), np.asarray(X_valid)])
        y_fit = np.concatenate([y_train, y_valid])
        final_rounds = max(1, int(best_iteration)) if best_iteration else num_boost_round
        final_adapter.fit(
            X_fit,
            y_fit,
            X_valid=None,
            y_valid=None,
            num_boost_round=final_rounds,
            callbacks=[],
        )
    else:
        logger.info("[train] fitting best configuration on train split with valid early stopping")
        final_adapter.fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            num_boost_round=max(1, best_iteration) if best_iteration else num_boost_round,
            callbacks=[],
        )
        selection_valid_probabilities = final_adapter.predict_proba(X_valid)

    elapsed = time.perf_counter() - started
    summary = {
        "model": model_name,
        "metric": metric,
        "direction": direction,
        "best_params": best_params,
        "best_iteration": best_iteration,
        "n_trials_requested": int(tuning_cfg.get("n_trials", 20)) if study else 0,
        "n_trials_run": len(study.trials) if study else 0,
        "n_trials_complete": (
            int(sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)) if study else 0
        ),
        "n_trials_pruned": (
            int(sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)) if study else 0
        ),
        "elapsed_seconds": round(elapsed, 4),
    }
    with (tuning_dir / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    logger.info("[train] final model trained in %.2fs", elapsed)
    return TuningResult(
        adapter=final_adapter,
        params=best_params,
        best_iteration=int(final_adapter.best_iteration),
        trials=trials,
        metric=metric,
        direction=direction,
        elapsed_seconds=elapsed,
        selection_valid_probabilities=selection_valid_probabilities,
    )


def _short_params(params: Dict[str, Any]) -> str:
    keys = [
        "num_leaves",
        "max_depth",
        "learning_rate",
        "eta",
        "min_data_in_leaf",
        "min_child_weight",
        "feature_fraction",
        "colsample_bytree",
        "bagging_fraction",
        "subsample",
        "lambda_l1",
        "alpha",
        "lambda_l2",
        "reg_lambda",
    ]
    values = []
    for key in keys:
        if key in params:
            value = params[key]
            values.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(values)
