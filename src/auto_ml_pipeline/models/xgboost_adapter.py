"""XGBoost model adapter (binary and multiclass classification)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb

from . import register_model
from .base import BaseModelAdapter


@register_model("xgboost")
class XGBoostAdapter(BaseModelAdapter):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model: Optional[xgb.Booster] = None
        self._early_stopped = False
        self._use_categorical = bool(self.categorical_indices)
        self._nthread = int(
            self.params.get(
                "nthread",
                self.params.get("num_threads", self.params.get("n_jobs", 0)),
            )
        )

    def _resolved_params(self) -> dict[str, Any]:
        params = dict(self.params)
        params.pop("early_stopping_rounds", None)
        params.pop("log_period", None)

        threads = params.pop("num_threads", None)
        if threads is None:
            threads = params.pop("n_jobs", 0)
        params.setdefault("nthread", int(threads))
        params["seed"] = self.seed
        params.setdefault("tree_method", "hist")
        params.setdefault("max_bin", 128)
        params.setdefault("verbosity", 0)

        # Accept common LightGBM names when switching model families, while
        # leaving native XGBoost names untouched when both are present.
        aliases = {
            "feature_fraction": "colsample_bytree",
            "bagging_fraction": "subsample",
            "lambda_l1": "alpha",
            "lambda_l2": "reg_lambda",
        }
        for old, new in aliases.items():
            if old in params:
                value = params.pop(old)
                params.setdefault(new, value)
        for unsupported in (
            "bagging_freq",
            "min_sum_hessian_in_leaf",
            "force_col_wise",
            "num_leaves",
            "min_data_in_leaf",
        ):
            params.pop(unsupported, None)
        max_depth = params.get("max_depth")
        if max_depth is not None and int(max_depth) <= 0:
            # LightGBM uses -1 for unlimited depth; XGBoost does not accept it.
            params.pop("max_depth", None)

        if "metric" in params:
            params.setdefault("eval_metric", params.pop("metric"))
        if self.task == "binary":
            params.setdefault("objective", "binary:logistic")
            params.pop("num_class", None)
            params.setdefault("eval_metric", "auc")
        else:
            # The pipeline expects probabilities for multiclass evaluation.
            params["objective"] = "multi:softprob"
            params["num_class"] = self.n_classes
            params.setdefault("eval_metric", "mlogloss")
        return params

    def _feature_types(self, categorical: bool) -> Optional[list[str]]:
        if not categorical or not self.categorical_indices:
            return None
        categorical_set = set(self.categorical_indices)
        return [
            "c" if index in categorical_set else "q"
            for index in range(len(self.feature_names))
        ]

    def _dmatrix(
        self,
        X: Any,
        label: Optional[np.ndarray],
        *,
        categorical: Optional[bool] = None,
    ) -> xgb.DMatrix:
        use_categorical = self._use_categorical if categorical is None else categorical
        feature_types = self._feature_types(use_categorical)
        if sp.issparse(X):
            feature_types = None
            use_categorical = False
        return xgb.DMatrix(
            X,
            label=None if label is None else np.asarray(label),
            feature_names=self.feature_names,
            feature_types=feature_types,
            enable_categorical=bool(feature_types),
            nthread=self._nthread,
        )

    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_valid: Any = None,
        y_valid: np.ndarray = None,
        num_boost_round: int = 1000,
        callbacks: Optional[list[Any]] = None,
    ) -> "XGBoostAdapter":
        params = self._resolved_params()
        early_stopping = int(self.params.get("early_stopping_rounds", 0) or 0)
        has_valid = X_valid is not None and y_valid is not None
        self._early_stopped = bool(has_valid and early_stopping > 0)

        def train(categorical: bool) -> xgb.Booster:
            train_matrix = self._dmatrix(
                X_train, y_train, categorical=categorical
            )
            evals = []
            if has_valid:
                evals.append(
                    (
                        self._dmatrix(
                            X_valid, y_valid, categorical=categorical
                        ),
                        "valid",
                    )
                )
            return xgb.train(
                params,
                train_matrix,
                num_boost_round=int(num_boost_round),
                evals=evals or None,
                early_stopping_rounds=early_stopping if self._early_stopped else None,
                verbose_eval=False,
                callbacks=list(callbacks or []),
            )

        try:
            self.model = train(self._use_categorical)
        except xgb.core.XGBoostError as exc:
            if not self._use_categorical or "categor" not in str(exc).lower():
                raise
            # Native categorical codes can be non-contiguous or not integer
            # valued. Fall back to numeric codes rather than failing the run.
            self._use_categorical = False
            self.model = train(False)
        return self

    @property
    def best_iteration(self) -> int:
        if self.model is None:
            return 0
        if self._early_stopped:
            # XGBoost best_iteration is zero-based.
            return int(self.model.best_iteration) + 1
        return int(self.model.num_boosted_rounds())

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        matrix = self._dmatrix(X, None)
        kwargs: dict[str, Any] = {}
        if self._early_stopped:
            kwargs["iteration_range"] = (0, self.best_iteration)
        prediction = self.model.predict(matrix, **kwargs)
        if self.task == "binary":
            probability = np.asarray(prediction, dtype=np.float64).reshape(-1)
            probability = np.clip(probability, 0.0, 1.0)
            return np.column_stack([1.0 - probability, probability])
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        return prediction

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        gain_map = self.model.get_score(importance_type="total_gain")
        split_map = self.model.get_score(importance_type="weight")
        gain = np.asarray(
            [float(gain_map.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        split = np.asarray(
            [int(split_map.get(name, 0)) for name in self.feature_names],
            dtype=np.int64,
        )
        frame = pd.DataFrame(
            {
                "feature": self.feature_names,
                "gain": gain,
                "split": split,
            }
        )
        total_gain = float(frame["gain"].sum())
        frame["gain_pct"] = (
            frame["gain"] / total_gain if total_gain > 0 else 0.0
        )
        return frame.sort_values("gain", ascending=False).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
