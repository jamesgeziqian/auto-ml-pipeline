"""LightGBM model adapter (binary and multiclass classification)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import register_model
from .base import BaseModelAdapter


@register_model("lightgbm")
class LightGBMAdapter(BaseModelAdapter):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model: Optional[lgb.Booster] = None

    def _resolved_params(self) -> dict[str, Any]:
        params = dict(self.params)
        params["verbose"] = -1
        params["seed"] = self.seed
        params.setdefault("num_threads", 0)
        params.setdefault("force_col_wise", True)
        params.pop("early_stopping_rounds", None)
        params.pop("log_period", None)
        if self.task == "binary":
            params["objective"] = params.get("objective", "binary")
            params.setdefault("metric", "auc")
            params.pop("num_class", None)
        else:
            params["objective"] = params.get("objective", "multiclass")
            params["num_class"] = self.n_classes
            params.setdefault("metric", "multi_logloss")
        return params

    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_valid: Any = None,
        y_valid: np.ndarray = None,
        num_boost_round: int = 1000,
        callbacks: Optional[list[Any]] = None,
    ) -> "LightGBMAdapter":
        params = self._resolved_params()
        categorical_feature: Any = self.categorical_indices if self.categorical_indices else []
        train_set = lgb.Dataset(
            X_train,
            label=np.asarray(y_train),
            categorical_feature=categorical_feature,
            free_raw_data=True,
        )
        valid_sets = []
        valid_names = []
        if X_valid is not None and y_valid is not None:
            valid_set = lgb.Dataset(
                X_valid,
                label=np.asarray(y_valid),
                reference=train_set,
                categorical_feature=categorical_feature,
                free_raw_data=True,
            )
            valid_sets.append(valid_set)
            valid_names.append("valid")

        user_callbacks = list(callbacks or [])
        early_stopping = self.params.get("early_stopping_rounds")
        log_period = int(self.params.get("log_period", 100))
        callback_list: list[Any] = []
        if valid_sets and early_stopping:
            callback_list.append(
                lgb.early_stopping(int(early_stopping), verbose=False)
            )
        callback_list.append(lgb.log_evaluation(period=max(1, log_period)))
        callback_list.extend(user_callbacks)

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=int(num_boost_round),
            valid_sets=valid_sets if valid_sets else None,
            valid_names=valid_names if valid_names else None,
            callbacks=callback_list,
        )
        return self

    @property
    def best_iteration(self) -> int:
        if self.model is None:
            return 0
        best = int(self.model.best_iteration or 0)
        return best if best > 0 else int(self.model.current_iteration())

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        prediction = self.model.predict(
            X, num_iteration=self.best_iteration if self.best_iteration > 0 else None
        )
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
        gain = self.model.feature_importance(importance_type="gain")
        split = self.model.feature_importance(importance_type="split")
        frame = pd.DataFrame(
            {
                "feature": self.feature_names,
                "gain": gain.astype(np.float64),
                "split": split.astype(np.int64),
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
