"""Base adapter contract for interchangeable model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd


class BaseModelAdapter(ABC):
    name = "base"

    def __init__(
        self,
        task: str,
        n_classes: int,
        params: dict[str, Any],
        seed: int,
        feature_names: Sequence[str],
        categorical_indices: Optional[Sequence[int]] = None,
    ):
        self.task = task
        self.n_classes = int(n_classes)
        self.params = dict(params)
        self.seed = int(seed)
        self.feature_names = list(feature_names)
        self.categorical_indices = list(categorical_indices or [])

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_valid: Any = None,
        y_valid: np.ndarray = None,
        num_boost_round: int = 1000,
        callbacks: Optional[list[Any]] = None,
    ) -> "BaseModelAdapter":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def best_iteration(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError
