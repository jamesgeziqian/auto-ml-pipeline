"""Classification metrics, threshold selection, and report plots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(scores)
    y_sorted = y_true[order]
    positives = y_sorted == 1
    negatives = ~positives
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")
    cum_positive = np.cumsum(positives) / positives.sum()
    cum_negative = np.cumsum(negatives) / negatives.sum()
    return float(np.max(np.abs(cum_positive - cum_negative)))


def select_binary_threshold(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return {"threshold": 0.5, "f1": float("nan")}
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return {"threshold": 0.5, "f1": float("nan")}
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best]),
        "f1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, np.floating):
        return _rounded(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


def _clean_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _rounded(value) for key, value in payload.items()}


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    predictions = (scores >= float(threshold)).astype(int)
    result: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "ks": ks_statistic(y_true, scores),
        "threshold": float(threshold),
        "log_loss": log_loss(y_true, scores, labels=[0, 1]),
        "brier": brier_score_loss(y_true, scores),
        "positive_rate": float(np.mean(predictions)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
    }
    try:
        result["auc"] = roc_auc_score(y_true, scores)
    except ValueError:
        result["auc"] = float("nan")
    return _clean_dict(result)


def multiclass_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    labels = list(range(probabilities.shape[1]))
    result: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_macro": precision_score(
            y_true, predictions, average="macro", zero_division=0, labels=labels
        ),
        "recall_macro": recall_score(
            y_true, predictions, average="macro", zero_division=0, labels=labels
        ),
        "f1_macro": f1_score(
            y_true, predictions, average="macro", zero_division=0, labels=labels
        ),
        "precision_weighted": precision_score(
            y_true, predictions, average="weighted", zero_division=0, labels=labels
        ),
        "recall_weighted": recall_score(
            y_true, predictions, average="weighted", zero_division=0, labels=labels
        ),
        "f1_weighted": f1_score(
            y_true, predictions, average="weighted", zero_division=0, labels=labels
        ),
        "log_loss": log_loss(y_true, probabilities, labels=labels),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "n": int(len(y_true)),
    }
    try:
        result["auc_ovr_macro"] = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=labels,
        )
    except ValueError:
        result["auc_ovr_macro"] = float("nan")
    return _clean_dict(result)


def aggregate_importance(
    importance: pd.DataFrame,
    output_specs: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    mapping = {
        spec["name"]: spec.get("source") or spec["name"] for spec in output_specs
    }
    frame = importance.copy()
    frame["source_feature"] = frame["feature"].map(mapping).fillna(frame["feature"])
    grouped = (
        frame.groupby("source_feature", as_index=False)
        .agg(
            gain=("gain", "sum"),
            split=("split", "sum"),
            output_columns=("feature", "count"),
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["gain"].sum())
    grouped["gain_pct"] = grouped["gain"] / total if total > 0 else 0.0
    return grouped


def build_predictions(
    keys: Sequence[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    task: str,
    threshold: float,
    split_name: str,
) -> pd.DataFrame:
    frame = pd.DataFrame({"key": list(keys), "split": split_name, "y_true": y_true})
    if task == "binary":
        scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        frame["y_prob"] = scores
        frame["y_pred"] = (scores >= float(threshold)).astype(int)
    else:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        for class_index in range(probabilities.shape[1]):
            frame[f"prob_{class_index}"] = probabilities[:, class_index]
        frame["y_pred"] = probabilities.argmax(axis=1)
    return frame


def save_metric_plots(
    task: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    importance: pd.DataFrame,
    report_dir: str | Path,
    top_importance: int,
    class_names: Sequence[str],
) -> List[str]:
    config_dir = Path(
        os.environ.get("MPLCONFIGDIR") or (Path(report_dir) / ".matplotlib")
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
    os.environ.setdefault("XDG_CACHE_HOME", str(config_dir.parent))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    if task == "binary":
        y_true = np.asarray(y_true).astype(int)
        scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, scores)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label="ROC")
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC curve")
            ax.legend()
            fig.tight_layout()
            path = report_dir / "roc_curve.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths.append(str(path))

            order = np.argsort(-scores)
            y_sorted = y_true[order]
            positives = y_sorted == 1
            cum_pos = np.cumsum(positives) / max(1, positives.sum())
            cum_neg = np.cumsum(~positives) / max(1, (~positives).sum())
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(cum_neg, cum_pos, label="KS curve")
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
            ax.set_xlabel("Cumulative good")
            ax.set_ylabel("Cumulative bad")
            ax.set_title("KS curve")
            ax.legend()
            fig.tight_layout()
            path = report_dir / "ks_curve.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            paths.append(str(path))

    if not importance.empty:
        top = importance.head(max(1, int(top_importance))).iloc[::-1]
        height = max(4.0, 0.28 * len(top) + 1.5)
        fig, ax = plt.subplots(figsize=(8, height))
        ax.barh(top["source_feature"], top["gain"])
        ax.set_xlabel("Gain")
        ax.set_title("Feature importance (top)")
        fig.tight_layout()
        path = report_dir / "feature_importance.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_metrics_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
