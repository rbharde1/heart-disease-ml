"""Metrics and small evaluation helpers."""
import json

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from config import RANDOM_STATE

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall (sensitivity)",
    "specificity": "Specificity",
    "f1": "F1 score",
    "roc_auc": "ROC-AUC",
    "avg_precision": "Average precision",
}


def predict_scores(model, X):
    """Probability of the positive class (falls back to decision_function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_score, threshold=0.5):
    """Threshold-dependent and threshold-free metrics for binary predictions."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    single_class = len(np.unique(y_true)) < 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": float("nan") if single_class else roc_auc_score(y_true, y_score),
        "avg_precision": float("nan") if single_class else average_precision_score(y_true, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, y_score, metric="roc_auc", n_boot=2000, alpha=0.05,
                 threshold=0.5, random_state=RANDOM_STATE):
    """Percentile bootstrap confidence interval for one metric on a fixed test set."""
    rng = np.random.default_rng(random_state)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(compute_metrics(y_true[idx], y_score[idx], threshold)[metric])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    return obj


def save_json(obj, path):
    path.write_text(json.dumps(_to_builtin(obj), indent=2))
