"""Validation-time threshold tuning and binary metrics (pickle-free helpers)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def tune_threshold_balanced_accuracy(
    y_true: np.ndarray,
    p_pneumonia: np.ndarray,
    *,
    n_grid: int = 99,
) -> tuple[float, float]:
    """Choose threshold on P(pneumonia) to maximize balanced accuracy on y_true."""
    y_true = np.asarray(y_true, dtype=np.int64)
    p_pneumonia = np.asarray(p_pneumonia, dtype=np.float64)
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, n_grid):
        pred = (p_pneumonia >= t).astype(np.int64)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score


def predict_from_threshold(proba: np.ndarray, threshold: float, *, pos_index: int = 1) -> np.ndarray:
    """Binary labels: 1 if P(pos_class) >= threshold else 0."""
    arr = np.asarray(proba, dtype=np.float64)
    p_pos = arr[:, pos_index] if arr.ndim == 2 else arr
    return (p_pos >= threshold).astype(np.int64)


def confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Assume classes 0=normal, 1=pneumonia."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    prec_pos = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "specificity_normal": spec,
        "npv_normal": npv,
        "sensitivity_pneumonia": sens,
        "precision_pneumonia": prec_pos,
    }
