"""
eeg_toolbox.evaluation
======================
Evaluation utilities for EEG classification models.
 
Handles two input formats:
    y_pred : (n_trials,)           — integer class predictions
    y_prob : (n_trials, n_classes) — class probabilities (softmax output)
 
Functions
---------
compute_metrics       — accuracy, f1, recall, confusion matrix, kappa, ROC AUC
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def _to_pred(y_prob_or_pred: np.ndarray) -> np.ndarray:
    """Convert probabilities (n, k) to integer predictions (n,)."""
    if y_prob_or_pred.ndim == 2:
        return np.argmax(y_prob_or_pred, axis=1)
    return y_prob_or_pred.astype(int)

def _check_labels(class_names: Optional[list], n_classes: int) -> list[str]:
    if class_names is not None:
        return class_names
    return [str(i) for i in range(n_classes)]

# ----------------------------------------------------------------------------------------
# 1. Metrics
# ----------------------------------------------------------------------------------------
def compute_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: Optional[list[str]] = None,
    average:     str = "macro"
) -> dict:
    """
    Compute classification metrics.
 
    Parameters
    ----------
    y_true : np.ndarray, shape (n_trials,)
        True integer labels.
    y_pred : np.ndarray, shape (n_trials,) or (n_trials, n_classes)
        Predicted labels or class probabilities.
    class_names : list[str] | None
        Class names for per-class metrics.
    average : str
        Averaging strategy for F1/recall: 'macro' (default) or 'weighted'.
 
    Returns
    -------
    dict with keys:
        accuracy, f1_macro, f1_weighted, recall_macro, recall_weighted,
        confusion_matrix, cohen_kappa, per_class (dict per class)
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score,
        confusion_matrix, cohen_kappa_score
    )

    y_pred_int = _to_pred(y_pred)
    n_classes  = len(np.unique(y_true))
    labels     = _check_labels(class_names, n_classes)
    classes    = np.unique(y_true)
    
    # ---- Global Metrics ----------------------------------------------------------
    acc     = accuracy_score(y_true, y_pred_int)
    f1_mac  = f1_score(y_true, y_pred_int, average="macro", zero_division=0)
    f1_w    = f1_score(y_true, y_pred_int, average="weighted", zero_division=0)
    rec_mac = recall_score(y_true, y_pred_int, average="macro", zero_division=0)
    rec_w   = recall_score(y_true, y_pred_int, average="weighted", zero_division=0)
    kappa   = cohen_kappa_score(y_true, y_pred_int)
    cm      = confusion_matrix(y_true, y_pred_int)

    # ---- Per-class metrics --------------------------------------------------------
    f1_per  = f1_score(y_true, y_pred_int, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred_int, average=None, zero_division=0)
    acc_per = cm.diagonal() / cm.sum(axis=1)

    per_class = {
        labels[i]: {
            "accuracy": float(acc_per[i]),
            "f1":       float(f1_per[i]),
            "recall":   float(rec_per[i])
        }
        for i, cls in enumerate(classes)
    }

    return {
        "accuracy":         float(acc),
        "f1_macro":         float(f1_mac),
        "f1_weighted":      float(f1_w),
        "recall_macro":     float(rec_mac),
        "recall_weighted":  float(rec_w),
        "cohen_kappa":      float(kappa),
        "confusion_matrix": cm,
        "per_class":        per_class,
    }

# Functions 
# ...

# Plots
# ...