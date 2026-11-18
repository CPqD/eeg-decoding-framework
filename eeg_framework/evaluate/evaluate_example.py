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

# ----------------------------------------------------------------------------------------
# 2. Plots
# ----------------------------------------------------------------------------------------
def plot_metrics_bar(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    metrics:     list[str]            = ["accuracy", "f1", "recall"],
    class_names: Optional[list[str]]  = None,
    title:       str                  = "Per-class metrics",
    savepath:    Optional[str]        = None,
    show:        bool                 = True,
) -> plt.Figure:
    """
    Bar chart of per-class metrics.
 
    Parameters
    ----------
    metrics : list[str]
        Any combination of 'accuracy', 'f1', 'recall'.
    """
    _valid = {"accuracy", "f1", "recall"}
    invalid = set(metrics) - _valid
    if invalid:
        raise ValueError(f"Invalid metrics: {invalid}. Choose from {_valid}")
    
    result    = compute_metrics(y_true, y_pred, class_names)
    per_class = result["per_class"]
    classes   = list(per_class.keys())
    n_classes = len(classes)
    n_metrics = len(metrics)

    x = np.arange(n_classes)
    width = 0.8 / n_metrics
    colors = ["#082bc5", "#f30e0e", "#09b917"]

    fig, ax = plt.subplots(figsize=(5, 4))

    for i, metric in enumerate(metrics):
        values = [per_class[cls][metric] for cls in classes]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width=width * 0.9,
                        label=metric.capitalize(), color=colors[i % len(colors)], 
                        edgecolor='k', lw=0.8, zorder=2)
    
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    
    ax.axhline(1 / n_classes, color="black", linewidth=1,
               linestyle="--", label=f"Chance ({1/n_classes:.2f})")
    ax.grid(True, ls='dotted')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    #ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, ncol = 2)
    fig.tight_layout()

    if show:
        plt.show()
    return fig

# Functions 
# ...
def summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> None:
    """Print a formatted summary of all metrics."""
    metrics = compute_metrics(y_true, y_pred, class_names)

    print("─" * 45)
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)      : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted)   : {metrics['f1_weighted']:.4f}")
    print(f"  Recall (macro)  : {metrics['recall_macro']:.4f}")
    print(f"  Cohen's Kappa   : {metrics['cohen_kappa']:.4f}")
    print("─" * 45)
    print(f"  {'Class':<14} {'Accuracy':>9} {'F1':>9} {'Recall':>9}")
    print("─" * 45)
    for cls, vals in metrics["per_class"].items():
        print(
            f"  {cls:<14} "
            f"{vals['accuracy']:>9.4f} "
            f"{vals['f1']:>9.4f} "
            f"{vals['recall']:>9.4f}"
        )
    print("─" * 45)

def plot_tsne(X_train: np.array, y_train: np.array):
    from sklearn.manifold import TSNE

    X_train_tsne = TSNE(n_components=2).fit_transform(X_train)

    # Convertendo y_train para um array NumPy caso ainda não seja
    y_train_np = np.array(y_train)

    fig, ax = plt.subplots(figsize=(5, 4))

    # 2. Plotamos APENAS a classe 1 (Right - Azul)
    ax.scatter(X_train_tsne[y_train_np == 1, 0], 
            X_train_tsne[y_train_np == 1, 1], 
            c='blue', 
            label='right',  # O nome já vai direto aqui!
            s=100, alpha=0.8, edgecolor='k', linewidth=0.5, zorder=20)

    # 1. Plotamos APENAS a classe 0 (Left - Vermelho)
    ax.scatter(X_train_tsne[y_train_np == 0, 0], 
            X_train_tsne[y_train_np == 0, 1], 
            c='red', 
            label='left',   # O nome já vai direto aqui!
            s=100, alpha=0.8, edgecolor='k', linewidth=0.5, zorder=20)

    ax.set_xlabel('t-SNE 1')#, fontsize=12, labelpad=10)
    ax.set_ylabel('t-SNE 2')#, fontsize=12, labelpad=10)

    # 3. Chama a legenda de forma simples (agora ela puxa os 'labels' de cima)
    ax.legend()
            #bbox_to_anchor=(1.05, 1), 
            #loc='upper left', 
            #frameon=True)

    ax.grid(True, ls='dotted')

    # 4. TRUQUE MÁGICO: O rect dá uma "esmagada" no gráfico para a esquerda, 
    # liberando espaço à direita para a legenda não ser cortada!
    plt.tight_layout()

    plt.show()