"""Reusable plotting helpers used across the projects.

These functions all return ``matplotlib.figure.Figure`` objects so the
caller can decide whether to display, save, or further customise the
figure. None of them call ``plt.show()`` themselves; that's the caller's
responsibility (notebook display vs. headless save).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc as _auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Iterable[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    cmap: str = "Blues",
    figsize: tuple[float, float] = (5.5, 4.5),
) -> plt.Figure:
    """Plot a (normalised) confusion matrix as a Seaborn heatmap."""
    class_names = list(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        with np.errstate(invalid="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
    fmt = ".2f" if normalize else "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves",
    figsize: tuple[float, float] = (6.0, 5.0),
) -> plt.Figure:
    """Plot ROC curves for one or more models on shared axes.

    Args:
        curves: Mapping of model name to ``(y_true, y_score)`` tuples,
            where ``y_score`` is the predicted probability (or any
            monotonic score) for the positive class.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = _auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_precision_recall_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Precision–Recall Curves",
    figsize: tuple[float, float] = (6.0, 5.0),
) -> plt.Figure:
    """Plot precision–recall curves for one or more models."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, (y_true, y_score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, label=name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_calibration_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
    title: str = "Calibration Curves",
    figsize: tuple[float, float] = (6.0, 5.0),
) -> plt.Figure:
    """Plot reliability diagrams for one or more probabilistic models.

    A perfectly calibrated model lies on the diagonal: when it predicts
    a probability of 0.7, the event occurs 70% of the time.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, (y_true, y_proba) in curves.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importances(
    importances: np.ndarray,
    feature_names: Iterable[str],
    top_k: int | None = None,
    title: str = "Feature Importances",
    figsize: tuple[float, float] = (7.0, 5.0),
) -> plt.Figure:
    """Plot a horizontal bar chart of feature importances, sorted."""
    feature_names = list(feature_names)
    order = np.argsort(importances)[::-1]
    if top_k is not None:
        order = order[:top_k]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_threshold_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Metrics vs. Decision Threshold",
    figsize: tuple[float, float] = (7.0, 5.0),
) -> plt.Figure:
    """Plot precision, recall, and F1 as a function of the decision threshold.

    Useful for picking an operating point that isn't the default 0.5.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1s, label="F1")
    ax.axvline(0.5, ls="--", color="gray", alpha=0.5, label="Default (0.5)")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a figure to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
