"""Reusable classification metric helpers.

These wrappers exist so that every project in this repo reports the same
set of metrics in the same shape, which makes cross-project comparison
in the README tables straightforward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationReport:
    """Container for the standard set of metrics we report.

    Attributes:
        accuracy: Fraction of correctly classified samples.
        precision: Precision (positive predictive value); ``None`` for
            multi-class results without an explicit averaging strategy.
        recall: Recall (sensitivity); ``None`` for multi-class without
            averaging.
        specificity: True negative rate. Only defined for binary tasks.
        f1: F1 score under the chosen averaging strategy.
        roc_auc: Area under the ROC curve (one-vs-rest for multi-class).
        average_precision: Area under the PR curve. Binary tasks only.
        brier_score: Mean squared error of probabilistic predictions
            against the true label. Binary tasks only.
    """

    accuracy: float
    precision: float | None
    recall: float | None
    specificity: float | None
    f1: float
    roc_auc: float | None
    average_precision: float | None
    brier_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_binary(y_true: np.ndarray) -> bool:
    return len(np.unique(y_true)) == 2


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    average: str = "binary",
) -> ClassificationReport:
    """Compute the standard set of classification metrics.

    Args:
        y_true: Ground-truth labels with shape ``(n_samples,)``.
        y_pred: Predicted labels with shape ``(n_samples,)``.
        y_proba: For binary tasks, the probability of the positive class
            with shape ``(n_samples,)``. For multi-class tasks, the full
            probability matrix with shape ``(n_samples, n_classes)``. May
            be ``None``, in which case probabilistic metrics are skipped.
        average: Averaging strategy passed through to ``precision_score``,
            ``recall_score``, and ``f1_score``. Use ``"binary"`` for
            two-class problems, ``"macro"`` or ``"weighted"`` for
            multi-class.

    Returns:
        A populated :class:`ClassificationReport`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    binary = _is_binary(y_true)

    accuracy = float(accuracy_score(y_true, y_pred))

    # precision / recall / f1
    if binary or average != "binary":
        precision = float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        )
        recall = float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        )
        f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    else:
        precision = recall = None
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    specificity: float | None = None
    if binary:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc: float | None = None
    average_precision: float | None = None
    brier: float | None = None
    if y_proba is not None:
        try:
            if binary:
                proba_pos = (
                    y_proba if y_proba.ndim == 1 else y_proba[:, 1]
                )
                roc_auc = float(roc_auc_score(y_true, proba_pos))
                average_precision = float(
                    average_precision_score(y_true, proba_pos)
                )
                brier = float(brier_score_loss(y_true, proba_pos))
            else:
                roc_auc = float(
                    roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro"
                    )
                )
        except ValueError as exc:
            # Common when only one class appears in y_true (e.g. tiny CV fold)
            logger.warning("Could not compute probabilistic metrics: %s", exc)

    return ClassificationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        roc_auc=roc_auc,
        average_precision=average_precision,
        brier_score=brier,
    )


def summarize_cv_scores(scores: dict[str, np.ndarray]) -> dict[str, str]:
    """Format a ``cross_validate`` result into ``"mean ± std"`` strings.

    Args:
        scores: Mapping from score name (e.g. ``"test_roc_auc"``) to an
            array of fold-level values, as returned by
            :func:`sklearn.model_selection.cross_validate`.

    Returns:
        Mapping from the same score names to formatted summary strings
        such as ``"0.872 ± 0.034"``.
    """
    summary = {}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        values = np.asarray(values)
        summary[key] = f"{values.mean():.3f} ± {values.std():.3f}"
    return summary
