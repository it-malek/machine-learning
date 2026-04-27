"""Cross-validated evaluation helpers for the heart failure project."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from utils.metrics import compute_classification_report

logger = logging.getLogger(__name__)

DEFAULT_SCORING = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)


@dataclass
class FoldResult:
    """Per-model cross-validated metrics + held-out predictions."""

    name: str
    cv_scores: dict[str, np.ndarray]  # raw fold scores per metric
    oof_predictions: np.ndarray  # out-of-fold class labels
    oof_probabilities: np.ndarray  # out-of-fold P(class=1)


def evaluate_models(
    models: Mapping[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: tuple[str, ...] = DEFAULT_SCORING,
) -> dict[str, FoldResult]:
    """Run stratified k-fold CV for every model in ``models``.

    Args:
        models: ``{name: pipeline}`` dictionary. Pipelines must include
            their own preprocessing so that the scaler is re-fit on each
            training fold (no leakage).
        X: Feature matrix.
        y: Target vector.
        n_splits: Number of stratified folds; defaults to 5.
        random_state: Seed for the fold splitter so results are
            reproducible across reruns.
        scoring: Iterable of sklearn scoring strings to compute.

    Returns:
        ``{name: FoldResult}`` keyed by model name.
    """
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    results: dict[str, FoldResult] = {}
    for name, pipe in models.items():
        logger.info("Cross-validating %s", name)
        cv_scores = cross_validate(
            pipe, X, y, cv=cv, scoring=list(scoring), n_jobs=-1
        )
        # Per-fold predictions for plotting (ROC, calibration, etc.)
        oof_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        oof_proba = cross_val_predict(
            pipe, X, y, cv=cv, n_jobs=-1, method="predict_proba"
        )[:, 1]
        results[name] = FoldResult(
            name=name,
            cv_scores={k: v for k, v in cv_scores.items() if k.startswith("test_")},
            oof_predictions=oof_pred,
            oof_probabilities=oof_proba,
        )
    return results


def results_to_dataframe(results: Mapping[str, FoldResult]) -> pd.DataFrame:
    """Format a ``{name: FoldResult}`` dict into a sortable summary table.

    Each cell is the per-fold mean (with standard deviation in
    parentheses), giving one row per model and one column per metric.
    """
    rows = []
    for name, res in results.items():
        row: dict[str, str] = {"model": name}
        for metric, values in res.cv_scores.items():
            metric_clean = metric.replace("test_", "")
            row[metric_clean] = f"{values.mean():.3f} ± {values.std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def best_model_by(
    results: Mapping[str, FoldResult], metric: str = "test_roc_auc"
) -> str:
    """Return the model name with the highest mean score on ``metric``."""
    return max(
        results.items(), key=lambda item: item[1].cv_scores[metric].mean()
    )[0]


def threshold_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Convenience: compute the standard metrics at an arbitrary threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    report = compute_classification_report(y_true, y_pred, y_proba)
    return report.to_dict()
