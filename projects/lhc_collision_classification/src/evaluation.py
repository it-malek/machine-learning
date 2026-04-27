"""Evaluation helpers for the LHC collision benchmark.

Mirrors the heart-failure project's evaluation API but uses macro
averaging because we have a 3-class problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

DEFAULT_SCORING = (
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "balanced_accuracy",
)


@dataclass
class FoldResult:
    """Per-model cross-validated metrics + held-out predictions."""

    name: str
    cv_scores: dict[str, np.ndarray]
    oof_predictions: np.ndarray


def evaluate_models(
    models: Mapping[str, Pipeline],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: tuple[str, ...] = DEFAULT_SCORING,
) -> dict[str, FoldResult]:
    """Run stratified k-fold CV for every model in ``models``."""
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    results: dict[str, FoldResult] = {}
    for name, pipe in models.items():
        logger.info("Cross-validating %s", name)
        cv_scores = cross_validate(
            pipe, X, y, cv=cv, scoring=list(scoring), n_jobs=-1
        )
        oof_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        results[name] = FoldResult(
            name=name,
            cv_scores={k: v for k, v in cv_scores.items() if k.startswith("test_")},
            oof_predictions=oof_pred,
        )
    return results


def results_to_dataframe(results: Mapping[str, FoldResult]) -> pd.DataFrame:
    """Format a ``{name: FoldResult}`` dict into a sortable summary table."""
    rows = []
    for name, res in results.items():
        row: dict[str, str] = {"model": name}
        for metric, values in res.cv_scores.items():
            metric_clean = metric.replace("test_", "")
            row[metric_clean] = f"{values.mean():.3f} ± {values.std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def best_model_by(
    results: Mapping[str, FoldResult], metric: str = "test_f1_macro"
) -> str:
    """Return the model name with the highest mean score on ``metric``."""
    return max(
        results.items(), key=lambda item: item[1].cv_scores[metric].mean()
    )[0]
