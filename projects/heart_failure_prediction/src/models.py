"""Model factory for the heart failure benchmarks.

Each builder returns a complete sklearn ``Pipeline`` with the standard
preprocessor up front, so the caller can simply call ``.fit(X, y)``
on raw DataFrames without worrying about leakage.

XGBoost is included as an optional model — we tolerate it not being
installed so the project still runs in lean environments.
"""

from __future__ import annotations

import logging
from typing import Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def _wrap(estimator) -> Pipeline:
    """Bolt the standard preprocessor onto an estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def make_logistic_regression() -> Pipeline:
    """L2-regularised logistic regression — the interpretable baseline."""
    return _wrap(
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )


def make_random_forest() -> Pipeline:
    """Random forest with class re-weighting for the imbalanced target."""
    return _wrap(
        RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )


def make_gradient_boosting() -> Pipeline:
    """Sklearn's gradient boosting — used when XGBoost is unavailable."""
    return _wrap(
        GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        )
    )


def make_xgboost() -> Pipeline | None:
    """XGBoost classifier; returns ``None`` if the package is missing."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed; skipping XGBoost model.")
        return None

    return _wrap(
        XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )


def make_svm_rbf() -> Pipeline:
    """RBF-kernel SVM with probability outputs enabled (slower but needed for AUC)."""
    return _wrap(
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )


def build_model_zoo() -> dict[str, Pipeline]:
    """Build the dictionary of {name: pipeline} we benchmark in the notebook."""
    builders: dict[str, Callable[[], Pipeline | None]] = {
        "Logistic Regression": make_logistic_regression,
        "Random Forest": make_random_forest,
        "Gradient Boosting": make_gradient_boosting,
        "XGBoost": make_xgboost,
        "SVM (RBF)": make_svm_rbf,
    }
    zoo: dict[str, Pipeline] = {}
    for name, builder in builders.items():
        pipe = builder()
        if pipe is not None:
            zoo[name] = pipe
    return zoo
