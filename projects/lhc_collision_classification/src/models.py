"""Model factory for the LHC collision classification benchmark.

We build everything as sklearn ``Pipeline`` objects with a shared
:class:`StandardScaler` head — important for SVMs in particular, since
HOG bins live on very different scales.
"""

from __future__ import annotations

import logging

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def _wrap(estimator, with_pca: bool = False, n_components: int = 100) -> Pipeline:
    """Bolt the standard scaler (and optional PCA) onto an estimator."""
    steps = [("scaler", StandardScaler())]
    if with_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def make_random_forest(with_pca: bool = False) -> Pipeline:
    """Random Forest with sensible defaults for HOG features."""
    return _wrap(
        RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        with_pca=with_pca,
    )


def make_linear_svm(with_pca: bool = False) -> Pipeline:
    """LinearSVC — fast baseline that the original notebook used."""
    return _wrap(
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        with_pca=with_pca,
    )


def make_rbf_svm(with_pca: bool = True) -> Pipeline:
    """RBF SVM. PCA is on by default because RBF is O(N^2) in features."""
    return _wrap(
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        with_pca=with_pca,
    )


def make_gradient_boosting(with_pca: bool = False) -> Pipeline:
    """Sklearn gradient boosting — used when XGBoost is unavailable."""
    return _wrap(
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        with_pca=with_pca,
    )


def make_xgboost(with_pca: bool = False) -> Pipeline | None:
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
            eval_metric="mlogloss",
            objective="multi:softprob",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        with_pca=with_pca,
    )


def build_model_zoo(with_pca: bool = False) -> dict[str, Pipeline]:
    """Return the standard {name: pipeline} dictionary we benchmark."""
    builders = {
        "Random Forest": lambda: make_random_forest(with_pca=with_pca),
        "Linear SVM": lambda: make_linear_svm(with_pca=with_pca),
        "RBF SVM (PCA)": lambda: make_rbf_svm(with_pca=True),
        "Gradient Boosting": lambda: make_gradient_boosting(with_pca=with_pca),
        "XGBoost": lambda: make_xgboost(with_pca=with_pca),
    }
    zoo: dict[str, Pipeline] = {}
    for name, builder in builders.items():
        pipe = builder()
        if pipe is not None:
            zoo[name] = pipe
    return zoo
