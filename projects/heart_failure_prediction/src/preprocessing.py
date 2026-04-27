"""Preprocessing pipeline for the heart failure dataset.

We standardise the continuous features and pass the already-binary
indicators through unchanged. Wrapping this as a ``ColumnTransformer``
inside an sklearn ``Pipeline`` is what prevents data leakage during
cross-validation: the scaler is re-fit on each training fold rather
than on the full dataset.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from .data_loader import BINARY_FEATURES, CONTINUOUS_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Construct the leakage-free preprocessing transformer.

    Returns:
        A fitted-on-demand ``ColumnTransformer`` that scales continuous
        features with :class:`StandardScaler` and passes binary
        indicators through unchanged. Use it as the first step of any
        downstream model :class:`~sklearn.pipeline.Pipeline`.
    """
    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), CONTINUOUS_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names() -> list[str]:
    """Return the names of features in the order the preprocessor emits them."""
    return list(CONTINUOUS_FEATURES) + list(BINARY_FEATURES)
