"""Shared utilities for the MachineLearning portfolio.

Modules:
    metrics:        Helpers to compute classification metrics consistently
                    across projects.
    visualization:  Reusable matplotlib/seaborn plotting helpers
                    (confusion matrices, ROC/PR curves, calibration).
"""

from utils import metrics, visualization

__all__ = ["metrics", "visualization"]
