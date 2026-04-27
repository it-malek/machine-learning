"""End-to-end CLI entry point for the heart failure project.

Usage::

    python -m projects.heart_failure_prediction.src.train

The script prints a CV summary table and writes ``results/cv_metrics.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running this file directly via ``python train.py`` from inside src/.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.heart_failure_prediction.src.data_loader import load_heart_failure
from projects.heart_failure_prediction.src.evaluation import (
    evaluate_models,
    results_to_dataframe,
    best_model_by,
)
from projects.heart_failure_prediction.src.models import build_model_zoo

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "results" / "cv_metrics.csv"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the metrics CSV.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional override for the heart failure CSV path.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    X, y, summary = load_heart_failure(args.data)
    logger.info("Dataset: %s", summary)

    zoo = build_model_zoo()
    logger.info("Evaluating %d models with %d-fold CV", len(zoo), args.folds)
    results = evaluate_models(zoo, X, y, n_splits=args.folds)

    table = results_to_dataframe(results)
    print("\n=== Cross-validated metrics ===")
    print(table.to_string())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output)
    logger.info("Wrote metrics to %s", args.output)

    winner = best_model_by(results, metric="test_roc_auc")
    logger.info("Best model by ROC-AUC: %s", winner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
