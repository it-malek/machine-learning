"""End-to-end CLI entry point for the LHC collision benchmark.

Usage::

    python -m projects.lhc_collision_classification.src.train \
        --train-dir path/to/Train \
        --test-dir  path/to/Test

Because the dataset (~30k images) is not bundled with the repository,
the script will fail fast with a clear error if the directories don't
exist. See the project README for download instructions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.lhc_collision_classification.src.data_loader import (
    CLASS_NAMES,
    load_dataset,
)
from projects.lhc_collision_classification.src.evaluation import (
    best_model_by,
    evaluate_models,
    results_to_dataframe,
)
from projects.lhc_collision_classification.src.models import build_model_zoo
from projects.lhc_collision_classification.src.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    HogParams,
    build_feature_matrix,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Directory containing one subfolder per class (training split).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional held-out test directory; if omitted, only CV is reported.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE[0],
        help=f"Square image side; default {DEFAULT_IMAGE_SIZE[0]}.",
    )
    parser.add_argument(
        "--with-pca",
        action="store_true",
        help="Insert a 100-component PCA between scaler and model.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Subsample to at most N images per class (debug aid).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    train_data = load_dataset(args.train_dir, class_names=CLASS_NAMES)

    if args.max_per_class is not None:
        keep = []
        for cls in range(len(CLASS_NAMES)):
            idx = (train_data.labels == cls).nonzero()[0][: args.max_per_class]
            keep.append(idx)
        keep_idx = sorted(int(i) for arr in keep for i in arr)
        train_data.filepaths = train_data.filepaths[keep_idx]
        train_data.labels = train_data.labels[keep_idx]
        logger.info("Subsampled to %d images total.", len(train_data))

    img_size = (args.image_size, args.image_size)
    logger.info("Extracting HOG features for %d training images at %s", len(train_data), img_size)
    X_train = build_feature_matrix(train_data.filepaths, image_size=img_size, hog_params=HogParams())
    y_train = train_data.labels

    zoo = build_model_zoo(with_pca=args.with_pca)
    logger.info("Evaluating %d models with %d-fold CV", len(zoo), args.folds)
    results = evaluate_models(zoo, X_train, y_train, n_splits=args.folds)

    table = results_to_dataframe(results)
    print("\n=== Cross-validated metrics ===")
    print(table.to_string())

    winner = best_model_by(results, metric="test_f1_macro")
    logger.info("Best model by macro-F1: %s", winner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
