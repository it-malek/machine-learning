"""Dataset walker for the LHC collision image set.

Expected on-disk layout::

    <root>/
        QCD/   image_001.png  image_002.png  ...
        TTbar/ image_001.png  ...
        WJets/ image_001.png  ...

The dataset is large (~30k 372x1196 PNGs) and not bundled with the
repo; download it from Kaggle:

    https://www.kaggle.com/datasets/farhanhubble/multimnistab/...
    (link in the project README)

This loader is deliberately streaming-friendly so we don't blow up RAM:
it returns lists of file paths and integer labels rather than the
decoded pixel arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


CLASS_NAMES: tuple[str, ...] = ("QCD", "TTbar", "WJets")


@dataclass
class CollisionDataset:
    """File-path index for one split (train, test, ...) of the dataset.

    Attributes:
        filepaths: Absolute path to every image, in deterministic order.
        labels: Integer class label per image, indexing :data:`CLASS_NAMES`.
        class_names: The class names corresponding to label indices.
        class_sizes: Number of images per class (same order as
            :attr:`class_names`).
    """

    filepaths: np.ndarray  # shape (N,), dtype object (str)
    labels: np.ndarray  # shape (N,), dtype int64
    class_names: tuple[str, ...] = field(default=CLASS_NAMES)
    class_sizes: np.ndarray = field(
        default_factory=lambda: np.zeros(len(CLASS_NAMES), dtype=np.int64)
    )

    def __len__(self) -> int:
        return len(self.filepaths)


def load_dataset(
    root: str | Path,
    class_names: tuple[str, ...] = CLASS_NAMES,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> CollisionDataset:
    """Walk the class folders under ``root`` and return file paths + labels.

    Args:
        root: Directory containing one subfolder per class.
        class_names: Tuple of expected class names. Subfolders are
            looked up in this exact order; missing folders raise.
        extensions: Image file extensions to include (lower case).

    Returns:
        A :class:`CollisionDataset` describing the split.

    Raises:
        FileNotFoundError: If ``root`` doesn't exist.
        ValueError: If any of the expected class folders is missing.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root '{root}' does not exist. See the project README "
            "for instructions on downloading the data."
        )

    paths: list[str] = []
    labels: list[int] = []
    sizes = np.zeros(len(class_names), dtype=np.int64)

    for label_idx, name in enumerate(class_names):
        class_dir = root / name
        if not class_dir.is_dir():
            raise ValueError(
                f"Expected class folder '{class_dir}' is missing."
            )
        files = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in extensions
        )
        for p in files:
            paths.append(str(p))
            labels.append(label_idx)
        sizes[label_idx] = len(files)
        logger.info("Class '%s': %d images", name, len(files))

    dataset = CollisionDataset(
        filepaths=np.array(paths, dtype=object),
        labels=np.array(labels, dtype=np.int64),
        class_names=tuple(class_names),
        class_sizes=sizes,
    )
    logger.info("Loaded %d images across %d classes", len(dataset), len(class_names))
    return dataset
