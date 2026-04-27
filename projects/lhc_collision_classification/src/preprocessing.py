"""Image preprocessing and HOG feature extraction for collision images."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from skimage import color, io, transform
from skimage.feature import hog

logger = logging.getLogger(__name__)


@dataclass
class HogParams:
    """Histogram of Oriented Gradients hyperparameters.

    The defaults mirror the original course notebook (8 orientations,
    32x32 pixel cells, 1x1 cell blocks) so the rebuilt pipeline gives
    comparable feature counts. Tweak these to trade dimensionality for
    spatial detail.
    """

    orientations: int = 8
    pixels_per_cell: tuple[int, int] = (32, 32)
    cells_per_block: tuple[int, int] = (1, 1)
    transform_sqrt: bool = False
    feature_vector: bool = True


DEFAULT_IMAGE_SIZE: tuple[int, int] = (256, 256)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Cast to float32 in ``[0, 1]`` and force grayscale to a single channel.

    Args:
        image: Image array as returned by :func:`skimage.io.imread`.
            Either ``(H, W)`` for grayscale or ``(H, W, 3)`` for RGB.

    Returns:
        ``float32`` array in ``[0, 1]`` with shape ``(H, W)``.
    """
    if image.ndim == 3:
        image = color.rgb2gray(image)
    image = image.astype(np.float32, copy=False)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def load_and_resize(
    path: str, size: tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Read an image from disk and resize it to ``size``.

    The resize uses ``anti_aliasing=True`` and preserves range, so we
    can keep a single normalisation step downstream.
    """
    image = io.imread(path)
    image = normalize_image(image)
    if image.shape != size:
        image = transform.resize(
            image, size, anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
    return image


def extract_hog(
    image: np.ndarray, params: HogParams | None = None
) -> np.ndarray:
    """Compute a 1-D HOG descriptor for a single grayscale image."""
    params = params or HogParams()
    return hog(
        image,
        orientations=params.orientations,
        pixels_per_cell=params.pixels_per_cell,
        cells_per_block=params.cells_per_block,
        transform_sqrt=params.transform_sqrt,
        feature_vector=params.feature_vector,
    ).astype(np.float32)


def build_feature_matrix(
    filepaths: np.ndarray,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    hog_params: HogParams | None = None,
    log_every: int = 500,
) -> np.ndarray:
    """Stream images from disk, compute HOG features, stack them.

    This intentionally avoids loading every image into memory at once;
    only the final ``(N, D)`` feature matrix is retained.

    Args:
        filepaths: Iterable of paths.
        image_size: Resize target.
        hog_params: HOG configuration. Defaults to :class:`HogParams`.
        log_every: Emit a progress log line every N images.

    Returns:
        ``float32`` array of shape ``(len(filepaths), D)``.
    """
    hog_params = hog_params or HogParams()
    features: list[np.ndarray] = []
    for i, path in enumerate(filepaths):
        image = load_and_resize(path, size=image_size)
        features.append(extract_hog(image, hog_params))
        if log_every and (i + 1) % log_every == 0:
            logger.info("Extracted HOG features: %d / %d", i + 1, len(filepaths))
    return np.stack(features, axis=0)
