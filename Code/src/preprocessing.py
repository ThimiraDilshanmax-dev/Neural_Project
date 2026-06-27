"""
Image preprocessing: patch extraction and tensor preparation.

AITEX images are very wide (4096 x 256).  Two complementary strategies are
supported throughout the project:

* **Patch-based** -- cut each strip into 256x256 tiles (16 per image).  This
  preserves the native resolution where defects (a few pixels wide) live, and
  lets a square-input CNN backbone work without distortion.
* **Full-image** -- resize the whole strip to a square the backbone accepts.
  Fast, but squashes the horizontal axis ~16x.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


@dataclass
class Patch:
    array: np.ndarray  # HxW uint8 grayscale
    x: int             # left coordinate in the source image
    y: int             # top coordinate in the source image


def extract_patches(
    img: np.ndarray, patch_size: int = 256, stride: int = 256
) -> List[Patch]:
    """Tile ``img`` into square patches.

    The last column/row is anchored to the right/bottom edge so the full image
    is covered even when the dimensions are not an exact multiple of ``stride``.
    """
    h, w = img.shape[:2]
    patch_size = min(patch_size, h, w)

    def positions(extent: int) -> List[int]:
        if extent <= patch_size:
            return [0]
        pos = list(range(0, extent - patch_size + 1, stride))
        if pos[-1] != extent - patch_size:
            pos.append(extent - patch_size)
        return pos

    patches: List[Patch] = []
    for y in positions(h):
        for x in positions(w):
            patches.append(Patch(img[y:y + patch_size, x:x + patch_size], x, y))
    return patches


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Replicate a single grayscale channel into 3 channels (uint8)."""
    if gray.ndim == 3:
        return gray
    return np.repeat(gray[:, :, None], 3, axis=2)


def resize_square(gray: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2-D grayscale image to ``size x size`` (uint8)."""
    im = Image.fromarray(gray).resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def prepare_patch_tensor(patch: np.ndarray, size: int) -> np.ndarray:
    """Grayscale patch -> ``size x size x 3`` uint8 ready for a CNN backbone."""
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = resize_square(patch, size)
    return to_rgb(patch)
