"""
Hand-crafted texture features for the classical image-processing baseline.

Combines three classic families of descriptors well suited to woven-fabric
texture analysis:

* **LBP**  -- Local Binary Pattern histogram (micro-texture / local structure).
* **GLCM** -- Gray-Level Co-occurrence Matrix Haralick properties
              (contrast, dissimilarity, homogeneity, energy, correlation, ASM).
* **First-order + edge statistics** -- mean, std, skew, kurtosis, entropy and
              Sobel edge density (defects break the regular weave, raising
              local edge energy and intensity variance).

Requires scikit-image and scipy (both pre-installed on Kaggle).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import Config
from .preprocessing import extract_patches


# ----------------------------------------------------------------------------
# Individual descriptor families
# ----------------------------------------------------------------------------
def lbp_histogram(gray: np.ndarray, points: int, radius: int) -> np.ndarray:
    from skimage.feature import local_binary_pattern

    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    n_bins = points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def glcm_features(
    gray: np.ndarray, distances: Tuple[int, ...], angles: Tuple[float, ...]
) -> np.ndarray:
    from skimage.feature import graycomatrix, graycoprops

    # 32 gray levels keeps the GLCM compact and robust on small images.
    levels = 32
    q = (gray.astype(np.uint16) * levels // 256).astype(np.uint8)
    glcm = graycomatrix(
        q, distances=list(distances), angles=list(angles),
        levels=levels, symmetric=True, normed=True,
    )
    props = ("contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "ASM")
    feats = [graycoprops(glcm, p).mean() for p in props]
    return np.asarray(feats, dtype=np.float32)


def gabor_features(gray: np.ndarray) -> np.ndarray:
    """Multi-scale, multi-orientation Gabor-filter energy statistics.

    Woven fabric has a strong oriented, periodic structure; Gabor filters are the
    classic textile-inspection descriptor for it, capturing frequency/orientation
    information that LBP and GLCM miss. Computed on a 64x64 resize for speed
    (energy statistics are scale-robust).
    """
    from skimage.filters import gabor
    from skimage.transform import resize

    g = resize(gray, (64, 64), preserve_range=True,
               anti_aliasing=True).astype(np.float32) / 255.0
    feats = []
    for freq in (0.15, 0.3):
        for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            real, imag = gabor(g, frequency=freq, theta=theta)
            mag = np.sqrt(real * real + imag * imag)
            feats += [float(mag.mean()), float(mag.std())]
    return np.asarray(feats, dtype=np.float32)


def fourier_features(gray: np.ndarray) -> np.ndarray:
    """Frequency-domain (periodicity) statistics from the 2-D FFT.

    A defect-free weave is highly periodic -> sharp spectral peaks; a defect
    disrupts that periodicity, redistributing spectral energy. Captures the
    low/mid/high radial energy split, spectral entropy and peak strength.
    """
    g = gray.astype(np.float32) / 255.0
    g = g - g.mean()
    F = np.fft.fftshift(np.abs(np.fft.fft2(g)))
    F2 = F * F
    total = float(F2.sum()) + 1e-9
    h, w = F.shape
    yy, xx = np.ogrid[:h, :w]
    rad = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    low, mid, high = rad <= 8, (rad > 8) & (rad <= 32), rad > 32
    p = (F2.ravel() / total)
    p = p[p > 0]
    spec_entropy = float(-(p * np.log2(p)).sum())
    fr = F.copy()
    fr[low] = 0
    peak_ratio = float(fr.max() / (fr.mean() + 1e-9))
    return np.asarray([
        F2[low].sum() / total, F2[mid].sum() / total, F2[high].sum() / total,
        spec_entropy / 20.0, peak_ratio / 1000.0,
    ], dtype=np.float32)


def intensity_edge_features(gray: np.ndarray) -> np.ndarray:
    from scipy.stats import skew, kurtosis
    from skimage.filters import sobel

    g = gray.astype(np.float32) / 255.0
    flat = g.ravel()
    # Shannon entropy of the intensity histogram.
    hist, _ = np.histogram(flat, bins=32, range=(0, 1), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())

    # skew/kurtosis are undefined for a constant patch (std == 0); guard them so
    # uniform defect-free regions don't produce NaNs.
    std = float(flat.std())
    sk = float(skew(flat)) if std > 1e-6 else 0.0
    ku = float(kurtosis(flat)) if std > 1e-6 else 0.0

    edges = sobel(g)
    feats = [
        float(flat.mean()),
        std,
        sk,
        ku,
        entropy,
        float(edges.mean()),          # mean edge magnitude
        float(edges.std()),
        float((edges > 0.1).mean()),  # edge density
    ]
    return np.asarray(feats, dtype=np.float32)


# ----------------------------------------------------------------------------
# Combined per-image descriptor
# ----------------------------------------------------------------------------
def image_feature_vector(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """Concatenate LBP + GLCM + intensity/edge features for one image.

    The image is split into patches and the per-patch features are aggregated
    with their mean **and max**.  The max is important: a defect occupies only a
    small fraction of a 4096x256 strip, so the *most anomalous* patch is far
    more informative than the average over the whole (mostly clean) image.
    """
    patches = extract_patches(gray, cfg.patch_size, cfg.patch_stride)
    # Skip blank white-padding patches (AITEX strips have ~1000px of uniform
    # border): including them dilutes the aggregation and measurably hurts
    # accuracy. Fall back to all patches only if every patch is blank.
    kept = [p for p in patches if float(p.array.std()) >= 5.0] or patches
    per_patch = []
    for p in kept:
        vec = np.concatenate([
            lbp_histogram(p.array, cfg.lbp_points, cfg.lbp_radius),
            glcm_features(p.array, cfg.glcm_distances, cfg.glcm_angles),
            intensity_edge_features(p.array),
            gabor_features(p.array),
            fourier_features(p.array),
        ])
        per_patch.append(vec)
    per_patch = np.stack(per_patch, axis=0)
    feats = np.concatenate([per_patch.mean(axis=0), per_patch.max(axis=0)])
    # GLCM correlation is NaN on a zero-variance patch; scrub any residual
    # non-finite values so downstream estimators stay happy.
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_feature_matrix(records, cfg: Config, log_every: int = 25
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the feature matrix ``X`` and label vector ``y`` for ``records``."""
    from .data import load_gray

    X: List[np.ndarray] = []
    y: List[int] = []
    for i, r in enumerate(records):
        X.append(image_feature_vector(load_gray(r.path), cfg))
        y.append(r.label)
        if log_every and (i + 1) % log_every == 0:
            print(f"  features: {i + 1}/{len(records)} images")
    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64)
