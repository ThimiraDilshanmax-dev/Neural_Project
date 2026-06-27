"""
Extended task: defect *localisation* on the original AITEX defective images.

Two localisation strategies, both producing a per-pixel defect heatmap that is
thresholded into a binary mask and bounding boxes:

1. ``cnn_heatmap``     -- slide the trained patch CNN over the strip with a
                          small stride; each patch's defect probability paints
                          its region.  Reuses the classifier with no extra
                          training (weakly-supervised localisation).
2. ``residual_heatmap`` -- purely classical: a defect breaks the periodic weave,
                          so subtracting a smoothed/median version of the image
                          leaves a bright residual at the defect.  No labels or
                          training required.

When the AITEX binary ground-truth masks are available the predicted masks are
scored with pixel-level IoU / precision / recall / F1.

Note on data: the Kaggle competition ships only image-level labels.  For this
component download the original *AITEX Fabric Image Database*, whose defective
images come with ``*_mask.png`` ground-truth masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import Config
from .data import load_gray
from .preprocessing import extract_patches, prepare_patch_tensor


# ----------------------------------------------------------------------------
# Heatmap generators
# ----------------------------------------------------------------------------
def cnn_heatmap(gray: np.ndarray, cnn_model, cfg: Config,
                stride: Optional[int] = None) -> np.ndarray:
    """Sliding-window CNN defect probability map, same HxW as ``gray``."""
    stride = stride or cfg.patch_size // 2
    heat = np.zeros(gray.shape, dtype=np.float32)
    count = np.zeros(gray.shape, dtype=np.float32)
    patches = extract_patches(gray, cfg.patch_size, stride)
    batch = np.stack([prepare_patch_tensor(p.array, cfg.cnn_input_size)
                      for p in patches]).astype("float32")
    probs = cnn_model.model.predict(batch, verbose=0).ravel()
    ps = cfg.patch_size
    for p, prob in zip(patches, probs):
        heat[p.y:p.y + ps, p.x:p.x + ps] += prob
        count[p.y:p.y + ps, p.x:p.x + ps] += 1.0
    return heat / np.maximum(count, 1e-6)


def residual_heatmap(gray: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Classical anomaly map: |image - median-smoothed image|, normalised 0..1."""
    from scipy.ndimage import median_filter, gaussian_filter

    g = gray.astype(np.float32)
    smooth = median_filter(g, size=(3, ksize))   # weave repeats along the width
    residual = np.abs(g - smooth)
    residual = gaussian_filter(residual, sigma=2)
    mn, mx = residual.min(), residual.max()
    return (residual - mn) / (mx - mn + 1e-6)


# ----------------------------------------------------------------------------
# Heatmap -> mask / boxes
# ----------------------------------------------------------------------------
def heatmap_to_mask(heat: np.ndarray, threshold: float = 0.5,
                    min_area: int = 64, percentile: float | None = None
                    ) -> np.ndarray:
    """Threshold + morphological clean-up + small-blob removal -> binary mask.

    If ``percentile`` is given the cut is taken at that percentile of the
    heatmap (robust for the min-max-normalised classical residual map, where a
    fixed absolute threshold is meaningless).  Otherwise ``threshold`` is used as
    an absolute value (appropriate for the CNN probability map).
    """
    from scipy.ndimage import binary_opening, binary_closing, label

    thr = np.percentile(heat, percentile) if percentile is not None else threshold
    mask = heat >= thr
    mask = binary_closing(mask, structure=np.ones((3, 3)))
    mask = binary_opening(mask, structure=np.ones((3, 3)))

    lbl, n = label(mask)
    out = np.zeros_like(mask)
    for i in range(1, n + 1):
        blob = lbl == i
        if blob.sum() >= min_area:
            out |= blob
    return out.astype(np.uint8)


def mask_to_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Connected-component bounding boxes as ``(x0, y0, x1, y1)``."""
    from scipy.ndimage import label, find_objects

    lbl, n = label(mask)
    boxes = []
    for sl in find_objects(lbl):
        if sl is None:
            continue
        ys, xs = sl
        boxes.append((xs.start, ys.start, xs.stop, ys.stop))
    return boxes


# ----------------------------------------------------------------------------
# Scoring against ground-truth masks
# ----------------------------------------------------------------------------
@dataclass
class LocalisationScore:
    iou: float
    precision: float
    recall: float
    f1: float


def score_mask(pred: np.ndarray, gt: np.ndarray) -> LocalisationScore:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    iou = inter / union if union else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return LocalisationScore(float(iou), float(precision), float(recall), float(f1))


def find_mask_pairs(image_dir: Path, mask_dir: Optional[Path] = None
                    ) -> List[Tuple[Path, Path]]:
    """Match AITEX defective images with their ``*_mask.png`` ground truth."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir) if mask_dir else image_dir
    pairs = []
    for img in sorted(image_dir.glob("*.png")):
        if img.stem.endswith("_mask"):
            continue
        cand = mask_dir / f"{img.stem}_mask.png"
        if cand.exists():
            pairs.append((img, cand))
    return pairs


def evaluate_localisation(pairs, cfg: Config, method: str = "residual",
                          cnn_model=None, threshold: float = 0.5,
                          residual_percentile: float = 99.5
                          ) -> Tuple[List[LocalisationScore], dict]:
    """Run a localisation method over image/mask pairs and aggregate IoU/F1.

    The CNN heatmap is thresholded at the absolute ``threshold`` (a probability);
    the classical residual map is thresholded at ``residual_percentile`` since it
    is only relative within an image.
    """
    scores: List[LocalisationScore] = []
    for img_path, mask_path in pairs:
        gray = load_gray(img_path)
        gt = (load_gray(mask_path) > 127).astype(np.uint8)
        if method == "cnn":
            if cnn_model is None:
                raise ValueError("method='cnn' requires a trained cnn_model")
            heat = cnn_heatmap(gray, cnn_model, cfg)
            pred = heatmap_to_mask(heat, threshold=threshold)
        else:
            heat = residual_heatmap(gray)
            pred = heatmap_to_mask(heat, percentile=residual_percentile)
        scores.append(score_mask(pred, gt))

    agg = {
        "mean_iou": float(np.mean([s.iou for s in scores])) if scores else 0.0,
        "mean_f1": float(np.mean([s.f1 for s in scores])) if scores else 0.0,
        "mean_precision": float(np.mean([s.precision for s in scores])) if scores else 0.0,
        "mean_recall": float(np.mean([s.recall for s in scores])) if scores else 0.0,
        "n": len(scores),
    }
    return scores, agg


def overlay(gray: np.ndarray, mask: np.ndarray, boxes=None):
    """Return an RGB visualisation: defect mask in red, optional boxes."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    alpha = (mask > 0)[:, :, None] * 0.45
    rgb = (rgb * (1 - alpha) + red * alpha)
    return rgb.astype(np.uint8)
