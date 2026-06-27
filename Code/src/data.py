"""
Dataset discovery, image loading and the reproducible train/validation split.

Only depends on numpy + Pillow so it runs anywhere.  Heavier libraries
(scikit-learn, TensorFlow, scikit-image) are imported lazily inside the
modules that actually need them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from .config import Config, DEFECTIVE, DEFECT_FREE


@dataclass
class ImageRecord:
    path: Path
    label: int          # 0 = defect_free, 1 = defective
    image_id: str       # original file name (e.g. defective_0001.png)


# ----------------------------------------------------------------------------
# Listing
# ----------------------------------------------------------------------------
def list_training_records(cfg: Config) -> List[ImageRecord]:
    """Return every labelled training image as an :class:`ImageRecord`."""
    train_dir = cfg.train_dir
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found. Looked for defective/defect_free under "
            f"{train_dir}. Set FABRIC_DATA_ROOT or check the dataset layout."
        )

    records: List[ImageRecord] = []
    for sub, label in (("defective", DEFECTIVE), ("defect_free", DEFECT_FREE)):
        folder = train_dir / sub
        for p in sorted(folder.glob("*.png")):
            records.append(ImageRecord(path=p, label=label, image_id=p.name))
    if not records:
        raise FileNotFoundError(f"No .png images found under {train_dir}.")
    return records


def list_test_records(cfg: Config) -> List[ImageRecord]:
    """Return the unlabelled test images (label set to -1).

    Falls back to the IDs in ``sample_submission.csv`` so the pipeline can be
    wired up before the Test/ folder is released.
    """
    test_dir = cfg.test_dir
    records: List[ImageRecord] = []
    if test_dir.exists():
        for p in sorted(test_dir.glob("*.png")):
            records.append(ImageRecord(path=p, label=-1, image_id=p.name))
    if records:
        return records

    # Test images not released yet -> use the sample submission IDs.
    if cfg.sample_submission.exists():
        import csv
        with open(cfg.sample_submission, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = (row.get("ID") or row.get("id") or "").strip().strip('"')
                if name:
                    records.append(
                        ImageRecord(path=test_dir / name, label=-1, image_id=name)
                    )
    return records


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_gray(path: Path) -> np.ndarray:
    """Load an image as a 2-D ``uint8`` grayscale array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    """Load an image as ``HxWx3`` uint8; optionally resize to ``size x size``."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


# ----------------------------------------------------------------------------
# Reproducible stratified split
# ----------------------------------------------------------------------------
def stratified_split(
    records: List[ImageRecord], val_fraction: float, seed: int
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Deterministic stratified train/validation split (no sklearn needed)."""
    rng = np.random.default_rng(seed)
    train: List[ImageRecord] = []
    val: List[ImageRecord] = []
    for label in (DEFECT_FREE, DEFECTIVE):
        group = [r for r in records if r.label == label]
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(group) * val_fraction)))
        val_idx = set(idx[:n_val].tolist())
        for i, r in enumerate(group):
            (val if i in val_idx else train).append(r)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def labels_array(records: List[ImageRecord]) -> np.ndarray:
    return np.array([r.label for r in records], dtype=np.int64)


def class_distribution(records: List[ImageRecord]) -> dict:
    arr = labels_array(records)
    return {
        "defect_free": int((arr == DEFECT_FREE).sum()),
        "defective": int((arr == DEFECTIVE).sum()),
        "total": int(arr.size),
    }
