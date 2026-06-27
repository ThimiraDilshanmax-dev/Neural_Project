"""
Central configuration for the Fabric Defect Detection project.

All paths and hyper-parameters live here so that the same code runs both
locally (this repository) and on Kaggle without editing the modules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _find_data_root() -> Path:
    """Locate the folder that contains ``Train/`` (and ``Test/`` when released).

    Resolution order:
      1. ``FABRIC_DATA_ROOT`` environment variable (explicit override).
      2. A Kaggle competition input folder under ``/kaggle/input``.
      3. The repository root (one level above ``Code/``).
    """
    env = os.environ.get("FABRIC_DATA_ROOT")
    if env:
        return Path(env)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for child in sorted(kaggle_input.iterdir()):
            if (child / "Train").exists() or (child / "train").exists():
                return child

    # Repository layout:  <repo>/Code/src/config.py  ->  <repo>
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ---- Paths -----------------------------------------------------------
    data_root: Path = field(default_factory=_find_data_root)
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs"
    )

    # ---- Reproducibility -------------------------------------------------
    seed: int = 42
    val_fraction: float = 0.25  # stratified hold-out for the validation protocol

    # ---- Image / patch geometry -----------------------------------------
    # AITEX images are 4096 x 256 (W x H), single channel.
    image_height: int = 256
    image_width: int = 4096
    patch_size: int = 256          # square patches -> 16 patches per image
    patch_stride: int = 256        # non-overlapping by default

    # Full-image CNN input (wide strip resized to a square the backbone accepts)
    cnn_input_size: int = 224

    # ---- CNN training ----------------------------------------------------
    backbone: str = "MobileNetV2"  # any keras.applications model name
    batch_size: int = 32
    head_epochs: int = 12          # train classifier head with frozen backbone
    finetune_epochs: int = 8       # then unfreeze top layers
    finetune_lr: float = 1e-5
    head_lr: float = 1e-3
    dropout: float = 0.3

    # ---- Classical baseline ---------------------------------------------
    lbp_radius: int = 3
    lbp_points: int = 24           # 8 * radius is a common choice
    glcm_distances: tuple = (1, 3, 5)
    glcm_angles: tuple = (0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345)

    def resolve(self) -> "Config":
        """Resolve the concrete Train/Test sub-folders and create outputs/."""
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    # ---- Derived directory helpers --------------------------------------
    @property
    def train_dir(self) -> Path:
        """Folder containing ``defective/`` and ``defect_free/``.

        The provided archive nests the data as ``Train/Train/<class>``; we
        support both ``Train/<class>`` and ``Train/Train/<class>``.
        """
        candidates = [
            self.data_root / "Train" / "Train",
            self.data_root / "Train",
            self.data_root / "train",
        ]
        for c in candidates:
            if (c / "defective").exists() and (c / "defect_free").exists():
                return c
        # Fall back to the first candidate for a clear error message later.
        return candidates[0]

    @property
    def test_dir(self) -> Path:
        """Folder of unlabelled test images (released in week 12)."""
        for c in [self.data_root / "Test", self.data_root / "test",
                  self.data_root / "Test" / "Test"]:
            if c.exists():
                return c
        return self.data_root / "Test"

    # ---- AITEX archive (extended localisation task) ---------------------
    @property
    def aitex_root(self) -> Path:
        """Folder holding the original AITEX archive (Defect_images, etc.)."""
        for c in [self.data_root / "archive", self.data_root / "AITEX",
                  self.data_root / "aitex"]:
            if (c / "Defect_images").exists():
                return c
        return self.data_root / "archive"

    @property
    def aitex_defect_dir(self) -> Path:
        return self.aitex_root / "Defect_images"

    @property
    def aitex_mask_dir(self) -> Path:
        return self.aitex_root / "Mask_images"

    @property
    def aitex_nodefect_dir(self) -> Path:
        return self.aitex_root / "NODefect_images"

    @property
    def sample_submission(self) -> Path:
        for c in [self.data_root / "sample_submission.csv",
                  self.data_root / "Test" / "sample_submission.csv"]:
            if c.exists():
                return c
        return self.data_root / "sample_submission.csv"


# Class label convention used everywhere in the project.
LABELS = {0: "defect_free", 1: "defective"}
DEFECT_FREE, DEFECTIVE = 0, 1
