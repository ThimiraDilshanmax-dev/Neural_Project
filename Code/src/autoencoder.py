"""
Extended Task 3 -- Convolutional autoencoder for unsupervised anomaly detection.

The autoencoder is trained on **defect-free images only**.  Having learned to
reconstruct the normal periodic weave, it reconstructs *defects* poorly, so the
per-pixel reconstruction error |x - x_hat| is an **anomaly score** that needs no
labelled defective examples at training time.

The same error map serves two purposes:
* **Detection**  -- aggregate the patch error into an image-level score and
  threshold it (defective vs defect-free).
* **Localisation** -- threshold the error map into a mask and score IoU/F1
  against the AITEX ground-truth masks (compared head-to-head with the U-Net).

Implemented in TensorFlow / Keras.  Operates on 256x256 grayscale patches.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import Config
from .data import load_gray
from .preprocessing import extract_patches, resize_square


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_autoencoder(input_size: int = 256, base: int = 32):
    """A compact symmetric conv autoencoder (grayscale in, grayscale out)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = tf.keras.Input((input_size, input_size, 1))
    # Encoder
    x = layers.Conv2D(base, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(base * 2, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(base * 4, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)                       # bottleneck (H/8)
    # Decoder
    x = layers.Conv2D(base * 4, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(base * 2, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(base, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    out = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inp, out, name="conv_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


# ----------------------------------------------------------------------------
# Patch helpers
# ----------------------------------------------------------------------------
def _img_to_patch_tensor(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """All patches of one image as ``(n, S, S, 1)`` float32 in [0, 1]."""
    s = cfg.patch_size
    patches = extract_patches(gray, s, s)
    arr = np.stack([
        (p.array if p.array.shape == (s, s) else resize_square(p.array, s))
        for p in patches
    ]).astype("float32") / 255.0
    return arr[..., None]


def _build_training_patches(paths: List[Path], cfg: Config) -> np.ndarray:
    xs = [ _img_to_patch_tensor(load_gray(p), cfg) for p in paths ]
    return np.concatenate(xs, axis=0)


# ----------------------------------------------------------------------------
# Trained-model wrapper
# ----------------------------------------------------------------------------
@dataclass
class AutoencoderModel:
    cfg: Config
    model: object

    def anomaly_map(self, gray: np.ndarray, use_ssim: bool = True) -> np.ndarray:
        """Per-pixel anomaly map, same HxW as ``gray`` (0..1).

        By default the error is **1 - SSIM(input, reconstruction)** rather than
        plain |x - x_hat|: SSIM compares local *structure* (windows of texture)
        instead of raw intensities, so it is far less distracted by small
        brightness shifts and localises broken-weave structure much better --
        a standard upgrade for autoencoder anomaly detection.
        """
        from scipy.ndimage import gaussian_filter

        s = self.cfg.patch_size
        # Overlapping windows -> each pixel is reconstructed in several tile
        # contexts and the errors average into a smoother, less blocky map.
        patches = extract_patches(gray, s, self.cfg.inference_stride)
        batch = np.stack([
            (p.array if p.array.shape == (s, s) else resize_square(p.array, s))
            for p in patches
        ]).astype("float32")[..., None] / 255.0
        recon = self.model.predict(batch, verbose=0)

        if use_ssim:
            from skimage.metrics import structural_similarity
            err = np.stack([
                1.0 - structural_similarity(batch[i, ..., 0], recon[i, ..., 0],
                                            data_range=1.0, full=True)[1]
                for i in range(len(batch))
            ])
        else:
            err = np.abs(batch - recon)[..., 0]        # (n, S, S)

        heat = np.zeros(gray.shape, np.float32)
        cnt = np.zeros(gray.shape, np.float32)
        for e, p in zip(err, patches):
            heat[p.y:p.y + s, p.x:p.x + s] += e
            cnt[p.y:p.y + s, p.x:p.x + s] += 1.0
        heat = heat / np.maximum(cnt, 1e-6)
        heat = gaussian_filter(heat, sigma=2)
        mn, mx = float(heat.min()), float(heat.max())
        return (heat - mn) / (mx - mn + 1e-6)

    def image_score(self, gray: np.ndarray) -> float:
        """Image-level anomaly score = mean error of the most anomalous patch."""
        s = self.cfg.patch_size
        patches = extract_patches(gray, s, s)
        batch = np.stack([
            (p.array if p.array.shape == (s, s) else resize_square(p.array, s))
            for p in patches
        ]).astype("float32")[..., None] / 255.0
        recon = self.model.predict(batch, verbose=0)
        per_patch = np.abs(batch - recon).mean(axis=(1, 2, 3))
        return float(per_patch.max())


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(defect_free_paths: List[Path], cfg: Config, epochs: int = 30,
          val_split: float = 0.1) -> AutoencoderModel:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(cfg.seed)
    print(f"[ae] building training patches from {len(defect_free_paths)} "
          f"defect-free images...")
    X = _build_training_patches(defect_free_paths, cfg)
    print(f"[ae] {X.shape[0]} patches of shape {X.shape[1:]}")

    model = build_autoencoder(cfg.patch_size)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X, X, epochs=epochs, batch_size=cfg.batch_size,
              validation_split=val_split, callbacks=[es], verbose=2)
    return AutoencoderModel(cfg=cfg, model=model)
