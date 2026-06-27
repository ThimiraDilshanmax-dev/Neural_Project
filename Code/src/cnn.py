"""
CNN classifier with transfer learning, data augmentation and patch / full-image
training strategies, implemented in TensorFlow / Keras.

Why these choices for a *small, wide-image* dataset (173 strips of 4096x256):

* **Transfer learning** from an ImageNet backbone (MobileNetV2 by default) gives
  a strong feature extractor without needing many images.
* **Data augmentation** (flips, small rotation/zoom/contrast) multiplies the
  effective dataset and combats over-fitting.
* **Patch-based training** turns every 4096x256 strip into 16 square 256x256
  tiles, giving ~16x more samples at native resolution.  At inference the patch
  probabilities are aggregated per image with a *max* rule: a strip is defective
  if its most-suspicious patch is defective.  (Patches inherit the image label,
  so positive patches are noisy -- the max rule is what makes this work.)
* **Two-phase schedule** -- train the head with the backbone frozen, then
  fine-tune the top backbone layers at a low learning rate.

The public API mirrors :class:`classical.ClassicalModel` (``predict_path`` etc.)
so both models are interchangeable in :mod:`predict` and :mod:`evaluate`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import Config
from .data import ImageRecord, load_gray
from .evaluate import EvalResult, evaluate, best_threshold
from .preprocessing import extract_patches, prepare_patch_tensor, resize_square, to_rgb


# ----------------------------------------------------------------------------
# Backbone / preprocessing registry
# ----------------------------------------------------------------------------
def _backbone(cfg: Config, input_shape):
    import tensorflow as tf

    name = cfg.backbone
    app = getattr(tf.keras.applications, name)
    preprocess = getattr(
        tf.keras.applications,
        {"MobileNetV2": "mobilenet_v2",
         "EfficientNetB0": "efficientnet",
         "ResNet50": "resnet50",
         "DenseNet121": "densenet"}.get(name, "mobilenet_v2"),
    ).preprocess_input
    base = app(include_top=False, weights="imagenet", input_shape=input_shape,
               pooling="avg")
    return base, preprocess


def build_model(cfg: Config, input_size: int):
    """Assemble augmentation + frozen backbone + classifier head."""
    import tensorflow as tf
    from tensorflow.keras import layers

    input_shape = (input_size, input_size, 3)
    base, preprocess = _backbone(cfg, input_shape)
    base.trainable = False

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="augment")

    inputs = tf.keras.Input(shape=input_shape)
    x = augment(inputs)
    x = layers.Lambda(preprocess, name="preprocess")(x)
    x = base(x, training=False)
    x = layers.Dropout(cfg.dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{cfg.backbone}_fabric")
    return model, base


# ----------------------------------------------------------------------------
# Building tf.data pipelines
# ----------------------------------------------------------------------------
def _records_to_patch_arrays(records: List[ImageRecord], cfg: Config
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Materialise every patch of every record as a uint8 tensor + label."""
    xs, ys = [], []
    for r in records:
        gray = load_gray(r.path)
        for p in extract_patches(gray, cfg.patch_size, cfg.patch_stride):
            xs.append(prepare_patch_tensor(p.array, cfg.cnn_input_size))
            ys.append(r.label)
    return np.asarray(xs, dtype=np.uint8), np.asarray(ys, dtype=np.float32)


def _records_to_full_arrays(records: List[ImageRecord], cfg: Config
                            ) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in records:
        gray = load_gray(r.path)
        xs.append(to_rgb(resize_square(gray, cfg.cnn_input_size)))
        ys.append(r.label)
    return np.asarray(xs, dtype=np.uint8), np.asarray(ys, dtype=np.float32)


def _make_dataset(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 2048), reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------
# Trained-model wrapper
# ----------------------------------------------------------------------------
@dataclass
class CNNModel:
    cfg: Config
    model: object
    mode: str            # "patch" or "full"
    threshold: float = 0.5

    # ---- per-image scoring ---------------------------------------------
    def score_image(self, gray: np.ndarray) -> float:
        if self.mode == "patch":
            patches = extract_patches(gray, self.cfg.patch_size, self.cfg.patch_stride)
            batch = np.stack([prepare_patch_tensor(p.array, self.cfg.cnn_input_size)
                              for p in patches]).astype("float32")
            probs = self.model.predict(batch, verbose=0).ravel()
            return float(probs.max())          # max-aggregation
        else:
            x = to_rgb(resize_square(gray, self.cfg.cnn_input_size))[None].astype("float32")
            return float(self.model.predict(x, verbose=0).ravel()[0])

    def predict_image(self, gray: np.ndarray) -> int:
        return int(self.score_image(gray) >= self.threshold)

    def predict_path(self, path: Path) -> int:
        return self.predict_image(load_gray(path))

    def score_paths(self, paths) -> np.ndarray:
        return np.array([self.score_image(load_gray(p)) for p in paths])


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _class_weights(y: np.ndarray) -> dict:
    n = len(y)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return {0: n / (2 * neg), 1: n / (2 * pos)}


def train(
    train_records: List[ImageRecord],
    val_records: List[ImageRecord],
    cfg: Config,
    mode: str = "patch",
) -> Tuple[CNNModel, EvalResult]:
    """Two-phase transfer-learning training; returns the model + val metrics."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(cfg.seed)
    materialise = _records_to_patch_arrays if mode == "patch" else _records_to_full_arrays

    print(f"[cnn:{mode}] preparing training tensors...")
    x_tr, y_tr = materialise(train_records, cfg)
    print(f"[cnn:{mode}] {x_tr.shape[0]} training samples of shape {x_tr.shape[1:]}")
    train_ds = _make_dataset(x_tr, y_tr, cfg.batch_size, shuffle=True)
    cw = _class_weights(y_tr)

    model, base = build_model(cfg, cfg.cnn_input_size)

    # Phase 1: frozen backbone, train the head.
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.head_lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=4, restore_best_weights=True)
    model.fit(train_ds, epochs=cfg.head_epochs, class_weight=cw,
              callbacks=[es], verbose=2)

    # Phase 2: unfreeze the top of the backbone and fine-tune slowly.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.finetune_lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=cfg.finetune_epochs, class_weight=cw,
              callbacks=[es], verbose=2)

    cnn = CNNModel(cfg=cfg, model=model, mode=mode)

    # Calibrate the decision threshold on the validation split (image level).
    val_scores = np.array([cnn.score_image(load_gray(r.path)) for r in val_records])
    val_true = np.array([r.label for r in val_records])
    cnn.threshold = best_threshold(val_true, val_scores)
    result = evaluate(val_true, (val_scores >= cnn.threshold).astype(int))
    print(f"[cnn:{mode}] calibrated threshold = {cnn.threshold:.3f}")
    return cnn, result
