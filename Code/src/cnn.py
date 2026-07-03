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
        # Weave is horizontally quasi-periodic: wrap-around translation creates
        # realistic new weave phases (physically-motivated augmentation).
        layers.RandomTranslation(0.0, 0.15, fill_mode="wrap"),
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
def _records_to_patch_arrays(records: List[ImageRecord], cfg: Config,
                             stride: int | None = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Materialise every patch of every record as a uint8 tensor + label.

    ``stride`` < patch_size yields overlapping patches — used for *training*
    only, roughly doubling the sample count on these wide strips.
    """
    stride = stride or cfg.patch_stride
    xs, ys = [], []
    for r in records:
        gray = load_gray(r.path)
        for p in extract_patches(gray, cfg.patch_size, stride):
            if float(p.array.std()) < 5.0:   # skip blank white-padding patches
                continue
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
    def _predict_tta(self, batch: np.ndarray) -> np.ndarray:
        """Flip-averaged (TTA) probabilities for a batch of tensors."""
        variants = [batch]
        if self.cfg.tta:
            variants += [batch[:, :, ::-1], batch[:, ::-1, :], batch[:, ::-1, ::-1]]
        probs = [self.model.predict(np.ascontiguousarray(v), verbose=0).ravel()
                 for v in variants]
        return np.mean(probs, axis=0)

    def score_image(self, gray: np.ndarray) -> float:
        if self.mode == "patch":
            # Overlapping inference windows: a defect straddling a tile border
            # is seen centred by some window instead of split in half. Blank
            # white-padding patches are skipped (never contain a defect).
            patches = extract_patches(gray, self.cfg.patch_size,
                                      self.cfg.inference_stride)
            kept = [p for p in patches if float(p.array.std()) >= 5.0] or patches
            batch = np.stack([prepare_patch_tensor(p.array, self.cfg.cnn_input_size)
                              for p in kept]).astype("float32")
            return float(self._predict_tta(batch).max())   # max-aggregation
        else:
            x = to_rgb(resize_square(gray, self.cfg.cnn_input_size))[None].astype("float32")
            return float(self._predict_tta(x)[0])

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
    if mode == "patch":
        x_tr, y_tr = _records_to_patch_arrays(train_records, cfg,
                                              stride=cfg.train_patch_stride)
    else:
        x_tr, y_tr = materialise(train_records, cfg)
    print(f"[cnn:{mode}] {x_tr.shape[0]} training samples of shape {x_tr.shape[1:]}")
    train_ds = _make_dataset(x_tr, y_tr, cfg.batch_size, shuffle=True)
    cw = _class_weights(y_tr)

    model, base = build_model(cfg, cfg.cnn_input_size)
    # Label smoothing softens the noisy patch labels (clean patches from a
    # defective strip inherit the positive label).
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg.label_smoothing)

    # Phase 1: frozen backbone, train the head.
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.head_lr),
                  loss=loss, metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=4, restore_best_weights=True)
    model.fit(train_ds, epochs=cfg.head_epochs, class_weight=cw,
              callbacks=[es], verbose=2)

    # Phase 2: unfreeze the top of the backbone and fine-tune slowly.
    base.trainable = True
    for layer in base.layers[:-cfg.unfreeze_layers]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.finetune_lr),
                  loss=loss, metrics=["accuracy"])
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


# ----------------------------------------------------------------------------
# Multi-backbone ensemble
# ----------------------------------------------------------------------------
@dataclass
class AveragedCNN:
    """Average of several trained CNNs (different backbones = diverse errors)."""
    members: list
    cfg: Config
    threshold: float = 0.5

    def score_image(self, gray: np.ndarray) -> float:
        return float(np.mean([m.score_image(gray) for m in self.members]))

    def predict_image(self, gray: np.ndarray) -> int:
        return int(self.score_image(gray) >= self.threshold)

    def predict_path(self, path: Path) -> int:
        return self.predict_image(load_gray(path))


def train_backbone_ensemble(train_records, val_records, cfg: Config,
                            backbones=None, mode: str = "patch"):
    """Train one CNN per backbone and average their per-image scores.

    Different architectures make different mistakes; averaging them is the
    strongest remaining small-data lever after TTA.  The averaged score gets its
    own calibrated threshold on the validation split.
    """
    from dataclasses import replace

    backbones = backbones or cfg.ensemble_backbones
    members = []
    for b in backbones:
        print(f"\n[cnn-ensemble] === training backbone: {b} ===")
        member_cfg = replace(cfg, backbone=b)
        member, res = train(train_records, val_records, member_cfg, mode=mode)
        print(f"[cnn-ensemble] {b} alone: acc={res.accuracy:.4f}")
        members.append(member)

    ens = AveragedCNN(members=members, cfg=cfg)
    val_true = np.array([r.label for r in val_records])
    val_scores = np.array([ens.score_image(load_gray(r.path)) for r in val_records])
    ens.threshold = best_threshold(val_true, val_scores)
    result = evaluate(val_true, (val_scores >= ens.threshold).astype(int))
    print(f"[cnn-ensemble] averaged ({'+'.join(backbones)}): "
          f"acc={result.accuracy:.4f} thr={ens.threshold:.3f}")
    return ens, result
