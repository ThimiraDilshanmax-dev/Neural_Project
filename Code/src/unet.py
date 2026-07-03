"""
Extended Task 4 -- U-Net for supervised defect localisation.

Trained on the AITEX defective images with the provided binary masks as ground
truth.  Because the strips are 4096x256, the network is trained patch-wise on
256x256 tiles and applied by tiling the full strip at inference.  Localisation
quality is measured with pixel-level **IoU** (and precision/recall/F1), and is
compared head-to-head with the autoencoder anomaly maps (Task 3).

Accuracy-oriented design choices:

* **Pretrained encoder** -- the U-Net encoder is an ImageNet MobileNetV2 (the
  same transfer-learning lever that lifted the classifier); a from-scratch
  variant is kept for comparison.
* **Focal-Tversky loss** (recall-weighted, beta > alpha) + BCE -- defect pixels
  are a tiny fraction of each strip, and plain BCE/Dice under-segments them.
* **Flip augmentation** of the (patch, mask) tiles (x4 the defect-bearing data).
* **Oversampling** patches that actually contain a defect (~1:1 with clean).
* **Threshold calibration** -- the binarisation threshold is chosen to maximise
  mean IoU on the *training* pairs instead of assuming 0.5.

Implemented in TensorFlow / Keras.
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
# Models
# ----------------------------------------------------------------------------
def build_unet_pretrained(input_size: int = 256):
    """U-Net with an ImageNet-pretrained MobileNetV2 encoder (grayscale in)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3), include_top=False,
        weights="imagenet")
    skip_names = ["block_1_expand_relu",   # 1/2
                  "block_3_expand_relu",   # 1/4
                  "block_6_expand_relu",   # 1/8
                  "block_13_expand_relu",  # 1/16
                  "out_relu"]              # 1/32
    encoder = tf.keras.Model(
        backbone.input, [backbone.get_layer(n).output for n in skip_names],
        name="mnv2_encoder")
    encoder.trainable = True                    # fine-tune end-to-end (low LR)

    def up_block(x, skip, filters):
        x = layers.UpSampling2D(2)(x)
        x = layers.concatenate([x, skip])
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    inp = tf.keras.Input((input_size, input_size, 1))
    x = layers.Concatenate()([inp, inp, inp])          # gray -> 3 channels
    x = layers.Rescaling(2.0, offset=-1.0)(x)          # [0,1] -> [-1,1]
    s1, s2, s3, s4, s5 = encoder(x)

    d = up_block(s5, s4, 256)
    d = up_block(d, s3, 128)
    d = up_block(d, s2, 64)
    d = up_block(d, s1, 32)
    d = layers.UpSampling2D(2)(d)
    d = layers.Conv2D(16, 3, padding="same", activation="relu")(d)
    out = layers.Conv2D(1, 1, activation="sigmoid")(d)
    return tf.keras.Model(inp, out, name="unet_mnv2")


def build_unet(input_size: int = 256, base: int = 32):
    """From-scratch U-Net (kept as a lighter/offline fallback)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    def conv_block(x, f):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        return x

    inp = tf.keras.Input((input_size, input_size, 1))
    c1 = conv_block(inp, base);          p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, base * 2);       p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, base * 4);       p3 = layers.MaxPooling2D(2)(c3)
    bn = conv_block(p3, base * 8)

    u3 = layers.UpSampling2D(2)(bn)
    u3 = layers.concatenate([u3, c3]);   c4 = conv_block(u3, base * 4)
    u2 = layers.UpSampling2D(2)(c4)
    u2 = layers.concatenate([u2, c2]);   c5 = conv_block(u2, base * 2)
    u1 = layers.UpSampling2D(2)(c5)
    u1 = layers.concatenate([u1, c1]);   c6 = conv_block(u1, base)

    out = layers.Conv2D(1, 1, activation="sigmoid")(c6)
    return tf.keras.Model(inp, out, name="unet")


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def focal_tversky_loss(alpha: float = 0.3, beta: float = 0.7,
                       gamma: float = 0.75, bce_weight: float = 0.5):
    """Recall-weighted Tversky (beta > alpha penalises false negatives) with a
    focal exponent, plus a small BCE term for stable gradients."""
    import tensorflow as tf

    def loss(y_true, y_pred):
        bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        yt = tf.reshape(y_true, [-1])
        yp = tf.reshape(y_pred, [-1])
        tp = tf.reduce_sum(yt * yp)
        fn = tf.reduce_sum(yt * (1.0 - yp))
        fp = tf.reduce_sum((1.0 - yt) * yp)
        tversky = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
        return tf.pow(1.0 - tversky, gamma) + bce_weight * bce
    return loss


def dice_bce_loss(y_true, y_pred):
    import tensorflow as tf

    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    y_t = tf.reshape(y_true, [-1])
    y_p = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_t * y_p)
    dice = 1.0 - (2.0 * inter + 1.0) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + 1.0)
    return bce + dice


# ----------------------------------------------------------------------------
# Patch dataset from (image, mask) pairs
# ----------------------------------------------------------------------------
def _patches_with_masks(pairs, cfg: Config, neg_per_pos: float = 1.0,
                        seed: int = 42, augment: bool = True,
                        stride: int | None = None, nodefect_paths=None):
    """(image_patch, mask_patch) tiles: oversample defect-bearing tiles and
    optionally add horizontal/vertical flips (x4 augmentation).

    ``stride`` < patch_size gives overlapping tiles, so each defect is seen at
    several horizontal offsets (more, and more varied, positive tiles).
    ``nodefect_paths`` adds pure-negative tiles from defect-free fabrics --
    a more diverse "clean" pool cuts false alarms.
    """
    s = cfg.patch_size
    stride = stride or cfg.train_patch_stride
    rng = np.random.default_rng(seed)
    pos_x, pos_y, neg_x, neg_y = [], [], [], []
    for img_path, mask_path in pairs:
        g = load_gray(img_path).astype("float32") / 255.0
        m = (load_gray(mask_path) > 127).astype("float32")
        for p in extract_patches((g * 255).astype("uint8"), s, stride):
            gp = g[p.y:p.y + s, p.x:p.x + s]
            mp = m[p.y:p.y + s, p.x:p.x + s]
            if gp.shape != (s, s):
                continue
            if mp.sum() > 0:
                pos_x.append(gp); pos_y.append(mp)
            else:
                neg_x.append(gp); neg_y.append(mp)
    # Extra clean tiles from defect-free fabrics (all-zero masks).
    if nodefect_paths:
        zero = np.zeros((s, s), "float32")
        for path in list(nodefect_paths)[::4]:
            g = load_gray(path).astype("float32") / 255.0
            for p in extract_patches((g * 255).astype("uint8"), s, s)[::5]:
                gp = g[p.y:p.y + s, p.x:p.x + s]
                if gp.shape == (s, s):
                    neg_x.append(gp); neg_y.append(zero)
    # keep all positives, subsample negatives
    n_neg = int(len(pos_x) * neg_per_pos) if pos_x else len(neg_x)
    idx = rng.permutation(len(neg_x))[:max(n_neg, 1)]
    X = np.asarray(pos_x + [neg_x[i] for i in idx], "float32")
    Y = np.asarray(pos_y + [neg_y[i] for i in idx], "float32")

    if augment and len(X):
        X = np.concatenate([X, X[:, :, ::-1], X[:, ::-1, :], X[:, ::-1, ::-1]])
        Y = np.concatenate([Y, Y[:, :, ::-1], Y[:, ::-1, :], Y[:, ::-1, ::-1]])

    X = X[..., None]
    Y = Y[..., None]
    perm = rng.permutation(len(X))
    return X[perm], Y[perm]


# ----------------------------------------------------------------------------
# Trained-model wrapper
# ----------------------------------------------------------------------------
@dataclass
class UNetModel:
    cfg: Config
    model: object
    threshold: float = 0.5

    def _predict_tiles(self, batch: np.ndarray, tta: bool = True) -> np.ndarray:
        """Per-tile probability maps, flip-TTA-averaged (flips are undone)."""
        p = self.model.predict(batch, verbose=0)[..., 0]
        if tta:
            for flip in [(slice(None), slice(None, None, -1)),      # horizontal
                         (slice(None, None, -1), slice(None)),      # vertical
                         (slice(None, None, -1), slice(None, None, -1))]:
                v = np.ascontiguousarray(batch[:, flip[0], flip[1]])
                p = p + self.model.predict(v, verbose=0)[..., 0][:, flip[0], flip[1]]
            p = p / 4.0
        return p

    def predict_map(self, gray: np.ndarray, stride: int | None = None,
                    tta: bool = True) -> np.ndarray:
        """Tile the strip (overlapping), predict each tile with flip-TTA, and
        average the overlaps into a smooth full-resolution probability map."""
        s = self.cfg.patch_size
        stride = stride or self.cfg.inference_stride
        patches = extract_patches(gray, s, stride)
        batch = np.stack([
            (p.array if p.array.shape == (s, s) else resize_square(p.array, s))
            for p in patches
        ]).astype("float32")[..., None] / 255.0
        preds = self._predict_tiles(batch, tta=tta)
        heat = np.zeros(gray.shape, np.float32)
        cnt = np.zeros(gray.shape, np.float32)
        for pr, p in zip(preds, patches):
            heat[p.y:p.y + s, p.x:p.x + s] += pr
            cnt[p.y:p.y + s, p.x:p.x + s] += 1.0
        return heat / np.maximum(cnt, 1e-6)


def calibrate_threshold(model: "UNetModel", pairs, thresholds=None) -> float:
    """Pick the binarisation threshold maximising mean IoU over ``pairs``.

    Calibrated on *training* pairs so the held-out evaluation stays honest.
    No min-area filtering here -- hairline defects must survive.
    """
    from .localisation import score_mask

    thresholds = thresholds if thresholds is not None else np.linspace(0.1, 0.9, 17)
    maps = []
    for img_path, mask_path in pairs:
        gray = load_gray(img_path)
        gt = (load_gray(mask_path) > 127).astype(np.uint8)
        maps.append((model.predict_map(gray), gt))
    best_t, best_iou = 0.5, -1.0
    for t in thresholds:
        iou = float(np.mean([score_mask((h >= t).astype(np.uint8), gt).iou
                             for h, gt in maps]))
        if iou > best_iou:
            best_iou, best_t = iou, float(t)
    print(f"[unet] calibrated threshold={best_t:.2f} (train mean IoU={best_iou:.3f})")
    return best_t


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(train_pairs, cfg: Config, epochs: int = 60,
          pretrained: bool = True, calibrate_on: int = 20,
          nodefect_paths=None) -> UNetModel:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(cfg.seed)
    print(f"[unet] building (patch, mask) tiles from {len(train_pairs)} pairs...")
    X, Y = _patches_with_masks(train_pairs, cfg, seed=cfg.seed, augment=True,
                               nodefect_paths=nodefect_paths)
    print(f"[unet] {X.shape[0]} training tiles (oversampled + flip-augmented), "
          f"shape {X.shape[1:]}")

    if pretrained:
        model = build_unet_pretrained(cfg.patch_size)
        lr = 1e-4                       # pretrained encoder -> gentler LR
    else:
        model = build_unet(cfg.patch_size)
        lr = 1e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=focal_tversky_loss())
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=8,
                                            restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5,
                                                patience=4, min_lr=1e-6)]
    model.fit(X, Y, epochs=epochs, batch_size=max(8, cfg.batch_size // 2),
              callbacks=cbs, verbose=2)

    unet = UNetModel(cfg=cfg, model=model)
    # Calibrate the binarisation threshold on (a subset of) the training pairs.
    unet.threshold = calibrate_threshold(unet, train_pairs[:calibrate_on])
    return unet
