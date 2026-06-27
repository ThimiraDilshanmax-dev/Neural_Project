# Fabric Defect Detection and Localisation Using Deep Neural Networks

**Module:** CO5420 — Project 3
**Team:** _[names + student IDs]_
**Date:** _[date]_
**Kaggle team / leaderboard name:** _[name]_  ·  **Public score:** _[acc]_  ·  **Private score:** _[acc]_

> Placeholders are written as _[…]_. Numbers come straight from the pipeline:
> `run_classical.py`, `run_cnn.py` and the Kaggle notebook print every metric,
> and the figures are saved under `outputs/`.

---

## Abstract
_[3–5 sentences. State the problem (automating AITEX fabric inspection), what you
built (a classical LBP/GLCM+SVM baseline vs a MobileNetV2 transfer-learning CNN),
the headline validation accuracy of each, which won, and the localisation result.
Fill once the numbers below are final.]_

## 1. Introduction
Quality control in textile manufacturing relies heavily on manual visual
inspection, which is slow, subjective and affected by operator fatigue. This
project automates the task on the **AITEX Fabric Image Database**: classifying
4096×256 grayscale fabric strips as **defective** (1) or **defect-free** (0), and
— as an extended task — **localising** the defect region at the pixel level.

We develop and compare two approaches under an identical validation protocol:
1. a **classical image-processing baseline** (hand-crafted texture features + SVM);
2. a **convolutional neural network** using transfer learning.

Because the dataset is small (173 labelled images) and the images are extremely
wide, we investigate **data augmentation**, **transfer learning** and
**patch-based training**.

## 2. Related Work
- **Silvestre-Blanes et al. (2019), _AFID: a public fabric image database for
  defect detection_, AUTEX Research Journal.** Introduces the AITEX/AFID
  database used here (7 fabrics, 9 defect categories, pixel masks) and a
  reference inspection pipeline. _[1–2 sentences on their method/result.]_
- _[Second paper — choose one and summarise in 2–3 sentences. Suggested options:
  a CNN/transfer-learning fabric-defect paper, or an LBP/GLCM texture-classification
  paper. State what they did and how it informs your design.]_

_[Optional: 1–2 sentences positioning your work relative to these — e.g. you
combine LBP+GLCM with a max-over-patches aggregation, and use patch-based
transfer learning to cope with the tiny, very wide images.]_

## 3. Dataset
The competition data derives from AITEX: 247 RGB strips (4096×256) from seven
fabric types, 106 defective / 141 defect-free, with a reproducible stratified
70/30 split into a **173-image labelled training set** and a **74-image hidden
test set**.

| Split | Source | Defective (1) | Defect-free (0) | Total |
|-------|--------|---------------|------------------|-------|
| Train (labelled) | `Train/Train/` | 74 | 99 | 173 |
| Validation (hold-out, ours) | seeded 25% of train | 18 | 25 | 43 |
| Test (hidden) | `Test/` (week 12) | — | — | 74 |

For the localisation task we use the original AITEX archive (`archive/`):
`Defect_images/` (106), `Mask_images/` (107 ground-truth masks), and
`NODefect_images/` (141). 103 image/mask pairs are matched for evaluation.

**Class imbalance** (74:99) is handled with `class_weight="balanced"` (classical)
and inverse-frequency class weights (CNN).

## 4. Methodology

### 4.1 Validation protocol
A single **seeded (42) stratified 75/25 split** of the 173 training images is
reused by *both* models, so the comparison is fair and reproducible. All metrics
below are on this held-out validation set; the test-set labels are hidden and
scored by Kaggle.

### 4.2 Classical baseline
Each 4096×256 strip is tiled into sixteen 256×256 patches. Per patch we extract:
- **LBP** — uniform Local Binary Pattern histogram (P=24, R=3): micro-texture.
- **GLCM** — gray-level co-occurrence Haralick properties (contrast,
  dissimilarity, homogeneity, energy, correlation, ASM) over distances {1,3,5}
  and 4 angles, on 32 quantised levels.
- **First-order + edge** — mean, std, skew, kurtosis, intensity entropy, and
  Sobel edge mean/std/density.

Patch features are aggregated across the strip by **mean and max**. The *max* is
key: a defect occupies only a small slice of a 4096-px-wide image, so the single
most-anomalous patch carries the signal. The resulting **80-dimensional** vector
feeds a `StandardScaler` + **SVM (RBF, C=10, balanced)** pipeline. (A
RandomForest variant is also available via `--classifier rf`.)

### 4.3 CNN (transfer learning)
- **Backbone:** ImageNet-pretrained **MobileNetV2** (`include_top=False`,
  global-average pooling) + dropout(0.3) + sigmoid head.
- **Augmentation:** random horizontal/vertical flips, ±5% rotation, ±10% zoom,
  ±10% contrast (applied on-the-fly during training).
- **Patch-based training** _(primary)_: every strip → 16 patches resized to
  224×224; patches inherit the image label. At inference the image score is the
  **max** patch probability and the decision threshold is **calibrated on the
  validation split**. A **full-image** mode (whole strip resized to 224×224) is
  provided for comparison (`--mode full`).
- **Two-phase schedule:** (1) train the head with the backbone frozen
  (lr=1e-3, ≤12 epochs); (2) unfreeze the top 30 backbone layers and fine-tune
  (lr=1e-5, ≤8 epochs). Early stopping on training loss.

### 4.4 Localisation (extended task)
Two heatmap methods produce a per-pixel defect map, thresholded + morphologically
cleaned into a binary mask and bounding boxes:
- **Classical residual** — `|image − median-smoothed image|`; a defect breaks the
  periodic weave and stands out. Thresholded at the 99.5th percentile.
  *Unsupervised, no training.*
- **CNN sliding window** — slide the trained patch classifier with a 128-px
  stride and paint each region with its defect probability. *Weakly supervised.*

Predicted masks are scored against AITEX `*_mask.png` ground truth with
**pixel-level IoU, precision, recall and F1**.

## 5. Results

### 5.1 Classification — validation set
| Model | Accuracy | Macro-F1 | Precision (defective) | Recall (defective) |
|-------|----------|----------|------------------------|---------------------|
| Classical (SVM, LBP+GLCM+edge) — hold-out | 0.698 | 0.672 | 0.692 | 0.500 |
| Classical (SVM) — **5-fold CV (all data)** | **0.832 ± 0.049** | **0.827 ± 0.051** | — | — |
| CNN (patch, MobileNetV2) | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ |
| CNN (full-image) | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ |

> Classical numbers above are **real runs** (`run_classical.py --classifier svm --cv`).
> The single 43-image hold-out is noisy; the 5-fold CV is the reliable estimate.
> CNN numbers: fill from `run_cnn.py --mode patch` / `--mode full` (or the notebook on Kaggle).

**Confusion matrices** (saved to `outputs/`):
- Classical: `outputs/confusion_classical_svm.png`
- CNN: `outputs/confusion_cnn_patch.png`

_[Embed both figures here.]_

### 5.2 Per-class analysis
On the hold-out split the classical SVM achieves balanced precision (0.70
defect-free, 0.69 defective) but **asymmetric recall: 0.84 defect-free vs 0.50
defective** — i.e. it misses half the defective strips (9 of 18 defective images
fell into the `pred_free` column of the confusion matrix). For quality control
this **false-negative** behaviour is the costly one: a missed defect ships a
faulty roll. _[Compare against the CNN's per-class recall once available — the
patch + max-aggregation CNN is expected to improve defective recall, since it
only needs one suspicious patch to flag a strip.]_

### 5.3 Classical vs CNN
_[2–4 sentences. Which won on accuracy / macro-F1? Did transfer learning +
patches beat hand-crafted features, and by how much? Was the gain worth the extra
compute given only 173 images? Reference the comparison bar chart from the
notebook.]_

### 5.4 Localisation (extended task)
| Method | Mean IoU | Mean F1 | Mean Precision | Mean Recall | #images |
|--------|----------|---------|----------------|-------------|---------|
| Classical residual (pct 99.5) | 0.015 | 0.026 | 0.026 | 0.070 | 103 |
| CNN sliding window | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ | 103 |

> Classical row is a **real run** (`run_localisation.py --method residual`, full
> 103 AITEX pairs). CNN row: fill after `--method cnn --model outputs/cnn_patch.keras`.
> Example overlays in `outputs/localisation/`. _[Embed 2–3 overlay images.]_

The classical residual localiser is **weak across the full defect set** (mean IoU
0.015). It performs noticeably better on large, high-contrast categories (e.g.
"broken end" subset ≈ 0.09 IoU) but poorly on small/subtle defects, and a single
global percentile cannot adapt to AITEX's very different defect sizes across the
9 categories. This motivates the learned alternatives: the **CNN sliding-window
heatmap** localises more precisely, and a fully supervised **U-Net trained on the
masks** would be the natural next step (see Future work). The key limitation
throughout is that patch labels are only image-level (weak supervision).

## 6. Discussion
- **Small-data strategies that helped:** _[transfer learning, augmentation,
  patch-based training (16× more samples), class weighting, threshold
  calibration — say which mattered most in your runs.]_
- **Limitations:** image-level labels for patches introduce noise (a defective
  strip's clean patches are labelled positive); the max-aggregation rule
  mitigates this. Very wide aspect ratio; subtle low-contrast defects. Tiny test
  set (74 images) → leaderboard variance.
- **Future work:** _[multiple-instance learning for patches; segmentation
  network (U-Net) trained on the AITEX masks for supervised localisation;
  anomaly-detection / autoencoder approaches; per-defect-type analysis.]_

## 7. Conclusion
_[3–4 sentences: restate what was built, the winning model and its validation
accuracy, the localisation outcome, and the key takeaway about classical vs deep
approaches on this small dataset.]_

## 8. Reproducibility
All randomness is seeded (`Config.seed = 42`): the stratified split, NumPy and
TensorFlow. The same split drives both models. Full instructions in `README.md`.

```bash
cd Code
pip install -r requirements.txt
python run_classical.py --classifier svm --cv
python run_cnn.py --mode patch
python run_predict.py --model cnn --mode patch        # -> outputs/submission.csv
python run_localisation.py --method residual
```
The Kaggle notebook `notebooks/fabric_defect_detection_kaggle.ipynb` reproduces
the entire workflow start-to-finish and generates the submission.

## References
1. J. Silvestre-Blanes, T. Albero-Albero, I. Miralles, R. Pérez-Llorens, J.
   Moreno. *AFID: a public fabric image database for defect detection.* AUTEX
   Research Journal, No. 4, 2019.
2. _[Second reference — full citation.]_
3. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen. *MobileNetV2:
   Inverted Residuals and Linear Bottlenecks.* CVPR, 2018.

## Appendix — AI tool usage (Level 3: AI Collaboration)
_[Per the brief: briefly note where AI assistance was used — idea generation,
coding, debugging, refinement — and confirm the team checked all outputs and
understands the submitted work.]_
