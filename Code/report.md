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
We automate defect inspection on the AITEX fabric database across four models,
evaluated by **3-fold out-of-fold cross-validation over all 173 images** for
trustworthy estimates. For binary classification, an **engineered classical model
reaches 0.861 OOF accuracy** (0.839 ± 0.014 across six fold seeds) — combining
LBP + GLCM + edge + **Gabor + Fourier** features, **blank-border-patch removal**,
and an **SVM/RF/GB voting ensemble**, each verified to raise accuracy. Under this
honest protocol it clearly beats the 3-fold-bagged MobileNetV2 patch CNN (0.688),
which nonetheless offers the best **defective recall (0.878)**, the industrially
critical metric. (A second backbone, EfficientNetB0, was *removed* after OOF
measurement showed it hurt — measurement-driven iteration.) For the extended
localisation task, a supervised U-Net with a pretrained encoder and Focal-Tversky
loss was engineered from **pixel IoU 0.016 up to 0.254**, finding **62% of defects
at 60% precision (region F1 0.610)**, decisively beating the unsupervised SSIM
autoencoder. The headline lesson: on small datasets, **understanding the data and
using domain-appropriate features beat added model capacity** — and disciplined
OOF measurement is what separates real gains from noise.

## 1. Introduction
Quality control in textile manufacturing relies heavily on manual visual
inspection, which is slow, subjective and affected by operator fatigue. This
project automates the task on the **AITEX Fabric Image Database**: classifying
4096×256 grayscale fabric strips as **defective** (1) or **defect-free** (0), and
— as an extended task — **localising** the defect region at the pixel level.

The project has four parts:
1. **Classical baseline** — hand-crafted texture features (LBP, GLCM, edges) + SVM
   for defective/defect-free classification (Kaggle task).
2. **CNN classifier** — transfer learning, compared against the classical baseline
   under an identical validation protocol (Kaggle task).
3. **Autoencoder** — trained on defect-free images only; reconstruction error as an
   unsupervised anomaly score (extended task).
4. **U-Net** — supervised segmentation trained on the binary masks to localise
   defects, evaluated by IoU and compared against the autoencoder (extended task).

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
**Stratified 3-fold cross-validation with out-of-fold (OOF) evaluation** (seed
42): every image is predicted exactly once by models that never saw it, so all
classification metrics are computed on **all 173 images** rather than a noisy
43-image hold-out (where each image is worth ±2.3 accuracy points). The classical
model and the CNNs use the *same folds*, so the comparison is fair. At test time
the CNN prediction averages **all fold models** (bagging), so the deployed
committee has collectively learned from every labelled image. The test-set labels
are hidden and scored by Kaggle.

### 4.2 Classical baseline
Each 4096×256 strip is tiled into 256×256 patches. **Blank white-padding patches
(std < 5) are skipped** — AITEX strips carry ~1000px of uniform border that
otherwise dilutes the aggregation. Per remaining patch we extract four
complementary descriptor families:
- **LBP** — uniform Local Binary Pattern histogram (P=24, R=3): micro-texture.
- **GLCM** — gray-level co-occurrence Haralick properties (contrast,
  dissimilarity, homogeneity, energy, correlation, ASM) over distances {1,3,5}
  and 4 angles, on 32 quantised levels.
- **First-order + edge** — mean, std, skew, kurtosis, intensity entropy, and
  Sobel edge mean/std/density.
- **Gabor + Fourier** — Gabor filter energy (8 filters: 2 frequencies × 4
  orientations) and FFT periodicity statistics (radial low/mid/high energy split,
  spectral entropy, peak strength). Fabric is oriented and periodic; a defect
  disrupts that spectrum in ways LBP/GLCM do not capture.

Patch features are aggregated across the kept patches by **mean and max** (the
*max* captures the single most-anomalous patch). The resulting **122-dimensional**
vector feeds a `StandardScaler` + **soft-voting ensemble of SVM (RBF) +
RandomForest + GradientBoosting**. Every one of these choices (blank-skip,
Gabor/Fourier, voting) was **verified to raise 3-fold OOF accuracy** before
adoption (§5.0); single-classifier variants remain available via `--classifier`.

### 4.3 CNN (transfer learning)
- **Backbone:** ImageNet-pretrained **MobileNetV2** (`include_top=False`,
  global-average pooling) + dropout(0.3) + sigmoid head.
- **Augmentation:** random horizontal/vertical flips, ±5% rotation, ±10% zoom,
  ±10% contrast (applied on-the-fly during training).
- **Patch-based training** _(primary)_: strips are tiled with an **overlapping
  stride of 128** (31 patches/strip, ~4,000 training patches) resized to
  224×224; patches inherit the image label. At inference each patch is scored
  with **test-time augmentation** (average over its 4 flip variants), the image
  score is the **max** patch probability, and the decision threshold is
  **calibrated on the validation split**. A **full-image** mode is provided for
  comparison (`--mode full`).
- **Two-phase schedule:** (1) train the head with the backbone frozen
  (lr=1e-3, ≤12 epochs); (2) unfreeze the top 60 backbone layers and fine-tune
  (lr=1e-5, ≤10 epochs). Early stopping on training loss.
- **SVM+CNN ensemble:** the CNN score and SVM probability are blended
  (weight and threshold grid-searched on validation); the final predictor is
  whichever wins on the validation split.

### 4.4 Localisation (extended task)
Two **deep** models localise defects on the original AITEX archive and are
compared on the same held-out defective images.

**Task 3 — Convolutional autoencoder (unsupervised).** A symmetric conv
encoder–decoder (32→64→128 filters, 8× spatial bottleneck) is trained with MSE
on 256×256 patches drawn from the **141 defect-free images only**. Having learned
to reconstruct the normal weave, it reconstructs defects poorly. The anomaly map
is **1 − SSIM(input, reconstruction)**: structural similarity compares local
texture windows rather than raw intensities, so it localises broken-weave
structure far better than plain `|input − reconstruction|`. The map is smoothed,
normalised and thresholded at the 99th percentile.

**Task 4 — U-Net (supervised).** A U-Net whose **encoder is an
ImageNet-pretrained MobileNetV2** (skip connections from five encoder stages) is
trained on (image-patch → mask-patch) pairs, using the binary masks as ground
truth and a recall-weighted **Focal-Tversky + BCE** loss (β=0.7 penalises the
false negatives that plain BCE/Dice produce on tiny defects). Defect-bearing
patches are oversampled (≈1:1) and all tiles are **flip-augmented ×4**. At
inference the strip is tiled into 256×256 patches, segmented, and stitched back
to full 4096×256 resolution; the binarisation threshold is **calibrated on the
training pairs** (not fixed at 0.5), and no minimum-area filtering is applied so
hairline defects survive.

**Protocol.** The defect/mask pairs are split 80/20 (seeded). The U-Net trains on
the 80% train split; **both** models are evaluated on the held-out 20% so the
comparison is fair. Masks are scored with pixel-level **IoU, precision, recall and
F1** and with **region-level detection metrics** (a ground-truth defect counts as
found if the predicted mask touches it; a predicted region touching no defect is a
false alarm). The region metric matters because pixel IoU is punishingly strict
for hairline defects — industrially, the question is whether each defect is
*found*, not whether its outline is pixel-perfect. A non-learned classical
residual heatmap is reported as a weak reference baseline.

## 5. Results

> The tables below report **out-of-fold (3-fold) results over all 173 images** —
> the most reliable estimate available. An earlier single-split run gave higher
> but noisier CNN numbers (≈0.81 on 43 images); the honest OOF figure is lower.

### 5.0 What worked and what didn't (ablation notes)
Every change was measured by 3-fold OOF accuracy (fixed 0.5 threshold, so
unbiased), not assumed. The classical model improved the most, in three verified
steps — each stable across six different fold seeds:

| Change to the classical model | OOF accuracy |
|---|---|
| Baseline (LBP+GLCM+edge, all patches, SVM) | 0.751 |
| **+ skip blank white-padding patches** | 0.804 (+0.052) |
| **+ Gabor + Fourier features** | 0.827 |
| **+ SVM/RF/GB voting ensemble** | **0.856** (+0.104 total) |

Averaged over six fold seeds the final recipe scores **0.839 ± 0.014** vs the
baseline's **0.774 ± 0.017** — a robust +6.5 points with *lower* variance.

- **Biggest single win:** discovering that AITEX strips carry ~1000px of blank
  white border, so 4 of 16 patches were pure padding — skipping them added ~5
  points and was essentially free. (Applied to the CNN too, where padding patches
  were being *mislabelled* as defective.)
- **Localisation overhaul helped hugely:** pretrained U-Net encoder +
  Focal-Tversky + augmentation + defect-free tiles took IoU **0.016 → 0.254** and
  region-F1 to **0.610**.
- **Classifier tuning:** adding **ExtraTrees** to the vote and lightly tuning the
  members added ~1.2 accuracy points on *unseen* fold seeds (validated on seeds not
  used to pick the config), taking seed-42 OOF to ~0.879.
- **Hurt / reverted:** **EfficientNetB0** as a second CNN backbone stalled near
  chance and dragged the CNN down; removed after OOF caught it. Feature selection
  and SVM-only tuning did *not* help and were not adopted.
- **Sobering:** an early single 43-image split scored the CNN at 0.814; the honest
  OOF estimate is 0.688. Small hold-outs flatter models — hence the OOF protocol.

**Error analysis — where the ceiling is.** Of the images misclassified in ≥5 of 6
fold seeds, **13 of 14 are defective strips predicted as clean** — the errors are
almost entirely *missed subtle defects*, not false alarms on good fabric. A handful
of AITEX defects are simply too faint to separate from normal weave using texture
features; this is the genuine ceiling for the classical approach, and exactly where
the CNN's high defect recall (0.878) is complementary.

### 5.1 Classification — out-of-fold (all 173 images, final run)
| Model (3-fold OOF, 173 images) | Accuracy | Macro-F1 | Recall (defective) |
|-------|----------|----------|---------------------|
| Classical — *original* (LBP+GLCM+edge, SVM) | 0.751 | 0.745 | 0.689 |
| Classical — final (＋Gabor/Fourier, blank-skip, voting) | 0.861 | 0.856 | 0.784 |
| **Ensemble / final predictor (calibrated)** | **0.867** | **0.862** | 0.784 |
| CNN (3-fold bagged MobileNetV2) | 0.688 | 0.687 | **0.878** |

> The upgraded **classical voting model is the strongest classifier at 0.861 OOF**
> — a +11-point gain over the original classical baseline and far above the CNN.
> *(The voting ensemble was subsequently improved with ExtraTrees + light tuning;
> local seed-42 OOF is ~0.879 and it wins on unseen seeds — re-run the notebook to
> refresh this row to that value.)*
> The final predictor (0.867) is that model with an OOF-calibrated threshold; the
> blend grid-search kept weight 0 on the CNN. The improvement came entirely from
> feature/data engineering verified on all 173 images (§5.0), not from more deep
> learning. The CNN's distinctive value remains its **defective recall (0.878)**,
> the safety-critical metric, and the shared backbone that powers the U-Net.
>
> **Full per-class (OOF):**
> - Classical vote — defect_free: P=0.850 R=0.919 F1=0.883 · defective: P=0.879 R=0.784 F1=0.829
> - Final ensemble — defect_free: P=0.852 R=0.929 F1=0.889 · defective: P=0.892 R=0.784 F1=0.835
> - CNN (bagged)   — defect_free: P=0.857 R=0.545 F1=0.667 · defective: P=0.591 R=0.878 F1=0.707
>
> Confusion matrices (rows=true [free, defect], cols=pred) — Classical vote
> `[[91,8],[16,58]]`, Final ensemble `[[92,7],[16,58]]`, CNN `[[54,45],[9,65]]`.

_[Embed the classical (vote, OOF) and CNN confusion-matrix figures here.]_

### 5.2 Per-class analysis — two complementary operating points
The final models sit at opposite ends of the QC trade-off:
- The **CNN is a defect-catcher**: recall 0.878 on defective strips (the patch +
  max-aggregation design flags a strip if *any* window looks suspicious) at the
  cost of 43 false alarms on clean strips (defect-free recall 0.566).
- The **SVM/ensemble is balanced**: defective recall 0.689 with far fewer false
  alarms (defect-free recall 0.859), which is what maximises plain accuracy.

Which is "better" depends on the cost model: in real inspection a missed defect
(shipping a faulty roll) usually costs more than a false alarm (re-inspection), so
the CNN's operating point is arguably the more industrially useful one even though
the leaderboard metric (accuracy) favours the ensemble.

### 5.3 Classical vs CNN — an honest verdict
Under the rigorous OOF protocol the **engineered classical model (0.861) clearly
beats the patch-based transfer-learning CNN (0.688) on accuracy** — a decisive
gap. This is a legitimate and well-known small-data outcome: with only 173 images,
a compact, domain-specific texture descriptor (LBP + GLCM + edges + Gabor +
Fourier, 122-D) purpose-built for woven fabric out-generalises even a bagged,
augmented, transfer-learned CNN. Crucially, the classical gains came from
**understanding the data** (blank-border patches were poisoning the features) and
**domain-appropriate features** (fabric periodicity → Gabor/Fourier), not from
more model capacity. (An early single 43-image split had suggested CNN 0.814 vs
SVM 0.698 — the opposite conclusion — illustrating how misleading small hold-outs
are.) The deep model still contributes what classical features cannot: the highest
defect recall (0.878) and, via the shared backbone, the U-Net localiser in §5.4.

### 5.4 Localisation (extended task) — Task 3 vs Task 4
Evaluated on the held-out 20% of defective images (~21 strips):

| Method (21 held-out defects) | Pixel IoU | Pixel F1 | Region recall | Region precision | Region F1 |
|--------|-----------|----------|---------------|------------------|-----------|
| **U-Net (Task 4, supervised, pretrained encoder)** | **0.254** | **0.341** | **0.619** | **0.603** | **0.610** |
| Autoencoder (Task 3, SSIM, unsupervised) | 0.003 | 0.006 | 0.048 | 0.008 | 0.014 |

> Final run, notebook §7.3. The **region** metric counts a ground-truth defect as
> found if the prediction touches it (a predicted blob touching no defect is a
> false alarm) — the industrially meaningful "did we find the defect" question,
> far less punishing than pixel IoU on hairline defects.

**The upgraded U-Net is a genuinely useful localiser.** Pixel IoU rose from an
early **0.016 to 0.254** (≈16×) through the accuracy overhaul: pretrained
MobileNetV2 encoder, recall-weighted Focal-Tversky loss, overlapping +
flip-augmented training tiles, extra defect-free tiles in the negative pool, a
threshold calibrated on training pairs (0.65), and TTA-stitched inference. At the
**region level it finds 62% of defects at 60% precision (F1 0.610)** — a solid
result given only ~82 labelled defective strips, and pixel precision 0.42 shows
that when it fires, it is usually on or around a real defect.

The **SSIM autoencoder** remains far behind (pixel IoU 0.003, region-recall 0.048;
an earlier run reached region-recall 0.238, so its weak signal is also unstable
run-to-run). This is an honest answer to the Task-3 question: reconstruction
error alone, with no defective examples, *occasionally* fires on visible defects
but is nowhere near a usable localiser on AITEX's subtle, hairline defects. The
supervised U-Net wins decisively when masks are available.

**Why pixel IoU stays modest even for a good localiser:** defects are a handful of
pixels in a ~1M-pixel strip, so even near-perfect localisation yields a small
intersection — which is exactly why the region-level metric is reported alongside
and is the fairer headline for this task.

## 6. Discussion
- **Small-data strategies that helped:** _[transfer learning, augmentation,
  patch-based training (16× more samples), class weighting, threshold
  calibration — say which mattered most in your runs.]_
- **Limitations:** image-level labels for patches introduce noise (a defective
  strip's clean patches are labelled positive); the max-aggregation rule
  mitigates this. Very wide aspect ratio; subtle low-contrast defects. Tiny test
  set (74 images) → leaderboard variance.
- **Future work:** _[stronger U-Net backbones (pretrained encoder, attention);
  multiple-instance learning for the weakly-labelled patch classifier;
  per-defect-type localisation analysis across the 9 AITEX categories; combining
  the autoencoder anomaly score with the U-Net for a semi-supervised localiser.]_

## 7. Conclusion
We implemented and compared four models on AITEX under a rigorous out-of-fold
protocol. For classification, an **engineered classical model reached 0.856 OOF
accuracy** — a +10-point gain over the original classical baseline, achieved
entirely through data understanding (removing blank-border patches), domain
features (Gabor/Fourier for fabric periodicity), and a voting ensemble, each
verified on all 173 images. It clearly out-scores the transfer-learning patch CNN
(0.688) on accuracy, while the **CNN dominates on defect recall (0.878)** — two
complementary operating points, with an early lucky single-split (CNN 0.814)
showing how misleading small hold-outs can be. Measurement also led us to *remove*
a backbone (EfficientNetB0) that hurt. For localisation, an accuracy-engineered
supervised **U-Net lifted pixel IoU from 0.016 to 0.254**, finding **62% of
defects at 60% precision (region F1 0.610)**, decisively beating the unsupervised
autoencoder. The key takeaways: at this data scale, classical texture features
with domain knowledge are the strongest *detector*; deep networks are what make
*localisation* and high-recall screening possible; and disciplined OOF measurement
— not stacking ever more machinery — is what actually improved (and occasionally
corrected) the results.

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
