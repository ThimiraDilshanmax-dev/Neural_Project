# Fabric Defect Detection and Localisation Using Deep Neural Networks

CO5420 — Project 3. Automatic quality control for woven fabric using the
**AITEX Fabric Image Database**. The project has two parts:

* **Kaggle task** — binary classification of 4096×256 fabric strips as
  `defective` (1) or `defect_free` (0). A *classical image-processing baseline*
  is compared against a *transfer-learning CNN* under an identical validation
  protocol, and a `submission.csv` is produced for the hidden test set.
* **Extended task** — *localisation* of defects on the original AITEX defective
  images using their binary ground-truth masks (pixel-level IoU / F1).

## Dataset

| Split | Folder | Count | Label |
|-------|--------|-------|-------|
| Train | `Train/Train/defective/`    | 74 | 1 |
| Train | `Train/Train/defect_free/`  | 99 | 0 |
| Test  | `Test/` *(released week 12)* | 74 | — |

Images are 4096×256, single-channel grayscale. `sample_submission.csv` defines
the required `ID,label` output format.

For the **extended localisation task** the original AITEX archive lives in
`archive/` at the repository root:

```
archive/
├── Defect_images/    # 106 defective strips  (nnnn_ddd_ff.png)
├── Mask_images/      # ground-truth masks    (nnnn_ddd_ff_mask.png)
└── NODefect_images/  # 141 defect-free strips (in per-fabric sub-folders)
```

`src/config.py` auto-detects this folder, so the localisation task needs no
extra paths.

## File structure

```
Code/
├── README.md
├── requirements.txt
├── run_classical.py        # train + evaluate the classical baseline
├── run_cnn.py              # train + evaluate the CNN
├── run_predict.py          # fit final model -> outputs/submission.csv
├── run_extended.py         # extended task: train AE + U-Net, compare IoU
├── run_localisation.py     # classical/CNN heatmap localisation (baseline)
├── notebooks/
│   └── fabric_defect_detection_kaggle.ipynb   # self-contained Kaggle notebook
├── outputs/                # models, figures, submission.csv (gitignored)
└── src/
    ├── config.py           # paths + hyper-parameters (Kaggle/local aware)
    ├── data.py             # discovery, loading, reproducible stratified split
    ├── preprocessing.py    # patch extraction, tensor prep
    ├── features.py         # LBP + GLCM + edge/intensity descriptors
    ├── classical.py        # features -> StandardScaler -> SVM / RandomForest
    ├── cnn.py              # MobileNetV2 transfer learning (patch / full image)
    ├── autoencoder.py      # Task 3: conv autoencoder, reconstruction-error anomaly
    ├── unet.py             # Task 4: U-Net segmentation (BCE+Dice) on masks
    ├── evaluate.py         # accuracy, confusion matrix, precision/recall/F1
    ├── predict.py          # submission writer (preserves sample IDs/order)
    └── localisation.py     # heatmaps -> masks -> boxes, IoU scoring
```

## Approach

### Classical baseline (`src/classical.py`, `src/features.py`)
Each strip is tiled into 256×256 patches; per patch we extract **LBP** histograms,
**GLCM** Haralick properties (contrast, dissimilarity, homogeneity, energy,
correlation, ASM) and **first-order + Sobel edge** statistics. Patch features are
aggregated by **mean and max** across the strip (the max captures the single most
anomalous region — defects cover only a small slice of a 4096-pixel-wide image).
The vector feeds a StandardScaler + SVM-RBF (or RandomForest), `class_weight="balanced"`.

### CNN (`src/cnn.py`)
Transfer learning from an ImageNet **MobileNetV2** backbone with on-the-fly data
augmentation. Two strategies for the very wide images:
* **patch** *(default)* — trained on **overlapping** patches (stride 128 → 31
  patches/strip, ~2× data); at inference each patch is scored with **test-time
  augmentation** (average over 4 flip variants) and the image score is the
  **max** patch probability, threshold calibrated on validation.
* **full** — resize the whole strip to a square input.

Training is two-phase: frozen-backbone head training, then fine-tuning the top
60 backbone layers at a low learning rate. Class imbalance is handled with class
weights. `run_predict.py --model ensemble` additionally blends the CNN score
with the SVM probability (weight + threshold grid-searched on validation).

### Extended task — localisation with deep networks
Two deep models localise defects on AITEX and are compared by pixel-level IoU/F1
on the same held-out defective images (`run_extended.py`):

* **Task 3 — Autoencoder** (`src/autoencoder.py`): a convolutional autoencoder
  trained on **defect-free patches only**. The anomaly map is **1 − SSIM(input,
  reconstruction)** — structural similarity localises broken weave far better
  than raw `|input − reconstruction|` (no defective labels at training time).
* **Task 4 — U-Net** (`src/unet.py`): supervised segmentation with an
  **ImageNet-pretrained MobileNetV2 encoder**, a recall-weighted
  **Focal-Tversky + BCE** loss, defect-bearing patches oversampled and
  **flip-augmented ×4**, and the binarisation threshold **calibrated on the
  training pairs**. At inference the strip is tiled, each tile segmented, and
  predictions stitched back to full resolution.

Both are scored with pixel IoU/precision/recall/F1 **and** region-level
detection metrics (a ground-truth defect counts as found if the prediction
touches it) — pixel IoU alone is punishingly strict for hairline defects.

A simpler classical/CNN heatmap localiser (`src/localisation.py`,
`run_localisation.py`) is retained as a non-learned baseline: `residual_heatmap`
(`|image − median-smoothed|`) and `cnn_heatmap` (sliding the patch classifier).

## Running locally

```bash
cd Code
pip install -r requirements.txt

python run_classical.py --classifier svm --cv
python run_cnn.py --mode patch
python run_predict.py --model cnn --mode patch      # writes outputs/submission.csv

# Extended task -- trains Task-3 autoencoder + Task-4 U-Net and compares IoU
# (auto-detects archive/Defect_images, Mask_images, NODefect_images):
python run_extended.py

# Optional non-learned baseline localiser:
python run_localisation.py --method residual
```

`run_extended.py` trains the autoencoder on the 141 defect-free images and the
U-Net on a train split of the defect/mask pairs, then reports IoU/precision/
recall/F1 for **both** on the same held-out defects and saves side-by-side
overlays (`input / ground-truth / autoencoder / U-Net`) to `outputs/extended/`.
The supervised U-Net localises more precisely; the autoencoder needs no defective
labels. (The classical `run_localisation.py` residual heatmap is a weak
non-learned reference, mean IoU ≈ 0.015 over 103 masks.)

`src/config.py` auto-detects the dataset: it uses this repository when run locally
and a `/kaggle/input/...` competition folder on Kaggle. Override with the
`FABRIC_DATA_ROOT` environment variable if needed.

## Running on Kaggle
Open `notebooks/fabric_defect_detection_kaggle.ipynb`, attach the competition
dataset, and *Run All*. It is self-contained (no repo imports), executes the full
workflow — load → preprocess → classical baseline → CNN → compare → predict — and
writes `submission.csv`, satisfying the reproducible-notebook requirement.

## Reproducibility
All randomness is seeded (`Config.seed = 42`): the stratified split, NumPy, and
TensorFlow. The same split is used for both models so the comparison is fair.
```
Evaluated with classification accuracy (leaderboard) plus confusion matrix,
precision, recall, F1 and per-class performance (report).
```
