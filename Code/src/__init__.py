"""Fabric Defect Detection and Localisation -- source package.

Modules
-------
config         project paths + hyper-parameters (Kaggle/local aware)
data           image discovery, loading, reproducible stratified split
preprocessing  patch extraction and tensor preparation
features       hand-crafted texture descriptors (LBP, GLCM, edge stats)
classical      classical baseline (features + SVM/RandomForest)
cnn            transfer-learning CNN (patch / full-image strategies)
autoencoder    extended Task 3: conv autoencoder, reconstruction-error anomaly
unet           extended Task 4: U-Net segmentation (BCE+Dice) on AITEX masks
evaluate       accuracy, confusion matrix, precision/recall/F1
predict        Kaggle submission CSV writer
localisation   defect heatmaps, masks, boxes and IoU scoring (shared helpers)
"""
from .config import Config, LABELS, DEFECTIVE, DEFECT_FREE  # noqa: F401

__all__ = ["Config", "LABELS", "DEFECTIVE", "DEFECT_FREE"]
