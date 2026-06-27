"""Fabric Defect Detection and Localisation -- source package.

Modules
-------
config         project paths + hyper-parameters (Kaggle/local aware)
data           image discovery, loading, reproducible stratified split
preprocessing  patch extraction and tensor preparation
features       hand-crafted texture descriptors (LBP, GLCM, edge stats)
classical      classical baseline (features + SVM/RandomForest)
cnn            transfer-learning CNN (patch / full-image strategies)
evaluate       accuracy, confusion matrix, precision/recall/F1
predict        Kaggle submission CSV writer
localisation   extended task: defect heatmaps, masks and IoU scoring
"""
from .config import Config, LABELS, DEFECTIVE, DEFECT_FREE  # noqa: F401

__all__ = ["Config", "LABELS", "DEFECTIVE", "DEFECT_FREE"]
