"""
Classical image-processing baseline.

Pipeline:  hand-crafted texture features (see :mod:`features`)  ->
StandardScaler  ->  classifier (SVM-RBF or RandomForest).

Exposes a small object that can both report validation metrics and predict the
label of an arbitrary image, so the same baseline drives the submission file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .config import Config
from .data import ImageRecord, load_gray
from .evaluate import EvalResult, evaluate
from .features import image_feature_vector, build_feature_matrix


def build_classifier(kind: str, seed: int):
    """Return an sklearn Pipeline (scaler + classifier)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "svm":
        from sklearn.svm import SVC
        clf = SVC(C=10.0, gamma="scale", kernel="rbf",
                  class_weight="balanced", probability=True, random_state=seed)
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier kind: {kind!r}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


@dataclass
class ClassicalModel:
    cfg: Config
    pipeline: object
    kind: str

    def predict_image(self, gray: np.ndarray) -> int:
        x = image_feature_vector(gray, self.cfg)[None, :]
        return int(self.pipeline.predict(x)[0])

    def predict_proba_image(self, gray: np.ndarray) -> float:
        x = image_feature_vector(gray, self.cfg)[None, :]
        return float(self.pipeline.predict_proba(x)[0, 1])

    def predict_path(self, path: Path) -> int:
        return self.predict_image(load_gray(path))


def cross_validate(X: np.ndarray, y: np.ndarray, kind: str, seed: int, folds: int = 5):
    """Stratified k-fold accuracy/F1 to gauge baseline stability on little data."""
    from sklearn.model_selection import StratifiedKFold, cross_validate as skcv

    pipe = build_classifier(kind, seed)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = skcv(pipe, X, y, cv=skf, scoring=["accuracy", "f1_macro"])
    return {
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "f1_macro_mean": float(scores["test_f1_macro"].mean()),
        "f1_macro_std": float(scores["test_f1_macro"].std()),
    }


def train_and_evaluate(
    train_records: List[ImageRecord],
    val_records: List[ImageRecord],
    cfg: Config,
    kind: str = "svm",
) -> tuple[ClassicalModel, EvalResult]:
    """Fit on the training split, evaluate on the validation split."""
    print(f"[classical] extracting features for {len(train_records)} train images...")
    X_tr, y_tr = build_feature_matrix(train_records, cfg)
    print(f"[classical] extracting features for {len(val_records)} val images...")
    X_va, y_va = build_feature_matrix(val_records, cfg)
    print(f"[classical] feature dimension = {X_tr.shape[1]}")

    pipe = build_classifier(kind, cfg.seed)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    result = evaluate(y_va, y_pred)

    model = ClassicalModel(cfg=cfg, pipeline=pipe, kind=kind)
    return model, result


def fit_on_all(records: List[ImageRecord], cfg: Config, kind: str = "svm"
               ) -> ClassicalModel:
    """Refit on *all* labelled data (for the final submission model)."""
    X, y = build_feature_matrix(records, cfg)
    pipe = build_classifier(kind, cfg.seed)
    pipe.fit(X, y)
    return ClassicalModel(cfg=cfg, pipeline=pipe, kind=kind)
