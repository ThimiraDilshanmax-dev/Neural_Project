"""
Train the chosen model on ALL labelled data and write submission.csv.

    python run_predict.py --model cnn --mode patch
    python run_predict.py --model classical --classifier svm
    python run_predict.py --model ensemble          # blended CNN + SVM scores
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import Config
from src.data import list_training_records, stratified_split, load_gray
from src import classical, cnn
from src.evaluate import calibrate_ensemble, evaluate
from src.predict import make_submission


@dataclass
class EnsembleModel:
    """Blend of CNN score and SVM probability: w*cnn + (1-w)*svm >= t."""
    cnn_model: object
    svm_model: object
    w: float
    threshold: float

    def predict_path(self, path: Path) -> int:
        gray = load_gray(path)
        s = (self.w * self.cnn_model.score_image(gray)
             + (1.0 - self.w) * self.svm_model.predict_proba_image(gray))
        return int(s >= self.threshold)


def main():
    ap = argparse.ArgumentParser(description="Generate the Kaggle submission")
    ap.add_argument("--model", choices=["classical", "cnn", "ensemble"],
                    default="cnn")
    ap.add_argument("--mode", choices=["patch", "full"], default="patch")
    ap.add_argument("--classifier", choices=["vote", "svm", "rf"], default="vote")
    args = ap.parse_args()

    cfg = Config().resolve()
    records = list_training_records(cfg)

    if args.model == "classical":
        model = classical.fit_on_all(records, cfg, args.classifier)
    elif args.model == "cnn":
        # Calibrate threshold on a held-out split, then keep that model.
        train_r, val_r = stratified_split(records, cfg.val_fraction, cfg.seed)
        model, _ = cnn.train(train_r, val_r, cfg, mode=args.mode)
    else:
        # Ensemble: dual-backbone CNN average + SVM, blended on the val split.
        train_r, val_r = stratified_split(records, cfg.val_fraction, cfg.seed)
        cnn_model, cnn_res = cnn.train_backbone_ensemble(
            train_r, val_r, cfg, mode=args.mode)
        svm_model, svm_res = classical.train_and_evaluate(
            train_r, val_r, cfg, args.classifier)
        y_val = np.array([r.label for r in val_r])
        cnn_scores = np.array([cnn_model.score_image(load_gray(r.path))
                               for r in val_r])
        svm_scores = np.array([svm_model.predict_proba_image(load_gray(r.path))
                               for r in val_r])
        w, t, acc = calibrate_ensemble(y_val, cnn_scores, svm_scores)
        print(f"[ensemble] w={w:.2f} threshold={t:.3f} val_acc={acc:.4f} "
              f"(cnn alone: {cnn_res.accuracy:.4f})")
        blend = w * cnn_scores + (1 - w) * svm_scores
        print(evaluate(y_val, (blend >= t).astype(int)).summary())
        model = EnsembleModel(cnn_model, svm_model, w, t)

    df = make_submission(model, cfg)
    print(df.head())


if __name__ == "__main__":
    main()
