"""
Train and evaluate the classical image-processing baseline.

    python run_classical.py --classifier svm
    python run_classical.py --classifier rf --cv
"""
from __future__ import annotations

import argparse

from src.config import Config
from src.data import (list_training_records, stratified_split,
                      class_distribution)
from src import classical
from src.evaluate import plot_confusion
from src.features import build_feature_matrix


def main():
    ap = argparse.ArgumentParser(description="Classical fabric-defect baseline")
    ap.add_argument("--classifier", choices=["svm", "rf"], default="svm")
    ap.add_argument("--cv", action="store_true", help="also run 5-fold cross-validation")
    args = ap.parse_args()

    cfg = Config().resolve()
    records = list_training_records(cfg)
    print("Class distribution:", class_distribution(records))

    train_r, val_r = stratified_split(records, cfg.val_fraction, cfg.seed)
    print(f"Train={len(train_r)}  Val={len(val_r)}")

    model, result = classical.train_and_evaluate(train_r, val_r, cfg, args.classifier)
    print("\n=== Classical baseline ({}) ===".format(args.classifier))
    print(result.summary())

    fig_path = cfg.output_dir / f"confusion_classical_{args.classifier}.png"
    plot_confusion(result, f"Classical ({args.classifier})", fig_path)
    print(f"Saved confusion matrix -> {fig_path}")

    if args.cv:
        print("\nRunning 5-fold cross-validation on all data...")
        X, y = build_feature_matrix(records, cfg)
        cv = classical.cross_validate(X, y, args.classifier, cfg.seed)
        print("Cross-validation:", cv)


if __name__ == "__main__":
    main()
