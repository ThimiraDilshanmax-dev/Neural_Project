"""
Train the chosen model on ALL labelled data and write submission.csv.

    python run_predict.py --model cnn --mode patch
    python run_predict.py --model classical --classifier svm
"""
from __future__ import annotations

import argparse

from src.config import Config
from src.data import list_training_records, stratified_split
from src import classical, cnn
from src.predict import make_submission


def main():
    ap = argparse.ArgumentParser(description="Generate the Kaggle submission")
    ap.add_argument("--model", choices=["classical", "cnn"], default="cnn")
    ap.add_argument("--mode", choices=["patch", "full"], default="patch")
    ap.add_argument("--classifier", choices=["svm", "rf"], default="svm")
    args = ap.parse_args()

    cfg = Config().resolve()
    records = list_training_records(cfg)

    if args.model == "classical":
        model = classical.fit_on_all(records, cfg, args.classifier)
    else:
        # Calibrate threshold on a held-out split, then keep that model.
        train_r, val_r = stratified_split(records, cfg.val_fraction, cfg.seed)
        model, _ = cnn.train(train_r, val_r, cfg, mode=args.mode)

    df = make_submission(model, cfg)
    print(df.head())


if __name__ == "__main__":
    main()
