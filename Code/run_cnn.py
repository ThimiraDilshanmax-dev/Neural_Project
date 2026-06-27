"""
Train and evaluate the transfer-learning CNN.

    python run_cnn.py --mode patch        # patch-based + max aggregation
    python run_cnn.py --mode full         # whole-strip resized
    python run_cnn.py --mode patch --backbone EfficientNetB0
"""
from __future__ import annotations

import argparse

from src.config import Config
from src.data import list_training_records, stratified_split, class_distribution
from src import cnn
from src.evaluate import plot_confusion


def main():
    ap = argparse.ArgumentParser(description="CNN fabric-defect classifier")
    ap.add_argument("--mode", choices=["patch", "full"], default="patch")
    ap.add_argument("--backbone", default=None, help="override the keras backbone")
    ap.add_argument("--epochs", type=int, default=None, help="override head epochs")
    args = ap.parse_args()

    cfg = Config().resolve()
    if args.backbone:
        cfg.backbone = args.backbone
    if args.epochs:
        cfg.head_epochs = args.epochs

    records = list_training_records(cfg)
    print("Class distribution:", class_distribution(records))
    train_r, val_r = stratified_split(records, cfg.val_fraction, cfg.seed)
    print(f"Train={len(train_r)}  Val={len(val_r)}  mode={args.mode}  backbone={cfg.backbone}")

    model, result = cnn.train(train_r, val_r, cfg, mode=args.mode)
    print(f"\n=== CNN ({args.mode}, {cfg.backbone}) ===")
    print(result.summary())

    fig_path = cfg.output_dir / f"confusion_cnn_{args.mode}.png"
    plot_confusion(result, f"CNN ({args.mode})", fig_path)
    print(f"Saved confusion matrix -> {fig_path}")

    model.model.save(cfg.output_dir / f"cnn_{args.mode}.keras")
    print(f"Saved model -> {cfg.output_dir / f'cnn_{args.mode}.keras'}")


if __name__ == "__main__":
    main()
