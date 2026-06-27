"""
Extended task: defect localisation on the original AITEX defective images.

    # Classical residual method (no training, no CNN needed):
    python run_localisation.py --images path/to/aitex/defects --method residual

    # CNN sliding-window method (needs a trained model saved by run_cnn.py):
    python run_localisation.py --images path/to/aitex/defects \
        --method cnn --model outputs/cnn_patch.keras

Each defective image ``NAME.png`` is matched with its ground-truth
``NAME_mask.png`` (AITEX convention) when present, and IoU/F1 are reported.
Example overlays are written to outputs/localisation/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import Config
from src.data import load_gray
from src import localisation as loc


def main():
    ap = argparse.ArgumentParser(description="Defect localisation (extended task)")
    ap.add_argument("--images", default=None,
                    help="folder of AITEX defective images (default: auto-detect archive/Defect_images)")
    ap.add_argument("--masks", default=None,
                    help="folder of mask images (default: auto-detect archive/Mask_images)")
    ap.add_argument("--method", choices=["residual", "cnn"], default="residual")
    ap.add_argument("--model", default=None, help="saved .keras model for --method cnn")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-overlays", type=int, default=12)
    args = ap.parse_args()

    cfg = Config().resolve()
    out_dir = cfg.output_dir / "localisation"
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images) if args.images else cfg.aitex_defect_dir
    masks_dir = Path(args.masks) if args.masks else cfg.aitex_mask_dir
    if not images_dir.exists():
        ap.error(f"Defective-image folder not found: {images_dir}. "
                 f"Pass --images or place the AITEX archive under {cfg.aitex_root}.")
    print(f"Images: {images_dir}\nMasks : {masks_dir} (exists={masks_dir.exists()})")

    cnn_model = None
    if args.method == "cnn":
        if not args.model:
            ap.error("--method cnn requires --model path/to/model.keras")
        import tensorflow as tf
        from src.cnn import CNNModel
        keras_model = tf.keras.models.load_model(args.model)
        cnn_model = CNNModel(cfg=cfg, model=keras_model, mode="patch")

    pairs = loc.find_mask_pairs(images_dir, masks_dir if masks_dir.exists() else None)
    print(f"Found {len(pairs)} image/mask pairs.")

    if pairs:
        scores, agg = loc.evaluate_localisation(
            pairs, cfg, method=args.method, cnn_model=cnn_model,
            threshold=args.threshold)
        print("Localisation metrics:", agg)

    # Save a few overlays for the report.
    sample = pairs[:args.max_overlays] if pairs else []
    if not sample:
        # No masks: still produce overlays from raw images.
        sample = [(p, None) for p in sorted(images_dir.glob("*.png"))
                  if not p.stem.endswith("_mask")][:args.max_overlays]

    for img_path, _ in sample:
        gray = load_gray(img_path)
        if args.method == "cnn":
            heat = loc.cnn_heatmap(gray, cnn_model, cfg)
            mask = loc.heatmap_to_mask(heat, threshold=args.threshold)
        else:
            heat = loc.residual_heatmap(gray)
            mask = loc.heatmap_to_mask(heat, percentile=99.5)
        boxes = loc.mask_to_boxes(mask)
        vis = loc.overlay(gray, mask, boxes)
        Image.fromarray(vis).save(out_dir / f"{img_path.stem}_overlay.png")
    print(f"Saved {len(sample)} overlays -> {out_dir}")


if __name__ == "__main__":
    main()
