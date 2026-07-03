"""
Extended task (3 + 4): train the autoencoder and the U-Net, then compare their
localisation against the AITEX ground-truth masks.

    python run_extended.py
    python run_extended.py --ae-epochs 40 --unet-epochs 80 --no-pretrained

Task 3 -- autoencoder trained on defect-free images only (unsupervised);
          SSIM reconstruction-error anomaly maps.
Task 4 -- U-Net with a pretrained MobileNetV2 encoder, Focal-Tversky loss,
          flip augmentation and a threshold calibrated on the training pairs.

Both are evaluated on the SAME held-out defective images, with pixel-level
IoU/precision/recall/F1 AND defect-level (region) detection metrics -- pixel IoU
is punishing for hairline defects, so the region metric reflects "did we find
each defect" the way an inspector would.
"""
from __future__ import annotations

import argparse
import numpy as np
from PIL import Image

from src.config import Config
from src.data import list_aitex_nodefect, load_gray
from src import autoencoder as ae
from src import unet as un
from src import localisation as loc


def split_pairs(pairs, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(round(len(pairs) * val_frac)))
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    return train, val


def eval_maps(model_kind, model, val_pairs, cfg, percentile=99.0):
    pix, reg = [], []
    for img_path, mask_path in val_pairs:
        gray = load_gray(img_path)
        gt = (load_gray(mask_path) > 127).astype(np.uint8)
        if model_kind == "ae":
            heat = model.anomaly_map(gray)                      # SSIM error map
            pred = (heat >= np.percentile(heat, percentile)).astype(np.uint8)
        else:                                                   # unet
            heat = model.predict_map(gray)
            pred = (heat >= model.threshold).astype(np.uint8)
        s = loc.score_mask(pred, gt)
        pix.append(s)
        reg.append(loc.region_metrics(pred, gt))
    out = {
        "mean_iou": float(np.mean([s.iou for s in pix])),
        "mean_f1": float(np.mean([s.f1 for s in pix])),
        "mean_precision": float(np.mean([s.precision for s in pix])),
        "mean_recall": float(np.mean([s.recall for s in pix])),
        "region_recall": float(np.mean([r["region_recall"] for r in reg])),
        "region_precision": float(np.mean([r["region_precision"] for r in reg])),
        "region_f1": float(np.mean([r["region_f1"] for r in reg])),
        "n": len(val_pairs),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Extended task: autoencoder + U-Net")
    ap.add_argument("--ae-epochs", type=int, default=30)
    ap.add_argument("--unet-epochs", type=int, default=60)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="use the from-scratch U-Net (offline fallback)")
    args = ap.parse_args()

    cfg = Config().resolve()
    out_dir = cfg.output_dir / "extended"
    out_dir.mkdir(parents=True, exist_ok=True)

    nodefect = list_aitex_nodefect(cfg)
    pairs = loc.find_mask_pairs(cfg.aitex_defect_dir, cfg.aitex_mask_dir)
    if not nodefect or not pairs:
        ap.error(f"AITEX archive incomplete. defect-free={len(nodefect)} "
                 f"defect/mask pairs={len(pairs)} under {cfg.aitex_root}")
    train_pairs, val_pairs = split_pairs(pairs, args.val_frac, cfg.seed)
    print(f"defect-free images={len(nodefect)}  "
          f"defect pairs: train={len(train_pairs)} val={len(val_pairs)}")

    # --- Task 3: autoencoder (trained on defect-free only) ---
    ae_model = ae.train(nodefect, cfg, epochs=args.ae_epochs)
    ae_res = eval_maps("ae", ae_model, val_pairs, cfg)
    print("\n[Task 3] Autoencoder localisation:", ae_res)

    # --- Task 4: U-Net (supervised on masks; clean tiles cut false alarms) ---
    unet_model = un.train(train_pairs, cfg, epochs=args.unet_epochs,
                          pretrained=not args.no_pretrained,
                          nodefect_paths=nodefect)
    unet_res = eval_maps("unet", unet_model, val_pairs, cfg)
    print("\n[Task 4] U-Net localisation:", unet_res)

    # --- Comparison ---
    print("\n=== Localisation comparison (same held-out defective images) ===")
    hdr = f"{'method':12s} {'IoU':>7s} {'F1':>7s} {'prec':>7s} {'recall':>7s}" \
          f" | {'regR':>6s} {'regP':>6s} {'regF1':>6s}"
    print(hdr)
    for name, r in [("Autoencoder", ae_res), ("U-Net", unet_res)]:
        print(f"{name:12s} {r['mean_iou']:7.3f} {r['mean_f1']:7.3f} "
              f"{r['mean_precision']:7.3f} {r['mean_recall']:7.3f} | "
              f"{r['region_recall']:6.3f} {r['region_precision']:6.3f} "
              f"{r['region_f1']:6.3f}")

    # --- Save side-by-side overlays for the report ---
    for img_path, mask_path in val_pairs[:8]:
        gray = load_gray(img_path)
        gt = (load_gray(mask_path) > 127).astype(np.uint8)
        heat = ae_model.anomaly_map(gray)
        ae_mask = (heat >= np.percentile(heat, 99.0)).astype(np.uint8)
        un_mask = (unet_model.predict_map(gray) >= unet_model.threshold
                   ).astype(np.uint8)
        Image.fromarray(loc.overlay(gray, gt)).save(out_dir / f"{img_path.stem}_gt.png")
        Image.fromarray(loc.overlay(gray, ae_mask)).save(out_dir / f"{img_path.stem}_ae.png")
        Image.fromarray(loc.overlay(gray, un_mask)).save(out_dir / f"{img_path.stem}_unet.png")
    print(f"\nSaved overlays -> {out_dir}")


if __name__ == "__main__":
    main()
