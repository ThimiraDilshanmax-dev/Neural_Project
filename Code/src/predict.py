"""
Generate the Kaggle submission CSV.

Works with any model exposing ``predict_path(path) -> int`` (both
:class:`classical.ClassicalModel` and :class:`cnn.CNNModel` qualify), and
preserves the exact ID order of ``sample_submission.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .config import Config
from .data import list_test_records


def make_submission(model, cfg: Config, out_path: Path | None = None) -> pd.DataFrame:
    """Predict every test image and write a 2-column ``ID,label`` CSV."""
    records = list_test_records(cfg)
    if not records:
        raise FileNotFoundError(
            "No test images and no sample_submission.csv found. "
            "The Test/ folder is released in week 12."
        )

    rows = []
    missing = 0
    for r in records:
        if r.path.exists():
            label = int(model.predict_path(r.path))
        else:
            # Test set not released yet: emit a placeholder so the format is valid.
            label = 0
            missing += 1
        rows.append({"ID": r.image_id, "label": label})

    df = pd.DataFrame(rows, columns=["ID", "label"])

    # Preserve the order/IDs from sample_submission.csv if present.
    if cfg.sample_submission.exists():
        sample = pd.read_csv(cfg.sample_submission)
        id_col = "ID" if "ID" in sample.columns else sample.columns[0]
        order = sample[id_col].astype(str).str.strip().str.strip('"')
        df = df.set_index("ID").reindex(order).reset_index()
        df["label"] = df["label"].fillna(0).astype(int)
        df.columns = ["ID", "label"]

    if out_path is None:
        out_path = cfg.output_dir / "submission.csv"
    df.to_csv(out_path, index=False)
    print(f"[submission] wrote {len(df)} rows to {out_path}"
          + (f"  ({missing} placeholders: Test/ not released)" if missing else ""))
    return df
