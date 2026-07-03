"""
Evaluation metrics shared by the classical baseline and the CNN.

Reports everything the brief asks for: accuracy, confusion matrix, precision,
recall, F1 and per-class performance.  Implemented with numpy so it has no hard
dependency on scikit-learn (which is used elsewhere for modelling).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

from .config import LABELS


@dataclass
class EvalResult:
    accuracy: float
    confusion: np.ndarray                 # rows = true, cols = pred; [neg, pos]
    per_class: Dict[str, Dict[str, float]]
    macro_f1: float
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        cm = self.confusion
        lines = [
            f"Accuracy : {self.accuracy:.4f}",
            f"Macro-F1 : {self.macro_f1:.4f}",
            "",
            "Confusion matrix (rows=true, cols=pred):",
            f"                pred_free  pred_defect",
            f"  true_free   {cm[0, 0]:9d}  {cm[0, 1]:11d}",
            f"  true_defect {cm[1, 0]:9d}  {cm[1, 1]:11d}",
            "",
            "Per-class metrics:",
        ]
        for name, m in self.per_class.items():
            lines.append(
                f"  {name:11s} precision={m['precision']:.3f} "
                f"recall={m['recall']:.3f} f1={m['f1']:.3f} support={int(m['support'])}"
            )
        return "\n".join(lines)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = float((y_true == y_pred).mean())

    per_class: Dict[str, Dict[str, float]] = {}
    f1s = []
    for cls in (0, 1):
        tp = cm[cls, cls]
        fp = cm[1 - cls, cls]
        fn = cm[cls, 1 - cls]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        per_class[LABELS[cls]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(cm[cls].sum()),
        }
        f1s.append(f1)

    return EvalResult(
        accuracy=accuracy,
        confusion=cm,
        per_class=per_class,
        macro_f1=float(np.mean(f1s)),
    )


def plot_confusion(result: EvalResult, title: str, save_path: Path | None = None):
    """Render the confusion matrix; returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    cm = result.confusion
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["defect_free", "defective"])
    ax.set_yticks([0, 1], ["defect_free", "defective"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\nacc={result.accuracy:.3f}  macroF1={result.macro_f1:.3f}")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=13)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick the probability threshold that maximises accuracy on (y_true, scores)."""
    order = np.unique(np.concatenate([[0.0, 1.0], scores]))
    best_t, best_acc = 0.5, -1.0
    for t in order:
        acc = ((scores >= t).astype(int) == y_true).mean()
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def calibrate_ensemble(y_true: np.ndarray, scores_a: np.ndarray,
                       scores_b: np.ndarray) -> tuple[float, float, float]:
    """Blend two per-image score vectors: score = w*a + (1-w)*b.

    Grid-searches the blend weight ``w`` and the decision threshold on the
    validation labels; returns ``(w, threshold, accuracy)``.  With ``w`` = 1 the
    ensemble degenerates to model A, so the result is never worse than the
    better single model *on the validation split* (use with the usual
    small-split caveat).
    """
    best = (1.0, 0.5, -1.0)
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * scores_a + (1.0 - w) * scores_b
        t = best_threshold(y_true, blend)
        acc = float(((blend >= t).astype(int) == y_true).mean())
        if acc > best[2]:
            best = (float(w), float(t), acc)
    return best
