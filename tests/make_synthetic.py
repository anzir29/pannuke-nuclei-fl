"""Write a tiny PanNuke-shaped dataset in the Kaggle folder layout (for smoke testing only)."""
import sys
from pathlib import Path

import numpy as np


def make(root, n_per_fold=12, seed=0):
    rng = np.random.default_rng(seed)
    tissues = np.array(["Breast", "Colon", "Lung"])
    yy, xx = np.mgrid[:256, :256]
    for f in (1, 2, 3):
        imgs = np.full((n_per_fold, 256, 256, 3), 230, np.float64)
        masks = np.zeros((n_per_fold, 256, 256, 6), np.float64)
        for i in range(n_per_fold):
            nid = 1
            for _ in range(rng.integers(6, 14)):
                c = rng.integers(0, 5)
                cy, cx, r = rng.integers(15, 241), rng.integers(15, 241), rng.integers(5, 11)
                m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
                m &= masks[i, ..., :5].sum(-1) == 0
                masks[i, ..., c][m] = nid
                imgs[i][m] = np.array([60 + 30 * c, 40, 120 + 20 * c])
                nid += 1
            masks[i, ..., 5] = (masks[i, ..., :5].sum(-1) == 0).astype(float)
        imgs += rng.normal(0, 8, imgs.shape)
        d_img = Path(root) / f"Fold {f}" / "images" / f"fold{f}"
        d_msk = Path(root) / f"Fold {f}" / "masks" / f"fold{f}"
        d_img.mkdir(parents=True, exist_ok=True)
        d_msk.mkdir(parents=True, exist_ok=True)
        np.save(d_img / "images.npy", np.clip(imgs, 0, 255))
        np.save(d_img / "types.npy", tissues[np.arange(n_per_fold) % 3])
        np.save(d_msk / "masks.npy", masks)


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
