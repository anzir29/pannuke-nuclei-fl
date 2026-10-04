"""PanNuke loading, the fixed 80/20 split, and nucleus crops for the classifier.

Kaggle layout (theredlad/pannuke-dataset-experimental-data): each fold has
images.npy (N,256,256,3), masks.npy (N,256,256,6), types.npy (N,).
Mask channels: 0 neoplastic, 1 inflammatory, 2 connective, 3 dead, 4 epithelial, 5 background;
channels 0-4 hold instance ids.
"""
import os
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

N_TYPES = 5


def _find_fold_files(root):
    root = Path(root)
    imgs = sorted(root.rglob("images.npy"))
    if not imgs:
        raise FileNotFoundError(f"No images.npy under {root}")
    folds = []
    for img in imgs:
        fold_dir_token = [p for p in img.parts if "fold" in p.lower()]
        token = fold_dir_token[-1].lower().replace(" ", "") if fold_dir_token else None
        cands = [m for m in root.rglob("masks.npy")
                 if token is None or token in "/".join(m.parts).lower().replace(" ", "")]
        typs = [t for t in root.rglob("types.npy")
                if token is None or token in "/".join(t.parts).lower().replace(" ", "")]
        if len(cands) != 1 or len(typs) != 1:
            raise RuntimeError(f"Could not match masks/types for {img}: {cands} {typs}")
        folds.append((img, cands[0], typs[0]))
    return folds


def _masks_to_maps(masks):
    """(n,H,W,6) PanNuke masks -> instance map int32 and type map uint8 (0=bg, 1..5)."""
    n, h, w, _ = masks.shape
    inst = np.zeros((n, h, w), np.int32)
    typ = np.zeros((n, h, w), np.uint8)
    for c in range(N_TYPES):
        ch = np.asarray(masks[..., c]).astype(np.int32)
        sel = ch > 0
        inst[sel] = ch[sel] + (c + 1) * 100000   # unique ids across channels
        typ[sel] = c + 1
    return inst, typ


def preprocess(root, cache, chunk=256):
    """Convert the raw float64 masks into compact arrays once (images, inst, type, tissue)."""
    cache = Path(cache)
    out = cache / "pannuke.npz"
    if out.exists():
        return out
    cache.mkdir(parents=True, exist_ok=True)
    images, insts, typs, tissues = [], [], [], []
    for img_f, msk_f, typ_f in _find_fold_files(root):
        im = np.load(img_f, mmap_mode="r")
        mk = np.load(msk_f, mmap_mode="r")
        ty = np.load(typ_f, allow_pickle=True)
        for s in range(0, len(im), chunk):
            images.append(np.asarray(im[s:s + chunk]).astype(np.uint8))
            i_map, t_map = _masks_to_maps(mk[s:s + chunk])
            insts.append(i_map)
            typs.append(t_map)
        tissues.append(np.asarray(ty).astype(str))
    tmp = cache / f"pannuke.tmp{os.getpid()}.npz"      # per-process temp file: safe if two runs overlap
    np.savez(tmp, images=np.concatenate(images), inst=np.concatenate(insts),
             type=np.concatenate(typs), tissue=np.concatenate(tissues))
    os.replace(tmp, out)
    return out


def load_split(cfg):
    """Returns dict with train/val/test index arrays plus the arrays themselves."""
    d = cfg["data"]
    path = preprocess(d["root"], d["cache"])
    z = np.load(path)
    images, inst, typ, tissue = z["images"], z["inst"], z["type"], z["tissue"]
    idx = np.arange(len(images))
    tr, te = train_test_split(idx, test_size=d["test_fraction"], stratify=tissue,
                              random_state=d["split_seed"])
    tr, va = train_test_split(tr, test_size=d["val_fraction"], stratify=tissue[tr],
                              random_state=d["split_seed"])
    return dict(images=images, inst=inst, type=typ, tissue=tissue,
                train=np.sort(tr), val=np.sort(va), test=np.sort(te))


def seg_arrays(data, part):
    ix = data[part]
    x = data["images"][ix]
    y = (data["type"][ix] > 0).astype(np.uint8)       # binary: nucleus vs background
    return x, y


def extract_crops(images, inst, typ, crop=64):
    """One crop per nucleus centred on its centroid; label = nucleus type 0..4."""
    half = crop // 2
    xs, ys = [], []
    for k in range(len(images)):
        img = np.pad(images[k], ((half, half), (half, half), (0, 0)), mode="reflect")
        ids = np.unique(inst[k])
        ids = ids[ids > 0]
        for i in ids:
            m = inst[k] == i
            rr, cc = np.nonzero(m)
            cy, cx = int(rr.mean()) + half, int(cc.mean()) + half
            xs.append(img[cy - half:cy + half, cx - half:cx + half])
            ys.append(int(np.bincount(typ[k][m]).argmax()) - 1)
    if not xs:
        return np.zeros((0, crop, crop, 3), np.uint8), np.zeros((0,), np.int64)
    return np.stack(xs).astype(np.uint8), np.asarray(ys, np.int64)


def crop_arrays(cfg, data, part):
    cache = Path(cfg["data"]["cache"]) / f"crops_{part}_s{cfg['data']['split_seed']}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    ix = data[part]
    x, y = extract_crops(data["images"][ix], data["inst"][ix], data["type"][ix],
                         cfg["data"]["crop_size"])
    tmp = cache.with_name(cache.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, x=x, y=y)
    os.replace(tmp, cache)
    return x, y


def partition_clients(n, n_clients, seed, tissue=None, mode="iid"):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if mode == "iid" or tissue is None:
        rng.shuffle(idx)
        return [np.sort(p) for p in np.array_split(idx, n_clients)]
    # tissue-based non-IID: tissues assigned round-robin to clients
    tissues = np.unique(tissue)
    rng.shuffle(tissues)
    parts = [[] for _ in range(n_clients)]
    for j, t in enumerate(tissues):
        parts[j % n_clients].extend(idx[tissue == t].tolist())
    return [np.sort(np.asarray(p)) for p in parts]
