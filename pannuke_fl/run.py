"""Entry point.

  python -m pannuke_fl.run prep                        # convert PanNuke once
  python -m pannuke_fl.run tune     --seed 0           # Hyperband for SegNet + DenseNet121 (shared by all seeds)
  python -m pannuke_fl.run seg      --seed S           # Tables 2-3: FCN / U-Net / SegNet / ResUNet
  python -m pannuke_fl.run cls      --seed S           # Table 6 + Fig 4: five classifiers
  python -m pannuke_fl.run opt      --seed S           # Table 7 rows 1-3 + quantization modes (Fig 6 size/accuracy)
  python -m pannuke_fl.run fl       --seed S           # Table 7 rows 4-6 (5 clients, INT8)
  python -m pannuke_fl.run flsweep  --seed S           # Fig 7: 5..30 clients
  python -m pannuke_fl.run all      --seed S           # everything above for one seed
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from .utils import gpu_setup, load_config, load_json, save_json, set_seed

STAGES = ["prep", "tune", "seg", "cls", "opt", "fl", "flsweep", "requant"]


def _eb(cfg):
    return cfg.get("eval_batch", 8)


def _cb(cfg):
    return cfg.get("cls_eval_batch", 128)


def _qargs(cfg):
    q = cfg["quantization"]
    return dict(n_rep=q["representative_samples"], balanced=q.get("balanced_calibration", False))


def _tta(cfg):
    return cfg.get("tta", False)


def _done(out, stage):
    return out / f"{stage}.done.json"


def _subset(arrs, frac, seed):
    if frac >= 1.0:
        return arrs
    n = len(arrs[0])
    ix = np.sort(np.random.default_rng(seed).choice(n, max(1, int(n * frac)), replace=False))
    return tuple(a[ix] for a in arrs)


def stage_prep(cfg, data_getter, **_):
    data = data_getter()
    from .data import crop_arrays
    for part in ["train", "val", "test"]:
        x, y = crop_arrays(cfg, data, part)
        print(f"[prep] {part}: {len(data[part])} images, {len(x)} nuclei, class counts {np.bincount(y, minlength=5)}")
    return {"n_images": {p: int(len(data[p])) for p in ["train", "val", "test"]}}


def stage_tune(cfg, data_getter, root, tasks=("seg", "cls"), config_path=None, **_):
    from .data import crop_arrays, seg_arrays
    from .train import tune
    data = data_getter()
    tdir = root / "tune"
    frac, s = cfg["tuning"]["train_subset"], cfg["tuning"]["seed"]
    res = {}
    if "seg" in tasks and not (tdir / "seg_best_hp.json").exists():
        res["seg"] = tune(cfg, "seg", _subset(seg_arrays(data, "train"), frac, s), seg_arrays(data, "val"), tdir,
                         config_path=config_path)
    if "cls" in tasks and not (tdir / "cls_best_hp.json").exists():
        res["cls"] = tune(cfg, "cls", _subset(crop_arrays(cfg, data, "train"), frac, s),
                          crop_arrays(cfg, data, "val"), tdir, input_size=cfg["data"]["cls_input_size"],
                          config_path=config_path)
    return res


def stage_seg(cfg, data_getter, out, seed, models=None, **_):
    from .data import seg_arrays
    from .train import eval_seg, keras_fn, train_seg
    data = data_getter()
    tr, va, te = (seg_arrays(data, p) for p in ["train", "val", "test"])
    res = {}
    for name in models or cfg["segmentation"]["models"]:
        mdir = out / "seg" / name
        if (mdir / "metrics.json").exists():
            res[name] = load_json(mdir / "metrics.json")
            continue
        model = train_seg(cfg, tr, va, name, mdir, seed)
        res[name] = eval_seg(keras_fn(model), *te, batch=_eb(cfg), tta=_tta(cfg))
        model.save(mdir / "model.keras")
        save_json(res[name], mdir / "metrics.json")
        print(f"[seg] {name}", {k: round(v, 4) for k, v in res[name].items() if isinstance(v, float)})
    return res


def stage_cls(cfg, data_getter, out, seed, models=None, **_):
    from .data import crop_arrays
    from .train import eval_cls, keras_fn, train_cls
    data = data_getter()
    tr, va, te = (crop_arrays(cfg, data, p) for p in ["train", "val", "test"])
    size = cfg["data"]["cls_input_size"]
    res = {}
    for name in models or cfg["classification"]["models"]:
        mdir = out / "cls" / name
        if (mdir / "metrics.json").exists():
            res[name] = load_json(mdir / "metrics.json")
            continue
        model = train_cls(cfg, tr, va, name, mdir, seed)
        res[name] = eval_cls(keras_fn(model), *te, size, batch=_cb(cfg), tta=_tta(cfg))
        model.save(mdir / "model.keras")
        save_json(res[name], mdir / "metrics.json")
        print(f"[cls] {name}", {k: round(v, 4) for k, v in res[name].items() if isinstance(v, float)})
    return res


def _best_hp(root, tasks=("seg", "cls")):
    """(seg_hp, cls_hp); entries for tasks not requested are None. Returns None if any requested is missing."""
    t = root / "tune"
    out = []
    for task in ("seg", "cls"):
        f = t / f"{task}_best_hp.json"
        if task in tasks and not f.exists():
            return None
        out.append(load_json(f) if task in tasks else None)
    return tuple(out)


def stage_opt(cfg, data_getter, out, root, seed, tasks=("seg", "cls"), **_):
    """Retrain SegNet / DenseNet121 with the tuned hyperparameters, then post-training quantization."""
    import keras
    from .data import crop_arrays, seg_arrays
    from .train import (eval_cls, eval_seg, keras_fn, quantize_and_eval, split_cls_hp,
                        split_seg_hp, train_cls, train_seg)
    eb, tta = _eb(cfg), _tta(cfg)
    best = _best_hp(root, tasks)
    if best is None:
        print("[opt] tuning results not found yet; run the 'tune' stage first.")
        return None
    data = data_getter()
    size = cfg["data"]["cls_input_size"]
    modes = cfg["quantization"]["modes"]
    n_rep = cfg["quantization"]["representative_samples"]
    res = {}

    seg_tr, seg_va, seg_te = (seg_arrays(data, p) for p in ["train", "val", "test"])
    sdir = out / "opt" / "segnet"
    if "seg" in tasks and not (sdir / "metrics.json").exists():
        hp, b = split_seg_hp(best[0])
        m = train_seg(cfg, seg_tr, seg_va, "segnet", sdir, seed, hp=hp, batch=b)
        met = eval_seg(keras_fn(m), *seg_te, batch=eb, tta=tta)
        met["hp"] = best[0]
        m.save(sdir / "model.keras")
        met["quant"] = quantize_and_eval(m, "seg", modes, seg_tr[0], seg_te, sdir / "tflite",
                                         eval_batch=eb, tta=tta, **_qargs(cfg))
        save_json(met, sdir / "metrics.json")
    if "seg" in tasks:
        res["segnet"] = load_json(sdir / "metrics.json")

    cls_tr, cls_va, cls_te = ((crop_arrays(cfg, data, p) for p in ["train", "val", "test"])
                              if "cls" in tasks else (None, None, None))
    cdir = out / "opt" / "densenet121"
    if "cls" in tasks and not (cdir / "metrics.json").exists():
        hp, b, opt_name, lr = split_cls_hp(best[1])
        m = train_cls(cfg, cls_tr, cls_va, "densenet121", cdir, seed, hp=hp, batch=b, opt_name=opt_name, lr=lr)
        met = eval_cls(keras_fn(m), *cls_te, size, batch=_cb(cfg), tta=tta)
        met["hp"] = best[1]
        m.save(cdir / "model.keras")
        met["quant"] = quantize_and_eval(m, "cls", modes, cls_tr[0], cls_te, cdir / "tflite",
                                         input_size=size, tta=tta, cls_eval_batch=_cb(cfg),
                                         rep_y=cls_tr[1], **_qargs(cfg))
        save_json(met, cdir / "metrics.json")
    if "cls" in tasks:
        res["densenet121"] = load_json(cdir / "metrics.json")

    # quantization of the un-tuned baseline models too (Fig 6 compares all modes)
    for task, name in [("seg", "segnet"), ("cls", "densenet121")]:
        if task not in tasks or not cfg["quantization"].get("include_baseline", True):
            continue
        bdir = out / task / name
        qf = bdir / "tflite" / f"{name}_quant.json"
        if (bdir / "model.keras").exists() and not qf.exists():
            m = keras.models.load_model(bdir / "model.keras")
            if task == "seg":
                quantize_and_eval(m, "seg", modes, seg_tr[0], seg_te, bdir / "tflite",
                                  eval_batch=eb, tta=tta, **_qargs(cfg))
            else:
                quantize_and_eval(m, "cls", modes, cls_tr[0], cls_te, bdir / "tflite", input_size=size,
                                  tta=tta, cls_eval_batch=_cb(cfg), rep_y=cls_tr[1], **_qargs(cfg))
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("quant", "hp")} for k, v in res.items()}


def _fl_task(cfg, data, task, best, algorithm, n_clients, seed, state_dir):
    import tensorflow as tf
    from .cls_models import build_cls
    from .data import crop_arrays, partition_clients, seg_arrays
    from .federated import run_federated
    from .recipe import make_optimizer
    from .seg_models import build_seg
    from .train import (TFLiteFn, cls_dataset, cls_loss, eval_cls, eval_seg, keras_fn, seg_dataset,
                        seg_loss, split_cls_hp, split_seg_hp, to_tflite)
    f = cfg["federated"]
    size = cfg["data"]["cls_input_size"]
    eb, tta = _eb(cfg), _tta(cfg)
    tissue_tr = data["tissue"][data["train"]]
    img_parts = partition_clients(len(data["train"]), n_clients, seed, tissue_tr, f["partition"])
    if task == "seg":
        c = cfg["segmentation"]
        hp, batch = split_seg_hp(best[0])
        tr, va, te = (seg_arrays(data, p) for p in ["train", "val", "test"])
        parts = img_parts
        dss = [seg_dataset(tr[0][p], tr[1][p], batch, shuffle=True, seed=seed + k, augment=c.get("augment", False))
               for k, p in enumerate(parts)]
        build = lambda: build_seg("segnet", hp=hp, pretrained=c["pretrained_encoder"])
        make_opt = lambda: make_optimizer(c["optimizer"], f.get("learning_rate", c["learning_rate"]),
                                          c.get("momentum", 0.0))
        loss = seg_loss(c)
        val_eval = lambda m: {k: v for k, v in eval_seg(keras_fn(m), *va, batch=eb).items()
                              if k in ("MPA", "CPA_nucleus")}
    else:
        c = cfg["classification"]
        hp, batch, opt_name, lr = split_cls_hp(best[1])
        tr, va, te = (crop_arrays(cfg, data, p) for p in ["train", "val", "test"])
        # crops inherit the client of their source image
        img_of_crop = _crop_image_index(cfg, data, "train")
        parts = [np.nonzero(np.isin(img_of_crop, p))[0] for p in img_parts]
        dss = [cls_dataset(tr[0][p], tr[1][p], batch, size, shuffle=True, seed=seed + k,
                           augment=c.get("augment", False)) for k, p in enumerate(parts)]
        build = lambda: build_cls("densenet121", size, hp=hp, pretrained=c["pretrained"])
        make_opt = lambda: make_optimizer(opt_name, lr, c.get("momentum", 0.0) if opt_name == "sgd" else 0.0)
        loss = cls_loss(c, tr[1])
        val_eval = lambda m: {k: v for k, v in eval_cls(keras_fn(m), *va, size, batch=_cb(cfg)).items() if k in ("ACC", "F1")}

    model, gw, cbn, client_weights, hist = run_federated(
        build, make_opt, loss, dss, [len(p) for p in parts], algorithm, f["rounds"], f["local_epochs"],
        f["fedprox_mu"], val_eval, state_dir, eval_every=f.get("eval_every", 5))

    # final test evaluation, float and INT8 (Table 7 reports the INT8-quantized models)
    from .train import calibration_set
    qa = _qargs(cfg)
    rep = calibration_set(task, tr[0], qa["n_rep"], size, tr[1] if task == "cls" else None, qa["balanced"])
    models_w = [gw] if cbn is None else [client_weights(k) for k in range(len(cbn))]
    per = []
    for w in models_w:
        model.set_weights(w)
        q = TFLiteFn(to_tflite(model, "int8", rep, workdir=str(state_dir)))
        if task == "seg":
            per.append({"float32": eval_seg(keras_fn(model), *te, batch=eb, tta=tta),
                        "int8": eval_seg(q, *te, batch=eb, tta=tta)})
        else:
            per.append({"float32": eval_cls(keras_fn(model), *te, size, batch=_cb(cfg), tta=tta),
                        "int8": eval_cls(q, *te, size, batch=_cb(cfg), tta=tta)})
    keys = ["MPA", "CPA_nucleus", "MIoU", "FWIoU"] if task == "seg" else ["ACC", "F1", "MCC", "SN", "SP"]
    final = {prec: {k: float(np.mean([p[prec][k] for p in per])) for k in keys} for prec in ["float32", "int8"]}
    final["history"] = hist
    if algorithm != "fedbn":
        model.set_weights(gw)
        model.save(Path(state_dir) / "global_model.keras")
    return final


def _crop_image_index(cfg, data, part):
    """Image index (position within data[part]) for every crop, in crop_arrays order."""
    ix = data[part]
    inst = data["inst"][ix]
    return np.concatenate([np.full(len(np.setdiff1d(np.unique(inst[k]), [0])), k) for k in range(len(ix))])


def stage_requant(cfg, data_getter, out, seed, tasks=("seg", "cls"), **_):
    """Re-run post-training quantization of already trained models with the current calibration
    settings (optimized models and FedAvg/FedProx global models). Previous results are kept as *_prev."""
    import keras
    from .data import crop_arrays, seg_arrays
    from .train import TFLiteFn, calibration_set, eval_cls, eval_seg, quantize_and_eval, to_tflite
    data = data_getter()
    size, eb, tta, qa = cfg["data"]["cls_input_size"], _eb(cfg), _tta(cfg), _qargs(cfg)
    modes = cfg["quantization"]["modes"]
    res = {}
    for task, name in [("seg", "segnet"), ("cls", "densenet121")]:
        if task not in tasks:
            continue
        tr = seg_arrays(data, "train") if task == "seg" else crop_arrays(cfg, data, "train")
        te = seg_arrays(data, "test") if task == "seg" else crop_arrays(cfg, data, "test")
        mdir = out / "opt" / name
        if (mdir / "model.keras").exists():
            met = load_json(mdir / "metrics.json")
            if "quant" in met and "quant_prev" not in met:
                met["quant_prev"] = met["quant"]
            m = keras.models.load_model(mdir / "model.keras")
            met["quant"] = quantize_and_eval(m, task, modes, tr[0], te, mdir / "tflite", input_size=size,
                                             eval_batch=eb, tta=tta, cls_eval_batch=_cb(cfg),
                                             rep_y=tr[1] if task == "cls" else None, **qa)
            met["calibration"] = qa
            save_json(met, mdir / "metrics.json")
            res[f"opt_{name}_int8"] = {k: v for k, v in met["quant"]["int8"].items() if isinstance(v, float)}
        rep = calibration_set(task, tr[0], qa["n_rep"], size, tr[1] if task == "cls" else None, qa["balanced"])
        for f in sorted((out / "fl").glob(f"*_{task}_c*/global_model.keras")):
            fin = load_json(f.parent / "final.json")
            if "int8_prev" not in fin:
                fin["int8_prev"] = fin["int8"]
            m = keras.models.load_model(f)
            q = TFLiteFn(to_tflite(m, "int8", rep, workdir=str(f.parent)))
            r = eval_seg(q, *te, batch=eb, tta=tta) if task == "seg" else eval_cls(q, *te, size, batch=_cb(cfg), tta=tta)
            keys = ["MPA", "CPA_nucleus", "MIoU", "FWIoU"] if task == "seg" else ["ACC", "F1", "MCC", "SN", "SP"]
            fin["int8"] = {k: float(r[k]) for k in keys}
            fin["calibration"] = qa
            save_json(fin, f.parent / "final.json")
            res[f.parent.name + "_int8"] = fin["int8"]
            print(f"[requant] {f.parent.name} int8:", {k: round(v, 4) for k, v in fin["int8"].items()})
    return res


def stage_fl(cfg, data_getter, out, root, seed, sweep=False, tasks=("seg", "cls"), **_):
    best = _best_hp(root, tasks)
    if best is None:
        print("[fl] tuning results not found yet; run the 'tune' stage first.")
        return None
    data = data_getter()
    f = cfg["federated"]
    clients = f["client_sweep"] if sweep else [f["clients"]]
    res = {}
    for n in clients:
        for alg in f["algorithms"]:
            for task in [t for t in ("seg", "cls") if t in tasks]:
                name = f"{alg}_{task}_c{n}"
                d = out / ("flsweep" if sweep else "fl") / name
                if (d / "final.json").exists():
                    res[name] = load_json(d / "final.json")
                    continue
                r = _fl_task(cfg, data, task, best, alg, n, seed, d)
                save_json(r, d / "final.json")
                res[name] = r
    return {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in res.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models for seg/cls")
    ap.add_argument("--tasks", nargs="+", choices=["seg", "cls"], default=["seg", "cls"],
                    help="restrict tune/opt/fl to segmentation (SegNet) or classification (DenseNet121)")
    ap.add_argument("--deterministic", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    from . import train as _train
    _train.JIT = cfg.get("jit_compile", "auto")
    gpu_setup()
    set_seed(a.seed, a.deterministic)
    root = Path(a.results)
    out = root / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_json({"config": cfg, "seed": a.seed, "argv": sys.argv}, out / "run_config.json")

    cache = {}

    def data_getter():
        if "d" not in cache:
            from .data import load_split
            cache["d"] = load_split(cfg)
        return cache["d"]

    fns = {"prep": stage_prep, "tune": stage_tune, "seg": stage_seg, "cls": stage_cls,
           "opt": stage_opt, "fl": stage_fl,
           "flsweep": lambda **kw: stage_fl(sweep=True, **kw), "requant": stage_requant}
    stages = STAGES if a.stage == "all" else [a.stage]
    tasks = tuple(a.tasks)
    tag = "" if set(tasks) == {"seg", "cls"} else "_" + "_".join(tasks)
    for st in stages:
        if a.stage == "all" and st == "requant":
            continue
        per_task = st in ("tune", "opt", "fl", "flsweep", "requant")
        name = st + (tag if per_task else "")
        marker = root / f"{name}.done.json" if st == "tune" else _done(out, name)
        if marker.exists():
            print(f"[{st}] already done ({marker})")
            continue
        set_seed(a.seed, a.deterministic)
        r = fns[st](cfg=cfg, data_getter=data_getter, out=out, root=root, seed=a.seed, models=a.models,
                    tasks=tasks, config_path=str(Path(a.config).resolve()))
        if r is None:
            if a.stage == "all":
                continue
            return 0              # not ready yet (e.g. waiting for tuning); a chained job retries
        save_json(r, marker)
        print(f"[{st}] done -> {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
