"""Collect results/seed*/ into CSVs (mean +- std over seeds) for every table and figure.

  python scripts/make_tables.py --results results --out results/tables
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEG = ["fcn", "unet", "segnet", "resunet"]
SEG_NAMES = {"fcn": "FCN", "unet": "U-Net", "segnet": "SegNet", "resunet": "ResUnet"}
CLS = ["vgg16", "vgg19", "resnet50", "densenet121", "inceptionv2"]
CLS_NAMES = {"vgg16": "VGG16", "vgg19": "VGG19", "resnet50": "ResNet50",
             "densenet121": "DenseNet121", "inceptionv2": "InceptionV2"}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def seeds(root):
    return sorted(int(p.name[4:]) for p in Path(root).glob("seed*") if p.name[4:].isdigit())


def summarise(rows, keys, group, pct=False):
    """rows: list of dicts with `group` columns, 'seed' and metric keys -> mean/std table."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    scale = 100.0 if pct else 1.0
    out = []
    for g, sub in df.groupby(group, sort=False):
        rec = dict(zip(group, g if isinstance(g, tuple) else (g,)))
        rec["n_seeds"] = len(sub)
        for k in keys:
            v = sub[k].astype(float) * scale
            rec[f"{k}_mean"] = v.mean()
            rec[f"{k}_std"] = v.std(ddof=1) if len(v) > 1 else 0.0
            rec[k] = f"{v.mean():.{1 if pct else 3}f} ± {rec[f'{k}_std']:.{1 if pct else 3}f}"
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    S = seeds(R)
    written = {}

    # ---- Tables 2-3: segmentation comparison
    rows = []
    for s in S:
        for m in SEG:
            r = load(R / f"seed{s}/seg/{m}/metrics.json")
            if r:
                rows.append({"Model": SEG_NAMES[m], "seed": s, **{k: r[k] for k in
                             ["MPA", "MIoU", "FWIoU", "PA", "CPA_nucleus", "IoU_nucleus", "PA_nucleus"]}})
    pd.DataFrame(rows).to_csv(O / "raw_segmentation_per_seed.csv", index=False)
    written["table2_segmentation.csv"] = summarise(rows, ["MPA", "MIoU", "FWIoU"], ["Model"], pct=True)
    written["table3_cpa_iou.csv"] = summarise(rows, ["CPA_nucleus", "IoU_nucleus"], ["Model"], pct=True)

    # ---- Table 6 + Fig 4: classifier comparison
    rows, cw = [], []
    for s in S:
        for m in CLS:
            r = load(R / f"seed{s}/cls/{m}/metrics.json")
            if r:
                rows.append({"Model": CLS_NAMES[m], "seed": s, **{k: r[k] for k in ["SP", "SN", "F1", "MCC", "ACC"]}})
                for c, v in r["per_class"].items():
                    cw.append({"Model": CLS_NAMES[m], "Class": c, "seed": s, **v})
    pd.DataFrame(rows).to_csv(O / "raw_classification_per_seed.csv", index=False)
    written["table6_classification.csv"] = summarise(rows, ["SP", "SN", "F1", "MCC", "ACC"], ["Model"])
    written["fig4_classwise.csv"] = summarise(cw, ["ACC", "F1", "MCC", "SP", "SN"], ["Model", "Class"])

    # ---- Table 7: centralized / optimized / quantized / federated (INT8)
    rows = []
    for s in S:
        seg_b = load(R / f"seed{s}/seg/segnet/metrics.json")
        cls_b = load(R / f"seed{s}/cls/densenet121/metrics.json")
        seg_o = load(R / f"seed{s}/opt/segnet/metrics.json")
        cls_o = load(R / f"seed{s}/opt/densenet121/metrics.json")

        def add(name, sg, cl):
            if sg or cl:
                nan = float("nan")
                rows.append({"Method": name, "seed": s,
                             "SegNet_MPA": sg["MPA"] if sg else nan, "SegNet_CPA": sg["CPA_nucleus"] if sg else nan,
                             "DenseNet121_ACC": cl["ACC"] if cl else nan, "DenseNet121_F1": cl["F1"] if cl else nan})
        add("Centralized Learning", seg_b, cls_b)
        add("Centralized Learning (Optimized)", seg_o, cls_o)
        add("Centralized Learning (Quantized)", seg_o["quant"]["int8"] if seg_o else None,
            cls_o["quant"]["int8"] if cls_o else None)
        for alg, label in [("fedavg", "FedAvg"), ("fedprox", "FedProx"), ("fedbn", "FedBN")]:
            fdir = R / f"seed{s}/fl"
            sg = load(next(iter(sorted(fdir.glob(f"{alg}_seg_c*/final.json"))), Path("/nonexistent")))
            cl = load(next(iter(sorted(fdir.glob(f"{alg}_cls_c*/final.json"))), Path("/nonexistent")))
            add(label, sg["int8"] if sg else None, cl["int8"] if cl else None)
    t7 = summarise(rows, ["SegNet_MPA", "SegNet_CPA"], ["Method"], pct=True)
    t7b = summarise(rows, ["DenseNet121_ACC", "DenseNet121_F1"], ["Method"])
    if not t7.empty:
        t7 = t7.merge(t7b.drop(columns=["n_seeds"]), on="Method")
    written["table7_federated_int8.csv"] = t7

    # ---- Fig 5: Hyperband trials
    for task, metric in [("seg", "MPA"), ("cls", "Accuracy")]:
        tr = load(R / f"tune/{task}_trials.json")
        if tr:
            written[f"fig5_hyperband_{task}_trials.csv"] = pd.DataFrame(
                [{"trial": t["trial_id"], "status": t["status"], "epochs": t["epochs"],
                  f"val_{metric}": t["score"], **{f"hp_{k}": v for k, v in t["hp"].items()
                                                  if not k.startswith("tuner/")}} for t in tr])

    # ---- Fig 6 (size / accuracy part): quantization modes
    rows = []
    for s in S:
        for variant, base in [("baseline", "seg/segnet"), ("optimized", "opt/segnet")]:
            q = load(R / f"seed{s}/{base}/tflite/segnet_quant.json")
            for mode, v in (q or {}).items():
                rows.append({"Model": "SegNet", "Variant": variant, "Mode": mode, "seed": s,
                             "Size_MB": v["size_mb"], "Metric": v["MPA"]})
        for variant, base in [("baseline", "cls/densenet121"), ("optimized", "opt/densenet121")]:
            q = load(R / f"seed{s}/{base}/tflite/densenet121_quant.json")
            for mode, v in (q or {}).items():
                rows.append({"Model": "DenseNet121", "Variant": variant, "Mode": mode, "seed": s,
                             "Size_MB": v["size_mb"], "Metric": v["ACC"]})
    written["fig6_quantization_size_accuracy.csv"] = summarise(rows, ["Size_MB", "Metric"], ["Model", "Variant", "Mode"])

    # ---- Fig 7: number of clients
    rows = []
    for s in S:
        for f in sorted((R / f"seed{s}/flsweep").glob("*/final.json")):
            alg, task, c = f.parent.name.split("_")
            r = load(f)
            for prec in ["float32", "int8"]:
                rows.append({"Algorithm": alg, "Model": "SegNet" if task == "seg" else "DenseNet121",
                             "Clients": int(c[1:]), "Precision": prec, "seed": s,
                             "Score": r[prec]["MPA" if task == "seg" else "ACC"]})
    written["fig7_clients.csv"] = summarise(rows, ["Score"], ["Algorithm", "Model", "Clients", "Precision"])

    md = []
    for name, df in written.items():
        if df is None or df.empty:
            print(f"(skipped {name}: no results yet)")
            continue
        df.to_csv(O / name, index=False)
        show = [c for c in df.columns if not c.endswith(("_mean", "_std")) and c != "seed"]
        md.append(f"## {name}\n\n" + df[show].to_markdown(index=False) + "\n")
        print(f"wrote {O / name}")
    (O / "summary.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
