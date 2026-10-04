"""Runs one Hyperband trial in its own process (called by train.tune).

  python -m pannuke_fl.tune_worker <trial_dir>/job.json
"""
import json
import math
import sys


def main(job_path):
    job = json.load(open(job_path))
    from .utils import gpu_setup, load_config, save_json, set_seed
    cfg = load_config(job["config"])
    from . import train as T
    T.JIT = cfg.get("jit_compile", "auto")
    gpu_setup()
    set_seed(job["seed"])
    import keras
    import numpy as np
    z = np.load(job["data"])
    task, hpv, size = job["task"], job["hp"], job["input_size"]
    model = T.build_trial_model(cfg, task, hpv, size, y_train=z["y_tr"] if task == "cls" else None)
    if job["load"]:
        model.load_weights(job["load"])
    b = int(hpv["batch_size"])
    import os
    if os.environ.get("PFL_TEST_FAIL_BATCH") == str(b):      # test hook: simulate an out-of-memory trial
        sys.exit(3)
    if task == "seg":
        aug = cfg["segmentation"].get("augment", False)
        tr = T.seg_dataset(z["x_tr"], z["y_tr"], b, shuffle=True, seed=job["seed"], augment=aug)
        va = T.seg_dataset(z["x_va"], z["y_va"], b)
        key = "val_mpa"
    else:
        aug = cfg["classification"].get("augment", False)
        tr = T.cls_dataset(z["x_tr"], z["y_tr"], b, size, shuffle=True, seed=job["seed"], augment=aug)
        va = T.cls_dataset(z["x_va"], z["y_va"], b, size)
        key = "val_accuracy"
    h = model.fit(tr, validation_data=va, epochs=job["epochs"], initial_epoch=job["initial_epoch"],
                  callbacks=[keras.callbacks.TerminateOnNaN()], verbose=2)
    vals = [v for v in h.history.get(key, []) if v is not None and not math.isnan(v)]
    best = max(vals) if vals else 0.0          # a diverged run scores 0
    model.save_weights(job["save"])
    save_json({"best": float(best), "history": h.history}, job["result"])


if __name__ == "__main__":
    main(sys.argv[1])
