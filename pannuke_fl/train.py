"""Centralized training, evaluation, Hyperband tuning and post-training quantization."""
import math
import os
import shutil
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from . import CLASS_NAMES
from .cls_models import build_cls
from .metrics import MeanPixelAccuracy, cls_metrics, seg_confusion, seg_metrics
from .recipe import (CEDiceLoss, SmoothedWeightedCE, augment_cls, augment_seg, class_weights,
                     make_optimizer, tta_cls, tta_seg)
from .seg_models import build_seg
from .utils import save_json

AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------------------------------------------ datasets
def seg_dataset(x, y, batch, shuffle=False, seed=0, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, tf.cast(b, tf.int32)),
                num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_seg, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def cls_dataset(x, y, batch, size, shuffle=False, seed=0, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 200_000), seed=seed, reshuffle_each_iteration=True)

    def prep(a, b):
        a = tf.image.resize(tf.cast(a, tf.float32) / 255.0, (size, size))
        return a, tf.cast(b, tf.int32)
    ds = ds.map(prep, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_cls, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


# ------------------------------------------------------------------ losses / compile
def seg_loss(c):
    return CEDiceLoss(c.get("dice_weight", 1.0)) if c.get("loss", "ce") == "ce_dice" \
        else keras.losses.SparseCategoricalCrossentropy()


def cls_loss(c, y_train=None):
    w = class_weights(y_train, c.get("class_weight")) if y_train is not None else None
    return SmoothedWeightedCE(5, c.get("label_smoothing", 0.0), w)


JIT = "auto"      # set from config `jit_compile` (auto | true | false); "auto" = XLA on GPU


def compile_seg(model, opt, loss):
    model.compile(optimizer=opt, loss=loss, metrics=[MeanPixelAccuracy(name="mpa")], jit_compile=JIT)


def compile_cls(model, opt, loss):
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"], jit_compile=JIT)


def recipe_optimizer(c, n_train, batch, name=None, lr=None):
    steps = math.ceil(n_train / batch)
    return make_optimizer(name or c["optimizer"], lr or c["learning_rate"], c.get("momentum", 0.0),
                          c.get("schedule"), steps * c["epochs"], steps * c.get("warmup_epochs", 0),
                          c.get("weight_decay"))


# ------------------------------------------------------------------ evaluation
def predict_seg(fn, x, batch=8):
    preds = []
    for s in range(0, len(x), batch):
        p = fn(x[s:s + batch].astype(np.float32) / 255.0)
        preds.append(np.argmax(np.asarray(p), -1).astype(np.uint8))
    return np.concatenate(preds)


def eval_seg(fn, x, y, batch=8, tta=False):
    fn = tta_seg(fn) if tta else fn
    cm = np.zeros((2, 2), np.int64)
    for s in range(0, len(x), batch):
        cm += seg_confusion(y[s:s + batch], predict_seg(fn, x[s:s + batch], batch))
    return seg_metrics(cm)


def predict_cls(fn, x, size, batch=256):
    out = []
    for s in range(0, len(x), batch):
        a = tf.image.resize(x[s:s + batch].astype(np.float32) / 255.0, (size, size)).numpy()
        out.append(np.argmax(np.asarray(fn(a)), -1))
    return np.concatenate(out)


def eval_cls(fn, x, y, size, batch=128, tta=False):
    fn = tta_cls(fn) if tta else fn
    return cls_metrics(y, predict_cls(fn, x, size, batch), 5, CLASS_NAMES)


def keras_fn(model):
    """Compiled inference function, cached on the model so repeated evaluations don't retrace."""
    fn = getattr(model, "_pfl_infer", None)
    if fn is None:
        @tf.function(reduce_retracing=True)
        def f(a):
            return model(a, training=False)
        fn = lambda a: f(tf.convert_to_tensor(a, tf.float32)).numpy()
        model._pfl_infer = fn
    return fn


# ------------------------------------------------------------------ training
def fit(model, train_ds, val_ds, epochs, monitor, out_dir, patience):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cbs = [
        keras.callbacks.BackupAndRestore(str(out_dir / "backup")),   # resume after walltime kill
        keras.callbacks.ModelCheckpoint(str(out_dir / "best.weights.h5"), monitor=monitor,
                                        mode="max", save_best_only=True, save_weights_only=True),
        keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=patience),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(str(out_dir / "history.csv"), append=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2)
    model.load_weights(str(out_dir / "best.weights.h5"))
    shutil.rmtree(out_dir / "backup", ignore_errors=True)
    return model


def train_seg(cfg, data_tr, data_va, name, out_dir, seed, hp=None, batch=None):
    c = cfg["segmentation"]
    keras.backend.clear_session()
    model = build_seg(name, hp=hp, pretrained=c["pretrained_encoder"])
    b = batch or c["batch_size"]
    compile_seg(model, recipe_optimizer(c, len(data_tr[0]), b), seg_loss(c))
    fit(model, seg_dataset(*data_tr, b, shuffle=True, seed=seed, augment=c.get("augment", False)),
        seg_dataset(*data_va, b), c["epochs"], "val_mpa", out_dir, c["early_stopping_patience"])
    return model


def train_cls(cfg, data_tr, data_va, name, out_dir, seed, hp=None, batch=None, opt_name=None, lr=None):
    c = cfg["classification"]
    size = cfg["data"]["cls_input_size"]
    keras.backend.clear_session()
    head = dict(c["head"])
    head.update(hp or {})
    model = build_cls(name, size, hp=head, pretrained=c["pretrained"])
    b = batch or c["batch_size"]
    compile_cls(model, recipe_optimizer(c, len(data_tr[0]), b, opt_name, lr), cls_loss(c, data_tr[1]))
    fit(model, cls_dataset(*data_tr, b, size, shuffle=True, seed=seed, augment=c.get("augment", False)),
        cls_dataset(*data_va, b, size), c["epochs"], "val_accuracy", out_dir,
        c["early_stopping_patience"])
    return model


# ------------------------------------------------------------------ Hyperband (Table 1)
SEG_SPACE = dict(
    filters=[32, 64, 128, 256],
    filter_len=[10, 15, 20, 25, 30],
    filter_wid=[2, 4, 8, 16],
    init=["uniform", "normal", "zero", "he_uniform"],
    activation=["softmax", "relu", "sigmoid", "tanh"],
    fc_neurons=[16, 32, 64, 128],
    dropout=[0.05, 0.1, 0.15, 0.2, 0.25],
)
CLS_SPACE = dict(
    units=[16, 32, 64, 128, 256],
    learning_rate=[1e-4, 1e-3, 1e-2],
    optimizer=["rmsprop", "adam", "sgd"],
    regularizer=["l1", "l2"],
    out_activation=["softmax", "relu", "tanh"],
    dense_units=[64, 128, 256],
    dropout=[0.0, 0.2, 0.3],
)
BATCH_SPACE = [32, 64, 128]


def _space(hp, space, restrict):
    # restrict=True keeps one value per hyperparameter; used only by the CPU smoke test
    return {k: hp.Choice(k, v[:1] if restrict else v) for k, v in space.items()}


def build_trial_model(cfg, task, hpv, input_size=None, y_train=None,
                      seg_name="segnet", cls_name="densenet121"):
    """Compiled model for one Hyperband configuration (plain dict of hyperparameter values)."""
    if task == "seg":
        c = cfg["segmentation"]
        m = build_seg(seg_name, hp={k: hpv[k] for k in SEG_SPACE}, pretrained=c["pretrained_encoder"])
        compile_seg(m, make_optimizer(c["optimizer"], c["learning_rate"], c.get("momentum", 0.0)), seg_loss(c))
    else:
        c = cfg["classification"]
        s = {k: hpv[k] for k in CLS_SPACE}
        m = build_cls(cls_name, input_size, hp=s, pretrained=c["pretrained"])
        mom = c.get("momentum", 0.0) if s["optimizer"] == "sgd" else 0.0
        compile_cls(m, make_optimizer(s["optimizer"], s["learning_rate"], mom), cls_loss(c, y_train))
    return m


def _search_space(cfg, task):
    t = cfg["tuning"]
    space = {k: list(v) for k, v in (SEG_SPACE if task == "seg" else CLS_SPACE).items()}
    if t.get("include_defaults", False):
        # optionally add the untuned baseline configuration to the candidates
        extra = (dict(filters=64, filter_len=3, filter_wid=3, init="he_uniform", activation="relu")
                 if task == "seg" else {})
        for k, v in extra.items():
            if v not in space[k]:
                space[k].append(v)
    if t.get("restrict_space_for_testing", False):
        space = {k: v[:1] for k, v in space.items()}
    return space


def tune(cfg, task, train, val, out_dir, input_size=None, config_path=None):
    """Hyperband (Keras Tuner) where every trial trains in a fresh subprocess.

    Running each trial in its own process means an out-of-memory configuration fails only that
    trial, and GPU memory / host RAM are fully released between trials.
    """
    import subprocess
    import sys
    import keras_tuner as kt
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")          # the coordinating process never touches the GPU
    t = cfg["tuning"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_f = out_dir / f"{task}_tune_data.npz"
    if not data_f.exists():
        tmp = out_dir / f"{task}_tune_data.tmp{os.getpid()}.npz"
        np.savez(tmp, x_tr=train[0], y_tr=train[1], x_va=val[0], y_va=val[1])
        os.replace(tmp, data_f)
    space = _search_space(cfg, task)
    restrict = t.get("restrict_space_for_testing", False)
    batches = t.get("batch_sizes", BATCH_SPACE)
    objective_name = "val_mpa" if task == "seg" else "val_accuracy"

    class HM(kt.HyperModel):
        def build(self, hp):           # only declares the search space in this process
            for k, v in space.items():
                hp.Choice(k, v)
            hp.Choice("batch_size", batches)
            return keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])

    class SubprocessHyperband(kt.Hyperband):
        def run_trial(self, trial, *a, **kw):
            hpv = dict(trial.hyperparameters.values)
            tdir = Path(self.get_trial_dir(trial.trial_id))
            tdir.mkdir(parents=True, exist_ok=True)
            load = None
            if "tuner/trial_id" in hpv:
                cand = Path(self.get_trial_dir(hpv["tuner/trial_id"])) / "w.weights.h5"
                load = str(cand) if cand.exists() else None
            job = {"config": str(config_path), "task": task, "hp": hpv, "data": str(data_f),
                   "epochs": int(hpv.get("tuner/epochs", t["max_epochs"])),
                   "initial_epoch": int(hpv.get("tuner/initial_epoch", 0)),
                   "load": load, "save": str(tdir / "w.weights.h5"), "input_size": input_size,
                   "seed": t["seed"], "result": str(tdir / "result.json")}
            save_json(job, tdir / "job.json")
            r = subprocess.run([sys.executable, "-u", "-m", "pannuke_fl.tune_worker", str(tdir / "job.json")])
            res = tdir / "result.json"
            if r.returncode != 0 or not res.exists():
                raise RuntimeError(f"trial {trial.trial_id} failed in worker (exit code {r.returncode}); "
                                   f"usually out of GPU memory for this configuration")
            import json
            score = json.loads(res.read_text())["best"]
            return {objective_name: score}

    tuner = SubprocessHyperband(HM(), objective=kt.Objective(objective_name, "max"),
                                max_epochs=t["max_epochs"], factor=t["factor"],
                                hyperband_iterations=t["hyperband_iterations"], seed=t["seed"],
                                directory=str(out_dir), project_name=task, overwrite=False,
                                max_consecutive_failed_trials=200)
    tuner.search(verbose=1)
    best = tuner.get_best_hyperparameters(1)[0].values
    trials = []
    for tr in tuner.oracle.trials.values():
        trials.append({"trial_id": tr.trial_id, "status": str(tr.status), "score": tr.score,
                       "epochs": tr.hyperparameters.values.get("tuner/epochs"),
                       "hp": tr.hyperparameters.values})
    save_json(trials, out_dir / f"{task}_trials.json")     # data behind Fig 5
    save_json(best, out_dir / f"{task}_best_hp.json")
    return best


def split_seg_hp(best):
    hp = {k: best[k] for k in SEG_SPACE}
    return hp, best["batch_size"]


def split_cls_hp(best):
    hp = {k: best[k] for k in ["units", "dense_units", "dropout", "regularizer", "out_activation"]}
    return hp, best["batch_size"], best["optimizer"], best["learning_rate"]


# ------------------------------------------------------------------ quantization (TFLite PTQ)
def to_tflite(model, mode, rep_x=None, workdir="/tmp"):
    sm = Path(workdir) / f"saved_{model.name}_{os.getpid()}"
    shutil.rmtree(sm, ignore_errors=True)
    model.export(str(sm), verbose=False)
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    if mode == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif mode == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep():
            for i in range(len(rep_x)):
                yield [rep_x[i:i + 1].astype(np.float32)]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    elif mode != "none":
        raise ValueError(mode)
    blob = conv.convert()
    shutil.rmtree(sm, ignore_errors=True)
    return blob


class TFLiteFn:
    """Callable batch inference over a TFLite flatbuffer (float or int8 I/O)."""
    def __init__(self, blob, threads=None):
        self.interp = tf.lite.Interpreter(model_content=blob, num_threads=threads or os.cpu_count())
        self.batch_shape = None

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        if self.batch_shape != x.shape:
            inp = self.interp.get_input_details()[0]
            self.interp.resize_tensor_input(inp["index"], list(x.shape))
            self.interp.allocate_tensors()
            self.inp = self.interp.get_input_details()[0]
            self.out = self.interp.get_output_details()[0]
            self.batch_shape = x.shape
        if self.inp["dtype"] == np.int8:
            s, z = self.inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])
        if self.out["dtype"] == np.int8:
            s, z = self.out["quantization"]
            y = (y.astype(np.float32) - z) * s
        return y


def calibration_set(task, rep_x, n_rep, input_size=None, rep_y=None, balanced=False, seed=0):
    """Training samples used to calibrate INT8 ranges. balanced=True draws (up to) equal numbers per
    class for the classifier, so rare nucleus types are represented in the calibration data."""
    rng = np.random.default_rng(seed)
    n_rep = min(n_rep, len(rep_x))
    if balanced and task == "cls" and rep_y is not None:
        classes = np.unique(rep_y)
        per = int(np.ceil(n_rep / len(classes)))
        idx = np.concatenate([rng.choice(np.nonzero(rep_y == c)[0], min(per, int((rep_y == c).sum())),
                                         replace=False) for c in classes])
        rng.shuffle(idx)
    else:
        idx = rng.choice(len(rep_x), n_rep, replace=False)
    rep = rep_x[idx].astype(np.float32) / 255.0
    if task == "cls":
        rep = tf.image.resize(rep, (input_size, input_size)).numpy()
    return rep


def quantize_and_eval(model, task, modes, rep_x, test, out_dir, input_size=None, n_rep=200,
                      eval_batch=8, tta=False, cls_eval_batch=128, rep_y=None, balanced=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = calibration_set(task, rep_x, n_rep, input_size, rep_y, balanced)
    results = {}
    for mode in modes:
        blob = to_tflite(model, mode, rep, workdir=str(out_dir))
        path = out_dir / f"{model.name}_{mode}.tflite"
        path.write_bytes(blob)
        fn = TFLiteFn(blob)
        if task == "seg":
            m = eval_seg(fn, *test, batch=eval_batch, tta=tta)
        else:
            m = eval_cls(fn, *test, input_size, batch=cls_eval_batch, tta=tta)
        m["size_mb"] = len(blob) / 2 ** 20
        results[mode] = m
        print(f"[quant] {model.name} {mode}: size={m['size_mb']:.1f}MB",
              {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    save_json(results, out_dir / f"{model.name}_quant.json")
    return results
