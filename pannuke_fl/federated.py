"""Federated simulation: FedAvg, FedProx, FedBN (float training), then INT8 export of the result.

Each round: every client starts from the global weights, trains `local_epochs` on its own data,
and the server averages the weights in proportion to client data size (Eq. 2 / McMahan et al.).
FedProx adds (mu/2)||w - w_global||^2 to the local loss. FedBN keeps BatchNorm parameters and
statistics local (Li et al., 2021); its personalised client models are each evaluated and averaged.
State is saved after every round so runs resume across 24-h jobs.
"""
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .utils import load_json, save_json


def _bn_mask(model):
    bn_vars = set()

    def walk(layer):
        if isinstance(layer, keras.layers.BatchNormalization):
            bn_vars.update(id(v) for v in layer.weights)
        for sub in getattr(layer, "layers", []):
            walk(sub)
    walk(model)
    return np.array([id(v) in bn_vars for v in model.weights])


class LocalTrainer:
    def __init__(self, model, optimizer, loss_fn, mu=0.0):
        self.model, self.opt, self.mu, self.loss_fn = model, optimizer, mu, loss_fn
        self.anchor = [tf.Variable(v, trainable=False) for v in model.trainable_variables]
        self.opt.build(model.trainable_variables)

    def reset_optimizer(self):
        """Each client starts a round with fresh optimizer state (momentum / Adam moments)."""
        for v in self.opt.variables:
            n = v.path if hasattr(v, "path") else v.name
            if "iteration" in n or "learning_rate" in n:
                continue
            v.assign(tf.zeros_like(v))

    @tf.function
    def _step(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(y, self.model(x, training=True))
            if self.model.losses:
                loss += tf.add_n(self.model.losses)
            if self.mu > 0:
                loss += 0.5 * self.mu * tf.add_n(
                    [tf.reduce_sum(tf.square(w - a)) for w, a in zip(self.model.trainable_variables, self.anchor)])
        g = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(g, self.model.trainable_variables))
        return loss

    def train(self, ds, epochs):
        self.reset_optimizer()
        for a, w in zip(self.anchor, self.model.trainable_variables):
            a.assign(w)
        for _ in range(epochs):
            for x, y in ds:
                self._step(x, y)


def _save_state(path, rnd, global_w, client_bn, history):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "state.tmp.npz"
    arrays = {f"g{i}": w for i, w in enumerate(global_w)}
    for k, ws in enumerate(client_bn or []):
        arrays.update({f"c{k}_{i}": w for i, w in enumerate(ws)})
    np.savez(tmp, **arrays)
    tmp.replace(path / "state.npz")
    save_json({"round": rnd, "history": history, "n_global": len(global_w),
               "n_clients_bn": len(client_bn or [])}, path / "state.json")


def _load_state(path):
    path = Path(path)
    if not (path / "state.json").exists():
        return None
    meta = load_json(path / "state.json")
    z = np.load(path / "state.npz")
    g = [z[f"g{i}"] for i in range(meta["n_global"])]
    cb = None
    if meta["n_clients_bn"]:
        cb = [[z[k] for k in sorted((k for k in z.files if k.startswith(f"c{c}_")),
                                    key=lambda s: int(s.split("_")[1]))]
              for c in range(meta["n_clients_bn"])]
    return meta["round"], g, cb, meta["history"]


def run_federated(build_fn, make_opt, loss_fn, client_datasets, client_sizes, algorithm, rounds,
                  local_epochs, mu, evaluate, state_dir, eval_every=5):
    """build_fn() -> compiled-free keras model; evaluate(model) -> dict of metrics."""
    keras.backend.clear_session()
    model = build_fn()
    bn = _bn_mask(model) if algorithm == "fedbn" else np.zeros(len(model.weights), bool)
    trainer = LocalTrainer(model, make_opt(), loss_fn, mu if algorithm == "fedprox" else 0.0)
    weights = np.asarray(client_sizes, np.float64) / np.sum(client_sizes)

    st = _load_state(state_dir)
    if st:
        start, global_w, client_bn, history = st
        print(f"[fl] resuming {algorithm} at round {start}")
    else:
        start, global_w, history = 0, model.get_weights(), []
        client_bn = [[w for w, b in zip(global_w, bn) if b] for _ in client_datasets] \
            if algorithm == "fedbn" else None

    def client_weights(k):
        if client_bn is None:
            return global_w
        it = iter(client_bn[k])
        return [next(it) if b else w for w, b in zip(global_w, bn)]

    for rnd in range(start, rounds):
        new = []
        for k, ds in enumerate(client_datasets):
            model.set_weights(client_weights(k))
            trainer.train(ds, local_epochs)
            w = model.get_weights()
            new.append(w)
            if client_bn is not None:
                client_bn[k] = [x for x, b in zip(w, bn) if b]
        global_w = [np.sum([wk * cw[i] for wk, cw in zip(weights, new)], axis=0).astype(new[0][i].dtype)
                    for i in range(len(global_w))]
        rec = {"round": rnd + 1}
        if (rnd + 1) % eval_every == 0 or rnd + 1 == rounds:
            rec.update(_evaluate_all(model, global_w, client_bn, client_weights, evaluate, algorithm))
        history.append(rec)
        print(f"[fl] {algorithm} round {rnd + 1}/{rounds}", {k: v for k, v in rec.items() if k != "round"})
        _save_state(state_dir, rnd + 1, global_w, client_bn, history)
    return model, global_w, client_bn, client_weights, history


def _evaluate_all(model, global_w, client_bn, client_weights, evaluate, algorithm):
    if algorithm != "fedbn":
        model.set_weights(global_w)
        return evaluate(model)
    per = []
    for k in range(len(client_bn)):
        model.set_weights(client_weights(k))
        per.append(evaluate(model))
    keys = [k for k, v in per[0].items() if isinstance(v, (int, float))]
    return {k: float(np.mean([p[k] for p in per])) for k in keys}
