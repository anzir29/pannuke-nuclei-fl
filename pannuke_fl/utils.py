import json
import os
import random
from pathlib import Path

import numpy as np
import yaml


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    """YAML config; a `base:` entry (path relative to this file) is loaded first and overridden."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(Path(path).parent / cfg.pop("base"))
        cfg = _merge(base, cfg)
    return cfg


def set_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    import keras
    keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def save_json(obj, path):
    """Atomic JSON write (safe against walltime kills)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _to_builtin(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def gpu_setup():
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
