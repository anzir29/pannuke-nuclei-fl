"""Training recipe shared by every model: augmentation, losses, LR schedule, TTA, encoder init.

These are standard practice for histopathology segmentation/classification and are applied
identically to all architectures within a comparison.
"""
import keras
import numpy as np
import tensorflow as tf
from keras import ops


# ------------------------------------------------------------------ augmentation
def _color_jitter(x):
    x = tf.image.random_brightness(x, 0.08)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    x = tf.image.random_saturation(x, 0.9, 1.1)
    x = tf.image.random_hue(x, 0.02)
    return tf.clip_by_value(x, 0.0, 1.0)


def augment_seg(x, y):
    """x: (H,W,3) float [0,1]; y: (H,W) int. Same geometric transform on both."""
    y = y[..., None]
    k = tf.random.uniform([], 0, 4, tf.int32)
    x, y = tf.image.rot90(x, k), tf.image.rot90(y, k)
    flip = tf.random.uniform([]) < 0.5
    x = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(flip, lambda: tf.image.flip_left_right(y), lambda: y)
    return _color_jitter(x), y[..., 0]


def augment_cls(x, y):
    x = tf.image.rot90(x, tf.random.uniform([], 0, 4, tf.int32))
    x = tf.image.random_flip_left_right(x)
    return _color_jitter(x), y


# ------------------------------------------------------------------ losses
@keras.saving.register_keras_serializable("pannuke_fl")
class CEDiceLoss(keras.losses.Loss):
    """Cross-entropy + soft Dice on the nucleus class (counters the background/nucleus imbalance)."""
    def __init__(self, dice_weight=1.0, name="ce_dice", **kw):
        super().__init__(name=name, **kw)
        self.dice_weight = dice_weight

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        y_pred = ops.clip(ops.cast(y_pred, "float32"), 1e-7, 1.0)
        ce = ops.mean(keras.losses.sparse_categorical_crossentropy(y_true, y_pred), axis=(1, 2))
        t = ops.cast(ops.equal(y_true, 1), "float32")
        p = y_pred[..., 1]
        inter = ops.sum(t * p, axis=(1, 2))
        dice = 1.0 - (2.0 * inter + 1.0) / (ops.sum(t, axis=(1, 2)) + ops.sum(p, axis=(1, 2)) + 1.0)
        return ce + self.dice_weight * dice

    def get_config(self):
        return {**super().get_config(), "dice_weight": self.dice_weight}


@keras.saving.register_keras_serializable("pannuke_fl")
class SmoothedWeightedCE(keras.losses.Loss):
    """Sparse cross-entropy with label smoothing and per-class weights."""
    def __init__(self, n_classes=5, label_smoothing=0.0, class_weights=None, name="ls_wce", **kw):
        super().__init__(name=name, **kw)
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        self.class_weights = list(class_weights) if class_weights is not None else None

    def call(self, y_true, y_pred):
        y_true = ops.reshape(ops.cast(y_true, "int32"), (-1,))
        oh = ops.one_hot(y_true, self.n_classes)
        oh = oh * (1.0 - self.label_smoothing) + self.label_smoothing / self.n_classes
        y_pred = ops.clip(ops.cast(y_pred, "float32"), 1e-7, 1.0)
        ce = -ops.sum(oh * ops.log(y_pred), axis=-1)
        if self.class_weights is not None:
            ce = ce * ops.take(ops.convert_to_tensor(self.class_weights, "float32"), y_true)
        return ce

    def get_config(self):
        return {**super().get_config(), "n_classes": self.n_classes,
                "label_smoothing": self.label_smoothing, "class_weights": self.class_weights}


def class_weights(y, mode, n=5):
    if not mode:
        return None
    cnt = np.bincount(y, minlength=n).astype(np.float64)
    w = 1.0 / np.maximum(cnt, 1)
    if mode == "sqrt_inv":
        w = np.sqrt(w)
    return (w / (w * cnt).sum() * cnt.sum()).tolist()      # mean weight over samples = 1


# ------------------------------------------------------------------ optimiser + schedule
def make_optimizer(name, lr, momentum=0.0, schedule=None, total_steps=None, warmup_steps=0,
                   weight_decay=None):
    if schedule == "cosine" and total_steps:
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr * 0.01 if warmup_steps else lr,
            decay_steps=max(1, total_steps - warmup_steps), alpha=0.01,
            warmup_target=lr if warmup_steps else None, warmup_steps=warmup_steps)
    name = name.lower()
    if name == "sgd":
        return keras.optimizers.SGD(lr, momentum=momentum, nesterov=momentum > 0,
                                    weight_decay=weight_decay)
    if name == "adam":
        return keras.optimizers.Adam(lr, weight_decay=weight_decay)
    if name == "adamw":
        return keras.optimizers.AdamW(lr, weight_decay=weight_decay or 1e-4)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(lr, weight_decay=weight_decay)
    raise ValueError(name)


# ------------------------------------------------------------------ test-time augmentation
def tta_seg(fn):
    """Average softmax maps over identity / h-flip / v-flip / both, undoing each flip."""
    def f(x):
        x = np.asarray(x, np.float32)
        acc = 0
        for ax in [(), (2,), (1,), (1, 2)]:
            xi = np.flip(x, ax) if ax else x
            p = np.asarray(fn(np.ascontiguousarray(xi)), np.float32)
            acc = acc + (np.flip(p, ax) if ax else p)
        return acc / 4.0
    return f


def tta_cls(fn):
    def f(x):
        x = np.asarray(x, np.float32)
        acc = 0
        for ax in [(), (2,), (1,), (1, 2)]:
            xi = np.flip(x, ax) if ax else x
            acc = acc + np.asarray(fn(np.ascontiguousarray(xi)), np.float32)
        return acc / 4.0
    return f


# ------------------------------------------------------------------ encoder init
def init_from_vgg16(model):
    """Copy ImageNet VGG16 conv kernels into the first 13 conv layers wherever shapes match
    (SegNet's encoder is the VGG16 conv topology; Badrinarayanan et al. 2017)."""
    vgg = keras.applications.VGG16(include_top=False, weights="imagenet")
    src = [l for l in vgg.layers if isinstance(l, keras.layers.Conv2D)]
    dst = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)][:13]
    copied = 0
    for s, d in zip(src, dst):
        k = s.get_weights()[0]
        if d.kernel.shape == k.shape:
            w = d.get_weights()
            w[0] = k
            d.set_weights(w)
            copied += 1
    del vgg
    return copied
