"""Segmentation metrics (Eqs. 3-7) from a pixel confusion matrix, and classification metrics (Eqs. 9-13)."""
import keras
import numpy as np
from keras import ops
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def seg_confusion(y_true, y_pred, n=2):
    return np.bincount(n * y_true.ravel().astype(np.int64) + y_pred.ravel(),
                       minlength=n * n).reshape(n, n)


def seg_metrics(cm):
    """cm[i, j] = pixels of true class i predicted as class j. Class 1 = nucleus."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    true_tot = cm.sum(1)
    pred_tot = cm.sum(0)
    iou = tp / np.maximum(true_tot + pred_tot - tp, 1)
    cls_acc = tp / np.maximum(true_tot, 1)              # per-class pixel accuracy (recall)
    cls_prec = tp / np.maximum(pred_tot, 1)
    freq = true_tot / cm.sum()
    return {
        "PA": tp.sum() / cm.sum(),
        "MPA": cls_acc.mean(),
        "MIoU": iou.mean(),
        "FWIoU": (freq * iou).sum(),
        "CPA_nucleus": cls_prec[1],                     # class pixel accuracy of nucleus predictions
        "PA_nucleus": cls_acc[1],
        "IoU_nucleus": iou[1],
        "IoU_background": iou[0],
        "confusion": cm.astype(np.int64).tolist(),
    }


def cls_metrics(y_true, y_pred, n=5, names=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n))).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    sn = tp / np.maximum(tp + fn, 1)
    sp = tn / np.maximum(tn + fp, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    acc_c = (tp + tn) / cm.sum()
    mcc_c = (tp * tn - fp * fn) / np.maximum(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 1e-12)
    out = {
        "ACC": float(tp.sum() / cm.sum()),
        "SN": float(sn.mean()), "SP": float(sp.mean()), "F1": float(f1.mean()),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "confusion": cm.astype(np.int64).tolist(),
        "per_class": {},
    }
    names = names or [str(i) for i in range(n)]
    for i, nm in enumerate(names):
        out["per_class"][nm] = {"ACC": acc_c[i], "F1": f1[i], "MCC": mcc_c[i], "SP": sp[i], "SN": sn[i]}
    return out


@keras.saving.register_keras_serializable("pannuke_fl")
class MeanPixelAccuracy(keras.metrics.Metric):
    """Streaming MPA for model selection / Hyperband objective."""
    def __init__(self, n=2, name="mpa", **kw):
        super().__init__(name=name, **kw)
        self.n = n
        self.cm = self.add_variable(shape=(n, n), initializer="zeros", name="cm", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # one-hot matmul instead of bincount so the metric compiles under XLA (jit_compile)
        t = ops.one_hot(ops.cast(ops.reshape(y_true, (-1,)), "int32"), self.n, dtype="float32")
        p = ops.one_hot(ops.cast(ops.reshape(ops.argmax(y_pred, -1), (-1,)), "int32"), self.n, dtype="float32")
        self.cm.assign_add(ops.matmul(ops.transpose(t), p))

    def result(self):
        d = ops.diagonal(self.cm)
        return ops.mean(d / ops.maximum(ops.sum(self.cm, axis=1), 1.0))

    def reset_state(self):
        self.cm.assign(ops.zeros_like(self.cm))

    def get_config(self):
        return {**super().get_config(), "n": self.n}
