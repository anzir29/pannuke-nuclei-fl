"""Segmentation models: FCN-8s, U-Net, SegNet, ResUNet (2 output classes: background, nucleus).

`hp` keys (Table 1, segmentation column):
  filters      base number of filters                      [32, 64, 128, 256]
  filter_len   stem kernel height                         [10, 15, 20, 25, 30]
  filter_wid   stem kernel width                          [2, 4, 8, 16]
  init         kernel initializer                         uniform | normal | zero | he_uniform
  activation   hidden activation                          softmax | relu | sigmoid | tanh
  fc_neurons   width of the 1x1 bottleneck layer          [16, 32, 64, 128]
  dropout      dropout at the bottleneck                  [0.05 .. 0.25]
"""
import keras
from keras import layers, ops

DEFAULT_HP = dict(filters=64, filter_len=3, filter_wid=3, init="he_uniform",
                  activation="relu", fc_neurons=None, dropout=0.0)

_INIT = {"uniform": "random_uniform", "normal": "random_normal", "zero": "zeros",
         "he_uniform": "he_uniform"}


def _hp(hp):
    out = dict(DEFAULT_HP)
    out.update(hp or {})
    return out


def _act(name):
    if name == "softmax":
        return layers.Softmax(axis=-1)
    return layers.Activation(name)


def conv_bn(x, f, hp, k=3):
    x = layers.Conv2D(f, k, padding="same", use_bias=False,
                      kernel_initializer=_INIT[hp["init"]])(x)
    x = layers.BatchNormalization()(x)
    return _act(hp["activation"])(x)


def stem(x, hp):
    """First convolution uses the tuned (filter_len x filter_wid) kernel."""
    return conv_bn(x, hp["filters"], hp, k=(hp["filter_len"], hp["filter_wid"]))


def bottleneck(x, hp):
    if hp["fc_neurons"]:
        x = layers.Conv2D(hp["fc_neurons"], 1, kernel_initializer=_INIT[hp["init"]])(x)
        x = _act(hp["activation"])(x)
    if hp["dropout"]:
        x = layers.Dropout(hp["dropout"])(x)
    return x


def head(x, n_classes=2):
    return layers.Conv2D(n_classes, 1, activation="softmax", dtype="float32", name="seg_out")(x)


# ---------------------------------------------------------------- SegNet
@keras.saving.register_keras_serializable("pannuke_fl")
class PoolWithIndices(layers.Layer):
    """2x2 max-pool that also returns a binary mask of the max locations (the pooling indices).

    Built only from MaxPool / nearest upsampling / equality, so it converts to TFLite INT8.
    """
    def call(self, x):
        p = ops.max_pool(x, 2, 2, padding="valid")
        up = ops.repeat(ops.repeat(p, 2, axis=1), 2, axis=2)
        return p, ops.cast(ops.equal(x, up), x.dtype)


@keras.saving.register_keras_serializable("pannuke_fl")
class UnpoolWithIndices(layers.Layer):
    def call(self, inputs):
        x, mask = inputs
        up = ops.repeat(ops.repeat(x, 2, axis=1), 2, axis=2)
        return up * mask


def segnet(input_shape=(256, 256, 3), hp=None, vgg_input=False):
    hp = _hp(hp)
    f = hp["filters"]
    widths = [f, 2 * f, 4 * f, 8 * f, 8 * f]
    reps = [2, 2, 3, 3, 3]
    inp = keras.Input(input_shape)
    x, masks = inp, []
    if vgg_input:     # match the input normalisation of the ImageNet VGG16 encoder weights
        x = _VGGPre()(layers.Rescaling(255.0)(x))
    for b, (w, r) in enumerate(zip(widths, reps)):
        for j in range(r):
            x = stem(x, hp) if (b == 0 and j == 0) else conv_bn(x, w, hp)
        x, m = PoolWithIndices()(x)
        masks.append(m)
    x = bottleneck(x, hp)
    if hp["fc_neurons"]:
        x = conv_bn(x, widths[-1], hp, k=1)
    dec_out = widths[::-1][1:] + [f]
    for b, (w, r, m) in enumerate(zip(widths[::-1], reps[::-1], masks[::-1])):
        x = UnpoolWithIndices()([x, m])
        for j in range(r - 1):
            x = conv_bn(x, w, hp)
        x = conv_bn(x, dec_out[b], hp)
    return keras.Model(inp, head(x), name="segnet")


# ---------------------------------------------------------------- U-Net
def unet(input_shape=(256, 256, 3), hp=None):
    hp = _hp(hp)
    f = hp["filters"]
    inp = keras.Input(input_shape)
    x = stem(inp, hp)
    x = conv_bn(x, f, hp)
    skips = []
    for w in [f, 2 * f, 4 * f, 8 * f]:
        skips.append(x)
        x = layers.MaxPool2D(2)(x)
        x = conv_bn(x, 2 * w, hp)
        x = conv_bn(x, 2 * w, hp)
    x = bottleneck(x, hp)
    for w, s in zip([8 * f, 4 * f, 2 * f, f], skips[::-1]):
        x = layers.Conv2DTranspose(w, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, s])
        x = conv_bn(x, w, hp)
        x = conv_bn(x, w, hp)
    return keras.Model(inp, head(x), name="unet")


# ---------------------------------------------------------------- ResUNet
def _res_block(x, w, hp, stride=1):
    sc = layers.Conv2D(w, 1, strides=stride, padding="same", use_bias=False)(x)
    sc = layers.BatchNormalization()(sc)
    y = layers.BatchNormalization()(x)
    y = _act(hp["activation"])(y)
    y = layers.Conv2D(w, 3, strides=stride, padding="same", kernel_initializer=_INIT[hp["init"]])(y)
    y = layers.BatchNormalization()(y)
    y = _act(hp["activation"])(y)
    y = layers.Conv2D(w, 3, padding="same", kernel_initializer=_INIT[hp["init"]])(y)
    return layers.Add()([y, sc])


def resunet(input_shape=(256, 256, 3), hp=None):
    hp = _hp(hp)
    f = hp["filters"]
    inp = keras.Input(input_shape)
    x = stem(inp, hp)
    x = layers.Conv2D(f, 3, padding="same")(x)
    sc = layers.Conv2D(f, 1, padding="same")(inp)
    x = layers.Add()([x, sc])
    skips = [x]
    for w in [2 * f, 4 * f, 8 * f]:
        x = _res_block(x, w, hp, stride=2)
        skips.append(x)
    x = bottleneck(_res_block(x, 16 * f, hp, stride=2), hp)
    for w, s in zip([8 * f, 4 * f, 2 * f, f], skips[::-1]):
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, s])
        x = _res_block(x, w, hp)
    return keras.Model(inp, head(x), name="resunet")


# ---------------------------------------------------------------- FCN-8s
def fcn(input_shape=(256, 256, 3), hp=None, pretrained=True):
    hp = _hp(hp)
    base = keras.applications.VGG16(include_top=False, input_shape=input_shape,
                                    weights="imagenet" if pretrained else None)
    p3 = base.get_layer("block3_pool").output
    p4 = base.get_layer("block4_pool").output
    p5 = base.get_layer("block5_pool").output
    w = hp["fc_neurons"] or 4 * hp["filters"]
    x = layers.Conv2D(w, 7, padding="same", kernel_initializer=_INIT[hp["init"]])(p5)
    x = _act(hp["activation"])(x)
    x = layers.Dropout(hp["dropout"] or 0.0)(x)
    x = layers.Conv2D(w, 1, kernel_initializer=_INIT[hp["init"]])(x)
    x = _act(hp["activation"])(x)
    s5 = layers.Conv2D(2, 1)(x)
    up = layers.Conv2DTranspose(2, 4, strides=2, padding="same")(s5)
    s4 = layers.Add()([up, layers.Conv2D(2, 1)(p4)])
    up = layers.Conv2DTranspose(2, 4, strides=2, padding="same")(s4)
    s3 = layers.Add()([up, layers.Conv2D(2, 1)(p3)])
    out = layers.Conv2DTranspose(2, 16, strides=8, padding="same")(s3)
    out = layers.Softmax(axis=-1, dtype="float32", name="seg_out")(out)
    # FCN takes VGG-preprocessed input; wrap so all models accept [0,1] RGB.
    inp = keras.Input(input_shape)
    x = layers.Rescaling(255.0)(inp)
    x = _VGGPre()(x)
    return keras.Model(inp, keras.Model(base.input, out)(x), name="fcn")


@keras.saving.register_keras_serializable("pannuke_fl")
class _VGGPre(layers.Layer):
    """RGB [0,255] -> VGG 'caffe' preprocessing (BGR, mean-subtracted)."""
    def call(self, x):
        x = ops.flip(x, axis=-1)
        return x - ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)


def build_seg(name, input_shape=(256, 256, 3), hp=None, pretrained=True):
    """pretrained=True: ImageNet VGG16 encoder for FCN and SegNet (their standard initialisation)."""
    name = name.lower()
    if name == "fcn":
        return fcn(input_shape, hp, pretrained)
    if name == "segnet":
        model = segnet(input_shape, hp, vgg_input=pretrained)
        if pretrained:
            from .recipe import init_from_vgg16
            n = init_from_vgg16(model)
            print(f"[segnet] initialised {n}/13 encoder convs from ImageNet VGG16")
        return model
    return {"unet": unet, "resunet": resunet}[name](input_shape, hp)
