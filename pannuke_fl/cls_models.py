"""Nucleus-type classifiers on instance crops (5 classes).

Head follows Fig 2: backbone -> pooling -> FC(units) -> Dropout -> FC(dense_units) -> FC(5).
`hp` keys (Table 1, classification column): units, dense_units, dropout, regularizer (l1|l2),
out_activation (softmax|relu|tanh), learning_rate, optimizer, batch_size.
"""
import keras
from keras import layers

N_CLASSES = 5

_BACKBONES = {
    "vgg16": (keras.applications.VGG16, keras.applications.vgg16.preprocess_input),
    "vgg19": (keras.applications.VGG19, keras.applications.vgg19.preprocess_input),
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
    "densenet121": (keras.applications.DenseNet121, keras.applications.densenet.preprocess_input),
    # Keras ships Inception-v3 and Inception-ResNet-v2; the latter is used as "InceptionV2".
    "inceptionv2": (keras.applications.InceptionResNetV2,
                    keras.applications.inception_resnet_v2.preprocess_input),
}

DEFAULT_HP = dict(units=1024, dense_units=1024, dropout=0.2, regularizer=None,
                  out_activation="softmax")


@keras.saving.register_keras_serializable("pannuke_fl")
class BackbonePreprocess(layers.Layer):
    def __init__(self, backbone, **kw):
        super().__init__(**kw)
        self.backbone = backbone

    def call(self, x):
        return _BACKBONES[self.backbone][1](x * 255.0)

    def get_config(self):
        return {**super().get_config(), "backbone": self.backbone}


def build_cls(name, input_size=96, hp=None, pretrained=True):
    h = dict(DEFAULT_HP)
    h.update(hp or {})
    name = name.lower()
    ctor, _ = _BACKBONES[name]
    reg = {"l1": keras.regularizers.L1(1e-4), "l2": keras.regularizers.L2(1e-4)}.get(
        (h["regularizer"] or "").lower())
    inp = keras.Input((input_size, input_size, 3))
    x = BackbonePreprocess(name)(inp)
    base = ctor(include_top=False, weights="imagenet" if pretrained else None,
                input_shape=(input_size, input_size, 3), pooling="avg")
    x = base(x)
    x = layers.Dense(h["units"], activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(h["dropout"])(x)
    x = layers.Dense(h["dense_units"], activation="relu", kernel_regularizer=reg)(x)
    out = layers.Dense(N_CLASSES, activation=h["out_activation"], dtype="float32",
                       name="cls_out")(x)
    return keras.Model(inp, out, name=name)
