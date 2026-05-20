"""
VGG16 with ImageNet pre-training.

Two-stage training works best here:
  1. Freeze the convolutional base, train only the classification head.
  2. Unfreeze the last `fine_tune_layers` and fine-tune with a tiny learning rate.

The training script handles stage 2.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16


def build_vgg16(
    input_shape=(224, 224, 3),
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_base: bool = True,
    dropout: float = 0.5,
):
    base = VGG16(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=input_shape,
    )
    base.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="vgg16_transfer")


def unfreeze_top_layers(model: models.Model, n_layers: int = 4):
    """Unfreeze the top N layers of the VGG base for fine-tuning."""
    vgg_base = next(layer for layer in model.layers if layer.name == "vgg16")
    vgg_base.trainable = True
    for layer in vgg_base.layers[:-n_layers]:
        layer.trainable = False
    return model
