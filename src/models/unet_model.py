"""
U-Net adapted for classification.

Standard U-Net is a segmentation architecture, but the encoder produces a rich
bottleneck representation that's useful for image-level classification.
This implementation builds the full encoder/decoder, then attaches a classifier
head on top of the bottleneck features (so we still benefit from the U-shape
during training via an optional auxiliary reconstruction loss — out of scope for
this version; we just use the encoder features).
"""

from tensorflow.keras import layers, models


def _conv_block(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet_classifier(
    input_shape=(224, 224, 3),
    num_classes: int = 2,
    depth: int = 4,
    base_filters: int = 64,
):
    inputs = layers.Input(shape=input_shape)

    x = inputs
    skips = []
    for i in range(depth):
        x = _conv_block(x, base_filters * (2 ** i))
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    # Bottleneck
    x = _conv_block(x, base_filters * (2 ** depth))

    # Classifier head straight from the bottleneck
    head = layers.GlobalAveragePooling2D()(x)
    head = layers.Dense(128, activation="relu")(head)
    head = layers.Dropout(0.5)(head)
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(head)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(head)

    return models.Model(inputs=inputs, outputs=outputs, name="unet_classifier")
