"""
Baseline CNN — a clean, deliberately simple architecture that serves as the
floor that the transfer-learning models need to beat.

4 conv blocks with BatchNorm + MaxPool, then GAP + Dense head with dropout.
Roughly 2M parameters.
"""

from tensorflow.keras import layers, models


def build_baseline_cnn(input_shape=(224, 224, 3), num_classes: int = 2, dropout: float = 0.5):
    inputs = layers.Input(shape=input_shape)

    x = _conv_block(inputs, 32)
    x = _conv_block(x, 64)
    x = _conv_block(x, 128)
    x = _conv_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")


def _conv_block(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    return x
