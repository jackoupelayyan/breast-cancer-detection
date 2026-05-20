"""
Keras training loop.

Used by baseline_cnn, vgg16, and unet. Handles:
  - Callbacks (early stopping, LR reduction, model checkpointing)
  - Class weighting for the imbalance between benign and malignant
  - Optional two-stage fine-tuning for VGG16
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ..models.vgg16_model import unfreeze_top_layers


def _compile(model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def _callbacks(config: dict, save_path: Path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=config["training"]["reduce_lr_patience"],
            factor=config["training"]["reduce_lr_factor"],
            min_lr=1e-7,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path), monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]


def train_keras_model(
    model,
    train_loader,
    val_loader,
    config: dict,
    model_name: str,
    fine_tune: bool = False,
):
    """Fit a Keras model. If fine_tune=True and model is VGG, runs the two-stage schedule."""
    save_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{model_name}.keras"

    # Class weights from the training labels
    train_labels = np.array([y for _, batch_y in (train_loader[i] for i in range(len(train_loader))) for y in batch_y])
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weight = dict(zip(classes.astype(int), weights))

    _compile(model, config["training"]["learning_rate"])
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=config["training"]["epochs"],
        callbacks=_callbacks(config, save_path),
        class_weight=class_weight,
        verbose=1,
    )

    if fine_tune and model.name == "vgg16_transfer":
        print("\n=== Stage 2: fine-tuning top VGG layers ===")
        model = unfreeze_top_layers(model, n_layers=config["models"]["vgg16"]["fine_tune_layers"])
        _compile(model, config["training"]["learning_rate"] * 0.1)  # smaller LR for fine-tuning
        ft_history = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=config["training"]["epochs"] // 2,
            callbacks=_callbacks(config, save_path),
            class_weight=class_weight,
            verbose=1,
        )
        # Merge histories
        for k, v in ft_history.history.items():
            history.history[k].extend(v)

    return model, history.history
