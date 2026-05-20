"""
Augmentation pipelines.

We expose two flavors:
  - Keras ImageDataGenerator-style augmenter (used inline by Keras loaders)
  - Albumentations pipeline (used by the PyTorch ResNet model)

Mammogram-specific notes:
  - We do NOT vertical-flip — mammogram orientation is medically meaningful.
  - We keep rotation modest (±20°) — large rotations create unrealistic views.
  - Brightness/contrast jitter is small — we already normalize via CLAHE.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_keras_augmenter(config: dict) -> ImageDataGenerator:
    """Build a Keras ImageDataGenerator from the augmentation config block."""
    aug = config["augmentation"]
    return ImageDataGenerator(
        rotation_range=aug["rotation_range"],
        horizontal_flip=aug["horizontal_flip"],
        vertical_flip=aug["vertical_flip"],
        zoom_range=aug["zoom_range"],
        brightness_range=tuple(aug["brightness_range"]),
        fill_mode="constant",
        cval=0.0,
    )


def get_albumentations_pipeline(config: dict, train: bool = True) -> A.Compose:
    """Albumentations pipeline for PyTorch training/inference."""
    aug = config["augmentation"]
    size = config["dataset"]["image_size"]

    if train:
        return A.Compose([
            A.Resize(size, size),
            A.Rotate(limit=aug["rotation_range"], p=0.5, border_mode=0),
            A.HorizontalFlip(p=0.5 if aug["horizontal_flip"] else 0.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
