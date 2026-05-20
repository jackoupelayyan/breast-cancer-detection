"""
Data loaders.

KerasDataLoader   — a tf.keras Sequence yielding (batch_images, batch_labels)
PyTorchDataset    — a torch Dataset that pairs preprocessed images with labels

Both are built on top of the preprocessing pipeline in preprocessing.py so the
two frameworks see exactly the same inputs (apart from augmentation flavors).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_mammogram


class KerasDataLoader(tf.keras.utils.Sequence):
    """Batched loader for Keras models."""

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        image_size: int = 224,
        augmenter=None,
        shuffle: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augmenter = augmenter
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx: int):
        batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch = self.df.iloc[batch_idx]

        images = np.stack([
            preprocess_mammogram(row.abs_path, target_size=self.image_size)
            for _, row in batch.iterrows()
        ])
        labels = batch["label"].to_numpy(dtype=np.float32)

        if self.augmenter is not None:
            images = np.stack([
                self.augmenter.random_transform(img) for img in images
            ])
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class PyTorchDataset(Dataset):
    """Torch Dataset for the PyTorch ResNet comparison model."""

    def __init__(self, df: pd.DataFrame, transform=None, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = preprocess_mammogram(
            row.abs_path,
            target_size=self.image_size,
            to_rgb=True,
        )
        # Albumentations expects uint8 HWC; convert here.
        img_uint8 = (img * 255).astype(np.uint8)
        if self.transform is not None:
            img_uint8 = self.transform(image=img_uint8)["image"]
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img_uint8, label
