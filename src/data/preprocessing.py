"""
Preprocessing utilities for CBIS-DDSM mammograms.

CBIS-DDSM images come as DICOM files with varied sizes, intensity ranges,
and a lot of background black space and labels. This module handles:
  1. Reading DICOM (or PNG fallback)
  2. Cropping to the breast region (removes labels/borders)
  3. Normalizing intensity to [0, 1]
  4. Resizing to the model input size
  5. Optional CLAHE contrast enhancement
"""

from pathlib import Path
import logging

import cv2
import numpy as np
import pandas as pd

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

logger = logging.getLogger(__name__)


def read_image(path: str | Path) -> np.ndarray:
    """Read a mammogram from DICOM or standard image format. Returns float32 in [0, 1]."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".dcm", ".dicom"}:
        if not HAS_PYDICOM:
            raise ImportError("pydicom is required to read DICOM files. Install with: pip install pydicom")
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = img.astype(np.float32)

    # Normalize to [0, 1] using the image's own range — DICOMs vary widely.
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return img


def crop_to_breast(img: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Crop the largest non-background connected component (the breast).
    Removes the black borders, annotations, and scanner artifacts that
    commonly appear in CBIS-DDSM mammograms.
    """
    binary = (img > threshold).astype(np.uint8)

    # Largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return img  # nothing to crop

    # stats[0] is the background — pick the largest non-background component
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h = stats[largest, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                  cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
    return img[y:y + h, x:x + w]


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization. Helps make masses more visible."""
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_uint8).astype(np.float32) / 255.0


def preprocess_mammogram(
    path: str | Path,
    target_size: int = 224,
    use_clahe: bool = True,
    to_rgb: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for one mammogram."""
    img = read_image(path)
    img = crop_to_breast(img)
    if use_clahe:
        img = apply_clahe(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    if to_rgb:
        img = np.stack([img] * 3, axis=-1)  # 3 channels for transfer-learning backbones
    return img.astype(np.float32)


def build_dataframe(metadata_csv: str | Path, image_root: str | Path) -> pd.DataFrame:
    """
    Build a clean DataFrame with image paths and binary labels from
    the CBIS-DDSM metadata CSV. Adapt column names to whichever CSV
    you download — calc_case_description / mass_case_description.
    """
    df = pd.read_csv(metadata_csv)
    image_root = Path(image_root)

    # Common CBIS-DDSM columns: 'image file path', 'pathology'
    # pathology is one of: BENIGN, BENIGN_WITHOUT_CALLBACK, MALIGNANT
    if "pathology" not in df.columns:
        raise KeyError("Expected a 'pathology' column in the metadata CSV.")

    df = df[df["pathology"].notna()].copy()
    df["label"] = (df["pathology"].str.upper() == "MALIGNANT").astype(int)

    path_col = next((c for c in df.columns if "image" in c.lower() and "path" in c.lower()), None)
    if path_col is None:
        raise KeyError("Could not find an image-path column in the metadata CSV.")

    df["abs_path"] = df[path_col].apply(lambda p: str(image_root / p))
    return df[["abs_path", "label", "pathology"]].reset_index(drop=True)
