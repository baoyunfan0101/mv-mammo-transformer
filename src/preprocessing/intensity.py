# src/preprocessing/intensity.py

from __future__ import annotations

import numpy as np
import cv2

from src.preprocessing.dicom import to_uint8

__all__ = ["zscore", "clahe_equalize"]


def zscore(img: np.ndarray) -> np.ndarray:
    mean, std = float(np.mean(img)), float(np.std(img))
    std = std if std > 1e-8 else 1e-8
    return (img - mean) / std


# Apply CLAHE local contrast enhancement in the float32 domain
def clahe_equalize_float32(
        img: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size=(16, 16),
        num_bins: int = 1024,
) -> np.ndarray:
    # Sanity check: enforce float32 input
    assert img.dtype == np.float32, "clahe_equalize_float32 expects float32 input"

    # Handle degenerate images (constant intensity)
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max <= img_min + 1e-8:
        return img.copy()

    # Normalize image to [0, 1] for quantization
    img_norm = (img - img_min) / (img_max - img_min)

    # Quantize to a fixed number of bins
    img_q = np.floor(img_norm * (num_bins - 1)).astype(np.uint16)

    # Apply CLAHE on the quantized index image
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    if num_bins <= 256:
        # Use uint8 mode if bins are small
        img_q8 = img_q.astype(np.uint8)
        img_eq = clahe.apply(img_q8).astype(np.float32)
        img_eq /= 255.0
    else:
        # Use uint16 mode for higher bin resolution
        scale = 65535 // num_bins
        img_q16 = (img_q * scale).astype(np.uint16)
        img_eq = clahe.apply(img_q16).astype(np.float32)
        img_eq /= 65535.0

    # Map enhanced values back to original float32 range
    img_eq = img_eq * (img_max - img_min) + img_min

    return img_eq.astype(np.float32)


def clahe_equalize(
        img: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size=(16, 16),
) -> np.ndarray:
    if img.dtype == np.float32:
        return clahe_equalize_float32(
            img,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )

    img_u8 = to_uint8(img)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    return clahe.apply(img_u8)
