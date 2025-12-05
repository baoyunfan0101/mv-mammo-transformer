# src/preprocessing/dicom.py

from __future__ import annotations
from typing import Tuple

import numpy as np
import pydicom
import cv2

__all__ = ["load_dicom", "to_uint8", "to_float32", "to_float16", "zscore", "remove_corner_text"]


def load_dicom(path: str, photometric_interpretation: str = "MONOCHROME2") -> np.ndarray:
    path = path.lower()
    if path.endswith(".dicom") or path.endswith(".dcm"):
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(np.float32)
    else:
        raise TypeError(f"[preprocessing.dicom] Invalid file type: {path}. Expected .dicom or .dcm")

    # Rescale slope/intercept
    slope = getattr(dicom, "RescaleSlope", 1.0)
    intercept = getattr(dicom, "RescaleIntercept", 0.0)
    img = img * slope + intercept

    # Handle inverted images (MONOCHROME1)
    if photometric_interpretation == "MONOCHROME1":
        img = np.max(img) - img

    return img


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    img -= img.min()
    img /= max(img.max(), 1e-8)
    return (img * 255).astype(np.uint8)


def to_float32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def to_float16(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float16)


def remove_corner_text(
        img: np.ndarray,
        h_corner_ratio: float = 0.15,
        w_corner_ratio: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    img_u8 = to_uint8(img)

    h, w = img_u8.shape
    mask = np.zeros_like(img_u8, dtype=np.uint8)

    # Define corner regions (15% of height, 30% of width)
    h_cut = int(h * h_corner_ratio)
    w_cut = int(w * w_corner_ratio)
    corners = [
        (0, 0, w_cut, h_cut),  # top-left
        (w - w_cut, 0, w, h_cut),  # top-right
        (0, h - h_cut, w_cut, h),  # bottom-left
        (w - w_cut, h - h_cut, w, h),  # bottom-right
    ]

    for (x1, y1, x2, y2) in corners:
        roi = img_u8[y1:y2, x1:x2]

        # Detect extremely bright text pixels
        _, bright = cv2.threshold(roi, 254, 255, cv2.THRESH_BINARY)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)
        mask[y1:y2, x1:x2] = cleaned

    # Apply mask to image
    img_clean = img.copy()
    img_clean[mask > 0] = 0

    return img_clean, mask
