# src/preprocessing/orientation.py

from __future__ import annotations

import numpy as np

__all__ = ["canonicalize_laterality"]


def canonicalize_laterality(
        img: np.ndarray,
        laterality: str,
) -> np.ndarray:
    if laterality.upper() == "R":
        # Horizontal flip
        img = np.flip(img, axis=1)

    return img
