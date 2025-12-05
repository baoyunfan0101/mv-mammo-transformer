# src/preprocessing/pipeline.py

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np

from src.preprocessing.dicom import load_dicom, remove_corner_text
from src.preprocessing.orientation import canonicalize_laterality
from src.preprocessing.crop import fixed_roi_crop
from src.preprocessing.intensity import clahe_equalize


def preprocess_one_image(
        dicom_path: str,
        metadata: Dict[str, str],
        target_size: Tuple[int, int] = (2560, 1440),
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (16, 16),
) -> np.ndarray:
    # Load dicom
    img = load_dicom(
        dicom_path,
        metadata.get("photometric_interpretation", "MONOCHROME1")
    )

    # Remove corner text
    img, _ = remove_corner_text(img)

    # Horizontal flip right breast
    img = canonicalize_laterality(
        img,
        metadata["laterality"],
    )

    # Fixed ROI crop
    img, _ = fixed_roi_crop(
        img,
        target_size=target_size,
    )

    # Local contrast normalization
    img = clahe_equalize(
        img,
        clip_limit=clahe_clip,
        tile_grid_size=clahe_grid,
    )

    return img
