# src/preprocessing/crop.py

from __future__ import annotations
from typing import Tuple

import numpy as np
import cv2


def get_breast_mask(
        img: np.ndarraym,
) -> np.ndarray:
    # Normalize to [0, 255] for thresholding
    img_min, img_max = float(img.min()), float(img.max())
    if img_max <= img_min + 1e-8:
        return np.zeros_like(img, dtype=np.uint8)

    img_norm = (img - img_min) / (img_max - img_min)
    img_u8 = (img_norm * 255).astype(np.uint8)

    # Otsu threshold to separate breast vs background
    _, th = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep only the largest connected component
    num_labels, labels = cv2.connectedComponents(th)
    if num_labels <= 1:
        return np.zeros_like(img_u8)

    max_area = 0
    max_label = 1
    for lab in range(1, num_labels):
        area = np.sum(labels == lab)
        if area > max_area:
            max_area = area
            max_label = lab

    mask = (labels == max_label).astype(np.uint8) * 255
    return mask


# Consider the laterality of the image is canonicalized (right -> left)
def fixed_roi_crop(
        img: np.ndarray,
        target_size: Tuple[int, int] = (2560, 1440),
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = img.shape
    target_h, target_w = target_size
    target_h, target_w = min(target_h, h), min(target_w, w)
    mask = get_breast_mask(img)

    # Vertical bounding box of the breast region
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        # Fallback: just center crop
        y_center = h // 2
    else:
        y_min, y_max = ys.min(), ys.max()
        y_center = (y_min + y_max) // 2

    # Compute vertical range
    y1 = int(y_center - target_h // 2)
    y1 = max(0, min(y1, h - target_h))
    y2 = y1 + target_h

    # Horizontal anchor: chest wall side
    x1 = 0
    x1 = max(0, min(x1, w - target_w))
    x2 = x1 + target_w

    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


# Crop breast using minimum bounding box
def min_bounding_box_crop(
        img: np.ndarray,
        ratio: float = 0.125,
        margin=200,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = img.shape

    # Project brightness along both axes
    y_sum = img.sum(axis=1)
    x_sum = img.sum(axis=0)

    # Thresholds (ratio of max sum)
    thres_y = ratio * np.max(y_sum)
    thres_x = ratio * np.max(x_sum)

    # Get indices where brightness > threshold
    y_idx = np.where(y_sum > thres_y)[0]
    x_idx = np.where(x_sum > thres_x)[0]

    # Handle edge cases
    if len(y_idx) == 0 or len(x_idx) == 0:
        return img.copy(), (0, 0, w, h)

    # Bounding box with margin
    y_min = max(0, int(y_idx[0]) - margin)
    y_max = min(h, int(y_idx[-1]) + margin)
    x_min = max(0, int(x_idx[0]) - margin)
    x_max = min(w, int(x_idx[-1]) + margin)

    cropped = img[y_min:y_max, x_min:x_max]
    return cropped, (x_min, y_min, x_max, y_max)
