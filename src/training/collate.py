# src/training/collate.py

from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional

import numpy as np
import torch

__all__ = ["CollateFn"]


def collate_batch(
        batch: List[Dict[str, Any]],
        transform: Optional[Callable] = None,
) -> Dict[str, Any]:
    # Batch is not empty
    assert len(batch) > 0, "Empty batch."
    first = batch[0]
    first_images = first["images"]

    # The first image(s) of the batch
    assert isinstance(first_images, list) and len(first_images) > 0
    num_views = len(first_images)

    # Key
    out: Dict[str, Any] = {
        "key": [sample["key"] for sample in batch],
    }

    # Images
    if num_views == 1:
        # single-view: images -> Tensor[B, C, H, W]
        imgs = []

        for sample in batch:
            img = sample["images"][0]

            # Apply transforms
            if transform is not None:
                assert isinstance(img, np.ndarray), "Single-view transform expects np.ndarray."
                # np.ndarray -> torch.Tensor
                img = transform(img)
            else:
                img = torch.from_numpy(img).float().unsqueeze(0)

            imgs.append(img)
        out["images"] = torch.stack(imgs)
    else:
        # multi-view: images -> List[Tensor[B, C, H, W]], len = num_views
        views: List[torch.Tensor] = []

        for sample in batch:
            imgs = sample["images"]

            # Apply transforms
            if transform is not None:
                assert all(isinstance(x, np.ndarray) for x in imgs), "Multi-view transform expects List[np.ndarray]."
                # List[np.ndarray] -> List[torch.Tensor]
                imgs = transform(imgs)
            else:
                imgs = [
                    torch.from_numpy(x).float().unsqueeze(0)
                    for x in imgs
                ]

            views.append(imgs)

        num_views = len(views[0])
        out["images"] = [
            torch.stack([views[b][v] for b in range(len(views))])
            for v in range(num_views)
        ]

    # Label
    if "label" in first and first["label"] is not None:
        labels: Dict[str, List[int]] = {}
        for sample in batch:
            for name, value in sample["label"].items():
                labels.setdefault(name, []).append(value)

        out["label"] = {
            name: torch.tensor(vals, dtype=torch.long)
            for name, vals in labels.items()
        }

    return out


class CollateFn:
    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_batch(batch, transform=self.transform)
