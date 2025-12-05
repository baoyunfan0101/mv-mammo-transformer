# src/transforms/sv_baseline.py

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
from torchvision.transforms import v2 as T

from src.transforms import register_transform

__all__ = ["SVBaselineTransform"]


@register_transform("sv_baseline")
class SVBaselineTransform:

    def __init__(
            self,
            image_size: Tuple[int, int] | None = None,
            augment: bool = True,
    ):
        self.image_size = image_size

        # Transform list
        ops: List[torch.nn.Module] = []

        # Resize if required
        if image_size is not None:
            ops.extend([
                # Resize
                T.Resize(self.image_size),
            ])

        # Augmentations
        if augment:
            ops.extend([
                # Small geometric jitter: rotation, translation, scale
                T.RandomApply([
                    T.RandomAffine(
                        degrees=7,  # +-7 degrees
                        translate=(0.05, 0.05),  # up to 5% shift
                        scale=(0.95, 1.05),  # zoom in/out
                        fill=0,  # fill background with black
                    )
                ], p=0.7),

                # Light brightness/contrast jitter
                T.RandomApply([
                    T.ColorJitter(
                        brightness=0.05,
                        contrast=0.10,
                    )
                ], p=0.5),

                # Optional slight blur
                T.RandomApply([
                    T.GaussianBlur(kernel_size=5)
                ], p=0.2),
            ])

        ops.extend([
            # Normalize into [-1,1]
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        # Final compose
        self.transform = T.Compose(ops)

    def __call__(
            self,
            img: np.ndarray,
    ) -> torch.Tensor:

        # np.ndarray -> torch.Tensor
        t = torch.from_numpy(img).float()
        if t.ndim == 2:
            t = t.unsqueeze(0)

        return self.transform(t)
