# src/transforms/sv_test.py

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch

from src.transforms import register_transform
from src.transforms.sv_baseline import SVBaselineTransform

__all__ = ["SVTestTransform"]


@register_transform("sv_test")
class SVTestTransform:

    def __init__(
            self,
            image_size: Tuple[int, int] = (512, 512),
    ):
        self.transform = SVBaselineTransform(
            image_size=image_size,
            augment=False,
        )

    def __call__(
            self,
            img: np.ndarray,
    ) -> torch.Tensor:
        return self.transform(img)
