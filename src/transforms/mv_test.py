# src/transforms/mv_test.py

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from src.transforms import register_transform
from src.transforms.sv_test import SVTestTransform

__all__ = ["MVTestTransform"]


@register_transform("mv_test")
class MVTestTransform:

    def __init__(
            self,
            image_size: Tuple[int, int] = (512, 512),
    ):
        self.sv_transform = SVTestTransform(
            image_size=image_size,
        )

    def __call__(
            self,
            imgs: List[np.ndarray],
    ) -> torch.Tensor:

        return [self.sv_transform(img) for img in imgs]
