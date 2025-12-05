# src/transforms/mv_consistent.py

from __future__ import annotations
from typing import List, Tuple

import random
import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

from src.transforms import register_transform

__all__ = ["MVConsistentTransform"]


# Consistent affine across 4 views
class RandomAffineConsistent:

    def __init__(
            self,
            degrees: float = 7,
            translate: Tuple[float, float] = (0.05, 0.05),
            scale: Tuple[float, float] = (0.95, 1.05),
            fill: float = 0.0,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.fill = fill

    def _sample_params(self, img_size):
        H, W = img_size
        angle, translations, scale, shear = T.RandomAffine.get_params(
            degrees=(-self.degrees, self.degrees),
            translate=self.translate,
            scale_ranges=self.scale,
            shears=(0.0, 0.0),
            img_size=(H, W),
        )
        return angle, translations, scale, shear

    def __call__(
            self,
            imgs: List[torch.Tensor]
    ) -> List[torch.Tensor]:

        if len(imgs) == 0:
            return imgs

        _, H, W = imgs[0].shape
        angle, translations, scale, shear = self._sample_params((H, W))

        out = []
        for img in imgs:
            new_img = F.affine(
                img,
                angle=angle,
                translate=list(translations),
                scale=scale,
                shear=list(shear),
                fill=self.fill,
            )
            out.append(new_img)
        return out


@register_transform("mv_consistent")
class MVConsistentTransform:

    def __init__(
            self,
            image_size: Tuple[int, int] | None = None,
            affine_prob: float = 0.7,
            global_flip_prob: float = 0.5,
            color_prob: float = 0.5,
            blur_prob: float = 0.2,
    ):
        self.image_size = image_size

        if self.image_size is not None:
            self.resize = T.Resize(self.image_size)

        # Consistent affine
        self.affine = RandomAffineConsistent(
            degrees=7,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=0,
        )
        self.affine_prob = affine_prob

        # Global flip probability
        self.global_flip_prob = global_flip_prob

        # Per-view photometric augmentation
        self.color = T.ColorJitter(brightness=0.05, contrast=0.10)
        self.color_prob = color_prob

        self.blur = T.GaussianBlur(kernel_size=5)
        self.blur_prob = blur_prob

        # Final conversion
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])

    def __call__(
            self,
            imgs: List[np.ndarray],
    ) -> List[torch.Tensor]:

        # np.ndarray -> torch.Tensor
        tensors: List[torch.Tensor] = []
        for img in imgs:
            t = torch.from_numpy(img).float()
            if t.ndim == 2:
                t = t.unsqueeze(0)
            tensors.append(t)

        imgs = tensors

        # Resize
        if self.image_size is None:
            global_size = imgs[0].shape
            for img in imgs:
                if img.shape != global_size:
                    raise ValueError(
                        f"[MVConsistentTransform] {len(imgs)} images have different size."
                    )
        else:
            imgs = [self.resize(img) for img in imgs]

        # Consistent affine across 4 views
        if random.random() < self.affine_prob:
            imgs = self.affine(imgs)

        # Global horizontal flip
        if random.random() < self.global_flip_prob:
            imgs = [F.hflip(img) for img in imgs]

        # Per-image photometric augmentation
        out = []
        for img in imgs:
            if random.random() < self.color_prob:
                img = self.color(img)

            if random.random() < self.blur_prob:
                img = self.blur(img)

            # Normalize into [-1,1]
            img = self.normalize(img)
            out.append(img)

        return out
