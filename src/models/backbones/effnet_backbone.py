# src/models/backbones/effnet_backbone.py

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.models.backbones import register_backbone
from src.models.backbones.base import BackboneBase

__all__ = ["EffNetV2SBackbone"]


@register_backbone("effnetv2_s")
class EffNetV2SBackbone(BackboneBase):

    def __init__(
            self,
            in_chans: int = 1,
            pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained torchvision EfficientNetV2-S
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)

        # Adapt first convolution layer to 1-channel grayscale
        if in_chans not in {1, 3}:
            raise ValueError(f"[EffNetV2SBackbone] in_chans must be 1 or 3, got {in_chans}")

        if in_chans != 3:
            # Conv2d layer inside features[0]
            old_conv = model.features[0][0]

            # Create new conv with same output channels
            new_conv = nn.Conv2d(
                in_chans,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            # Convert RGB weights to grayscale mean
            new_conv.weight.data = old_conv.weight.mean(dim=1, keepdim=True)
            model.features[0][0] = new_conv

        # Backbone body: (B, C, H, W) -> (B, F, H', W'), F = 1280
        self.body = model.features

        # (B, F, H', W') -> (B, F, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # EfficientNetV2-S final channel dim
        self.out_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, F, H', W')
        feats = self.body(x)

        # (B, F, H', W') -> (B, F, 1, 1) -> (B, F)
        feats = self.pool(feats).flatten(1)
        return feats

    @property
    def blocks(self):
        return list(self.body)
