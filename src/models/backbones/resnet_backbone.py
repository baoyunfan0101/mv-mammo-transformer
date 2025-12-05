# src/models/backbones/resnet_backbone.py

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.models.backbones import register_backbone
from src.models.backbones.base import BackboneBase

__all__ = ["ResNet50Backbone"]


@register_backbone("resnet50")
class ResNet50Backbone(BackboneBase):

    def __init__(
            self,
            in_chans: int = 1,
            pretrained: bool = True
    ):
        super().__init__()

        # Load pretrained torchvision ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)

        # Adapt first convolution layer to 1-channel grayscale
        if in_chans not in {1, 3}:
            raise ValueError(f"[ResNetBackbone] in_chans must be 1 or 3, got {in_chans}")

        if in_chans != 3:
            old_conv = model.conv1

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
            model.conv1 = new_conv

        # Backbone body: (B, in_chans, H, W) -> (B, F, H', W'), F = 2048
        # Batch size: correspond to loss function
        # For a conv kernel (in_chans, kernel_h, kernel_w): output the sum of the convolution of all input channels
        # For a conv layer (out_chans, in_chans, kernel_h, kernel_w): number of conv kernels == number of output channels
        # Convolutional layers => bottleneck blocks => model.layer
        stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

        self.body = nn.Sequential(
            stem,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        # (B, F, H', W') -> (B, F, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ResNet50 final channel dim (model.layer4)
        self.out_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_chans, H, W) -> (B, F, H', W')
        feats = self.body(x)

        # (B, F, H', W') -> (B, F, 1, 1) -> (B, F)
        feats = self.pool(feats).flatten(1)
        return feats

    @property
    def blocks(self):
        return list(self.body)
