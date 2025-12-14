# src/models/backbones/swin_backbone.py

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from src.models.backbones import register_backbone
from src.models.backbones.base import BackboneBase

__all__ = ["SwinTBackbone", "SwinSBackbone", "SwinBBackbone"]


class SwinBackbone(BackboneBase):
    MODEL_NAME = None

    def __init__(
            self,
            in_chans: int = 1,
            pretrained: bool = True,
    ):
        super().__init__()

        # Load model according to MODEL_NAME
        self.model = timm.create_model(
            self.MODEL_NAME,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=False,
        )

        # Conv + Norm
        self.patch_embed = self.model.patch_embed

        # 4 transformer stages
        self.layers = self.model.layers

        # Final LN
        self.norm = self.model.norm

        # (B, F, H', W') -> (B, F, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Final feature dimension (encoder output dim)
        self.out_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H', W', F)
        x = self.patch_embed(x)

        # (B, H', W', F) -> (B, H', W', F)
        for layer in self.layers:
            x = layer(x)

        # (B, H', W', F) -> (B, H', W', F)
        x = self.norm(x)

        # (B, H', W', F) -> (B, F, H', W')
        x = x.permute(0, 3, 1, 2)

        # (B, F, H', W') -> (B, F, 1, 1) -> (B, F)
        x = self.pool(x).flatten(1)
        return x

    @property
    def blocks(self):
        # Sequential -> list of SwinTransformerStage
        return list(self.layers)


@register_backbone("swin_t")
class SwinTBackbone(SwinBackbone):
    MODEL_NAME = "swin_tiny_patch4_window7_224"


@register_backbone("swin_s")
class SwinSBackbone(SwinBackbone):
    MODEL_NAME = "swin_small_patch4_window7_224"


@register_backbone("swin_b")
class SwinBBackbone(SwinBackbone):
    MODEL_NAME = "swin_base_patch4_window7_224"
