# src/models/singleview/sv_baseline.py

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models import register_model
from src.models.backbones import get_backbone
from src.models.heads import get_head

__all__ = ["SVBaseline"]


@register_model("sv_baseline")
class SVBaseline(nn.Module):

    def __init__(
            self,
            backbone: str = "resnet50",
            head: str = "softmax",
            in_chans: int = 1,
            birads_classes: int = 5,
            density_classes: int = 4,
            backbone_kwargs: Optional[Dict] = None,
            head_kwargs: Optional[Dict] = None,
    ):
        super().__init__()

        # Get backbone
        # (B, in_chans, H, W) -> (B, feat_dim)
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = get_backbone(
            backbone,
            in_chans=in_chans,
            **backbone_kwargs,
        )
        feat_dim = self.backbone.out_dim

        # Get head
        # (B, feat_dim) -> (B, Dict)
        head_kwargs = head_kwargs or {}
        self.head = get_head(
            head,
            in_dim=feat_dim,
            birads_classes=birads_classes,
            density_classes=density_classes,
            **head_kwargs,
        )

    def forward(self, x: torch.Tensor):
        # (B, 1, H, W) -> (B, F)
        feats = self.backbone(x)

        # (B, F) -> {breast_birads: {logits: (B, 5)}, breast_density: {logits: (B, 4)}}
        return self.head(feats)
