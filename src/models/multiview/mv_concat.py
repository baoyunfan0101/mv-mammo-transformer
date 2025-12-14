# src/models/multiview/mv_concat.py

from __future__ import annotations
from typing import List, Dict, Optional

import torch
import torch.nn as nn

from src.models import register_model
from src.models.backbones import get_backbone
from src.models.heads import get_head

__all__ = ["MVConcatFusion"]


@register_model("mv_concat")
class MVConcatFusion(nn.Module):

    def __init__(
            self,
            backbone: str = "resnet50",
            head: str = "softmax",
            in_chans: int = 1,
            birads_classes: int = 5,
            density_classes: int = 4,
            proj_dim: int | None = None,
            dropout: float = 0.1,
            backbone_kwargs: Optional[Dict] = None,
            head_kwargs: Optional[Dict] = None,
    ):
        super().__init__()

        # Get backbone (shared across 4 views)
        # (B, C, H, W) -> (B, F)
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = get_backbone(
            backbone,
            in_chans=in_chans,
            **backbone_kwargs,
        )
        feat_dim = self.backbone.out_dim

        # Optional projection layer
        if proj_dim is not None:
            self.proj = nn.Linear(feat_dim, proj_dim)
            feat_dim = proj_dim
        else:
            self.proj = None

        # Concatenate 4 views
        fused_dim = feat_dim * 4

        # Randomly assign 10% features to 0
        self.drop = nn.Dropout(dropout)

        # Get head
        # (B, fused_dim) -> (B, Dict)
        head_kwargs = head_kwargs or {}
        self.head = get_head(
            head,
            in_dim=fused_dim,
            birads_classes=birads_classes,
            density_classes=density_classes,
            **head_kwargs,
        )

    def forward(self, images: List[torch.Tensor]):
        # Batch size
        B = images[0].shape[0]

        # List[4 * (B, C, H, W)] -> (4B, C, H, W)
        x = torch.cat(images, dim=0)

        # (4B, C, H, W) -> (4B, F)
        feats = self.backbone(x)

        if self.proj is not None:
            # (4B, F) -> (4B, D)
            feats = self.proj(feats)

        # (4B, D) -> (4, B, D)
        feats = feats.view(4, B, -1)

        # (4, B, D) -> (B, 4D)
        fused = torch.cat(
            [feats[i] for i in range(4)],
            dim=1
        )

        # Randomly assign 10% features to 0
        fused = self.drop(fused)

        # (B, 4D) -> {breast_birads: {logits: (B, 5)}, breast_density: {logits: (B, 4)}}
        return self.head(fused)
