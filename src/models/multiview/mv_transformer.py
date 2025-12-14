# src/models/multiview/mv_transformer.py

from __future__ import annotations
from typing import List, Dict, Optional

import torch
import torch.nn as nn

from src.models import register_model
from src.models.backbones import get_backbone
from src.models.heads import get_head

__all__ = ["MVTransformerFusion"]


@register_model("mv_transformer")
class MVTransformerFusion(nn.Module):

    def __init__(
            self,
            backbone: str = "resnet50",
            head: str = "softmax",
            in_chans: int = 1,
            birads_classes: int = 5,
            density_classes: int = 4,
            token_dim: int = 512,
            nhead: int = 8,
            expansion: int = 4,
            dropout: float = 0.1,
            num_layers: int = 2,
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

        # (B, F) -> (B, D)
        self.proj = nn.Linear(feat_dim, token_dim)

        # Learnable view embedding for 4 views
        # {0: (D,), 1: (D,), 2: (D,), 3: (D,)}
        self.view_embed = nn.Embedding(4, token_dim)

        # Transformer Encoder Layer
        # Self-Attention: (B, 4, D) -> (B, 4, D)
        # FFN: (B, 4, D) -> (B, 4, 4D) -> (B, 4, D)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=token_dim * expansion,
            dropout=dropout,
            batch_first=True,
        )

        # Transformer Encoder Layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Normalize each token
        self.norm = nn.LayerNorm(token_dim)

        # Get head
        # (B, fused_dim) -> (B, Dict)
        head_kwargs = head_kwargs or {}
        self.head = get_head(
            head,
            in_dim=token_dim,
            birads_classes=birads_classes,
            density_classes=density_classes,
            **head_kwargs,
        )

    # Encode 4 views
    def _encode_views(self, images: List[torch.Tensor]) -> torch.Tensor:
        # Batch size
        B = images[0].shape[0]

        # List[4 * (B, C, H, W)] -> (4B, 1, H, W)
        x = torch.cat(images, dim=0)

        # (4B, 1, H, W) -> (4B, F)
        feats = self.backbone(x)

        # (4B, F) -> (4B, D)
        feats = self.proj(feats)

        # (4B, D) -> (4, B, D) -> (B, 4, D)
        tokens = feats.view(4, B, -1).permute(1, 0, 2)

        # embedding -> List[embedding] -> List[B * embedding]
        # (4,) -> (1, 4) -> (B, 4)
        pos_ids = torch.arange(4, device=tokens.device).unsqueeze(0).expand(B, 4)

        # (B, 4, D) + (B, 4, D)
        tokens = tokens + self.view_embed(pos_ids)

        return tokens

    def forward(self, images: List[torch.Tensor]):
        # List[4 * (B, C, H, W)] -> (B, 4, D)
        tokens = self._encode_views(images)

        # (B, 4, D) -> (B, 4, D)
        z = self.encoder(tokens)

        # Normalize each token
        z = self.norm(z)

        # (B, 4, D) -> (B, D)
        fused = z.mean(dim=1)

        # (B, D) -> {breast_birads: {logits: (B, 5)}, breast_density: {logits: (B, 4)}}
        return self.head(fused)
