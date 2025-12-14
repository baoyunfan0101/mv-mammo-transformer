# src/models/heads/softmax_head.py

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn

from src.models.heads import register_head

__all__ = ["MultiTaskSoftmaxHead"]


class SoftmaxHead(nn.Module):

    def __init__(
            self,
            in_dim: int,
            num_classes: int,
            dropout: float = 0.1
    ):
        super().__init__()
        # Randomly assign 10% features to 0
        self.dropout = nn.Dropout(dropout)

        # (B, D) -> (B, num_classes)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Randomly assign 10% features to 0
        z = self.dropout(feats)

        # (B, D) -> (B, num_classes)
        logits = self.fc(z)
        return {
            "logits": logits,
        }


@register_head("softmax")
class MultiTaskSoftmaxHead(nn.Module):

    def __init__(
            self,
            in_dim: int,
            birads_classes: int = 5,
            density_classes: int = 4,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.head_b = SoftmaxHead(in_dim, birads_classes, dropout)
        self.head_d = SoftmaxHead(in_dim, density_classes, dropout)

    def forward(self, feats: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "breast_birads": self.head_b.forward(feats),
            "breast_density": self.head_d.forward(feats),
        }
