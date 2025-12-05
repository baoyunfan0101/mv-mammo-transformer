# src/models/heads/evidential_head.py

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads import register_head

__all__ = ["MultiTaskEvidentialHead"]


class EvidentialHead(nn.Module):

    def __init__(
            self,
            in_dim: int,
            num_classes: int,
            hidden: int = 512,
            dropout: float = 0.1,
    ):
        super().__init__()

        # (B, in_dim) -> (B, 512) -> (B, num_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        # (B, in_dim) -> (B, num_classes)
        logits = self.net(feats)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0

        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
        }


@register_head("evidential")
class MultiTaskEvidentialHead(nn.Module):

    def __init__(
            self,
            in_dim: int,
            birads_classes: int = 5,
            density_classes: int = 4,
            hidden: int = 512,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.head_b = EvidentialHead(in_dim, birads_classes, hidden, dropout)
        self.head_d = EvidentialHead(in_dim, density_classes, hidden, dropout)

    def forward(self, feats: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "breast_birads": self.head_b.forward(feats),
            "breast_density": self.head_d.forward(feats),
        }
