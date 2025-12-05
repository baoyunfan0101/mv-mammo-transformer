# src/models/backbones/base.py

from __future__ import annotations
from typing import List

import torch.nn as nn

__all__ = ["BackboneBase"]


class BackboneBase(nn.Module):

    def __init__(self):
        super().__init__()

    @property
    def blocks(self) -> List[nn.Module]:
        raise NotImplementedError(
            "[BackboneBase] Subclasses must implement `.blocks` to define freeze units."
        )

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_until(self, n: int):
        for i, block in enumerate(self.blocks):
            if i < n:
                for p in block.parameters():
                    p.requires_grad = False

    def unfreeze_from(self, n: int):
        for i, block in enumerate(self.blocks):
            if i >= n:
                for p in block.parameters():
                    p.requires_grad = True

    def freeze_blocks(self, idxs: List[int]):
        for i, block in enumerate(self.blocks):
            if i in idxs:
                for p in block.parameters():
                    p.requires_grad = False
