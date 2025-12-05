# src/training/forward.py

from __future__ import annotations
from typing import Dict, Any, Mapping

import torch

__all__ = ["forward_batch"]


def forward_batch(
        model: torch.nn.Module,
        batch: Mapping[str, Any]
) -> Dict[str, Any]:
    if "images" in batch:
        return model(batch["images"])

    raise KeyError(
        "[training.forward_batch] Expected one of keys: 'image', 'images', or 'inputs'. "
        f"Got: {list(batch.keys())}"
    )
