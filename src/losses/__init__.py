# src/losses/__init__.py

from __future__ import annotations
from typing import Callable, Dict

import inspect
import torch

from src.utils.log_utils import log

# Loss Registry
LOSS_REGISTRY: Dict[str, Callable] = {}

__all__ = ["build_weight_tensor"]


def build_weight_tensor(
        weight_dict: Dict[int, float],
        num_classes: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    weight = torch.ones(num_classes, device=device, dtype=dtype)
    for cls, w in weight_dict.items():
        weight[int(cls)] = float(w)
    return weight


def register_loss(name: str):
    def wrap(fn):
        key = name.lower()
        if key in LOSS_REGISTRY:
            raise ValueError(f"[LossRegistry] Duplicate loss name: {name}")
        LOSS_REGISTRY[key] = fn
        return fn

    return wrap


def get_loss(name: str, **kwargs):
    key = name.lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(
            f"[LossFactory] Unknown loss: '{name}'. "
            f"Available losses: {list(LOSS_REGISTRY.keys())}"
        )

    loss_cls = LOSS_REGISTRY[name]

    sig = inspect.signature(loss_cls.__init__)
    valid_params = sig.parameters

    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in valid_params
    }

    log(f"Using loss: {name}", "LossFactory")
    return LOSS_REGISTRY[key](**filtered_kwargs)


# Import losses to ensure registration
from .softmax_loss import *  # noqa
from .evidential_loss import *  # noqa
from .softmax_gradcam_loss import *  # noqa
from .evidential_gradcam_loss import *  # noqa
