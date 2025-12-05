# src/models/backbones/__init__.py

from __future__ import annotations
from typing import Callable, Dict

from src.utils.log_utils import log

# Backbone Registry
BACKBONE_REGISTRY: Dict[str, Callable] = {}


def register_backbone(name: str):
    def wrap(fn):
        key = name.lower()
        if key in BACKBONE_REGISTRY:
            raise ValueError(f"[BackboneRegistry] Duplicate backbone name: {name}")
        BACKBONE_REGISTRY[key] = fn
        return fn

    return wrap


def get_backbone(name: str, **kwargs):
    key = name.lower()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(
            f"[BackboneFactory] Unknown backbone: '{name}'. "
            f"Available backbones: {list(BACKBONE_REGISTRY.keys())}"
        )

    log(f"Using backbone: {name}", "BackboneFactory")
    return BACKBONE_REGISTRY[key](**kwargs)


# Import backbones to ensure registration
from .resnet_backbone import *  # noqa
from .effnet_backbone import *  # noqa
from .swin_backbone import *  #noqa
