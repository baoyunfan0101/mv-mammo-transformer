# src/transforms/__init__.py

from __future__ import annotations
from typing import Callable, Dict

from src.utils.log_utils import log

# Transform Registry
TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def register_transform(name: str):
    def wrap(fn):
        key = name.lower()
        if key in TRANSFORM_REGISTRY:
            raise ValueError(f"[TransformRegistry] Duplicate transform name: {name}")
        TRANSFORM_REGISTRY[key] = fn
        return fn

    return wrap


def get_transform(name: str, **kwargs):
    key = name.lower()
    if key not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"[TransformFactory] Unknown transform: '{name}'. "
            f"Available transforms: {list(TRANSFORM_REGISTRY.keys())}"
        )

    log(f"Using transform: {name}", "TransformFactory")
    return TRANSFORM_REGISTRY[key](**kwargs)


# Import transforms to ensure registration
from .mv_consistent import *  # noqa
from .mv_test import *  # noqa
from .sv_baseline import *  # noqa
from .sv_test import *  # noqa
