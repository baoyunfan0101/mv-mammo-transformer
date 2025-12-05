# src/models/heads/__init__.py

from __future__ import annotations
from typing import Callable, Dict

from src.utils.log_utils import log

# Head Registry
HEAD_REGISTRY: Dict[str, Callable] = {}


def register_head(name: str):
    def wrap(fn):
        key = name.lower()
        if key in HEAD_REGISTRY:
            raise ValueError(f"[HeadRegistry] Duplicate head name: {name}")
        HEAD_REGISTRY[key] = fn
        return fn

    return wrap


def get_head(name: str, **kwargs):
    key = name.lower()
    if key not in HEAD_REGISTRY:
        raise ValueError(
            f"[HeadFactory] Unknown head: '{name}'. "
            f"Available heads: {list(HEAD_REGISTRY.keys())}"
        )

    log(f"Using head: {name}", "HeadFactory")
    return HEAD_REGISTRY[key](**kwargs)


# Import heads to ensure registration
from .softmax_head import MultiTaskSoftmaxHead  # noqa
from .evidential_head import MultiTaskEvidentialHead  # noqa
