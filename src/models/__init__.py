# src/models/__init__.py

from __future__ import annotations
from typing import Callable, Dict

from src.utils.log_utils import log

# Model registry
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    def wrap(fn):
        key = name.lower()
        if key in MODEL_REGISTRY:
            raise ValueError(f"[ModelRegistry] Duplicate model name: {name}")
        MODEL_REGISTRY[key] = fn
        return fn

    return wrap


def get_model(name: str, **kwargs):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"[ModelFactory] Unknown model: '{name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    log(f"Using model: {name}", "ModelFactory")
    return MODEL_REGISTRY[key](**kwargs)


# get model mode: "single" or "multi"
def get_model_mode(model_name: str) -> str:
    model_name = model_name.lower()
    if model_name.startswith("sv_"):
        return "single"
    elif model_name.startswith("mv_"):
        return "multi"
    else:
        raise ValueError(
            f"[ModelFactory] Unknown model prefix in '{model_name}'. "
            f"Use 'sv_' for single-view or 'mv_' for multi-view models."
        )


# Import models to ensure registration
from src.models.singleview.sv_baseline import SVBaseline  # noqa
from src.models.multiview.mv_concat import MVConcatFusion  # noqa
from src.models.multiview.mv_transformer import MVTransformerFusion  # noqa
