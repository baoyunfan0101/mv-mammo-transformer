# src/utils/device_utils.py

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Union

import torch

from src.utils.log_utils import log

__all__ = ["get_device", "move_to_device"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        log("Using Apple MPS", "Device")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        log("Using CUDA", "Device")
    else:
        dev = torch.device("cpu")
        log("Using CPU", "Device")

    return dev


def move_to_device(
        obj: Any,
        device: Union[str, torch.device]
) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, Dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, (List, Tuple)):
        converted = [move_to_device(v, device) for v in obj]
        return type(obj)(converted)

    return obj
