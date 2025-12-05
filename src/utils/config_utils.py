# src/utils/config_utils.py

from __future__ import annotations
from typing import List, Dict, Any

__all__ = ["format_config", "apply_overrides"]


# Format a nested dictionary
def format_config(config: Any, indent: int = 0) -> str:
    space = "  " * indent
    out_lines = []

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict):
                out_lines.append(f"{space}{key}:")
                out_lines.append(format_config(value, indent + 1))
            else:
                out_lines.append(f"{space}{key}: {value}")
    elif isinstance(config, list):
        for item in config:
            if isinstance(item, dict):
                out_lines.append(f"{space}-")
                out_lines.append(format_config(item, indent + 1))
            else:
                out_lines.append(f"{space}- {item}")
    else:
        out_lines.append(f"{space}{config}")

    return "\n".join(out_lines)


# Apply overrides to config
def apply_overrides(config: Dict[str, Any], overrides: List[str]):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"[scripts.train] Invalid override format: {item}")
        key, value = item.split("=", 1)

        # Cast string to proper type
        try:
            value = eval(value)
        except:
            raise ValueError(f"[scripts.train] Failed to overrides: {item}")

        # Dot notation
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            # If key does not exist, create a new dict
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    return config
