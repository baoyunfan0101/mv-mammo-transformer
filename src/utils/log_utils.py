# src/utils/log_utils.py

from __future__ import annotations
from typing import Dict, Any

import sys

from src.utils.config_utils import format_config

__all__ = ["log", "log_section", "log_config"]


# everyday log utility
def log(msg: str, tag: str = "INFO", file=sys.stdout):
    print(f"[{tag}] {msg}", file=file)


# log section header
def log_section(title: str):
    print(f"\n====== {title} ======")


# log config
def log_config(cfg: Dict[str, Any], tag: str = "CONFIG"):
    msg = "\n" + format_config(cfg)
    log(msg, tag)
