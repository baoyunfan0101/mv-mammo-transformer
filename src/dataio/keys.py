# src/dataio/keys.py

from __future__ import annotations
from typing import Tuple

from dataclasses import dataclass

__all__ = [
    "ImageKey",
    "MultiViewKey",
]


@dataclass(frozen=True)
class ImageKey:
    study_id: str
    laterality: str  # "L" / "R"
    view: str  # "CC" / "MLO"


@dataclass(frozen=True)
class MultiViewKey:
    study_id: str

    def views(self) -> Tuple[ImageKey, ImageKey, ImageKey, ImageKey]:
        return (
            ImageKey(self.study_id, "L", "CC"),
            ImageKey(self.study_id, "L", "MLO"),
            ImageKey(self.study_id, "R", "CC"),
            ImageKey(self.study_id, "R", "MLO"),
        )
