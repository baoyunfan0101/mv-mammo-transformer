# src/dataio/label_provider.py

from __future__ import annotations
from typing import List, Dict, Any, Optional

from src.data.breast_level import BreastLevel
from src.dataio.keys import ImageKey, MultiViewKey

__all__ = ["LabelProvider"]


class LabelProvider:

    def __init__(
            self,
            breast_level: BreastLevel,
            *,
            columns: List[str],
            mapping: Optional[Dict[str, Dict[Any, int]]] = None,
    ):
        self.breast_level = breast_level
        self.columns = list(columns)
        self.mapping = mapping or {}

    def get_single_label(self, key: ImageKey) -> Dict[str, Any]:
        image_id = self.breast_level.by_study_id.get_image_id(
            key.study_id
        )[(key.laterality, key.view)]

        row = self.breast_level.by_image_id.get(image_id)

        out = {}
        for c in self.columns:
            val = row[c]
            if c in self.mapping:
                val = self.mapping[c][val]
            out[c] = val
        return out

    def get_multi_label(self, key: MultiViewKey) -> Dict[str, Any]:
        rows = self.breast_level.by_study_id.get(key.study_id)

        out = {}
        for row in rows:
            for c in self.columns:
                val = row[c]
                if c in self.mapping:
                    val = self.mapping[c][val]
                out[c] = max(out.get(c, 0), val)
        return out
